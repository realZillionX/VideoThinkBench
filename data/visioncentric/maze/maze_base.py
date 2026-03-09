"""Shared scaffolding for maze-style puzzle generators and evaluation.

These base classes implement reusable logic for maze puzzles where a solver must
trace a red path from a designated start to a goal while avoiding walls drawn in
black. Subclasses can focus on maze layout generation and rendering while
inheriting dataset serialization, CLI wiring, and pixel-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import uuid
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from data.base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike


def _strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    stripped = prompt.strip()
    suffixes = (
        " In portrait, static camera, no zoom, no pan.",
        " In portrait. Static camera.",
        " In portrait. Static Camera. No zoom.",
        " In portrait. Static Camera. No zoom, no pan.",
    )
    for suffix in suffixes:
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].strip()
    return stripped


def _draw_path_segment_rect(
    draw: ImageDraw.ImageDraw,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: Any,
    thickness: int,
) -> None:
    """Draw a path segment as a filled rectangle."""
    x1, y1 = p1
    x2, y2 = p2
    half = thickness / 2

    if math.isclose(x1, x2, abs_tol=1e-6) or math.isclose(y1, y2, abs_tol=1e-6):
        left = min(x1, x2) - half
        top = min(y1, y2) - half
        right = max(x1, x2) + half
        bottom = max(y1, y2) + half
        draw.rectangle((left, top, right, bottom), fill=color)
        return

    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-6:
        draw.rectangle((x1 - half, y1 - half, x1 + half, y1 + half), fill=color)
        return

    # Other maze variants in this shared base use angled paths, so keep them
    # artifact-free with an oriented rectangle instead of a stroked line.
    ux = dx / seg_len
    uy = dy / seg_len
    px = -uy * half
    py = ux * half
    start_x = x1 - ux * half
    start_y = y1 - uy * half
    end_x = x2 + ux * half
    end_y = y2 + uy * half
    draw.polygon(
        (
            (start_x + px, start_y + py),
            (start_x - px, start_y - py),
            (end_x - px, end_y - py),
            (end_x + px, end_y + py),
        ),
        fill=color,
    )


def draw_path_line(
    image: Image.Image,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draws a path (solution line) on the given image."""
    draw = ImageDraw.Draw(image)
    if len(points) >= 2:
        for i in range(len(points) - 1):
            _draw_path_segment_rect(draw, points[i], points[i + 1], color, thickness)
    elif len(points) == 1:
        _draw_path_segment_rect(draw, points[0], points[0], color, thickness)

@dataclass
class MazePuzzleRecord:
    """Serializable metadata for a maze puzzle asset pair."""

    id: str
    ti2v_prompt: str
    vlm_prompt: str
    ti2i_prompt: str
    canvas_dimensions: Tuple[int, int]
    start_point: Tuple[float, float]
    goal_point: Tuple[float, float]
    image: str
    solution_image_path: str
    vlm_answer: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    solution_video_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "ti2v_prompt": self.ti2v_prompt,
            "vlm_prompt": self.vlm_prompt,
            "ti2i_prompt": self.ti2i_prompt,
            "vlm_answer": self.vlm_answer,
            "canvas_dimensions": [int(self.canvas_dimensions[0]), int(self.canvas_dimensions[1])],
            "start_point": [float(self.start_point[0]), float(self.start_point[1])],
            "goal_point": [float(self.goal_point[0]), float(self.goal_point[1])],
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


class MazePuzzleGenerator(AbstractPuzzleGenerator[MazePuzzleRecord]):
    """Base generator providing canvas configuration and asset management."""

    DEFAULT_OUTPUT_DIR: Optional[PathLike] = "data/visioncentric/maze"
    DEFAULT_TI2V_PROMPT: Optional[str] = "Draw a red path connecting two red dots without touching the black walls. In portrait. Static camera."
    DEFAULT_VLM_PROMPT: Optional[str] = "Find a path connecting the two red dots without touching the black walls in the maze. Each traversable region has its ID printed on it. Present your answer as a list of IDs. Example: [1, 4, 3, 2]. Must answer now without asking for clarifications."
    DEFAULT_TI2I_PROMPT: Optional[str] = _strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        size: int = 32,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        if resolved_output is None:
            raise ValueError("output_dir must be provided either via argument or DEFAULT_OUTPUT_DIR")
        super().__init__(resolved_output)

        if canvas_width <= 0:
            raise ValueError("canvas_width must be positive")
        width = int(canvas_width)
        if aspect is not None and aspect <= 0:
            raise ValueError("aspect must be positive when provided")
        if aspect is None:
            height = width
        else:
            height = int(round(width / float(aspect)))
            if height <= 0:
                raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions: Tuple[int, int] = (width, height)

        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        resolved_ti2v_prompt = ti2v_prompt if ti2v_prompt is not None else self.DEFAULT_TI2V_PROMPT
        self.seed = seed
        self.ti2v_prompt = resolved_ti2v_prompt or ""
        self.vlm_prompt = self.DEFAULT_VLM_PROMPT or self.ti2v_prompt
        self.ti2i_prompt = (
            self.DEFAULT_TI2I_PROMPT
            or _strip_video_instruction(self.ti2v_prompt)
            or self.ti2v_prompt
        )
        self.prompt = self.ti2v_prompt
        self.show_cell_id = show_cell_id
        self.video = video
        self._rng = random.Random(seed)

        root = Path(self.output_dir)
        self.puzzle_dir = root / "puzzles"
        self.solution_dir = root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def canvas_width(self) -> int:
        return self.canvas_dimensions[0]

    @property
    def canvas_height(self) -> int:
        return self.canvas_dimensions[1]

    def next_id(self) -> str:
        return str(uuid.uuid4())

    @staticmethod
    def strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
        return _strip_video_instruction(prompt)

    def save_images(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        solution_image: Image.Image,
    ) -> Tuple[Path, Path]:
        puzzle_path = self.puzzle_dir / f"{record_id}_puzzle.png"
        solution_path = self.solution_dir / f"{record_id}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        return puzzle_path, solution_path

    def save_video(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        points: List[Tuple[float, float]],
        thickness: int = 5,
        color: Tuple[int, int, int] = (220, 30, 30),
        fps: int = 30,
        duration: float = 6.4,
    ) -> Optional[Path]:
        """Generates a solution video if video output is enabled."""
        if not self.video:
            return None
        
        try:
            import cv2
        except ImportError:
            print("Warning: opencv-python not installed, skipping video generation")
            return None

        video_path = self.solution_dir / f"{record_id}_solution.mp4"
        width, height = puzzle_image.size
        # Video codecs require even dimensions; pad the image if odd
        padded = False
        vid_w = width + (width % 2)
        vid_h = height + (height % 2)
        if vid_w != width or vid_h != height:
            padded_img = Image.new("RGB", (vid_w, vid_h), (0, 0, 0))
            padded_img.paste(puzzle_image, (0, 0))
            puzzle_image = padded_img
            width, height = vid_w, vid_h
            padded = True
        
        # Codec fallback chain: avc1 (H.264) → vp09 (VP9) → mp4v (MPEG-4)
        out = None
        for codec in ('avc1', 'vp09', 'mp4v'):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if out.isOpened():
                break
            out.release()
            out = None
        
        if out is None or not out.isOpened():
            print(f"Warning: Could not open video writer for {video_path} (tried avc1/vp09/mp4v)")
            return None

        n_points = len(points)
        if n_points < 2:
            # Just write static frame
            frame_np = np.array(puzzle_image)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            for _ in range(fps):
                out.write(frame_bgr)
            out.release()
            return video_path

        # Determine reasonable duration, capped at 10s
        eff_duration = min(duration, 10.0)
        total_frames = int(fps * eff_duration)
        
        # Calculate segments and total length
        segments = []
        total_len = 0.0
        cumulative_lengths = []
        for i in range(n_points - 1):
            p1 = points[i]
            p2 = points[i+1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            segments.append((dist, p1, p2))
            total_len += dist
            cumulative_lengths.append(total_len)

        full_frame = puzzle_image.copy()
        full_draw = ImageDraw.Draw(full_frame)
        for i in range(len(points) - 1):
            _draw_path_segment_rect(full_draw, points[i], points[i + 1], color, thickness)

        revealed_mask = Image.new("L", (width, height), 0)
        _draw_path_segment_rect(ImageDraw.Draw(revealed_mask), points[0], points[0], 255, thickness)
        prev_segment = 0
        prev_tip: Tuple[float, float] = points[0]

        # Generate frames
        for f in range(total_frames + 1):
            progress = f / total_frames if total_frames > 0 else 1.0
            cur_dist = total_len * progress

            full_segments = 0
            while full_segments < len(cumulative_lengths) and cumulative_lengths[full_segments] <= cur_dist + 1e-6:
                full_segments += 1

            if full_segments >= len(segments):
                current_segment = len(segments) - 1
                visible_tip = points[-1]
            else:
                current_segment = full_segments
                prev_len = cumulative_lengths[current_segment - 1] if current_segment > 0 else 0.0
                seg_dist, p1, p2 = segments[current_segment]
                remain = max(0.0, cur_dist - prev_len)
                if seg_dist > 0 and remain > 0:
                    ratio = min(1.0, remain / seg_dist)
                    visible_tip = (
                        round(p1[0] + (p2[0] - p1[0]) * ratio),
                        round(p1[1] + (p2[1] - p1[1]) * ratio),
                    )
                else:
                    visible_tip = p1

            revealed_draw = ImageDraw.Draw(revealed_mask)
            if current_segment == prev_segment:
                if visible_tip != prev_tip:
                    _draw_path_segment_rect(revealed_draw, prev_tip, visible_tip, 255, thickness)
            else:
                prev_end = segments[prev_segment][2]
                if prev_end != prev_tip:
                    _draw_path_segment_rect(revealed_draw, prev_tip, prev_end, 255, thickness)
                for seg_idx in range(prev_segment + 1, current_segment):
                    _, seg_p1, seg_p2 = segments[seg_idx]
                    _draw_path_segment_rect(revealed_draw, seg_p1, seg_p2, 255, thickness)
                current_start = segments[current_segment][1]
                if visible_tip != current_start:
                    _draw_path_segment_rect(revealed_draw, current_start, visible_tip, 255, thickness)

            frame_mask = revealed_mask.copy()
            mask_draw = ImageDraw.Draw(frame_mask)
            _draw_path_segment_rect(mask_draw, visible_tip, visible_tip, 255, thickness)

            frame_img = Image.composite(full_frame, puzzle_image, frame_mask)

            frame_np = np.array(frame_img)
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            prev_segment = current_segment
            prev_tip = visible_tip
            
        # Hold end
        for _ in range(int(fps * 1.0)):
            out.write(frame_bgr)

        out.release()
        return video_path

    @abstractmethod
    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        """Return the pixel center for a cell id."""

    def build_record(
        self,
        record_id: str,
        *,
        start_point: Tuple[float, float],
        goal_point: Tuple[float, float],
        puzzle_path: Path,
        solution_path: Path,
        ti2v_prompt: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        video_path: Optional[Path] = None,
    ) -> MazePuzzleRecord:
        record_ti2v_prompt = ti2v_prompt if ti2v_prompt is not None else self.ti2v_prompt
        extra_payload = extra if extra is not None else {}
        video_rel: Optional[str] = None
        if video_path is not None:
            video_rel = self.relativize_path(video_path)
        # Derive vlm_answer from solution path cell IDs
        vlm_answer: Optional[str] = None
        path_ids = extra_payload.get("solution_path_cell_ids")
        if isinstance(path_ids, list):
            vlm_answer = str(path_ids)
        return MazePuzzleRecord(
            id=record_id,
            ti2v_prompt=record_ti2v_prompt,
            vlm_prompt=self.vlm_prompt,
            ti2i_prompt=_strip_video_instruction(record_ti2v_prompt) or self.ti2i_prompt,
            canvas_dimensions=self.canvas_dimensions,
            start_point=start_point,
            goal_point=goal_point,
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            vlm_answer=vlm_answer,
            extra=extra_payload,
            solution_video_path=video_rel,
        )

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate maze puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to create")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--canvas-width", type=int, default=550)
        parser.add_argument("--aspect", type=float, default=0.55)
        parser.add_argument("--size", type=int, default=32, help="Primary maze sizing parameter (e.g., cell size or radius)")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--show-cell-id", action="store_true", help="Draw cell IDs on the maze")
        parser.add_argument("--use-gpt-5", action="store_true", help="Use DEFAULT_VLM_PROMPT and show cell IDs for VLM-style maze solving.")
        parser.add_argument("--video", action="store_true", help="Generate solution video")
        namespace=parser.parse_args(argv)
        if namespace.use_gpt_5:
            namespace.show_cell_id = True
        return namespace

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        prompt_arg = args.prompt if args.prompt is not None else (
            cls.DEFAULT_VLM_PROMPT if args.use_gpt_5 else cls.DEFAULT_TI2V_PROMPT
        )
        output_arg = args.output_dir if args.output_dir is not None else cls.DEFAULT_OUTPUT_DIR
        generator = cls(
            output_dir=output_arg,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            size=args.size,
            seed=args.seed,
            ti2v_prompt=prompt_arg,
            show_cell_id=args.show_cell_id,
            video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


@dataclass
class MazeEvaluationResult:
    puzzle_id: str
    red_pixel_count: int
    overlaps_walls: bool
    touches_start: bool
    touches_goal: bool
    connected: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "red_pixel_count": self.red_pixel_count,
            "overlaps_walls": self.overlaps_walls,
            "touches_start": self.touches_start,
            "touches_goal": self.touches_goal,
            "connected": self.connected,
            "message": self.message,
        }


class MazePuzzleEvaluator(AbstractPuzzleEvaluator):
    """Pixel-based evaluation for maze puzzles."""

    RED_THRESHOLD: int = 150
    RED_DOMINANCE: int = 70
    WALL_VALUE_THRESHOLD: int = 40
    ENDPOINT_SEARCH_RADIUS: int = 4

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
    ) -> MazeEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        
        if not candidate_path.exists():
            self._attempt_reconstruction(record, candidate_path)

        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        source_path = self.resolve_path(record["image"])
        if not source_path.exists():
            raise FileNotFoundError(f"Puzzle image not found: {source_path}")

        with Image.open(source_path) as src_image:
            puzzle_image = src_image.convert("RGB")
        source_canvas_size = puzzle_image.size
        with Image.open(candidate_path) as cand_image:
            candidate_image_rgb = cand_image.convert("RGB")

        if candidate_image_rgb.size != puzzle_image.size:
            puzzle_image = puzzle_image.resize(candidate_image_rgb.size, Image.NEAREST)

        puzzle_pixels = np.asarray(puzzle_image, dtype=np.uint8)
        candidate_pixels = np.asarray(candidate_image_rgb, dtype=np.uint8)

        red_mask = self._red_mask(candidate_pixels)
        red_pixel_count = int(red_mask.sum())
        if red_pixel_count == 0:
            self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
            self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)
            return MazeEvaluationResult(
                puzzle_id=puzzle_id,
                red_pixel_count=0,
                overlaps_walls=False,
                touches_start=False,
                touches_goal=False,
                connected=False,
                message="No red path detected.",
            )

        start_point = self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
        goal_point = self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)

        wall_mask = self._wall_mask(puzzle_pixels)
        safe_radius = self._endpoint_safe_radius(record, source_canvas_size, candidate_image_rgb.size)
        wall_mask = self._suppress_endpoint_walls(wall_mask, [start_point, goal_point], safe_radius)

        # Exempt the red endpoint marker regions drawn on puzzle images.
        marker_bboxes = self._endpoint_marker_bboxes(puzzle_pixels, [start_point, goal_point], safe_radius)
        if marker_bboxes:
            marker_margin = max(1, min(4, int(round(safe_radius * 0.25))))
            wall_mask = self._suppress_endpoint_bboxes(wall_mask, marker_bboxes, marker_margin)

        endpoint_bboxes = self._endpoint_bboxes(record, source_canvas_size, candidate_image_rgb.size)
        if endpoint_bboxes:
            bbox_margin = max(2, min(6, int(round(safe_radius * 0.5))))
            wall_mask = self._suppress_endpoint_bboxes(wall_mask, endpoint_bboxes, bbox_margin)
        overlaps_walls = bool(np.any(red_mask & wall_mask))

        endpoint_search_radius = max(self.ENDPOINT_SEARCH_RADIUS, safe_radius)
        start_seed = self._nearest_red(red_mask, start_point, radius=endpoint_search_radius)
        goal_seed = self._nearest_red(red_mask, goal_point, radius=endpoint_search_radius)
        touches_start = start_seed is not None
        touches_goal = goal_seed is not None

        connected = False
        if touches_start and touches_goal and not overlaps_walls:
            connected = self._connected(red_mask, start_seed, goal_seed)

        if overlaps_walls:
            message = "Red path overlaps walls."
        elif not touches_start:
            message = "Red path does not reach the start."
        elif not touches_goal:
            message = "Red path does not reach the goal."
        elif not connected:
            message = "Red path is not continuous between start and goal."
        else:
            message = "Red path successfully connects start to goal."

        return MazeEvaluationResult(
            puzzle_id=puzzle_id,
            red_pixel_count=red_pixel_count,
            overlaps_walls=overlaps_walls,
            touches_start=touches_start,
            touches_goal=touches_goal,
            connected=connected,
            message=message,
        )

    def _attempt_reconstruction(
        self,
        record: Dict[str, Any],
        candidate_path: Path,
    ) -> None:
        """Attempt to reconstruct a candidate solution image from a content.txt file containing a path of cell IDs."""
        content_path = candidate_path.parent / "content.txt"
        if not content_path.exists():
            return

        try:
            content = content_path.read_text(encoding="utf-8")
        except Exception:
            return

        # Find list pattern: [1, 2, 3]
        matches = re.findall(r"\[([\d,\s]+)\]", content)
        if not matches:
            return
        
        # Use the last match found
        raw_list = matches[-1]
        try:
            path_ids = [int(x.strip()) for x in raw_list.split(",") if x.strip()]
        except ValueError:
            return

        if not path_ids:
            return

        # Load original image as canvas
        source_path = self.resolve_path(record.get("image"))

        try:
            with Image.open(source_path) as src:
                canvas = src.convert("RGB")
        except Exception:
            return
        
        generator = self._build_generator(record)
        points = [generator._cell_center(pid) for pid in path_ids]
        
        if len(points) >= 2:
            # Draw a thick red line
            # Thickness can be estimated from cell size if available, or static
            thickness = 5
            if "cell_size" in record:
                thickness = max(3, int(record["cell_size"]) // 3)
            elif "cell_radius" in record:
                thickness = max(3, int(record["cell_radius"]) // 3)
            elif "ring_width" in record:
                thickness = max(3, int(record["ring_width"]) // 4)
            
            draw_path_line(canvas, points, (220, 30, 30), thickness)
            # Save the reconstructed image
            try:
                canvas.save(candidate_path)
            except Exception:
                pass

    @abstractmethod
    def _build_generator(self, record: Dict[str, Any]) -> MazePuzzleGenerator:
        """Return a generator configured to match the puzzle record."""
        raise NotImplementedError("Subclasses must implement _build_generator")

    def _red_mask(self, pixels: np.ndarray) -> np.ndarray:
        red = pixels[:, :, 0].astype(np.int32)
        green = pixels[:, :, 1].astype(np.int32)
        blue = pixels[:, :, 2].astype(np.int32)
        dominance = red - np.maximum(green, blue)
        mask = (red >= self.RED_THRESHOLD) & (dominance >= self.RED_DOMINANCE)
        return mask

    def _wall_mask(self, pixels: np.ndarray) -> np.ndarray:
        max_channel = pixels.max(axis=2)
        return max_channel <= self.WALL_VALUE_THRESHOLD

    def _endpoint_safe_radius(
        self,
        record: Dict[str, Any],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> int:
        base_radius = self.ENDPOINT_SEARCH_RADIUS
        if "cell_size" in record:
            base_radius = max(base_radius, int(round(float(record["cell_size"]) * 0.45)))
        elif "cell_radius" in record:
            base_radius = max(base_radius, int(round(float(record["cell_radius"]) * 0.8)))
        elif "ring_width" in record:
            base_radius = max(base_radius, int(round(float(record["ring_width"]) * 0.6)))

        src_w, src_h = source_size
        tgt_w, tgt_h = target_size
        if src_w <= 0 or src_h <= 0:
            return base_radius
        scale = min(tgt_w / float(src_w), tgt_h / float(src_h))
        return max(self.ENDPOINT_SEARCH_RADIUS, int(round(base_radius * scale)))

    def _suppress_endpoint_walls(
        self,
        wall_mask: np.ndarray,
        endpoints: Iterable[Tuple[float, float]],
        radius: int,
    ) -> np.ndarray:
        if radius <= 0:
            return wall_mask
        height, width = wall_mask.shape
        y_grid, x_grid = np.ogrid[:height, :width]
        for point in endpoints:
            if point is None:
                continue
            x, y = point
            dist2 = (x_grid - x) ** 2 + (y_grid - y) ** 2
            wall_mask[dist2 <= (radius * radius)] = False
        return wall_mask

    def _suppress_endpoint_bboxes(
        self,
        wall_mask: np.ndarray,
        bboxes: Iterable[Tuple[float, float, float, float]],
        margin: int,
    ) -> np.ndarray:
        if margin < 0:
            margin = 0
        height, width = wall_mask.shape
        for left, top, right, bottom in bboxes:
            x0 = max(0, int(math.floor(left - margin)))
            y0 = max(0, int(math.floor(top - margin)))
            x1 = min(width, int(math.ceil(right + margin)))
            y1 = min(height, int(math.ceil(bottom + margin)))
            if x0 < x1 and y0 < y1:
                wall_mask[y0:y1, x0:x1] = False
        return wall_mask

    def _endpoint_bboxes(
        self,
        record: Dict[str, Any],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> List[Tuple[float, float, float, float]]:
        if "cell_bboxes" not in record:
            return []
        cell_bboxes = record.get("cell_bboxes")
        if not isinstance(cell_bboxes, Iterable):
            return []
        bboxes: List[Tuple[float, float, float, float]] = []
        for key in ("start", "goal"):
            if key not in record:
                continue
            cell = record.get(key)
            if not isinstance(cell, Iterable):
                continue
            cell_list = list(cell)
            if len(cell_list) < 2:
                continue
            row = int(cell_list[0])
            col = int(cell_list[1])
            bbox_rows = list(cell_bboxes)
            if row < 0 or row >= len(bbox_rows):
                continue
            row_data = bbox_rows[row]
            if not isinstance(row_data, Iterable):
                continue
            row_cells = list(row_data)
            if col < 0 or col >= len(row_cells):
                continue
            bbox = row_cells[col]
            if not isinstance(bbox, Iterable):
                continue
            bbox_values = list(bbox)
            if len(bbox_values) < 4:
                continue
            scaled = self._scale_bbox(
                (bbox_values[0], bbox_values[1], bbox_values[2], bbox_values[3]),
                source_size,
                target_size,
            )
            bboxes.append(scaled)
        return bboxes

    def _resolve_endpoint(
        self,
        record: Dict[str, Any],
        label: str,
        puzzle_size: Tuple[int, int],
        candidate_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        pixel_key = f"{label}_pixel"
        if pixel_key in record:
            point = self._coerce_point(record[pixel_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        point_key = f"{label}_point"
        if point_key in record:
            point = self._coerce_point(record[point_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        if label in record and "cell_bboxes" in record:
            cell = record[label]
            if not isinstance(cell, Iterable):
                raise ValueError(f"Record contains invalid '{label}' entry")
            cell_list = list(cell)
            if len(cell_list) < 2:
                raise ValueError(f"Record '{label}' does not contain row and column")
            row = int(cell_list[0])
            col = int(cell_list[1])
            bbox_rows = record["cell_bboxes"]
            if not isinstance(bbox_rows, Iterable):
                raise ValueError("cell_bboxes must be iterable")
            bbox_list = list(bbox_rows)
            if row < 0 or row >= len(bbox_list):
                raise ValueError("start or goal row index out of range")
            row_data = bbox_list[row]
            if not isinstance(row_data, Iterable):
                raise ValueError("cell row is not iterable")
            row_cells = list(row_data)
            if col < 0 or col >= len(row_cells):
                raise ValueError("start or goal column index out of range")
            bbox = row_cells[col]
            if not isinstance(bbox, Iterable):
                raise ValueError("cell bbox is not iterable")
            bbox_values = list(bbox)
            if len(bbox_values) < 4:
                raise ValueError("cell bbox must contain four coordinates")
            left = float(bbox_values[0])
            top = float(bbox_values[1])
            right = float(bbox_values[2])
            bottom = float(bbox_values[3])
            center = ((left + right) * 0.5, (top + bottom) * 0.5)
            return self._scale_point(center, puzzle_size, candidate_size)
        raise KeyError(f"Record is missing endpoint information for '{label}'")

    def _scale_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        left, top, right, bottom = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        p1 = self._scale_point((left, top), source_size, target_size)
        p2 = self._scale_point((right, bottom), source_size, target_size)
        return (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1]))

    def _coerce_point(self, value: Any) -> Tuple[float, float]:
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        raise ValueError("Endpoint entries must contain two numeric values")

    def _scale_point(
        self,
        point: Tuple[float, float],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        source_width, source_height = source_size
        target_width, target_height = target_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source dimensions must be positive")
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        return point[0] * scale_x, point[1] * scale_y

    def _nearest_red(
        self,
        red_mask: np.ndarray,
        point: Tuple[float, float],
        radius: Optional[int] = None,
    ) -> Optional[Tuple[int, int]]:
        height, width = red_mask.shape
        cx = int(round(point[0]))
        cy = int(round(point[1]))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        if red_mask[cy, cx]:
            return (cy, cx)
        max_radius = self.ENDPOINT_SEARCH_RADIUS if radius is None else max(0, int(radius))
        for delta in range(1, max_radius + 1):
            min_x = max(0, cx - delta)
            max_x = min(width - 1, cx + delta)
            min_y = max(0, cy - delta)
            max_y = min(height - 1, cy + delta)
            for y in range(min_y, max_y + 1):
                if red_mask[y, min_x]:
                    return (y, min_x)
                if red_mask[y, max_x]:
                    return (y, max_x)
            for x in range(min_x + 1, max_x):
                if red_mask[min_y, x]:
                    return (min_y, x)
                if red_mask[max_y, x]:
                    return (max_y, x)
        return None

    def _component_from_seed(
        self,
        mask: np.ndarray,
        seed: Tuple[int, int],
    ) -> np.ndarray:
        height, width = mask.shape
        component = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([seed])
        component[seed[0], seed[1]] = True
        while queue:
            y, x = queue.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if mask[ny, nx] and not component[ny, nx]:
                        component[ny, nx] = True
                        queue.append((ny, nx))
        return component

    def _endpoint_marker_bboxes(
        self,
        puzzle_pixels: np.ndarray,
        endpoints: Iterable[Tuple[float, float]],
        safe_radius: int,
    ) -> List[Tuple[float, float, float, float]]:
        puzzle_red_mask = self._red_mask(puzzle_pixels)
        if not puzzle_red_mask.any():
            return []

        height, width = puzzle_red_mask.shape
        padding = max(2, min(8, int(round(safe_radius * 0.5))))
        bboxes: List[Tuple[float, float, float, float]] = []
        for endpoint in endpoints:
            if endpoint is None:
                continue
            seed = self._nearest_red(
                puzzle_red_mask,
                endpoint,
                radius=max(self.ENDPOINT_SEARCH_RADIUS, safe_radius * 2),
            )
            if seed is None:
                continue
            component = self._component_from_seed(puzzle_red_mask, seed)
            ys, xs = np.nonzero(component)
            if ys.size == 0 or xs.size == 0:
                continue
            left = max(0, int(xs.min()) - padding)
            top = max(0, int(ys.min()) - padding)
            right = min(width, int(xs.max()) + padding + 1)
            bottom = min(height, int(ys.max()) + padding + 1)
            bboxes.append((float(left), float(top), float(right), float(bottom)))
        return bboxes

    def _connected(
        self,
        red_mask: np.ndarray,
        start_seed: Tuple[int, int],
        goal_seed: Tuple[int, int],
    ) -> bool:
        height, width = red_mask.shape
        visited = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([start_seed])
        visited[start_seed[0], start_seed[1]] = True
        while queue:
            y, x = queue.popleft()
            if (y, x) == goal_seed:
                return True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if red_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        return False

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Evaluate maze puzzles")
        parser.add_argument("metadata", type=Path, help="Path to maze metadata JSON")
        parser.add_argument("puzzle_id", type=str, help="Identifier of the puzzle to evaluate")
        parser.add_argument("candidate", type=Path, help="Candidate solution image path")
        parser.add_argument("--base-dir", type=Path, default=None)
        return parser.parse_args(argv)

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        evaluator = cls(args.metadata, base_dir=args.base_dir)
        result = evaluator.evaluate(args.puzzle_id, args.candidate)
        print(json.dumps(result.to_dict(), indent=2))


__all__ = [
    "MazePuzzleRecord",
    "MazePuzzleGenerator",
    "MazeEvaluationResult",
    "MazePuzzleEvaluator",
]
