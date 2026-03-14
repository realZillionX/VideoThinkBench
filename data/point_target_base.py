"""Shared scaffolding for point-target option puzzles.

These puzzles feature a hidden or implicit key point on the canvas and a fixed
set of labeled candidate markers positioned nearby. Solvers indicate the
correct marker by speaking, writing, or highlighting it in red. Generators and
Evaluators implementing this pattern can derive from the classes here to reuse
candidate placement and scoring logic.
"""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import uuid

import cv2
import numpy as np

from .base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike

from PIL import Image, ImageFont, ImageDraw


def _strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
    if prompt is None:
        return None
    stripped = prompt.strip()
    suffixes = (
        " In portrait, static camera, no zoom, no pan.",
        " In portrait. Static camera.",
        " In portrait. Static Camera. No zoom.",
        " In portrait. Static Camera. No zoom, no pan.",
        " Static camera perspective, no zoom or pan.",
    )
    for suffix in suffixes:
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].strip()
    return stripped


@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]

@dataclass
class PointCandidate:
    """Serializable representation of a labeled candidate point."""

    x: float
    y: float
    label: str

    def to_dict(self) -> Dict[str, object]:
        return {"x": self.x, "y": self.y, "label": self.label}

@dataclass
class PointTargetPuzzleRecord:
    """Base record fields for point-target puzzles."""

    id: str
    ti2v_prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    candidates: List[PointCandidate]
    correct_option: str
    image: str
    solution_image_path: str
    point_radius: int
    line_width: int
    vlm_prompt: Optional[str] = None
    ti2i_prompt: Optional[str] = None
    vlm_answer: Optional[str] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    solution_video_path: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "id": self.id,
            "ti2v_prompt": self.ti2v_prompt,
            "vlm_prompt": self.vlm_prompt,
            "ti2i_prompt": self.ti2i_prompt,
            "vlm_answer": self.vlm_answer,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            # Handle list of dataclass objects
            "candidates": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates],
            "correct_option": self.correct_option,
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "solution_video_path": self.solution_video_path,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
            "point_radius": self.point_radius,
            "line_width": self.line_width,
            "seed": self.seed,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload

class PointTargetPuzzleGenerator(AbstractPuzzleGenerator):
    """Base generator providing canvas configuration and candidate placement."""

    POINT_RADIUS: int = 10
    LINE_WIDTH: int = 5
    CANDIDATE_OUTLINE_COLOR: Tuple[int, int, int] = (32, 32, 32)
    CANDIDATE_HIGHLIGHT_COLOR: Tuple[int, int, int] = (198, 24, 24)
    CANDIDATE_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    CANDIDATE_BASE_FILL: Tuple[int, int, int] = (255, 255, 255)
    CANDIDATE_HIGHLIGHT_FILL: Tuple[int, int, int] = (255, 220, 220)
    CANDIDATE_OUTLINE_WIDTH: int = 4
    CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH: int = 4
    CANDIDATE_LABEL_OFFSET_Y: int = 0
    MAX_VIDEO_FRAMES: int = 193
    DEFAULT_OUTPUT_DIR: str = None
    DEFAULT_TI2V_PROMPT: str = None
    DEFAULT_VLM_PROMPT: str = None
    DEFAULT_TI2I_PROMPT: str = None

    def __init__(
        self,
        output_dir: PathLike,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        option_labels: Sequence[str] = ("A", "B", "C", "D", "E"),
        margin_ratio: float = 0.06,
        record_video: bool = False,
        point_radius: Optional[int] = None,
        line_width: Optional[int] = None,
    ) -> None:
        output_dir = output_dir if output_dir is not None else Path(self.DEFAULT_OUTPUT_DIR)
        resolved_ti2v_prompt = ti2v_prompt if ti2v_prompt is not None else self.DEFAULT_TI2V_PROMPT
        super().__init__(output_dir)
        width = int(canvas_width)
        if width <= 0:
            raise ValueError("canvas_width must be positive")
        if aspect and aspect > 0:
            height = round(width / float(aspect))
        else:
            height = width
        if height <= 0:
            raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions = (width, height)
        margin_base = min(width, height)
        computed_margin = round(margin_base * max(0.0, margin_ratio))
        self.margin = max(18, computed_margin)
        self._rng = random.Random(seed)
        if not option_labels:
            raise ValueError("option_labels must contain at least one label")
        self.option_labels = tuple(option_labels)
        self.seed = seed
        self.ti2v_prompt = resolved_ti2v_prompt or ""
        self.vlm_prompt = self.DEFAULT_VLM_PROMPT
        self.ti2i_prompt = (
            self.DEFAULT_TI2I_PROMPT
            or _strip_video_instruction(self.ti2v_prompt)
            or self.ti2v_prompt
        )
        self.prompt = self.ti2v_prompt
        self._candidate_font: Optional[Any] = None
        self.point_radius = int(point_radius) if point_radius is not None else int(self.POINT_RADIUS)
        if self.point_radius <= 0:
            raise ValueError("point_radius must be positive")
        self.line_width = int(line_width) if line_width is not None else int(self.LINE_WIDTH)
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self._recording_active = False
        self._recorder: Optional[DrawingRecorder] = None

    @staticmethod
    def strip_video_instruction(prompt: Optional[str]) -> Optional[str]:
        return _strip_video_instruction(prompt)

    @property
    def rng(self) -> random.Random:
        return self._rng

    def canvas_bounds(self) -> Tuple[int, int, int, int]:
        width, height = self.canvas_dimensions
        left = self.margin
        top = self.margin
        right = width - self.margin
        bottom = height - self.margin
        return left, top, right, bottom

    def _point_within_canvas_coords(
        self,
        x: float,
        y: float,
        *,
        padding_x: float = 0.0,
        padding_y: Optional[float] = None,
    ) -> bool:
        if padding_y is None:
            padding_y = padding_x
        left, top, right, bottom = self.canvas_bounds()
        return (
            left + padding_x <= x <= right - padding_x
            and top + padding_y <= y <= bottom - padding_y
        )
    
    def inside_canvas(
        self,
        point: Point,
        *,
        padding: float = 0.0,
    ) -> bool:
        x, y = point.to_list()
        return self._point_within_canvas_coords(x, y, padding_x=padding, padding_y=padding)
    
    def distance(
        self,
        p1: Point,
        p2: Point,
    ) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def distance_to_canvas_edge(
        self,
        point: Point,
    ) -> float:
        left, top, right, bottom = self.canvas_bounds()
        return min(
            point.x - left,
            right - point.x,
            point.y - top,
            bottom - point.y,
        )

    def angle_between(
        self,
        angle1: float,
        angle2: float,
    ) -> float:
        return abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)

    def point_on_ray(
        self,
        origin: Point,
        angle: float,
        distance: float,
    ) -> Point:
        return Point(
            x=origin.x + distance * math.cos(angle),
            y=origin.y + distance * math.sin(angle),
        )

    def project_point_onto_line(
        self,
        point: Point,
        line_point1: Point,
        line_point2: Point,
    ) -> Tuple[Point, float]:
        dx = line_point2.x - line_point1.x
        dy = line_point2.y - line_point1.y
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return line_point1, 0.0
        t = (
            (point.x - line_point1.x) * dx
            + (point.y - line_point1.y) * dy
        ) / denom
        projection = Point(
            x=line_point1.x + dx * t,
            y=line_point1.y + dy * t,
        )
        return projection, t

    def distance_point_to_line(
        self,
        point: Point,
        line_point1: Point,
        line_point2: Point,
    ) -> float:
        projection, _ = self.project_point_onto_line(point, line_point1, line_point2)
        return self.distance(point, projection)

    def angle_at_vertex(
        self,
        p1: Point,
        vertex: Point,
        p2: Point,
    ) -> float:
        v1x = p1.x - vertex.x
        v1y = p1.y - vertex.y
        v2x = p2.x - vertex.x
        v2y = p2.y - vertex.y
        mag1 = math.hypot(v1x, v1y)
        mag2 = math.hypot(v2x, v2y)
        if mag1 <= 1e-6 or mag2 <= 1e-6:
            return 0.0
        cosine = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
        return math.degrees(math.acos(cosine))

    def triangle_angles(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> Tuple[float, float, float]:
        return (
            self.angle_at_vertex(p2, p1, p3),
            self.angle_at_vertex(p1, p2, p3),
            self.angle_at_vertex(p1, p3, p2),
        )
    
    @property
    def canvas_short_side(self) -> int:
        width, height = self.canvas_dimensions
        return min(width, height)

    def triangle_area(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> float:
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) / 2.0

    def triangle_side_lengths(
        self,
        p1: Point,
        p2: Point,
        p3: Point,
    ) -> Tuple[float, float, float]:
        return (
            self.distance(p1, p2),
            self.distance(p2, p3),
            self.distance(p3, p1),
        )

    def sample_triangle_vertices(
        self,
        *,
        jitter_ratio: float = 0.8,
        min_side_ratio: float = 0.18,
        min_area_ratio: float = 0.03,
        min_altitude_ratio: float = 0.12,
        min_angle_deg: Optional[float] = None,
        max_angle_deg: Optional[float] = None,
        forbidden_angle_windows: Sequence[Tuple[float, float]] = (),
        max_attempts: int = 999,
        validator: Optional[Callable[[Point, Point, Point], bool]] = None,
    ) -> Tuple[Point, Point, Point]:
        min_side = self.canvas_short_side * min_side_ratio
        min_area = (self.canvas_short_side ** 2) * min_area_ratio
        min_altitude = self.canvas_short_side * min_altitude_ratio
        for _ in range(max_attempts):
            p1 = self.pick_target_point(jitter_ratio)
            p2 = self.pick_target_point(jitter_ratio)
            p3 = self.pick_target_point(jitter_ratio)
            area = self.triangle_area(p1, p2, p3)
            if area < min_area:
                continue
            side_lengths = self.triangle_side_lengths(p1, p2, p3)
            if min(side_lengths) < min_side:
                continue
            longest_side = max(side_lengths)
            if longest_side <= 1e-6:
                continue
            if (2.0 * area / longest_side) < min_altitude:
                continue
            angles = self.triangle_angles(p1, p2, p3)
            if min_angle_deg is not None and min(angles) < min_angle_deg:
                continue
            if max_angle_deg is not None and max(angles) > max_angle_deg:
                continue
            if any(
                lower <= angle <= upper
                for angle in angles
                for lower, upper in forbidden_angle_windows
            ):
                continue
            if validator is not None and not validator(p1, p2, p3):
                continue
            return p1, p2, p3
        raise RuntimeError("Failed to sample a valid triangle configuration")

    def candidate_anchor_padding(
        self,
        *,
        extra: float = 0.0,
    ) -> float:
        pad_x, pad_y = self._candidate_safe_padding()
        return max(pad_x, pad_y) + float(extra)

    def minimum_candidate_spacing(
        self,
        *,
        scale: float = 1.0,
    ) -> float:
        pad_x, pad_y = self._candidate_safe_padding()
        base = max(self.point_radius * 2.8, pad_x * 1.9, pad_y * 1.9)
        return base * max(0.1, scale)

    def point_can_host_candidate(
        self,
        point: Point,
        *,
        extra_padding: float = 0.0,
    ) -> bool:
        padding = self.candidate_anchor_padding(extra=extra_padding)
        return self.inside_canvas(point, padding=padding)

    def points_are_well_spaced(
        self,
        points: Sequence[Point],
        *,
        min_distance: Optional[float] = None,
    ) -> bool:
        threshold = self.minimum_candidate_spacing() if min_distance is None else float(min_distance)
        for idx, point in enumerate(points):
            for other in points[idx + 1 :]:
                if self.distance(point, other) < threshold:
                    return False
        return True

    def validate_candidate_layout(
        self,
        candidates: Sequence[PointCandidate],
        *,
        min_distance: Optional[float] = None,
    ) -> bool:
        if not all(self._candidate_fits(candidate) for candidate in candidates):
            return False
        points = [Point(candidate.x, candidate.y) for candidate in candidates]
        threshold = self.minimum_candidate_spacing(scale=0.82) if min_distance is None else min_distance
        return self.points_are_well_spaced(points, min_distance=threshold)

    def circle_fits(
        self,
        center: Point,
        radius: float,
        *,
        extra_padding: float = 0.0,
    ) -> bool:
        return self._point_within_canvas_coords(
            center.x,
            center.y,
            padding_x=radius + extra_padding,
            padding_y=radius + extra_padding,
        )

    def sample_point_along_direction(
        self,
        origin: Point,
        angle: float,
        *,
        min_distance: float,
        max_distance: Optional[float] = None,
        padding: float = 0.0,
    ) -> Point:
        budget = self._max_travel_distance(origin, angle, padding_x=padding, padding_y=padding)
        if budget < min_distance:
            raise RuntimeError("No feasible distance along direction inside canvas")
        safe_budget = max(0.0, budget * 0.98)
        upper = safe_budget if max_distance is None else min(safe_budget, max_distance)
        if upper < min_distance:
            raise RuntimeError("Requested distance range does not fit inside canvas")
        distance = self._rng.uniform(min_distance, upper)
        return self.point_on_ray(origin, angle, distance)

    def sample_symmetric_segment(
        self,
        midpoint: Point,
        angle: float,
        *,
        min_half_length: float,
        max_half_length: Optional[float] = None,
        padding: float = 0.0,
    ) -> Tuple[Point, Point]:
        forward_budget = self._max_travel_distance(
            midpoint, angle, padding_x=padding, padding_y=padding,
        )
        backward_budget = self._max_travel_distance(
            midpoint, angle + math.pi, padding_x=padding, padding_y=padding,
        )
        symmetric_budget = min(forward_budget, backward_budget)
        if symmetric_budget < min_half_length:
            raise RuntimeError("No feasible symmetric segment fits inside canvas")
        upper = symmetric_budget if max_half_length is None else min(symmetric_budget, max_half_length)
        if upper < min_half_length:
            raise RuntimeError("Requested symmetric segment range does not fit inside canvas")
        half_length = self._rng.uniform(min_half_length, upper * 0.98 if upper > min_half_length else upper)
        return (
            self.point_on_ray(midpoint, angle + math.pi, half_length),
            self.point_on_ray(midpoint, angle, half_length),
        )

    def segment_fits(
        self,
        p1: Point,
        p2: Point,
        *,
        padding: float = 0.0,
    ) -> bool:
        return self.inside_canvas(p1, padding=padding) and self.inside_canvas(p2, padding=padding)

    def clip_line_to_canvas(
        self,
        anchor: Point,
        angle: float,
        *,
        padding: float = 0.0,
    ) -> Tuple[Point, Point]:
        forward = self._max_travel_distance(anchor, angle, padding_x=padding, padding_y=padding)
        backward = self._max_travel_distance(anchor, angle + math.pi, padding_x=padding, padding_y=padding)
        return (
            self.point_on_ray(anchor, angle + math.pi, backward),
            self.point_on_ray(anchor, angle, forward),
        )

    def _candidate_safe_padding(
        self,
        label: Optional[str] = None,
    ) -> Tuple[float, float]:
        font = self._get_candidate_font()
        labels = [label] if label is not None else list(self.option_labels)
        max_text_width = 0
        max_text_height = 0
        for item in labels:
            text_bbox = font.getbbox(item)
            max_text_width = max(max_text_width, text_bbox[2] - text_bbox[0])
            max_text_height = max(max_text_height, text_bbox[3] - text_bbox[1])
        pad_x = max(
            self.point_radius + self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH,
            math.ceil(max_text_width / 2),
        )
        pad_y = max(
            self.point_radius + self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH,
            max_text_height + max(0, -self.CANDIDATE_LABEL_OFFSET_Y),
        )
        return float(pad_x + 1), float(pad_y + 1)

    def _candidate_fits(self, candidate: PointCandidate) -> bool:
        pad_x, pad_y = self._candidate_safe_padding(candidate.label)
        return self._point_within_canvas_coords(
            candidate.x,
            candidate.y,
            padding_x=pad_x,
            padding_y=pad_y,
        )

    def _max_travel_distance(
        self,
        origin: Point,
        angle: float,
        *,
        padding_x: float = 0.0,
        padding_y: Optional[float] = None,
    ) -> float:
        if padding_y is None:
            padding_y = padding_x
        left, top, right, bottom = self.canvas_bounds()
        dx = math.cos(angle)
        dy = math.sin(angle)
        limits: List[float] = []
        if dx > 1e-6:
            limits.append((right - padding_x - origin.x) / dx)
        elif dx < -1e-6:
            limits.append((left + padding_x - origin.x) / dx)
        if dy > 1e-6:
            limits.append((bottom - padding_y - origin.y) / dy)
        elif dy < -1e-6:
            limits.append((top + padding_y - origin.y) / dy)
        positives = [value for value in limits if value >= 0.0]
        return min(positives) if positives else 0.0

    def pick_target_point(
        self,
        jitter_ratio: float = 0.36,
        padding: float = 0.0,
    ) -> Point:
        jitter_ratio/=2 # jitter_ratio = 1 means full spread across the canvas
        left, top, right, bottom = self.canvas_bounds()
        left += padding
        top += padding
        right -= padding
        bottom -= padding
        if left >= right or top >= bottom:
            raise ValueError("padding leaves no drawable area on canvas")
        width, height = right - left, bottom - top
        center_x = left + width * 0.5
        center_y = top + height * 0.5
        jitter_x = self._rng.uniform(-jitter_ratio * width, jitter_ratio * width)
        jitter_y = self._rng.uniform(-jitter_ratio * height, jitter_ratio * height)
        x = center_x + jitter_x
        y = center_y + jitter_y
        return Point(x, y)
    
    def place_candidates_line(self,true_point: Point,angle:float|None=None)->None:
        radius = self.point_radius
        base_x, base_y = true_point.x, true_point.y
        labels = list(self.option_labels)
        target_count = len(labels)
        if target_count == 0:
            raise RuntimeError("option_labels must contain at least one label")
        if angle is None:
            angle = self._rng.uniform(0.0, math.tau)
        pad_x, pad_y = self._candidate_safe_padding()
        if not self._point_within_canvas_coords(base_x, base_y, padding_x=pad_x, padding_y=pad_y):
            raise RuntimeError("Candidate anchor is too close to the canvas boundary")

        default_spread = self.minimum_candidate_spacing()
        min_spread = max(default_spread * 0.92, radius * 1.8)
        forward_budget = self._max_travel_distance(true_point, angle, padding_x=pad_x, padding_y=pad_y)
        backward_budget = self._max_travel_distance(true_point, angle + math.pi, padding_x=pad_x, padding_y=pad_y)

        feasible: List[Tuple[float, int]] = []
        roomy: List[Tuple[float, int]] = []
        for correct_index in range(target_count):
            forward_steps = target_count - 1 - correct_index
            backward_steps = correct_index
            allowed_spread = default_spread
            if forward_steps > 0:
                allowed_spread = min(allowed_spread, forward_budget / forward_steps)
            if backward_steps > 0:
                allowed_spread = min(allowed_spread, backward_budget / backward_steps)
            if allowed_spread >= min_spread:
                feasible.append((allowed_spread, correct_index))
                if allowed_spread >= default_spread:
                    roomy.append((allowed_spread, correct_index))
        if not feasible:
            raise RuntimeError("Failed to fit line candidates within the canvas")

        chosen_allowed_spread, correct_index = self._rng.choice(roomy or feasible)
        spread = default_spread if chosen_allowed_spread >= default_spread else max(min_spread, chosen_allowed_spread * 0.98)
        dx,dy=math.cos(angle)*spread, math.sin(angle)*spread
        correct_label = labels[correct_index]
        candidates: List[PointCandidate] = []
        for i in range(target_count):
            cx = base_x + dx*(i-correct_index)
            cy = base_y + dy*(i-correct_index)
            label = labels[i]
            candidates.append(PointCandidate(x=cx, y=cy, label=label))
        if not self.validate_candidate_layout(candidates, min_distance=min_spread * 0.95):
            raise RuntimeError("Line candidates still exceed the canvas after fitting")
        self.candidates, self.correct_label= candidates, correct_label

    def check_candidates_inside(self)->bool:
        return all(self._candidate_fits(candidate) for candidate in self.candidates)
    
    def place_candidates(
        self,
        true_point: Point,
    ) -> None:
        radius = self.point_radius
        left, top, right, bottom = self.canvas_bounds()
        pad_x, pad_y = self._candidate_safe_padding()
        base_x, base_y = true_point.x, true_point.y
        labels = list(self.option_labels)
        self._rng.shuffle(labels)
        correct_label = labels[0]
        candidates: List[PointCandidate] = []
        candidates.append(PointCandidate(x=base_x, y=base_y, label=correct_label))
        target_count = len(labels)
        max_attempts = 600
        attempt = 0
        spread = self.minimum_candidate_spacing()
        min_candidate_distance = max(radius * 2.6, spread * 0.95)
        while len(candidates) < target_count and attempt < max_attempts:
            attempt += 1
            angle = self._rng.uniform(0.0, math.tau)
            distance = self._rng.uniform(spread * 0.8, spread * 1.8)
            cx = base_x + math.cos(angle) * distance
            cy = base_y + math.sin(angle) * distance
            inside_bounds = self._point_within_canvas_coords(
                cx,
                cy,
                padding_x=pad_x,
                padding_y=pad_y,
            )
            if not inside_bounds:
                continue
            too_close = False
            for existing in candidates:
                if math.hypot(existing.x - cx, existing.y - cy) < min_candidate_distance:
                    too_close = True
                    break
            if too_close:
                continue
            label = labels[len(candidates)]
            candidates.append(PointCandidate(x=cx, y=cy, label=label))
            if attempt == 1 and self._rng.random() < 0.8:
                base_x = cx
                base_y = cy
        if len(candidates) < target_count:
            needed = target_count - len(candidates)
            fallback_offsets = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
                (2, 0),
                (-2, 0),
            ]
            for ox, oy in fallback_offsets:
                if len(candidates) >= target_count:
                    break
                cx = base_x + ox * spread
                cy = base_y + oy * spread
                if not self._point_within_canvas_coords(cx, cy, padding_x=pad_x, padding_y=pad_y):
                    continue
                point = Point(cx, cy)
                if any(self.distance(point, Point(existing.x, existing.y)) < min_candidate_distance for existing in candidates):
                    continue
                label = labels[len(candidates)]
                candidates.append(PointCandidate(x=cx, y=cy, label=label))
            if len(candidates) < target_count:
                raise RuntimeError("Failed to place a non-overlapping candidate layout")
        if not self.validate_candidate_layout(candidates, min_distance=min_candidate_distance * 0.95):
            raise RuntimeError("Candidate layout exceeds canvas or becomes too crowded")
        self.candidates, self.correct_label= candidates, correct_label

    def draw_candidates(
        self,
        draw: Any,
        *,
        highlight_label: Optional[str] = None,
    ) -> None:
        if isinstance(draw, DrawingRecorder):
            draw.add_high_level_command("draw_candidates", highlight_label=highlight_label)
            return

        if ImageDraw is None:
            raise RuntimeError("Pillow is required to draw candidates but is not installed")
        font = self._get_candidate_font()
        active_highlight = highlight_label.upper() if isinstance(highlight_label, str) else None

        point_radius = self.point_radius
        for candidate in sorted(self.candidates, key=lambda c: c.label):
            cx = round(candidate.x)
            cy = round(candidate.y)
            bbox = (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius)
            is_highlight = active_highlight is not None and candidate.label.upper() == active_highlight
            outline = self.CANDIDATE_HIGHLIGHT_COLOR if is_highlight else self.CANDIDATE_OUTLINE_COLOR
            width = self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH if is_highlight else self.CANDIDATE_OUTLINE_WIDTH
            fill = self.CANDIDATE_HIGHLIGHT_FILL if is_highlight else self.CANDIDATE_BASE_FILL
            draw.ellipse(bbox, fill=fill, outline=outline, width=width)
            text_bbox = font.getbbox(candidate.label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            tx = cx - text_width // 2
            ty = cy - text_height + self.CANDIDATE_LABEL_OFFSET_Y
            draw.text((tx, ty), candidate.label, fill=self.CANDIDATE_TEXT_COLOR, font=font)

    def draw_line(self,draw,points:List[Point],width_factor:float=1)->None:
        width = max(1, round(self.line_width * width_factor))
        if isinstance(draw, DrawingRecorder):
            pts = [[round(p.x), round(p.y)] for p in points]
            draw.add_high_level_command("draw_line", points=pts, width_factor=width_factor,
                                        fill=self.CANDIDATE_OUTLINE_COLOR, width=width)
            return

        draw.line(
            [(round(p.x), round(p.y)) for p in points],
            fill=self.CANDIDATE_OUTLINE_COLOR,
            width=width,
        )
        
    def draw_circle(self,draw,center:Point,radius:int)->None:
        if isinstance(draw, DrawingRecorder):
            cx, cy = round(center.x), round(center.y)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.add_high_level_command("draw_circle", bbox=bbox, outline=self.CANDIDATE_OUTLINE_COLOR, width=self.line_width)
            return

        cx,cy=round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, outline=self.CANDIDATE_OUTLINE_COLOR, width=self.line_width)

    def draw_anchor_marker(
        self,
        draw,
        center: Point,
        radius: int,
    ) -> None:
        if isinstance(draw, DrawingRecorder):
            cx, cy = round(center.x), round(center.y)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.add_high_level_command(
                "draw_anchor_marker",
                bbox=bbox,
                fill=self.CANDIDATE_BASE_FILL,
                outline=self.CANDIDATE_OUTLINE_COLOR,
                width=max(2, self.line_width),
            )
            return

        cx, cy = round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(
            bbox,
            fill=self.CANDIDATE_BASE_FILL,
            outline=self.CANDIDATE_OUTLINE_COLOR,
            width=max(2, self.line_width),
        )

    def trim_segment(
        self,
        start: Point,
        end: Point,
        *,
        start_offset: float = 0.0,
        end_offset: float = 0.0,
    ) -> Tuple[Point, Point]:
        dx = end.x - start.x
        dy = end.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return start, end
        ux = dx / length
        uy = dy / length
        trimmed_start = Point(
            x=start.x + ux * max(0.0, start_offset),
            y=start.y + uy * max(0.0, start_offset),
        )
        trimmed_end = Point(
            x=end.x - ux * max(0.0, end_offset),
            y=end.y - uy * max(0.0, end_offset),
        )
        return trimmed_start, trimmed_end

    def _get_candidate_font(self) -> Any:
        if self._candidate_font is None:
            self._candidate_font = ImageFont.load_default(15)
        return self._candidate_font
    
    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        """Hook: draw extra elements on top of animated lines but below candidates.

        Subclasses override this to re-draw static geometry (e.g. anchor circles)
        that must stay visible above animated solution lines in video frames.
        """
        pass

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        raise NotImplementedError("Subclasses must implement _render method")
    
    def get_draw_base(self) -> Tuple[ImageDraw.ImageDraw, Image.Image]:
        width, height = self.canvas_dimensions
        if self._recording_active:
            if self._recorder is None:
                self._recorder = DrawingRecorder(width, height)
            return self._recorder, self._recorder.base_image
            
        base = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base)
        return draw, base

    def build_record_extra(self) -> Dict[str, Any]:
        return {}
    
    def save_puzzle(self) -> PointTargetPuzzleRecord:
        pid = str(uuid.uuid4())
        self.pid=pid
        puzzle_img = self._render(
            highlight_label=None,
        )
        solution_img = self._render(
            highlight_label=self.correct_label,
        )

        self.puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        self.solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(self.puzzle_path)
        solution_img.save(self.solution_path)

        video_rel_path: Optional[str] = None
        video_fps: Optional[int] = None
        video_num_frames: Optional[int] = None
        if self.record_video:
            try:
                video_num_frames = self.save_video_solution(pid)
                video_abs = self.solution_dir / f"{pid}_solution.mp4"
                if video_abs.exists():
                    video_rel_path = self.relativize_path(video_abs)
                    video_fps = 16
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Video generation failed for {pid}: {e}")

        return PointTargetPuzzleRecord(
            id=self.pid,
            ti2v_prompt=self.ti2v_prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            candidates=self.candidates,
            correct_option=self.correct_label,
            image=self.relativize_path(self.puzzle_path),
            solution_image_path=self.relativize_path(self.solution_path),
            point_radius=self.point_radius,
            line_width=self.line_width,
            vlm_prompt=self.vlm_prompt,
            ti2i_prompt=self.ti2i_prompt,
            vlm_answer=self.correct_label,
            seed=self.seed,
            extra=self.build_record_extra(),
            solution_video_path=video_rel_path,
            video_fps=video_fps,
            video_num_frames=video_num_frames,
        )

    def save_video_solution(self, pid: str) -> Optional[int]:
        # Phase 1: Record trace without highlights
        self._recording_active = True
        self._recorder = None # Reset
        self._render(highlight_label=None) # This populates self._recorder
        trace_base = self._recorder.commands if self._recorder else []
        
        # Phase 2: Record trace with highlights
        self._recorder = None # Reset
        self._render(highlight_label=self.correct_label)
        trace_solution = self._recorder.commands if self._recorder else []
        self._recording_active = False

        if not trace_base and not trace_solution:
            # Generator likely doesn't use get_draw_base
            return None

        # Diff commands
        # Use simple queue based diffing against the non-candidate base geometry.
        # Any command in 'trace_solution' that matches the head of 'geometry_cmds' is skipped (part of base).
        # Any command that doesn't match is considered new solution geometry or the final highlighted candidates.
        
        geometry_cmds = [cmd for cmd in trace_base if cmd["type"] != "draw_candidates"]
        base_candidates_cmd = next(
            (
                cmd
                for cmd in trace_base
                if cmd["type"] == "draw_candidates" and cmd.get("highlight_label") is None
            ),
            None,
        )
        base_queue = list(geometry_cmds)
        solution_diff = []
        
        for cmd in trace_solution:
            if base_queue and cmd == base_queue[0]:
                base_queue.pop(0)
            else:
                solution_diff.append(cmd)
        
        solution_steps = []
        final_candidates_cmd = None
        
        for cmd in solution_diff:
            if cmd['type'] == 'draw_candidates':
                final_candidates_cmd = cmd
            else:
                solution_steps.append(cmd)
        
        # Generate Video
        video_path = self.solution_dir / f"{pid}_solution.mp4"
        width, height = self.canvas_dimensions
        fps = 16
        
        base_hold = 8
        end_hold = 16
        step_frames = 12

        estimated_frames = base_hold + len(solution_steps) * step_frames + end_hold
        if estimated_frames > self.MAX_VIDEO_FRAMES:
            available = self.MAX_VIDEO_FRAMES - base_hold - end_hold
            if len(solution_steps) > 0:
                step_frames = max(1, int(available / len(solution_steps)))
            else:
                step_frames = 1

        # Render the recorded commands into an in-memory frame sequence.
        video_renderer = VideoRenderer(width, height, self)

        def candidates_overlay(
            frame: Image.Image,
            highlight: Optional[str] = None,
        ) -> Image.Image:
            out = frame.copy()
            draw = ImageDraw.Draw(out)
            self._video_overlay_extras(draw)
            self.draw_candidates(draw, highlight_label=highlight)
            return out

        def overlay_no_highlight(frame: Image.Image) -> Image.Image:
            return candidates_overlay(frame, highlight=None)
        
        # 1. Base Frame (Static)
        video_renderer.execute_commands(geometry_cmds)
        # Hold base frame for 1 second
        for _ in range(base_hold):
            if base_candidates_cmd is not None:
                video_renderer.add_pil_frame(candidates_overlay(video_renderer.canvas))
            else:
                video_renderer.write_frame()
            
        # 2. Animate Solution Steps
        for cmd in solution_steps:
            video_renderer.animate_command(
                cmd,
                duration_frames=step_frames,
                overlay_callback=overlay_no_highlight if base_candidates_cmd is not None else None,
            )
             
        # 3. Animate Answer (Candidates)
        if final_candidates_cmd:
            highlight = final_candidates_cmd.get("highlight_label")
            for _ in range(end_hold):
                video_renderer.add_pil_frame(candidates_overlay(video_renderer.canvas, highlight=highlight))

        video_renderer.save(video_path)
        if not video_path.exists():
            return None
        return len(video_renderer.frames)

    
    @staticmethod
    def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate point target puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to create")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--canvas-width", type=int, default=512)
        parser.add_argument("--aspect", type=float, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--use-gpt-5", action="store_true", help="Use DEFAULT_VLM_PROMPT defined by the puzzle generator. Will be overridden by --prompt if both are provided.")
        parser.add_argument("--video", action="store_true", help="Generate video solution")
        return parser.parse_args(argv)

    @staticmethod
    def main(cls: PointTargetPuzzleGenerator, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        generator = cls(
            output_dir=args.output_dir,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            seed=args.seed,
            ti2v_prompt=cls.DEFAULT_VLM_PROMPT if args.use_gpt_5 and not args.prompt else args.prompt,
            record_video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")

class DrawingRecorder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.commands = []
        self.base_image = Image.new("RGB", (width, height), (255, 255, 255))
        
    def add_high_level_command(self, type_name, **kwargs):
         self.commands.append({"type": type_name, **kwargs})

    def line(self, xy, fill=None, width=0, joint=None):
        self.commands.append({"type": "line", "xy": xy, "fill": fill, "width": width})
        
    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "ellipse", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align="left", direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False):
        self.commands.append({"type": "text", "xy": xy, "text": text, "fill": fill, "font": font}) # Simplify capture

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "rectangle", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def arc(self, xy, start, end, fill=None, width=1):
        self.commands.append({"type": "arc", "xy": xy, "start": start, "end": end, "fill": fill, "width": width})
        
    # Pillow Draw methods proxy
    def point(self, xy, fill=None): pass
    def polygon(self, xy, fill=None, outline=None): pass
    def chord(self, xy, start, end, fill=None, outline=None, width=1): pass
    def pieslice(self, xy, start, end, fill=None, outline=None, width=1): pass


class VideoRenderer:
    def __init__(self, width, height, generator: PointTargetPuzzleGenerator):
        self.width = width
        self.height = height
        self.generator = generator
        self.frames = []
        # Current state as PIL image
        self.canvas = Image.new("RGB", (width, height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.canvas)
        
    def execute_commands(self, commands):
        for cmd in commands:
            self.execute_command_instant(cmd)
            
    def execute_command_instant(self, cmd):
        t = cmd['type']
        if t == 'draw_candidates':
            self.generator.draw_candidates(self.draw, highlight_label=cmd.get('highlight_label'))
        elif t == 'draw_line':
            # Reconstruct args for draw_line call, but better to just draw it directly here
            # to avoid recursion into recording logic (which shouldn't happen as self.draw is real)
            # cmd keys: points, width_factor, fill, width
             pts = cmd['points']
             # flatten if needed or list of [x,y]
             # PIL line expects [(x,y), (x,y)] or [x,y,x,y]
             flat_list = [tuple(p) for p in pts]
             self.draw.line(flat_list, fill=cmd['fill'], width=cmd['width'])
        elif t == 'draw_circle':
             self.draw.ellipse(cmd['bbox'], outline=cmd['outline'], width=cmd['width'])
        elif t == 'draw_anchor_marker':
             self.draw.ellipse(
                 cmd['bbox'],
                 fill=cmd.get('fill'),
                 outline=cmd.get('outline'),
                 width=cmd.get('width', 1),
             )
        
        # Native PIL commands
        elif t == 'line':
            self.draw.line(cmd['xy'], fill=cmd.get('fill'), width=cmd.get('width', 0))
        elif t == 'ellipse':
            self.draw.ellipse(cmd['xy'], fill=cmd.get('fill'), outline=cmd.get('outline'), width=cmd.get('width', 1))
        # ... other PIL types support if needed for complex animations
    
    def animate_command(
        self,
        cmd,
        duration_frames=16,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        t = cmd['type']
        # Currently only animating lines and circles for smooth effect
        if t == 'draw_line' or t == 'line':
            self.animate_line(cmd, duration_frames, overlay_callback=overlay_callback)
        elif t == 'draw_circle' or t == 'ellipse':
             self.animate_circle(cmd, duration_frames, overlay_callback=overlay_callback)
        else:
            self.execute_command_instant(cmd)
            for _ in range(duration_frames):
                if overlay_callback is None:
                    self.write_frame()
                else:
                    self.add_pil_frame(overlay_callback(self.canvas))

    @staticmethod
    def _count_completed_segments(cumulative_lengths, current_len):
        completed = 0
        while completed < len(cumulative_lengths) and cumulative_lengths[completed] <= current_len + 1e-6:
            completed += 1
        return completed

    def animate_line(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if frames <= 0:
            self.execute_command_instant(cmd)
            return

        # Extract points
        if cmd['type'] == 'draw_line':
            points = [tuple(p) for p in cmd['points']] # [(x,y), (x,y), ...]
            width = cmd['width']
            fill = cmd['fill']
        else:
            xy = cmd['xy']
            # xy can be [x,y, x,y...] or [(x,y), (x,y)...]
            if isinstance(xy[0], (int, float)):
                points = [(xy[i], xy[i + 1]) for i in range(0, len(xy), 2)]
            else:
                points = [(p[0], p[1]) for p in xy]
            width = cmd.get('width', 1)
            fill = cmd.get('fill')

        if len(points) < 2:
            self.execute_command_instant(cmd)
            return

        # Calculate total length
        total_len = 0
        segments = []
        cumulative_lengths = []
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            segments.append((dist, p1, p2))
            total_len += dist
            cumulative_lengths.append(total_len)
        
        if total_len == 0:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        ImageDraw.Draw(final_frame).line(points, fill=fill, width=width)
        revealed_mask = Image.new("L", (self.width, self.height), 0)
        prev_segment = 0
        prev_tip = points[0]
        last_composite = final_frame

        for f in range(frames):
            progress = (f + 1) / frames
            current_len = total_len * progress
            full_segments = self._count_completed_segments(cumulative_lengths, current_len)

            if full_segments >= len(segments):
                current_segment = len(segments) - 1
                current_tip = points[-1]
            else:
                current_segment = full_segments
                prev_len = cumulative_lengths[current_segment - 1] if current_segment > 0 else 0.0
                seg_dist, p1, p2 = segments[current_segment]
                remain = max(0.0, current_len - prev_len)
                if seg_dist > 0 and remain > 0:
                    ratio = min(1.0, remain / seg_dist)
                    current_tip = (
                        round(p1[0] + (p2[0] - p1[0]) * ratio),
                        round(p1[1] + (p2[1] - p1[1]) * ratio),
                    )
                else:
                    current_tip = p1

            mask_draw = ImageDraw.Draw(revealed_mask)
            if current_segment == prev_segment:
                if current_tip != prev_tip:
                    mask_draw.line([prev_tip, current_tip], fill=255, width=width)
            else:
                prev_end = segments[prev_segment][2]
                if prev_tip != prev_end:
                    mask_draw.line([prev_tip, prev_end], fill=255, width=width)
                for seg_idx in range(prev_segment + 1, current_segment):
                    _, seg_p1, seg_p2 = segments[seg_idx]
                    mask_draw.line([seg_p1, seg_p2], fill=255, width=width)
                current_start = segments[current_segment][1]
                if current_tip != current_start:
                    mask_draw.line([current_start, current_tip], fill=255, width=width)

            temp_canvas = Image.composite(final_frame, base_frame, revealed_mask)
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)
            prev_segment = current_segment
            prev_tip = current_tip
        
        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)
        
    def animate_circle(
        self,
        cmd,
        frames,
        overlay_callback: Optional[Callable[[Image.Image], Image.Image]] = None,
    ):
        if frames <= 0:
            self.execute_command_instant(cmd)
            return

        if cmd['type'] == 'draw_circle':
             bbox = cmd['bbox']
             width_px = cmd['width']
             outline = cmd['outline']
        else:
             bbox = cmd['xy'] # [x0, y0, x1, y1]
             width_px = cmd.get('width', 1)
             outline = cmd.get('outline')

        if outline is None:
            self.execute_command_instant(cmd)
            return

        base_frame = self.canvas.copy()
        final_frame = base_frame.copy()
        ImageDraw.Draw(final_frame).ellipse(bbox, outline=outline, width=width_px)
        completed_mask = Image.new("L", (self.width, self.height), 0)
        x0, y0, x1, y1 = bbox
        mask_bbox = (x0 - width_px, y0 - width_px, x1 + width_px, y1 + width_px)
        prev_end_angle = 0.0
        last_composite = final_frame

        for f in range(frames):
            end_angle = 360 * (f + 1) / frames
            frame_mask = completed_mask.copy()
            ImageDraw.Draw(frame_mask).pieslice(
                mask_bbox,
                start=prev_end_angle,
                end=end_angle,
                fill=255,
            )
            temp_canvas = Image.composite(final_frame, base_frame, frame_mask)
            last_composite = temp_canvas
            frame_to_write = temp_canvas if overlay_callback is None else overlay_callback(temp_canvas)
            self.add_pil_frame(frame_to_write)
            completed_mask = frame_mask
            prev_end_angle = end_angle

        self.canvas = last_composite.copy()
        self.draw = ImageDraw.Draw(self.canvas)

    def write_frame(self):
        self.add_pil_frame(self.canvas)
        
    def add_pil_frame(self, pil_img):
        self.frames.append(pil_img.convert("RGB").copy())

    def save(self, path: Path):
        if not self.frames:
            return

        fps = 16
        path.parent.mkdir(parents=True, exist_ok=True)
        target_width = self.width + (self.width % 2)
        target_height = self.height + (self.height % 2)
        rgb_frames: List[np.ndarray] = []
        for frame in self.frames:
            rgb_frame = np.ascontiguousarray(np.array(frame.convert("RGB"), dtype=np.uint8))
            if (rgb_frame.shape[1], rgb_frame.shape[0]) != (self.width, self.height):
                raise ValueError("VideoRenderer frame size does not match renderer dimensions")
            if target_width != self.width or target_height != self.height:
                padded = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
                padded[: self.height, : self.width] = rgb_frame
                rgb_frame = padded
            rgb_frames.append(rgb_frame)

        if shutil.which("ffmpeg") is not None:
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{target_width}x{target_height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                str(path),
            ]
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            except OSError as exc:
                print(f"Error: Failed to launch ffmpeg for {path}: {exc}", flush=True)
                return
            stderr_output = b""
            try:
                for frame in rgb_frames:
                    if proc.stdin is None:
                        raise BrokenPipeError("ffmpeg stdin pipe is unavailable")
                    proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                if proc.stdin is not None:
                    proc.stdin.close()
                if proc.stderr is not None:
                    stderr_output = proc.stderr.read()
                proc.wait()
            else:
                if proc.stdin is not None:
                    proc.stdin.close()
                if proc.stderr is not None:
                    stderr_output = proc.stderr.read()
                proc.wait()
            finally:
                if proc.stderr is not None:
                    proc.stderr.close()

            if proc.returncode == 0 and path.exists():
                return

            error_message = stderr_output.decode("utf-8", errors="replace").strip()
            print(
                f"Error: ffmpeg failed to encode {path}: {error_message or f'return code {proc.returncode}'}",
                flush=True,
            )
            return

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(path), fourcc, float(fps), (target_width, target_height))
        if not out.isOpened():
            out.release()
            print(f"Error: Failed to open OpenCV avc1 writer for {path}", flush=True)
            return

        for frame in rgb_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()



class PointTargetPuzzleEvaluator(AbstractPuzzleEvaluator):
    """Base evaluator utilities for point-target option puzzles."""

    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")

    def image_option_from_path(
        self,
        candidate_image: PathLike,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        candidate_path = Path(candidate_image)
        loaded_frame = cv2.imread(candidate_path.as_posix(), cv2.IMREAD_COLOR)
        if loaded_frame is None:
            return None, 0, None
        rgb_frame = cv2.cvtColor(loaded_frame, cv2.COLOR_BGR2RGB)
        return self.image_option_from_frame(rgb_frame, record)

    def image_option_from_frame(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        return self._score_red_point(frame, record)

    def video_option_from_attempt(
        self,
        attempt_dir: Path,
        record: Dict[str, object],
        sample_stride: int,
    ) -> Optional[str]:
        stride = sample_stride if sample_stride > 0 else 1
        counts: Dict[str, int] = {}
        for video_path in self._iter_video_files(attempt_dir):
            capture = cv2.VideoCapture(video_path.as_posix())
            if not capture.isOpened():
                capture.release()
                continue
            frame_index = 0
            success, frame = capture.read()
            while success:
                if frame_index % stride == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, _, _ = self._score_red_point(rgb_frame, record)
                    if label:
                        key = label.upper()
                        counts[key] = counts.get(key, 0) + 1
                frame_index += 1
                success, frame = capture.read()
            capture.release()
        if not counts:
            return None
        best_count = max(counts.values())
        best_labels = [label for label, count in counts.items() if count == best_count]
        best_labels.sort()
        return best_labels[0]

    def transcript_option_from_attempt(self, attempt_dir: Path) -> Optional[str]:
        transcript_result = self.transcribe_video(attempt_dir)
        value = transcript_result.get("first_nato_word")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped.upper()
        return None

    def text_option_from_attempt(self, attempt_dir: Path) -> Optional[str]:
        text_path = attempt_dir / "content.txt"
        if not text_path.exists() or not text_path.is_file():
            raise FileNotFoundError(f"Text not found: {text_path}")
        text_payload = text_path.read_text(encoding="utf-8")
        return self.extract_first_nato_word(text_payload)

    def _score_red_point(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        height = frame.shape[0]
        width = frame.shape[1]
        canvas_dims_obj = record.get("canvas_dimensions")
        scale = self._extract_scale(canvas_dims_obj, width, height)
        if scale is None:
            return None, 0, None
        scale_x, scale_y = scale
        red_mask = self._red_mask(frame)
        red_pixels = np.column_stack(np.nonzero(red_mask > 0.5))
        candidates_raw = record.get("candidates") or []
        scaled_candidates: List[Dict[str, object]] = []
        for entry in candidates_raw:
            label = entry.get("label")
            x, y = entry.get("x"), entry.get("y")

            cx = float(x) * scale_x
            cy = float(y) * scale_y
            scaled_candidates.append({"label": label, "x": cx, "y": cy})
        # if want to exclude red pixels far from candidates: 
        # threshold = min(height, width) / 5.0
        # if red_pixels.size and scaled_candidates:
        #     coords = red_pixels.astype(np.float32)
        #     candidate_coords = np.array([[c["y"], c["x"]] for c in scaled_candidates], dtype=np.float32)
        #     if candidate_coords.size:
        #         distance_sq = np.sum((coords[:, None, :] - candidate_coords[None, :, :]) ** 2, axis=2)
        #         mask = np.min(distance_sq, axis=1) <= (threshold * threshold)
        #         red_pixels = red_pixels[mask]
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None
        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        red_point = (mean_x, mean_y)
        # print(f"Detected {red_count} red pixels, centroid at ({mean_x:.1f}, {mean_y:.1f})")
        
        best_label: Optional[str] = None
        best_distance: float = math.inf
        for scaled in scaled_candidates:
            label, cx, cy = scaled['label'], scaled['x'], scaled['y']
            distance = math.hypot(cx - mean_x, cy - mean_y)
            # print(f"Candidate {label}: position ({cx:.1f}, {cy:.1f}), distance {distance:.1f}")

            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label, red_count, red_point

    def _extract_scale(
        self,
        canvas_dims_obj: object,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        if isinstance(canvas_dims_obj, (list, tuple)) and len(canvas_dims_obj) >= 2:
            raw_width = canvas_dims_obj[0]
            raw_height = canvas_dims_obj[1]
        elif isinstance(canvas_dims_obj, dict) and {"width", "height"} <= set(canvas_dims_obj):
            raw_width = canvas_dims_obj["width"]
            raw_height = canvas_dims_obj["height"]
        else:
            return None
        if not isinstance(raw_width, (int, float)) or not isinstance(raw_height, (int, float)):
            return None
        canvas_width = float(raw_width)
        canvas_height = float(raw_height)
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        return scale_x, scale_y

    def _iter_video_files(self, attempt_dir: Path) -> List[Path]:
        seen = set()
        videos: List[Path] = []
        for pattern in self.VIDEO_GLOBS:
            for candidate in attempt_dir.glob(pattern):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    videos.append(candidate)
        videos.sort(key=lambda path: path.name)
        return videos

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        blue = frame[:, :, 2].astype(np.float32)
        dominance = red - np.maximum(green, blue)
        mask = (
            (red >= 140.0) &
            (dominance >= 40.0) &
            (green <= 130.0) &
            (blue <= 130.0)
        )
        return mask.astype(np.float32)

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        video_sample_stride: int = 5,
    ) -> AbstractPuzzleEvaluator.OptionEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper()
        if not correct or len(correct) != 1:
            raise ValueError("Puzzle record missing valid 'correct_option' (single letter)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent

        transcript_option = self.transcript_option_from_attempt(attempt_dir)
        text_option = self.text_option_from_attempt(attempt_dir)
        video_option = self.video_option_from_attempt(attempt_dir, record, video_sample_stride)
        image_option, red_pixel_count, red_centroid = self.image_option_from_path(candidate_path, record)

        result = AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            video_option=video_option,
            image_option=image_option,
            text_option=text_option,
            attempt_dir=attempt_dir.as_posix(),
        )
        result.red_pixel_count = red_pixel_count
        result.red_centroid = red_centroid
        return result
    
    @staticmethod
    def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Evaluate point target puzzles")
        parser.add_argument("metadata", type=Path)
        parser.add_argument("puzzle_id", type=str)
        parser.add_argument("candidate", type=Path)
        parser.add_argument("--base-dir", type=Path, default=None)
        parser.add_argument("--video-stride", dest="video_sample_stride", type=int, default=5)
        return parser.parse_args(argv)


    @staticmethod
    def main(argv: Optional[list[str]] = None) -> None:
        args = PointTargetPuzzleEvaluator._parse_args(argv)
        evaluator = PointTargetPuzzleEvaluator(args.metadata, base_dir=args.base_dir)
        result = evaluator.evaluate(
            args.puzzle_id,
            args.candidate,
            video_sample_stride=args.video_sample_stride,
        )
        print(json.dumps(result.to_dict(), indent=2))

__all__ = [
    "PointCandidate",
    "PointTargetPuzzleGenerator",
    "PointTargetPuzzleEvaluator",
]
