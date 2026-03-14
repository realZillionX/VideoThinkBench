"""Circular labyrinth maze generator for radial path puzzles."""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from data.video_encoding import encode_rgb_frames_to_mp4
from data.visioncentric.maze.maze_base import MazePuzzleGenerator, MazePuzzleRecord

# Colors used for rendering.
PATH_COLOR = (240, 240, 240)
WALL_COLOR = (0, 0, 0)
START_COLOR = (220, 30, 30)
GOAL_COLOR = START_COLOR # (40, 180, 80)
LINE_COLOR = (220, 30, 30)
BACKGROUND_COLOR = (16, 16, 16)
TEXT_COLOR = (0, 0, 255)

Cell = Tuple[int, int]


class MazeLabyrinthGenerator(MazePuzzleGenerator):
    """Generate mazes arranged on concentric rings with angular segments."""

    DEFAULT_OUTPUT_DIR = "data/visioncentric/maze/maze_labyrinth"
    DEFAULT_TI2V_PROMPT = (
        "A square-format video shows a circular labyrinth on a very dark charcoal background. The maze is made of pale "
        "gray concentric ring corridors and a pale gray central disk, separated by thick black circular walls and black "
        "radial spokes, with a solid red start dot on the outer ring and a solid red goal dot at the center. The animation "
        "begins with the static labyrinth, then a thick bright red path is progressively drawn through the centers of "
        "connected ring segments and radial passages without crossing any black wall. After the line reaches the center, the "
        "completed route holds for one second. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT = (
        "A circular labyrinth is shown on a dark background with light gray ring-shaped cells, black radial and circular "
        "walls, a red start dot on the outer ring, a red goal dot in the center, and blue cell ID numbers. Find a valid "
        "path from the outer start cell to the center goal cell by moving only between adjacent labyrinth cells through open "
        "passages. Return the answer as a Python-style list of cell IDs, for example [12, 25, 26, 9, 0]."
    )
    DEFAULT_TI2I_PROMPT = MazePuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    DEFAULT_RINGS = 3  # Number of rings excluding the central cell.
    DEFAULT_SEGMENTS = 8
    DEFAULT_RING_WIDTH = 42

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        rings: int = DEFAULT_RINGS,
        segments: int = DEFAULT_SEGMENTS,
        ring_width: Optional[int] = None,
        wall_thickness: Optional[int] = None,
        size: Optional[int] = None,
        seed: Optional[int] = None,
        ti2v_prompt: Optional[str] = None,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        if rings < 2:
            raise ValueError("rings must be at least 2 to form an interesting maze")
        if segments < 6:
            raise ValueError("segments must be at least 6 for smooth angular resolution")

        self.rings = int(rings)
        self.total_rings = self.rings + 1  # Include central cell as ring 0.
        self.segments = int(segments)

        target_rw = int(ring_width if ring_width is not None else (size if size is not None else self.DEFAULT_RING_WIDTH))
        is_user_set = (ring_width is not None) or (size is not None)

        def _get_layout_params(rw: int) -> Tuple[int, int, int]:
            wt = int(wall_thickness if wall_thickness is not None else max(6, rw // 4))
            io = max(12, wt * 2, rw // 2)
            # max_radius calculation: (rings-1)*spacing + ring_width
            # spacing = rw + wt
            mr = (self.total_rings - 1) * (rw + wt) + rw
            # Canvas size covers diameter + 2*inner_offset (no outer margin)
            sz = int(math.ceil((mr + io) * 2))
            return wt, io, sz

        def _check_fits(rw: int, width_limit: Optional[int] = None) -> bool:
            limit = width_limit if width_limit is not None else canvas_width
            if limit is None:
                return True
            _, _, sz = _get_layout_params(rw)
            w = int(limit)
            
            if aspect is None:
                return w >= sz
                
            asp = float(aspect)
            if asp >= 1.0:
                mw = int(math.ceil(sz * asp))
            else:
                mw = sz
            return w >= mw - 1

        if not is_user_set:
            target_w = int(canvas_width) if canvas_width is not None else 512
            low, high = 7, max(7, target_w)
            best = 7
            while low <= high:
                mid = (low + high) // 2
                if _check_fits(mid, target_w):
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            target_rw = best

        if target_rw <= 6:
            raise ValueError("ring_width (or resulting auto-sized width) must be greater than 6 pixels")

        self.ring_width = target_rw
        self.wall_thickness, self.inner_offset, canvas_size = _get_layout_params(self.ring_width)

        if self.wall_thickness <= 0:
            raise ValueError("wall_thickness must be positive")
        self.ring_spacing = self.ring_width + self.wall_thickness
        
        self.max_radius = (self.total_rings - 1) * self.ring_spacing + self.ring_width
        self.canvas_radius = self.max_radius + self.inner_offset

        final_width, final_height = self._resolve_canvas_dimensions(canvas_size, canvas_width, aspect)
        final_aspect = final_width / final_height

        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=final_aspect,
            size=self.ring_width,
            seed=seed,
            ti2v_prompt=ti2v_prompt,
            show_cell_id=show_cell_id,
            video=video,
        )

        self.canvas_size = self.canvas_width
        self.center = (self.canvas_width / 2.0, self.canvas_height / 2.0)
        self.cells_per_ring: List[int] = [1] + [self.segments for _ in range(self.rings)]

    # ------------------------------------------------------------------
    # Public puzzle creation

    def _resolve_canvas_dimensions(
        self,
        base_size: int,
        canvas_width: Optional[int],
        aspect: Optional[float],
    ) -> Tuple[int, int]:
        """Determine final canvas width and height while ensuring the maze fits."""

        min_size = int(base_size)
        if min_size <= 0:
            raise ValueError("Base canvas size must be positive")

        if aspect is None:
            width_hint = int(canvas_width) if canvas_width is not None else min_size
            if width_hint < min_size:
                raise ValueError("canvas_width is too small to contain the labyrinth geometry")
            return width_hint, width_hint

        aspect_value = float(aspect)
        if aspect_value <= 0:
            raise ValueError("aspect must be positive")

        if aspect_value >= 1.0:
            min_width = int(math.ceil(min_size * aspect_value))
        else:
            min_width = min_size

        if canvas_width is not None:
            width = int(canvas_width)
            # FIXED: Added tolerance for floating point rounding errors (e.g. 640 vs 641)
            # and allow the user to proceed if they are very close.
            if width < min_width - 1:
                raise ValueError(
                    f"canvas_width ({width}) is too small to satisfy the requested aspect ratio "
                    f"({aspect_value}) without clipping the labyrinth (needs ~{min_width})"
                )
        else:
            width = min_width

        width = max(width, min_size)
        height = int(math.ceil(width / aspect_value))
        
        # FIXED: Allow slight tolerance for height checks too.
        if height < min_size - 1:
            raise ValueError("Resolved canvas height is too small for the labyrinth geometry")
        return width, height

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MazePuzzleRecord:
        puzzle_uuid = puzzle_id or self.next_id()
        passages = self._generate_passages()

        outer_ring = self.total_rings - 1
        start_segment = self.rng.randrange(self.cells_per_ring[outer_ring])
        start_cell: Cell = (outer_ring, start_segment)
        goal_cell: Cell = (0, 0)

        path_cells = self._shortest_path(passages, start_cell, goal_cell)
        if not path_cells:
            raise RuntimeError("Failed to compute solution path for labyrinth maze")

        puzzle_image = self._render_maze(passages, start_cell, goal_cell, path=None)
        solution_image = self._render_maze(passages, start_cell, goal_cell, path=path_cells)
        puzzle_path, solution_path = self.save_images(puzzle_uuid, puzzle_image, solution_image)

        video_path = None
        if self.video:
            path_points = self._build_path_points(path_cells)
            thickness = max(3, self.ring_width // 5)
            video_path = self.save_video(puzzle_uuid, puzzle_image, path_points, thickness=thickness, duration=5.0)

        start_point = self._cell_center_from_cell(start_cell)
        goal_point = self._cell_center_from_cell(goal_cell)

        record = self.build_record(
            puzzle_uuid,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            ti2v_prompt=self.ti2v_prompt,
            extra={
                "total_rings": self.total_rings,
                "rings": self.rings,
                "segments": self.segments,
                "ring_width": self.ring_width,
                "wall_thickness": self.wall_thickness,
                "cells_per_ring": self.cells_per_ring,
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
                "solution_path_cell_ids": [self._get_cell_id(cell) for cell in path_cells],
            },
            video_path=video_path,
        )
        return record

    def _get_cell_id(self, cell: Cell) -> int:
        ring, segment = cell
        if ring == 0:
            return 0
        # Ring 0 has 1 cell.
        # Rings 1..(ring-1) have self.segments cells each.
        return 1 + (ring - 1) * self.segments + segment

    # ------------------------------------------------------------------
    # Graph construction and traversal

    def _generate_passages(self) -> Dict[Cell, Set[Cell]]:
        graph = self._build_graph()
        passages: Dict[Cell, Set[Cell]] = {cell: set() for cell in graph}
        visited: Set[Cell] = set()

        def dfs(cell: Cell) -> None:
            visited.add(cell)
            neighbors = list(graph[cell])
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    passages[cell].add(neighbor)
                    passages[neighbor].add(cell)
                    dfs(neighbor)

        dfs((0, 0))
        return passages

    def _build_graph(self) -> Dict[Cell, Set[Cell]]:
        graph: Dict[Cell, Set[Cell]] = {}
        for ring in range(self.total_rings):
            count = self.cells_per_ring[ring]
            for idx in range(count):
                cell = (ring, idx)
                graph[cell] = set()

        for ring in range(self.total_rings):
            count = self.cells_per_ring[ring]
            for idx in range(count):
                cell = (ring, idx)

                # Same-ring neighbors (angular direction)
                if count > 1:
                    left = (ring, (idx - 1) % count)
                    right = (ring, (idx + 1) % count)
                    graph[cell].add(left)
                    graph[cell].add(right)

                # Outward neighbors
                if ring + 1 < self.total_rings:
                    for neighbor in self._adjacent_cells(ring, idx, ring + 1):
                        graph[cell].add(neighbor)
                        graph[neighbor].add(cell)

                # Inward neighbors
                if ring > 0:
                    for neighbor in self._adjacent_cells(ring, idx, ring - 1):
                        graph[cell].add(neighbor)
                        graph[neighbor].add(cell)
        return graph

    def _adjacent_cells(self, ring: int, idx: int, target_ring: int) -> List[Cell]:
        current_count = self.cells_per_ring[ring]
        target_count = self.cells_per_ring[target_ring]
        start_frac = idx / current_count
        end_frac = (idx + 1) / current_count
        start_idx = int(math.floor(start_frac * target_count))
        end_idx = int(math.ceil(end_frac * target_count)) - 1
        if end_idx < start_idx:
            end_idx = start_idx
        result = []
        for t in range(start_idx, end_idx + 1):
            result.append((target_ring, t % target_count))
        if not result:
            result.append((target_ring, int(round(start_frac * target_count)) % target_count))
        return result

    def _shortest_path(self, passages: Dict[Cell, Set[Cell]], start: Cell, goal: Cell) -> List[Cell]:
        queue: Deque[Cell] = deque([start])
        parents: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for neighbor in passages[cell]:
                if neighbor not in parents:
                    parents[neighbor] = cell
                    queue.append(neighbor)
        if goal not in parents:
            return []
        path: List[Cell] = []
        node: Optional[Cell] = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Rendering helpers

    def _render_maze(
        self,
        passages: Dict[Cell, Set[Cell]],
        start_cell: Cell,
        goal_cell: Cell,
        *,
        path: Optional[Sequence[Cell]] = None,
    ) -> Image.Image:
        canvas = Image.new("RGB", (self.canvas_width, self.canvas_height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        self._fill_walkways(draw)

        self._draw_walls(draw, passages)
        self._draw_markers(draw, start_cell, goal_cell)
        if path:
            self._draw_solution(canvas, path)
        
        if self.show_cell_id:
            self._draw_cell_ids(draw)
            
        return canvas

    def _fill_walkways(self, draw: ImageDraw.ImageDraw) -> None:
        for ring in range(self.total_rings - 1, -1, -1):
            inner, outer = self._ring_bounds(ring)
            outer_radius = self._radius_to_pixel(outer)
            bbox_outer = self._bbox(outer_radius)
            draw.ellipse(bbox_outer, fill=PATH_COLOR)
            if inner > 0:
                inner_radius = self._radius_to_pixel(inner)
                bbox_inner = self._bbox(inner_radius)
                draw.ellipse(bbox_inner, fill=BACKGROUND_COLOR)

    def _draw_walls(self, draw: ImageDraw.ImageDraw, passages: Dict[Cell, Set[Cell]]) -> None:
        # Angular walls within the same ring
        for ring in range(self.total_rings):
            count = self.cells_per_ring[ring]
            inner, outer = self._ring_bounds(ring)
            inner_pix = self._radius_to_pixel(inner)
            outer_pix = self._radius_to_pixel(outer)
            if count > 1:
                for idx in range(count):
                    cell = (ring, idx)
                    neighbor = (ring, (idx + 1) % count)
                    if neighbor not in passages[cell]:
                        boundary_angle = self._segment_angles_deg(ring, idx)[1]
                        self._draw_radial_wall(draw, inner_pix, outer_pix, boundary_angle)

        # Circular walls between rings
        for ring in range(1, self.total_rings):
            inner, _ = self._ring_bounds(ring)
            wall_center = inner - self.wall_thickness 
            boundary_radius = self._radius_to_pixel(wall_center)
            bbox = self._bbox(boundary_radius)
            draw.ellipse(bbox, outline=WALL_COLOR, width=self.wall_thickness)
            count = self.cells_per_ring[ring]
            for idx in range(count):
                cell = (ring, idx)
                neighbors = self._adjacent_cells(ring, idx, ring - 1)
                for neighbor in neighbors:
                    if neighbor in passages[cell]:
                        overlap_mid = self._overlap_mid_angle(ring, idx, neighbor)
                        self._draw_radial_gap(draw, boundary_radius, overlap_mid)

        # Outer boundary ring (ensures maze is enclosed)
        outer_radius = self._radius_to_pixel(self._ring_bounds(self.total_rings - 1)[1])
        outer_bbox = self._bbox(outer_radius)
        draw.ellipse(outer_bbox, outline=WALL_COLOR, width=self.wall_thickness)

    def _draw_radial_wall(
        self,
        draw: ImageDraw.ImageDraw,
        inner_radius: float,
        outer_radius: float,
        angle_deg: float,
    ) -> None:
        angle_rad = math.radians(angle_deg)
        pad = self.wall_thickness / 2.0
        start = self._polar_to_cartesian(inner_radius - pad, angle_rad)
        end = self._polar_to_cartesian(outer_radius + pad, angle_rad)
        draw.line([start, end], fill=WALL_COLOR, width=self.wall_thickness)

    def _draw_radial_gap(
        self,
        draw: ImageDraw.ImageDraw,
        radius: float,
        angle_deg: float,
    ) -> None:
        angle_rad = math.radians(angle_deg)
        half_gap = max(self.wall_thickness * 1.5, 4.0)
        inner = radius - half_gap
        outer = radius + half_gap
        start = self._polar_to_cartesian(inner, angle_rad)
        end = self._polar_to_cartesian(outer, angle_rad)
        draw.line([start, end], fill=PATH_COLOR, width=int(self.wall_thickness * 1.5))

    def _draw_markers(self, draw: ImageDraw.ImageDraw, start_cell: Cell, goal_cell: Cell) -> None:
        start_point = self._cell_center_from_cell(start_cell)
        goal_point = self._cell_center_from_cell(goal_cell)
        self._draw_marker(draw, start_point, START_COLOR, self._marker_radius_for_cell(start_cell))
        self._draw_marker(draw, goal_point, GOAL_COLOR, self._marker_radius_for_cell(goal_cell))

    def _draw_marker(
        self,
        draw: ImageDraw.ImageDraw,
        point: Tuple[float, float],
        color: Tuple[int, int, int],
        radius: int,
    ) -> None:
        x, y = point
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=color)

    def _draw_solution(self, image: Image.Image, path: Sequence[Cell]) -> None:
        if len(path) < 2:
            return
        thickness = max(3, self.ring_width // 5)
        points = self._build_path_points(path)
        ImageDraw.Draw(image).line(points, fill=LINE_COLOR, width=thickness, joint="curve")

    def _build_path_points(self, path: Sequence[Cell]) -> List[Tuple[float, float]]:
        if not path:
            return []
        points: List[Tuple[float, float]] = [self._cell_center_from_cell(path[0])]
        for current, nxt in zip(path, path[1:]):
            if current[0] == nxt[0]:
                points.extend(self._arc_points_between_cells(current, nxt)[1:])
                continue

            overlap_angle_deg = self._overlap_mid_angle(current[0], current[1], nxt)
            current_anchor = self._cell_anchor_at_angle(current, overlap_angle_deg)
            next_anchor = self._cell_anchor_at_angle(nxt, overlap_angle_deg)
            if points[-1] != current_anchor:
                points.append(current_anchor)
            for point in self._radial_points_between(current_anchor, next_anchor)[1:]:
                if point != points[-1]:
                    points.append(point)
            next_center = self._cell_center_from_cell(nxt)
            if points[-1] != next_center:
                points.append(next_center)
        return points

    def _cell_anchor_at_angle(self, cell: Cell, angle_deg: float) -> Tuple[float, float]:
        ring, _ = cell
        if ring == 0:
            return self.center
        return self._polar_to_cartesian(self._cell_path_radius(ring), math.radians(angle_deg))

    def _arc_points_between_cells(self, current: Cell, nxt: Cell) -> List[Tuple[float, float]]:
        ring, idx = current
        radius = self._cell_path_radius(ring)
        start_deg = sum(self._segment_angles_deg(ring, idx)) * 0.5
        end_deg = sum(self._segment_angles_deg(nxt[0], nxt[1])) * 0.5
        delta = (end_deg - start_deg + 180.0) % 360.0 - 180.0
        steps = max(8, int(abs(delta) / 2.0))
        arc_points: List[Tuple[float, float]] = []
        for step in range(steps + 1):
            ratio = step / steps
            angle_deg = start_deg + delta * ratio
            arc_points.append(self._polar_to_cartesian(radius, math.radians(angle_deg)))
        return arc_points

    def _radial_points_between(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(4, int(distance / max(3.0, self.ring_width * 0.15)))
        radial_points: List[Tuple[float, float]] = []
        for step in range(steps + 1):
            ratio = step / steps
            radial_points.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
        return radial_points

    def _marker_radius_for_cell(self, cell: Cell) -> int:
        ring, _ = cell
        if ring == 0:
            safe = (self.ring_width - self.wall_thickness * 1.8) / 2.0
            return max(4, int(safe))

        inner, outer = self._ring_bounds(ring)
        radius_mid = self._radius_to_pixel((inner + outer) * 0.5)
        radial_clearance = (outer - inner) * 0.5 - self.wall_thickness * 0.9
        angle_span = (2.0 * math.pi) / self.cells_per_ring[ring]
        angular_clearance = radius_mid * math.sin(angle_span * 0.5) - self.wall_thickness * 0.9
        safe = min(radial_clearance, angular_clearance) - 1.0
        return max(4, int(safe))

    def _draw_cell_ids(self, draw: ImageDraw.ImageDraw) -> None:
        font_size = max(8, int(self.ring_width * 0.4))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        # Iterate all cells
        for ring in range(self.total_rings):
            count = self.cells_per_ring[ring]
            for segment in range(count):
                cell = (ring, segment)
                text = str(self._get_cell_id(cell))
                cx, cy = self._cell_center_from_cell(cell)
                draw.text((cx, cy), text, fill=TEXT_COLOR, anchor="mm", font=font)

    # ------------------------------------------------------------------
    # Geometry utilities

    def _ring_bounds(self, ring: int) -> Tuple[float, float]:
        inner = ring * self.ring_spacing
        outer = inner + self.ring_width
        return inner, outer

    def _segment_angles_deg(self, ring: int, idx: int) -> Tuple[float, float]:
        count = self.cells_per_ring[ring]
        start = (idx / count) * 360.0
        end = ((idx + 1) / count) * 360.0
        return start, end

    def _overlap_mid_angle(self, ring: int, idx: int, neighbor: Cell) -> float:
        count_a = self.cells_per_ring[ring]
        count_b = self.cells_per_ring[neighbor[0]]
        start_a = idx / count_a
        end_a = (idx + 1) / count_a
        start_b = neighbor[1] / count_b
        end_b = (neighbor[1] + 1) / count_b
        start = max(start_a, start_b)
        end = min(end_a, end_b)
        if end <= start:
            midpoint = (start_a + end_a) * 0.5
        else:
            midpoint = (start + end) * 0.5
        return midpoint * 360.0

    def _radius_to_pixel(self, radius: float) -> float:
        return self.inner_offset + radius

    def _bbox(self, radius: float) -> Tuple[float, float, float, float]:
        cx, cy = self.center
        return (cx - radius, cy - radius, cx + radius, cy + radius)

    def _polar_to_cartesian(self, radius: float, angle_rad: float) -> Tuple[float, float]:
        cx, cy = self.center
        x = cx + radius * math.cos(angle_rad)
        y = cy - radius * math.sin(angle_rad)
        return (x, y)

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        if cell_id == 0:
            cell = (0, 0)
        else:
            adjusted = cell_id - 1
            ring = adjusted // self.segments + 1
            segment = adjusted % self.segments
            cell = (ring, segment)
        return self._cell_center_from_cell(cell)

    def _cell_center_from_cell(self, cell: Cell) -> Tuple[float, float]:
        ring, idx = cell
        if ring == 0:
            return self.center
        radius = self._cell_path_radius(ring)
        start_deg, end_deg = self._segment_angles_deg(ring, idx)
        angle_rad = math.radians((start_deg + end_deg) * 0.5)
        return self._polar_to_cartesian(radius, angle_rad)

    def _cell_path_radius(self, ring: int) -> float:
        inner, outer = self._ring_bounds(ring)
        return self._radius_to_pixel((inner + outer - self.wall_thickness) * 0.5)

    def save_video(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        points: List[Tuple[float, float]],
        thickness: int = 5,
        color: Tuple[int, int, int] = (220, 30, 30),
        fps: int = 16,
        duration: float = 6.4,
        joint_style: str = "round",
    ) -> Optional[Path]:
        self._last_video_fps = None
        self._last_video_num_frames = None
        if not self.video:
            return None

        video_path = self.solution_dir / f"{record_id}_solution.mp4"
        if len(points) < 2:
            frames = [puzzle_image.copy() for _ in range(fps)]
            if not encode_rgb_frames_to_mp4(frames, video_path, fps=fps):
                return None
            self._last_video_fps = fps
            self._last_video_num_frames = fps
            return video_path

        eff_duration = min(duration, 10.0)
        total_frames = int(fps * eff_duration)
        cumulative = [0.0]
        for p1, p2 in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        total_len = cumulative[-1]

        frames: List[Image.Image] = []
        for frame_idx in range(total_frames + 1):
            progress = frame_idx / total_frames if total_frames > 0 else 1.0
            target_len = total_len * progress
            visible: List[Tuple[float, float]] = [points[0]]
            for seg_idx, (p1, p2) in enumerate(zip(points, points[1:]), start=1):
                seg_start = cumulative[seg_idx - 1]
                seg_end = cumulative[seg_idx]
                if target_len >= seg_end - 1e-6:
                    visible.append(p2)
                    continue
                if target_len > seg_start:
                    ratio = (target_len - seg_start) / max(1e-6, seg_end - seg_start)
                    visible.append(
                        (
                            p1[0] + (p2[0] - p1[0]) * ratio,
                            p1[1] + (p2[1] - p1[1]) * ratio,
                        )
                    )
                break

            frame = puzzle_image.copy()
            if len(visible) >= 2:
                ImageDraw.Draw(frame).line(visible, fill=color, width=thickness, joint="curve")
            frames.append(frame)

        end_hold_frames = int(fps * 1.0)
        final_frame = puzzle_image.copy()
        ImageDraw.Draw(final_frame).line(points, fill=color, width=thickness, joint="curve")
        for _ in range(end_hold_frames):
            frames.append(final_frame.copy())

        if not encode_rgb_frames_to_mp4(frames, video_path, fps=fps):
            return None
        self._last_video_fps = fps
        self._last_video_num_frames = len(frames)
        return video_path

    # ------------------------------------------------------------------
    # CLI helpers

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate circular labyrinth maze puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to generate")
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--rings", type=int, default=cls.DEFAULT_RINGS, help="Number of rings excluding the center")
        parser.add_argument("--segments", type=int, default=cls.DEFAULT_SEGMENTS, help="Number of angular segments per ring")
        parser.add_argument("--ring-width", type=int, default=None, help="Thickness of each ring walkway in pixels")
        parser.add_argument("--wall-thickness", type=int, default=None, help="Thickness of black walls in pixels")
        parser.add_argument("--size", type=int, default=None, help="Alias for --ring-width to align with shared base interface")
        parser.add_argument("--canvas-width", type=int, default=None, help="Final canvas width in pixels")
        parser.add_argument("--aspect", type=float, default=None, help="Desired width/height ratio")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--show-cell-id", action="store_true", help="Draw cell IDs on the maze")
        parser.add_argument("--use-gpt-5", action="store_true", help="Use DEFAULT_VLM_PROMPT and show cell IDs.")
        parser.add_argument("--video", action="store_true", help="Generate solution video")
        namespace=parser.parse_args(argv)
        if namespace.use_gpt_5:
            namespace.show_cell_id = True
        return namespace

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        ring_width = args.ring_width if args.ring_width is not None else args.size
        prompt_arg = args.prompt if args.prompt is not None else (
            cls.DEFAULT_VLM_PROMPT if args.use_gpt_5 else cls.DEFAULT_TI2V_PROMPT
        )
        generator = cls(
            output_dir=args.output_dir,
            rings=args.rings,
            segments=args.segments,
            ring_width=ring_width,
            wall_thickness=args.wall_thickness,
            seed=args.seed,
            ti2v_prompt=prompt_arg,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            show_cell_id=args.show_cell_id,
            video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


__all__ = ["MazeLabyrinthGenerator"]


def main(argv: Optional[List[str]] = None) -> None:
    MazeLabyrinthGenerator.main(argv)


if __name__ == "__main__":
    MazeLabyrinthGenerator.main()
