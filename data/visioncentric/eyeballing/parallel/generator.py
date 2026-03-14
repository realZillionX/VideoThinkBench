"""Parallel puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

def distanceToLine(point: Point, line_point1: Point, line_point2: Point) -> float:
    """Calculate the distance from a point to a line defined by two points."""
    numerator = abs(
        (line_point2.y - line_point1.y) * point.x -
        (line_point2.x - line_point1.x) * point.y +
        line_point2.x * line_point1.y -
        line_point2.y * line_point1.x
    )
    denominator = math.sqrt(
        (line_point2.y - line_point1.y) ** 2 +
        (line_point2.x - line_point1.x) ** 2
    )
    return numerator / denominator

class ParallelGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/parallel"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows one black reference segment of medium thickness and one separate small hollow marker with white fill and a thick black outline marking the point that a new segment must pass through. "
        "Five candidate markers A-E sit near the hidden parallel direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the reference segment, the through-point marker, and the candidate row, then draws a single black segment of medium thickness starting just outside that hollow marker and extending toward the correct candidate in a direction parallel to the reference segment. "
        "In the final frame, only the candidate on that new parallel segment changes to pale red fill with a dark red outline, while the other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows one black reference line, one separate small hollow white marker outlined in black indicating a required through-point, "
        "and five labeled candidate circles A-E. Determine which candidate lies on the line through the small circle that is "
        "parallel to the reference line. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 72
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            line_angle = self._rng.uniform(0.0, math.tau)
            normal_angle = line_angle + math.pi / 2 * self._rng.choice([-1, 1])
            through_point = self.pick_target_point(0.46, padding=anchor_padding + self.canvas_short_side * 0.18)
            try:
                reference_mid = self.sample_point_along_direction(
                    through_point,
                    normal_angle,
                    min_distance=self.canvas_short_side * 0.2,
                    max_distance=self.canvas_short_side * 0.3,
                    padding=self.line_width,
                )
                p1 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                p2 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle + math.pi,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                target = self.sample_point_along_direction(
                    through_point,
                    line_angle,
                    min_distance=self.canvas_short_side * 0.3,
                    max_distance=self.canvas_short_side * 0.4,
                    padding=anchor_padding,
                )
                self.place_candidates_line(target, line_angle + math.pi / 2 + self._rng.uniform(-0.05, 0.05))
            except RuntimeError:
                continue
            if self.distance(p1, p2) < self.canvas_short_side * 0.28:
                continue
            self.points = (through_point, p1, p2)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to find valid parallel geometry after 999 tries")

    def build_record_extra(self) -> dict[str, object]:
        return {
            "reference_line_endpoints": [self.points[1].to_list(), self.points[2].to_list()],
            "through_point": self.points[0].to_list(),
        }

    def _draw_through_point(self, draw) -> None:
        self.draw_anchor_marker(draw, self.points[0], 7)

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        self._draw_through_point(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points[1:])
        if highlight_label:
            line_start, line_end = self.trim_segment(
                self.points[0],
                self.target_point,
                start_offset=9.0,
                end_offset=float(self.point_radius),
            )
            self.draw_line(draw,[line_start, line_end])
        self._draw_through_point(draw)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ParallelGenerator.main(ParallelGenerator, argv)

if __name__ == "__main__":
    main()
