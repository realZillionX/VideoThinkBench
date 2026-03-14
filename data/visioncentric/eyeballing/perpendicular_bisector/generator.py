"""Perpendicular bisector puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

def distance(p1: Point, p2: Point) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

class PerpendicularBisectorGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find a point on the perpendicular bisector of a line segment."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/perpendicular_bisector"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows two small black endpoint circles with thick outlines connected by one black segment of medium thickness. "
        "Five candidate markers A-E are arranged near the hidden perpendicular bisector; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the endpoint circles, the connecting segment, and the candidate markers, then draws a long black bisector line of medium thickness through the segment midpoint in the perpendicular direction so it runs across the canvas. "
        "In the final frame, only the candidate lying on that perpendicular bisector changes to pale red fill with a dark red outline, while all other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows two small endpoint circles connected by a black segment and five labeled candidate circles "
        "A-E. Determine which candidate lies on the perpendicular bisector of the segment, meaning it is on the line "
        "through the midpoint that is perpendicular to the segment. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 72
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        tries = 0
        while tries < 999:
            midpoint = self.pick_target_point(
                0.48, padding=candidate_padding + self.canvas_short_side * 0.14,
            )
            segment_angle = self._rng.uniform(0.0, math.tau)
            try:
                p1, p2 = self.sample_symmetric_segment(
                    midpoint,
                    segment_angle,
                    min_half_length=self.canvas_short_side * 0.13,
                    max_half_length=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
            except RuntimeError:
                tries += 1
                continue

            target_angle = segment_angle + math.pi / 2
            if self._rng.random() < 0.5:
                target_angle += math.pi
            try:
                target = self.sample_point_along_direction(
                    midpoint,
                    target_angle,
                    min_distance=self.canvas_short_side * 0.26,
                    max_distance=self.canvas_short_side * 0.38,
                    padding=candidate_padding,
                )
            except RuntimeError:
                tries += 1
                continue

            self.points = (p1, p2)
            self.target_point = target
            
            # Place candidates along a line roughly perpendicular to the bisector itself
            # (i.e., parallel to the original p1-p2 segment)
            try:
                self.place_candidates_line(target, segment_angle + self._rng.uniform(-0.08, 0.08))
            except RuntimeError:
                tries += 1
                continue
            
            if not self.check_candidates_inside():
                tries += 1
                continue
            break

        else:
            raise RuntimeError("Failed to find valid perpendicular bisector geometry after 999 tries")

        record = self.save_puzzle()
        record.points = self.points
        return record

    def _draw_endpoints(self, draw) -> None:
        p1, p2 = self.points
        self.draw_anchor_marker(draw, p1, 7)
        self.draw_anchor_marker(draw, p2, 7)

    def _video_overlay_extras(self, draw: ImageDraw.ImageDraw) -> None:
        self._draw_endpoints(draw)

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1, p2 = self.points

        # Draw the original segment
        seg_start, seg_end = self.trim_segment(
            p1,
            p2,
            start_offset=9.0,
            end_offset=9.0,
        )
        self.draw_line(draw, [seg_start, seg_end])
        self._draw_endpoints(draw)
        if highlight_label:

            # Draw the perpendicular bisector line
            midpoint = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            line_angle = math.atan2(dx, -dy)
            line_p1, line_p2 = self.clip_line_to_canvas(midpoint, line_angle, padding=self.line_width)
            self.draw_line(draw, [line_p1, line_p2])
            self.draw_circle(draw, self.target_point, 5)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    PerpendicularBisectorGenerator.main(PerpendicularBisectorGenerator, argv)

if __name__ == "__main__":
    main()
