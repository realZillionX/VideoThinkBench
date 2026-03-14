"""Isosceles Trapezoid puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class IsoscelesTrapezoidGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the fourth vertex of an isosceles trapezoid."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/isosceles_trapezoid"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows three known vertices of an almost-complete isosceles trapezoid as an open black polyline of medium thickness with two connected segments already visible and one corner missing. "
        "Five candidate markers A-E are placed near that missing fourth vertex; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the open three-vertex polyline and the candidate markers, then draws two black segments of medium thickness from the visible right endpoint to the correct candidate and from that candidate to the upper-left visible vertex, completing a trapezoid whose two bases are parallel and whose two legs are equal. "
        "In the final frame, only the correct missing-corner marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows three connected black vertices that almost form an isosceles trapezoid, plus five labeled "
        "candidate circles A-E near the missing corner. Determine which candidate completes the figure so the two bases are "
        "parallel and the two non-parallel sides are equal in length. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            center = self.pick_target_point(0.45, padding=anchor_padding + self.canvas_short_side * 0.12)
            base_angle = self._rng.uniform(0.0, math.tau)
            axis_angle = base_angle + math.pi / 2
            base_half = self.canvas_short_side * self._rng.uniform(0.2, 0.26)
            top_half = base_half * self._rng.uniform(0.52, 0.72)
            height = self.canvas_short_side * self._rng.uniform(0.2, 0.28)

            bottom_center = self.point_on_ray(center, axis_angle + math.pi, height * 0.5)
            top_center = self.point_on_ray(center, axis_angle, height * 0.5)
            p1 = self.point_on_ray(bottom_center, base_angle + math.pi, base_half)
            p2 = self.point_on_ray(bottom_center, base_angle, base_half)
            p3 = self.point_on_ray(top_center, base_angle + math.pi, top_half)
            target_point = self.point_on_ray(top_center, base_angle, top_half)

            points = (p1, p2, p3, target_point)
            if not all(self.point_can_host_candidate(point) for point in points):
                continue
            if height < self.canvas_short_side * 0.2:
                continue
            if 2 * top_half < self.canvas_short_side * 0.18:
                continue
            if 2 * (base_half - top_half) < self.canvas_short_side * 0.12:
                continue
            self.trapezoid_points = (p1, p2, p3, target_point)
            self.place_candidates(target_point)
            record = self.save_puzzle()
            record.trapezoid_points = self.trapezoid_points
            return record
        raise RuntimeError("Failed to generate a valid isosceles trapezoid puzzle.")

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1, p2, p3, target = self.trapezoid_points
        
        # Draw the three given vertices/sides
        self.draw_line(draw, [p3, p1, p2])
        
        # In the solution image, complete the trapezoid
        if highlight_label:
            self.draw_line(draw, [p2, target, p3])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    IsoscelesTrapezoidGenerator.main(IsoscelesTrapezoidGenerator, argv)

if __name__ == "__main__":
    main()
