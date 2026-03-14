"""Perpendicular puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class PerpendicularGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/perpendicular"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows one black reference segment of medium thickness and one small black circle with a thick outline centered on the marked through-point. "
        "Five candidate markers A-E sit near the hidden perpendicular direction on a short straight row; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the reference segment, the marked through-point circle, and the candidate row, then draws one black segment of medium thickness starting at that marked point and extending toward the correct candidate at a right angle to the reference segment. "
        "In the final state, only the candidate on this perpendicular segment changes to pale red fill with a dark red outline, while the other markers stay white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows one black reference line, one small black circle marking a through-point, and five labeled "
        "candidate circles A-E. Determine which candidate lies on the line through the small circle that is perpendicular to "
        "the reference line. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 64
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            line_angle = self._rng.uniform(0.0, math.tau)
            normal_angle = line_angle + math.pi / 2 * self._rng.choice([-1, 1])
            through_point = self.pick_target_point(0.46, padding=anchor_padding + self.canvas_short_side * 0.16)
            try:
                reference_mid = self.sample_point_along_direction(
                    through_point,
                    normal_angle,
                    min_distance=self.canvas_short_side * 0.2,
                    max_distance=self.canvas_short_side * 0.3,
                    padding=self.line_width,
                )
                p_ref1 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                p_ref2 = self.sample_point_along_direction(
                    reference_mid,
                    line_angle + math.pi,
                    min_distance=self.canvas_short_side * 0.14,
                    max_distance=self.canvas_short_side * 0.22,
                    padding=self.line_width,
                )
                target = self.sample_point_along_direction(
                    through_point,
                    normal_angle,
                    min_distance=self.canvas_short_side * 0.28,
                    max_distance=self.canvas_short_side * 0.38,
                    padding=anchor_padding,
                )
                self.place_candidates_line(target, line_angle + self._rng.uniform(-0.05, 0.05))
            except RuntimeError:
                continue
            if self.distance(p_ref1, p_ref2) < self.canvas_short_side * 0.28:
                continue
            self.points = (through_point, p_ref1, p_ref2)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to find valid perpendicular geometry after 999 tries")

    def build_record_extra(self) -> dict[str, object]:
        return {
            "reference_line_endpoints": [self.points[1].to_list(), self.points[2].to_list()],
            "through_point": self.points[0].to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points[1:])
        self.draw_circle(draw,self.points[0],7)
        if highlight_label:
            self.draw_line(draw,[self.points[0],self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    PerpendicularGenerator.main(PerpendicularGenerator, argv)

if __name__ == "__main__":
    main()
