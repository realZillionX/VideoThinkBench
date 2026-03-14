"""Parallelogram puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class ParallelogramGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that hide the parallelogram of a segment."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/parallelogram"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows three known vertices of a parallelogram as an open black broken line of medium thickness shaped like a V, with two adjacent sides already visible and one opposite corner missing. "
        "Five candidate markers A-E are placed near the missing corner; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the open broken line and the candidate markers, then draws two black closing segments of medium thickness from one visible endpoint to the correct candidate and from that candidate to the other visible endpoint so the full quadrilateral becomes a parallelogram. "
        "In the final state, only the correct missing-vertex marker changes to pale red fill with a dark red outline, while the other four candidates remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows three black vertices connected as two adjacent sides of a parallelogram, plus five labeled "
        "candidate circles A-E near the missing corner. Determine which candidate is the fourth vertex that closes the "
        "parallelogram. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            target_point = self.pick_target_point(0.45, padding=anchor_padding + self.canvas_short_side * 0.1)
            base_angle = self._rng.uniform(0.0, math.tau)
            angle_delta = math.radians(self._rng.uniform(54.0, 116.0))
            second_angle = base_angle + angle_delta
            side_a = self.canvas_short_side * self._rng.uniform(0.2, 0.3)
            side_b = self.canvas_short_side * self._rng.uniform(0.2, 0.3)
            p1 = self.point_on_ray(target_point, base_angle, side_a)
            p2 = self.point_on_ray(target_point, second_angle, side_b)
            p3 = Point(
                x=p1.x + p2.x - target_point.x,
                y=p1.y + p2.y - target_point.y,
            )
            points = (p1, p2, p3, target_point)
            if not all(self.point_can_host_candidate(point, extra_padding=self.canvas_short_side * 0.01) for point in points):
                continue
            area = abs(
                (p1.x - target_point.x) * (p2.y - target_point.y)
                - (p1.y - target_point.y) * (p2.x - target_point.x)
            )
            if area < (self.canvas_short_side ** 2) * 0.05:
                continue
            self.parallelogram_points = (p1, p2, p3, target_point)
            self.place_candidates(target_point)
            return self.save_puzzle()
        raise RuntimeError("Failed to generate a valid parallelogram puzzle.")

    def build_record_extra(self) -> dict[str, object]:
        return {
            "known_vertices": [point.to_list() for point in self.parallelogram_points[:3]],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1,p2,p3,target=self.parallelogram_points
        self.draw_line(draw,[p1,p3,p2])
        if highlight_label:
            self.draw_line(draw,[p2,target,p1],)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ParallelogramGenerator.main(ParallelogramGenerator, argv)

if __name__ == "__main__":
    main()
