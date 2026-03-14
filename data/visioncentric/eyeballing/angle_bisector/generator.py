"""AngleBisector puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

def mid_angle(angle1: float, angle2: float) -> float:
    """Calculate the bisector angle between two angles."""
    x = math.cos(angle1) + math.cos(angle2)
    y = math.sin(angle1) + math.sin(angle2)
    return math.atan2(y, x)

class AngleBisectorGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/angle_bisector"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows two solid black line segments of medium thickness meeting at one shared vertex and opening into a clear angle, with no arrows and no extra marks. "
        "Five small candidate markers A-E sit near the hidden bisector on a short straight row: each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. "
        "The video holds this angle-and-candidates puzzle first, then draws one black bisector segment of medium thickness outward from the shared vertex through the interior of the angle toward the correct marker. "
        "In the final state, only the correct candidate changes to pale red fill with a dark red outline, while all other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows two black rays meeting at one vertex and five labeled candidate circles A-E near the interior "
        "of the angle. Identify which candidate lies on the exact angle bisector, meaning the line from the vertex that "
        "splits the angle into two equal angles. Answer with a single option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)


    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 64
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.06)
        for _ in range(999):
            vertex = self.pick_target_point(0.5, padding=anchor_padding + self.canvas_short_side * 0.16)
            opening = math.radians(self._rng.uniform(54.0, 112.0))
            bisector_angle = self._rng.uniform(0.0, math.tau)
            angle1 = bisector_angle - opening / 2.0
            angle2 = bisector_angle + opening / 2.0
            ray_min = self.canvas_short_side * 0.24
            ray_max = self.canvas_short_side * 0.36
            target_min = self.canvas_short_side * 0.30
            target_max = self.canvas_short_side * 0.40
            try:
                p1 = self.sample_point_along_direction(
                    vertex, angle1, min_distance=ray_min, max_distance=ray_max, padding=self.line_width,
                )
                p2 = self.sample_point_along_direction(
                    vertex, angle2, min_distance=ray_min, max_distance=ray_max, padding=self.line_width,
                )
                target = self.sample_point_along_direction(
                    vertex, bisector_angle, min_distance=target_min, max_distance=target_max, padding=anchor_padding,
                )
                self.place_candidates_line(target, bisector_angle + math.pi / 2 + self._rng.uniform(-0.06, 0.06))
            except RuntimeError:
                continue
            if min(self.distance(vertex, p1), self.distance(vertex, p2)) < self.canvas_short_side * 0.24:
                continue
            self.points = (p1, vertex, p2)
            self.target_point = target
            return self.save_puzzle()
        raise RuntimeError("Failed to find valid angle bisector geometry after 999 tries")

    def build_record_extra(self) -> dict[str, object]:
        angle1 = math.atan2(self.points[0].y - self.points[1].y, self.points[0].x - self.points[1].x)
        angle2 = math.atan2(self.points[2].y - self.points[1].y, self.points[2].x - self.points[1].x)
        angle_radians = abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)
        return {
            "vertex": self.points[1].to_list(),
            "ray_endpoints": [self.points[0].to_list(), self.points[2].to_list()],
            "angle_degrees": math.degrees(angle_radians),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points)
        if highlight_label:
            self.draw_line(draw,[self.points[1],self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    AngleBisectorGenerator.main(AngleBisectorGenerator, argv)

if __name__ == "__main__":
    main()
