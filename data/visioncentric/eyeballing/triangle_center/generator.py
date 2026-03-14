"""TriangleCenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class TriangleCenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/triangle_center"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows a triangle drawn only as a black outline of medium thickness, with no midpoint marks and no interior construction lines at first. "
        "Five candidate markers A-E are clustered near the hidden centroid; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the bare triangle and the five markers, then draws three black medians of medium thickness, each running from one triangle vertex to the midpoint of the opposite side so all three meet at one point. "
        "In the final state, only the centroid marker changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A black triangle outline is shown on a white canvas with five labeled candidate circles A-E near its interior. "
        "Determine which candidate is the triangle's center of mass, namely the centroid where the three medians intersect. "
        "Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        p1, p2, p3 = self.sample_triangle_vertices(
            jitter_ratio=0.68,
            min_side_ratio=0.24,
            min_area_ratio=0.055,
            min_altitude_ratio=0.17,
            min_angle_deg=38.0,
            max_angle_deg=108.0,
        )
        center=Point(
            x=(p1.x + p2.x + p3.x) / 3,
            y=(p1.y + p2.y + p3.y) / 3,
        )
        p12=Point(x=(p1.x + p2.x)/2, y=(p1.y + p2.y)/2)
        p23=Point(x=(p2.x + p3.x)/2, y=(p2.y + p3.y)/2)
        p31=Point(x=(p3.x + p1.x)/2, y=(p3.y + p1.y)/2)
        self.triangle_points = (p1, p2, p3)
        self.midpoints = (p23, p31, p12)
        self.target_point = center
        self.place_candidates(center)
        return self.save_puzzle()

    def build_record_extra(self) -> dict[str, object]:
        return {
            "triangle_vertices": [point.to_list() for point in self.triangle_points],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,list(self.triangle_points)+[self.triangle_points[0]])
        if highlight_label:
            for vertex, midpoint in zip(self.triangle_points, self.midpoints):
                self.draw_line(draw,[vertex, midpoint])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    TriangleCenterGenerator.main(TriangleCenterGenerator, argv)

if __name__ == "__main__":
    main()
