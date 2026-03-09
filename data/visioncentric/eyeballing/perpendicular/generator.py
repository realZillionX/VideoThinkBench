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
    DEFAULT_TI2V_PROMPT="Draw a black line perpendicular to the existing line and passing the small circle, then mark the correct option red. In portrait, static camera, no zoom, no pan."
    DEFAULT_VLM_PROMPT="Which option lies on the line that passes through the small circle and is perpendicular to the existing line? Answer an option in A-E."
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin=50
        tries=0
        while tries<999:
            p1,p2=self.pick_target_point(0.8), self.pick_target_point(0.8)
            angle=math.atan2(p2.y - p1.y, p2.x - p1.x)+math.pi/2*self._rng.choice([-1,1])
            length=self.canvas_short_side*self._rng.uniform(0.3,0.6)
            target=Point(
                x=p1.x+length*math.cos(angle),
                y=p1.y+length*math.sin(angle),
            )
            distance=self.distance(p1,p2)
            if distance<self.canvas_short_side*0.3 or not self.inside_canvas(target):
                tries+=1
                continue
            break
        self.points = (p1, p2)
        self.target_point = target
        self.place_candidates_line(target,angle+math.pi/2)
        return self.save_puzzle()

    def build_record_extra(self) -> dict[str, object]:
        return {
            "reference_line_endpoints": [self.points[0].to_list(), self.points[1].to_list()],
            "through_point": self.points[0].to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points)
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
