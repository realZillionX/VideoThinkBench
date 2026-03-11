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
        "A 512x512 white canvas shows two solid black 5 px line segments meeting at one shared vertex and opening into a clear angle, with no arrows and no extra marks. "
        "Five small candidate markers A-E sit near the hidden bisector on a short straight row: each marker is a 10 px circle with white fill, a 4 px dark gray outline RGB(32,32,32), and a black uppercase letter. "
        "The video holds this angle-and-candidates puzzle first, then draws one black 5 px bisector segment outward from the shared vertex through the interior of the angle toward the correct marker. "
        "In the final state, only the correct candidate changes to pale red fill RGB(255,220,220) with a dark red outline RGB(198,24,24), while all other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows two black rays meeting at one vertex and five labeled candidate circles A-E near the interior "
        "of the angle. Identify which candidate lies on the exact angle bisector, meaning the line from the vertex that "
        "splits the angle into two equal angles. Answer with a single option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)


    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin=50
        tries=0
        distance_threshold = self.canvas_short_side * 0.3
        while tries<999:
            p0, p1, p2 = self.pick_target_point(0.7), self.pick_target_point(0.7), self.pick_target_point(0.7)
            angle1,angle2=math.atan2(p1.y-p0.y,p1.x-p0.x), math.atan2(p2.y-p0.y,p2.x-p0.x)
            mid_angle_value=mid_angle(angle1,angle2)
            length=self.canvas_short_side*self._rng.uniform(0.5,1.0)
            target=Point(
                x=p0.x+length*math.cos(mid_angle_value),
                y=p0.y+length*math.sin(mid_angle_value),
            )
            if not self.inside_canvas(target) or self.distance(p0, p1) < distance_threshold or self.distance(p0, p2) < distance_threshold or abs(math.sin(angle2-angle1))<0.3:
                tries+=1
                continue
            break
        self.points = (p2, p0, p1)
        self.target_point = target
        self.place_candidates_line(target,mid_angle_value+math.pi/2+self._rng.uniform(-0.1,0.1))
        return self.save_puzzle()

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
