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
        "On a white square canvas, draw one existing black reference line and one separate small black outlined circle that "
        "marks the point a new line must pass through. Place five labeled candidate circles A-E near the hidden parallel "
        "line, each with white fill, dark gray outline, and a black letter. Animate the solution by first holding the "
        "reference line and the through-point circle, then drawing a solid black line through the small circle that stays "
        "parallel to the reference line, and finally changing the candidate lying on that new parallel line to pale red "
        "with a dark red outline while the remaining candidates stay white. In portrait, static camera, no zoom, no pan."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows one black reference line, one separate small black circle indicating a required through-point, "
        "and five labeled candidate circles A-E. Determine which candidate lies on the line through the small circle that is "
        "parallel to the reference line. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin=80
        tries=0
        distance_threshold = self.canvas_short_side * 0.3
        while tries<9999:
            p0, p1, p2 = self.pick_target_point(0.7), self.pick_target_point(0.7), self.pick_target_point(0.7)
            distance=distanceToLine(p0, p1, p2)
            angle=math.atan2(p2.y - p1.y, p2.x - p1.x)
            length=self.canvas_short_side*self._rng.uniform(0.5,1.0)
            target=Point(
                x=p0.x+length*math.cos(angle),
                y=p0.y+length*math.sin(angle),
            )
            if not self.inside_canvas(target) or self.distance(p1, p2) < distance_threshold or distance < distance_threshold:
                tries+=1
                continue
            self.points = (p0, p1, p2)
            self.target_point = target
            self.place_candidates_line(target,angle+math.pi/2+self._rng.uniform(-0.1,0.1))
            if not self.check_candidates_inside():
                tries += 1
                continue
            break
        else:
            raise RuntimeError("Failed to find valid parallel geometry after 9999 tries")
        return self.save_puzzle()

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
    ParallelGenerator.main(ParallelGenerator, argv)

if __name__ == "__main__":
    main()
