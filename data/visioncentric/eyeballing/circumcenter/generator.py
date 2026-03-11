"""Circumcenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

def calculate_circumcenter(p1: Point, p2: Point, p3: Point):
    """
    Calculates the circumcenter and circumradius of a triangle defined by three points.

    Args:
        p1 (Point): The first point.
        p2 (Point): The second point.
        p3 (Point): The third point.

    Returns:
        A tuple containing:
        - Point: The circumcenter of the triangle.
        - float: The circumradius of the triangle.
        Returns (None, None) if the points are collinear.
    """
    # For clarity, unpack the coordinates
    ax, ay = p1.x, p1.y
    bx, by = p2.x, p2.y
    cx, cy = p3.x, p3.y

    # Calculate the common denominator D
    # D is proportional to the signed area of the triangle.
    # If D is 0, the points are collinear.
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    
    # Use a small epsilon for floating-point comparison to check for collinearity
    if abs(D) < 1e-9:
        return 0, 0

    # Pre-calculate squares of coordinates
    sq_a = ax**2 + ay**2
    sq_b = bx**2 + by**2
    sq_c = cx**2 + cy**2

    # Calculate the coordinates of the circumcenter (Ux, Uy)
    Ux = (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / D
    Uy = (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / D

    circumcenter = Point(Ux, Uy)

    # The circumradius is the distance from the circumcenter to any of the vertices.
    # We'll use p1.
    radius = math.sqrt((ax - Ux)**2 + (ay - Uy)**2)

    return circumcenter, radius
class CircumcenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the circumcenter of a triangle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/circumcenter"
    DEFAULT_TI2V_PROMPT=(
        "A 512x512 white canvas shows a triangle drawn only as a black 5 px outline, with no fill and no center marks. "
        "Five candidate markers A-E sit near the hidden circumcenter; each marker is a 10 px white circle with a 4 px dark gray outline RGB(32,32,32) and a black uppercase letter. "
        "The video first holds the triangle and the candidate markers, then draws one black 5 px circumcircle centered on the correct option so that the circle passes exactly through all three triangle vertices. "
        "In the final state, the circumcenter marker alone changes to pale red fill RGB(255,220,220) with a dark red outline RGB(198,24,24), while the other candidates remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A black triangle outline is shown on a white canvas with five labeled candidate circles A-E near its middle. Find "
        "the circumcenter, the point that is equally distant from all three vertices and would be the center of the circle "
        "through the three vertices. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        width, height = self.canvas_dimensions
        tries=0
        while tries<999:
            p1,p2,p3=self.pick_target_point(0.8), self.pick_target_point(0.8), self.pick_target_point(0.8)
            circumcenter,r=calculate_circumcenter(p1,p2,p3)
            circle_fits = (
                circumcenter.x - r >= self.margin and
                circumcenter.y - r >= self.margin and
                circumcenter.x + r <= width - self.margin and
                circumcenter.y + r <= height - self.margin
            )
            if r<self.canvas_short_side*0.1 or not self.inside_canvas(circumcenter) or not circle_fits:
                tries+=1
                continue
            break
        self.triangle_points = (p1, p2, p3)
        self.target_point = circumcenter
        self.r = r
        self.place_candidates(circumcenter)
        record = self.save_puzzle()
        record.triangle_points=self.triangle_points
        record.r=self.r
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,list(self.triangle_points)+[self.triangle_points[0]])
        if highlight_label:
            self.draw_circle(draw,self.target_point,self.r)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircumcenterGenerator.main(CircumcenterGenerator, argv)

if __name__ == "__main__":
    main()
