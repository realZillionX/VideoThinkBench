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
        "A 512x512 white canvas shows two small 7 px outlined black endpoint circles connected by one black 5 px segment. "
        "Five candidate markers A-E are arranged near the hidden perpendicular bisector; each marker is a 10 px white circle with a 4 px dark gray outline RGB(32,32,32) and a black uppercase letter. "
        "The video first holds the endpoint circles, the connecting segment, and the candidate markers, then draws a long black 5 px bisector line through the segment midpoint in the perpendicular direction so it runs across the canvas. "
        "In the final frame, only the candidate lying on that perpendicular bisector changes to pale red fill RGB(255,220,220) with a dark red outline RGB(198,24,24), while all other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows two small endpoint circles connected by a black segment and five labeled candidate circles "
        "A-E. Determine which candidate lies on the perpendicular bisector of the segment, meaning it is on the line "
        "through the midpoint that is perpendicular to the segment. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 80
        tries = 0
        min_dist = self.canvas_short_side * 0.25
        while tries < 999:
            p1, p2 = self.pick_target_point(0.7), self.pick_target_point(0.7)
            if distance(p1, p2) < min_dist:
                tries += 1
                continue

            # Calculate midpoint
            midpoint = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            
            # Calculate perpendicular vector
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            perp_dx, perp_dy = -dy, dx

            # Normalize the perpendicular vector
            length = math.sqrt(perp_dx**2 + perp_dy**2)
            if length == 0:
                tries += 1
                continue
            
            u_perp_dx, u_perp_dy = perp_dx / length, perp_dy / length

            # Pick a random distance along the bisector
            dist_from_mid = self._rng.choice([-1, 1]) * self._rng.uniform(
                self.canvas_short_side * 0.4, self.canvas_short_side * 1.0
            )
            
            target = Point(
                midpoint.x + dist_from_mid * u_perp_dx,
                midpoint.y + dist_from_mid * u_perp_dy
            )

            self.points = (p1, p2)
            self.target_point = target
            
            # Place candidates along a line roughly perpendicular to the bisector itself
            # (i.e., parallel to the original p1-p2 segment)
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            self.place_candidates_line(target, angle + self._rng.uniform(-0.1, 0.1))
            
            if not self.check_candidates_inside():
                tries += 1
                continue
            break

        record = self.save_puzzle()
        record.points = self.points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1, p2 = self.points
        self.draw_circle(draw, p1, 7)
        self.draw_circle(draw, p2, 7)

        # Draw the original segment
        self.draw_line(draw, [p1, p2])
        if highlight_label:

            # Draw the perpendicular bisector line
            midpoint = Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            perp_dx, perp_dy = -dy, dx
            
            # Extend the bisector line far off-canvas to ensure it crosses the entire image
            line_p1 = Point(midpoint.x - perp_dx * 2, midpoint.y - perp_dy * 2)
            line_p2 = Point(midpoint.x + perp_dx * 2, midpoint.y + perp_dy * 2)
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
