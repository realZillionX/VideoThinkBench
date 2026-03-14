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
        "A square white canvas shows two small black endpoint circles with thick outlines connected by one black segment of medium thickness. "
        "Five candidate markers A-E are arranged near the hidden perpendicular bisector; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the endpoint circles, the connecting segment, and the candidate markers, then draws a long black bisector line of medium thickness through the segment midpoint in the perpendicular direction so it runs across the canvas. "
        "In the final frame, only the candidate lying on that perpendicular bisector changes to pale red fill with a dark red outline, while all other markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows two small endpoint circles connected by a black segment and five labeled candidate circles "
        "A-E. Determine which candidate lies on the perpendicular bisector of the segment, meaning it is on the line "
        "through the midpoint that is perpendicular to the segment. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 80
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        tries = 0
        min_dist = self.canvas_short_side * 0.25
        while tries < 999:
            p1 = self.pick_target_point(0.55, padding=candidate_padding)
            p2 = self.pick_target_point(0.55, padding=candidate_padding)
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

            target_angle = math.atan2(u_perp_dy, u_perp_dx)
            if self._rng.random() < 0.5:
                target_angle += math.pi
            try:
                target = self.sample_point_along_direction(
                    midpoint,
                    target_angle,
                    min_distance=self.canvas_short_side * 0.24,
                    max_distance=self.canvas_short_side * 0.36,
                    padding=candidate_padding,
                )
            except RuntimeError:
                tries += 1
                continue

            self.points = (p1, p2)
            self.target_point = target
            
            # Place candidates along a line roughly perpendicular to the bisector itself
            # (i.e., parallel to the original p1-p2 segment)
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)
            try:
                self.place_candidates_line(target, angle + self._rng.uniform(-0.1, 0.1))
            except RuntimeError:
                tries += 1
                continue
            
            if not self.check_candidates_inside():
                tries += 1
                continue
            break

        else:
            raise RuntimeError("Failed to find valid perpendicular bisector geometry after 999 tries")

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
            line_angle = math.atan2(dx, -dy)
            line_p1, line_p2 = self.clip_line_to_canvas(midpoint, line_angle, padding=self.line_width)
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
