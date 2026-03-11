"""Circle Tangent Line puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class CircleTangentLineGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that require identifying a line tangent to a circle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/circle_tangent_line"
    DEFAULT_TI2V_PROMPT=(
        "A 512x512 white canvas shows one large unfilled circle drawn with a black 5 px outline, plus one small 7 px outlined black circle sitting exactly on the circumference to mark the tangency point. "
        "Five candidate markers A-E form a short straight row near the hidden tangent direction; each is a 10 px white circle with a 4 px dark gray outline RGB(32,32,32) and a black uppercase letter. "
        "The video first holds the circle, the marked contact point, and the candidate row, then draws a single black 5 px tangent segment starting at the contact point and extending in one direction toward the correct candidate. "
        "In the final frame, only the candidate on that tangent segment changes to pale red fill RGB(255,220,220) with a dark red outline RGB(198,24,24), while the others remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A large black circle is shown on a white canvas with one marked point on the circumference and five labeled "
        "candidate circles A-E nearby. Determine which candidate lies on the line tangent to the circle at the marked "
        "contact point, where the tangent is perpendicular to the radius at that point. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        width, height = self.canvas_dimensions
        tries=0
        while tries<999:
            # 1. Define the circle's center and radius
            center = self.pick_target_point(0.6) # Keep center away from edges
            min_radius = self.canvas_short_side * 0.2
            max_radius = self.canvas_short_side * 0.4
            radius = self._rng.uniform(min_radius, max_radius)
            circle_fits = (
                center.x - radius >= self.margin and
                center.y - radius >= self.margin and
                center.x + radius <= width - self.margin and
                center.y + radius <= height - self.margin
            )
            if not circle_fits:
                tries += 1
                continue

            # 2. Define the point on the circle's circumference
            radius_angle = self._rng.uniform(0, math.tau)
            point_on_circle = Point(
                x=center.x + radius * math.cos(radius_angle),
                y=center.y + radius * math.sin(radius_angle)
            )

            # 3. The tangent line is perpendicular to the radius
            tangent_angle = radius_angle + math.pi / 2

            # 4. Define the target point on the tangent line
            # Choose a distance from the point on the circle to the target
            dist_from_poc = self._rng.uniform(0.2, 0.4) * self.canvas_short_side
            # Randomly pick a direction along the tangent line
            dist_from_poc *= self._rng.choice([-1, 1])

            target_point = Point(
                x=point_on_circle.x + dist_from_poc * math.cos(tangent_angle),
                y=point_on_circle.y + dist_from_poc * math.sin(tangent_angle)
            )

            self.circle_center = center
            self.circle_radius = radius
            self.point_on_circle = point_on_circle
            self.target_point = target_point
            
            self.place_candidates_line(target_point, tangent_angle+math.pi/2+self._rng.uniform(-0.1,0.1))
            # 5. Ensure the target point is within the canvas
            if not self.check_candidates_inside():
                tries += 1
                continue
            
            # If everything is valid, break the loop
            break

        
        record = self.save_puzzle()
        record.circle_center = self.circle_center
        record.circle_radius = self.circle_radius
        record.point_on_circle = self.point_on_circle
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        # Draw the main circle
        self.draw_circle(draw, self.circle_center, self.circle_radius)
        # Highlight the point on the circumference where the tangent touches
        self.draw_circle(draw, self.point_on_circle, 7)

        # In the solution image, draw the tangent line segment
        if highlight_label:
            self.draw_line(draw, [self.point_on_circle, self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircleTangentLineGenerator.main(CircleTangentLineGenerator, argv)

if __name__ == "__main__":
    main()
