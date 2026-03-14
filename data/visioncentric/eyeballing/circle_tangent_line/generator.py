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
        "A square white canvas shows one large unfilled circle drawn with a black outline of medium thickness, plus one small black circle with a thick outline sitting exactly on the circumference to mark the tangency point. "
        "Five candidate markers A-E form a short straight row near the hidden tangent direction; each is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the circle, the marked contact point, and the candidate row, then draws a single black tangent segment of medium thickness starting at the contact point and extending in one direction toward the correct candidate. "
        "In the final frame, only the candidate on that tangent segment changes to pale red fill with a dark red outline, while the others remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A large black circle is shown on a white canvas with one marked point on the circumference and five labeled "
        "candidate circles A-E nearby. Determine which candidate lies on the line tangent to the circle at the marked "
        "contact point, where the tangent is perpendicular to the radius at that point. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        tries=0
        anchor_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)
        while tries<999:
            # 1. Define the circle's center and radius
            center = self.pick_target_point(0.42, padding=anchor_padding + self.canvas_short_side * 0.08)
            min_radius = self.canvas_short_side * 0.22
            max_radius = self.canvas_short_side * 0.36
            radius = self._rng.uniform(min_radius, max_radius)
            circle_fits = self.circle_fits(center, radius, extra_padding=max(self.line_width, anchor_padding * 0.2))
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

            if self._rng.random() < 0.5:
                tangent_angle += math.pi
            try:
                target_point = self.sample_point_along_direction(
                    point_on_circle,
                    tangent_angle,
                    min_distance=self.canvas_short_side * 0.26,
                    max_distance=self.canvas_short_side * 0.38,
                    padding=anchor_padding,
                )
            except RuntimeError:
                tries += 1
                continue

            if not self.point_can_host_candidate(target_point, extra_padding=self.canvas_short_side * 0.03):
                tries += 1
                continue
            if self.distance(point_on_circle, target_point) < self.canvas_short_side * 0.26:
                tries += 1
                continue
            self.circle_center = center
            self.circle_radius = radius
            self.point_on_circle = point_on_circle
            self.target_point = target_point
            
            try:
                self.place_candidates_line(target_point, tangent_angle + math.pi / 2 + self._rng.uniform(-0.05, 0.05))
            except RuntimeError:
                tries += 1
                continue
            
            # If everything is valid, break the loop
            break
        else:
            raise RuntimeError("Failed to find valid tangent-line geometry after 999 tries")

        
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
