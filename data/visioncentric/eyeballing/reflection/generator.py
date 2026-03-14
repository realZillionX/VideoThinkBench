"""Reflection puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class ReflectionGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the reflection of a point across a line."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/reflection"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas shows one black reflection-axis segment of medium thickness and one small black source point with a thick outline on one side of that axis. "
        "Five candidate markers A-E are clustered near the hidden reflected location; each marker is a small white circle with a thin dark gray outline and a black uppercase letter. "
        "The video first holds the reflection-axis segment, the source point, and the candidate markers, then draws one black connector segment of medium thickness from the source point straight across the axis to its reflected position. "
        "In the final state, only the reflected-position candidate changes to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows one black line, one small black source point, and five labeled candidate circles A-E on the "
        "other side or nearby. Determine which candidate is the mirror reflection of the source point across the black line. "
        "Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates a puzzle by defining a line and a point, then calculating the
        reflection of that point across the line.
        """
        tries = 0
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.05)

        while tries < 999:
            tries += 1
            axis_angle = self._rng.uniform(0.0, math.tau)
            normal_angle = axis_angle + math.pi / 2 * self._rng.choice([-1, 1])
            axis_midpoint = self.pick_target_point(
                0.55, padding=candidate_padding + self.canvas_short_side * 0.14,
            )

            try:
                p1, p2 = self.sample_symmetric_segment(
                    axis_midpoint,
                    axis_angle,
                    min_half_length=self.canvas_short_side * 0.16,
                    max_half_length=self.canvas_short_side * 0.28,
                    padding=self.line_width,
                )
                projection_point = self.sample_point_along_direction(
                    axis_midpoint,
                    axis_angle,
                    min_distance=0.0,
                    max_distance=self.canvas_short_side * 0.12,
                    padding=self.line_width,
                )
            except RuntimeError:
                continue

            projection_point = Point(
                x=(projection_point.x + axis_midpoint.x) * 0.5,
                y=(projection_point.y + axis_midpoint.y) * 0.5,
            )
            _, t = self.project_point_onto_line(projection_point, p1, p2)
            if not 0.2 <= t <= 0.8:
                continue

            offset_distance = self._rng.uniform(
                self.canvas_short_side * 0.22,
                self.canvas_short_side * 0.34,
            )
            source_point = self.point_on_ray(projection_point, normal_angle, offset_distance)
            target_point = self.point_on_ray(projection_point, normal_angle + math.pi, offset_distance)

            if not self.point_can_host_candidate(target_point):
                continue
            if not self.inside_canvas(source_point, padding=self.line_width):
                continue
            if self.distance_point_to_line(source_point, p1, p2) < self.canvas_short_side * 0.2:
                continue
            if self.distance(source_point, target_point) < self.canvas_short_side * 0.42:
                continue
            if min(self.distance(projection_point, p1), self.distance(projection_point, p2)) < self.canvas_short_side * 0.1:
                continue
            break
        else:
            raise RuntimeError("Failed to generate a valid reflection puzzle.")
        
        # Store the geometric elements for rendering.
        self.line_points = (p1, p2)
        self.source_point = source_point
        self.target_point = target_point
        
        self.place_candidates(target_point)
        return self.save_puzzle()

    def build_record_extra(self) -> dict[str, object]:
        return {
            "reflection_axis": [point.to_list() for point in self.line_points],
            "source_point": self.source_point.to_list(),
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        """Renders the reflection puzzle on an image canvas."""
        draw, base = self.get_draw_base()
        
        # Draw the line of reflection.
        self.draw_line(draw, self.line_points)
        
        # Draw the source point to be reflected.
        self.draw_circle(draw, self.source_point, 7)
        
        # If rendering the solution, draw the line connecting the source to its reflection.
        if highlight_label:
            self.draw_line(draw, [self.source_point, self.target_point])

        # Draw the multiple-choice candidate points.
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ReflectionGenerator.main(ReflectionGenerator, argv)

if __name__ == "__main__":
    main()
