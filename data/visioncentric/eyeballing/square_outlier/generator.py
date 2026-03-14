"""Square outlier puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import itertools
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point, PointCandidate

class SquareOutlierGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles where four of five points form a square."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/square_outlier"
    DEFAULT_TI2V_PROMPT=(
        "A square white canvas initially shows only five candidate markers A-E and no lines or helper shapes. "
        "Each candidate marker is a small white circle with a thin dark gray outline and a black uppercase letter, and exactly four of the five points are the vertices of one square while the fifth point is the outlier. "
        "The video first holds the five markers alone, then draws a black square outline of medium thickness through the four matching vertices, including rotation if the square is tilted. "
        "In the final state, only the single outlier marker changes to pale red fill with a dark red outline, while the four square-vertex markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows five labeled candidate circles A-E. Four of them can be connected to form a square. Determine "
        "which labeled point is the outlier that does not belong to that square. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def _forms_square(self, points: List[Point]) -> bool:
        if len(points) != 4:
            return False
        distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                dx = points[i].x - points[j].x
                dy = points[i].y - points[j].y
                distances.append(dx * dx + dy * dy)
        distances.sort()
        if distances[0] <= 1e-6:
            return False
        side = distances[0]
        diag = distances[-1]
        side_tol = side * 0.08
        diag_tol = diag * 0.08
        return (
            all(abs(value - side) <= side_tol for value in distances[:4])
            and abs(distances[4] - diag) <= diag_tol
            and abs(distances[5] - diag) <= diag_tol
            and abs(diag - 2.0 * side) <= max(side_tol * 2.0, diag_tol)
        )

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates a puzzle with five points, where four form a square and one is an outlier.
        The outlier is the target.
        """
        tries = 0
        candidate_padding = self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04)
        min_point_distance = max(self.minimum_candidate_spacing(scale=1.05), self.canvas_short_side * 0.16)
        while tries < 999:
            # 1. Generate a square with a random size, rotation, and position.
            
            # The side length of the square.
            side_length = self._rng.uniform(self.canvas_short_side * 0.28, self.canvas_short_side * 0.42)
            half_side = side_length / 2
            
            center = self.pick_target_point(0.5, padding=candidate_padding + side_length * 0.75)
            
            # The rotation angle of the square.
            angle = self._rng.uniform(0, 2 * math.pi)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Vertices of a non-rotated square centered at the origin.
            unrotated_corners = [
                Point(half_side, half_side),
                Point(-half_side, half_side),
                Point(-half_side, -half_side),
                Point(half_side, -half_side),
            ]
            
            square_points = []
            valid_square = True
            for p in unrotated_corners:
                # Apply rotation and then translation to get final vertex coordinates.
                rotated_x = p.x * cos_a - p.y * sin_a
                rotated_y = p.x * sin_a + p.y * cos_a
                final_point = Point(center.x + rotated_x, center.y + rotated_y)
                
                if not self.point_can_host_candidate(final_point):
                    valid_square = False
                    break
                square_points.append(final_point)
            
            if not valid_square:
                tries += 1
                continue

            # 2. Generate the outlier point (the target).
            target_point = None
            outlier_tries = 0
            while outlier_tries < 100:
                potential_outlier = self.pick_target_point(0.5, padding=candidate_padding)
                
                # Ensure the outlier is not too close to any of the square's vertices.
                min_dist = min(self.distance(potential_outlier, v) for v in square_points)
                if min_dist > max(min_point_distance, self.canvas_short_side * 0.18):
                    target_point = potential_outlier
                    break
                outlier_tries += 1
            
            if target_point is None: # Failed to place an outlier
                tries += 1
                continue

            labels = ['A', 'B', 'C', 'D', 'E']
            all_points = square_points + [target_point]
            if not self.points_are_well_spaced(all_points, min_distance=min_point_distance):
                tries += 1
                continue
            square_count = 0
            for combo in itertools.combinations(all_points, 4):
                combo_points = list(combo)
                if self._forms_square(combo_points):
                    square_count += 1
            if square_count != 1:
                tries += 1
                continue

            # A valid configuration has been found.
            self.target_point = target_point
            self.square_points = square_points
            break
        
        if tries >= 999:
             raise RuntimeError("Failed to generate a valid square outlier puzzle.")

        # 3. Manually create the list of candidates.
        # This is required because place_candidates would place distractors
        # according to some rule, but here our distractors are part of the core puzzle.
        all_points = self.square_points + [self.target_point]
        self._rng.shuffle(all_points)
        
        labels = ['A', 'B', 'C', 'D', 'E']
        self.candidates = []
        for i, p in enumerate(all_points):
            label = labels[i]
            self.candidates.append(PointCandidate(x=p.x, y=p.y, label=label))
            if p == self.target_point:
                self.correct_label = label
        if not self.validate_candidate_layout(self.candidates, min_distance=min_point_distance * 0.95):
            raise RuntimeError("Generated square outlier candidates do not fit the canvas")

        # 4. Save the puzzle data and return the record.
        record = self.save_puzzle()
        record.square_points = self.square_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        """
        Renders the puzzle image. For the solution, it draws the square.
        """
        draw, base = self.get_draw_base()
        
        # For the solution image, draw the square formed by the four non-outlier points.
        if highlight_label:
            # The points are already ordered. Add the first point to the end to close the shape.
            points_to_draw = self.square_points + [self.square_points[0]]
            self.draw_line(draw, points_to_draw)

        # Draw all five candidate points and their labels.
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    SquareOutlierGenerator.main(SquareOutlierGenerator, argv)

if __name__ == "__main__":
    main()
