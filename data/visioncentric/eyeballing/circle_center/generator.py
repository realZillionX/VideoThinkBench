"""CircleCenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class CircleCenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a circle."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/circle_center"
    DEFAULT_TI2V_PROMPT=(
        "On a white square canvas, draw one large black circle with no fill. Around its hidden center, place five small "
        "labeled candidate circles A-E with white fill, dark gray outlines, and black letters inside. Animate the solution "
        "by keeping the large circle fixed and then turning the candidate exactly at the center of the black circle into a "
        "pale red marker with a dark red outline while the other candidates remain white. In portrait, static camera, no "
        "zoom, no pan."
    )
    DEFAULT_VLM_PROMPT=(
        "A single large black circle is drawn on a white canvas, and five labeled candidate circles A-E are placed near its "
        "middle. Determine which candidate marks the exact center of the circle. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        target_point = self.pick_target_point()
        self.target_point = target_point
        r_max = min(
            target_point.x,
            target_point.y,
            self.canvas_dimensions[0] - target_point.x,
            self.canvas_dimensions[1] - target_point.y,
        )
        r=(self._rng.random()*0.5+0.25)*r_max
        self.r=r
        self.place_candidates(target_point)
        return self.save_puzzle()

    def build_record_extra(self) -> dict[str, object]:
        return {
            "circle_center": self.target_point.to_list(),
            "radius": self.r,
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        draw.ellipse(
            [
                (self.target_point.x - self.r, self.target_point.y - self.r),
                (self.target_point.x + self.r, self.target_point.y + self.r),
            ],
            outline="black",
            width=3,
        )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircleCenterGenerator.main(CircleCenterGenerator, argv)

if __name__ == "__main__":
    main()
