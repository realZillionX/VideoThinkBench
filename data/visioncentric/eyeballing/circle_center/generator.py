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
        "A square white canvas shows one large unfilled circle centered somewhere on the page, drawn with a thin black outline and no other geometry. "
        "Five candidate markers A-E are clustered around the hidden center of that circle; each marker is a small circle with white fill, a thin dark gray outline, and a black uppercase letter. "
        "The video begins with a short hold on the static circle and the five markers, and there is no construction line or extra shape added during the middle phase. "
        "The final state simply changes the exact center marker to pale red fill with a dark red outline, while the other four markers remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A single large black circle is drawn on a white canvas, and five labeled candidate circles A-E are placed near its "
        "middle. Determine which candidate marks the exact center of the circle. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        target_point = self.pick_target_point(0.45, padding=self.candidate_anchor_padding(extra=self.canvas_short_side * 0.04))
        self.target_point = target_point
        left, top, right, bottom = self.canvas_bounds()
        r_max = min(
            target_point.x - left,
            target_point.y - top,
            right - target_point.x,
            bottom - target_point.y,
        )
        r=self._rng.uniform(0.35, 0.58) * r_max
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
