"""Parallelogram puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from data.point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class ParallelogramGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that hide the parallelogram of a segment."""
    DEFAULT_OUTPUT_DIR="data/visioncentric/eyeballing/parallelogram"
    DEFAULT_TI2V_PROMPT=(
        "A 512x512 white canvas shows three known vertices of a parallelogram as an open black 5 px broken line shaped like a V, with two adjacent sides already visible and one opposite corner missing. "
        "Five candidate markers A-E are placed near the missing corner; each marker is a 10 px white circle with a 4 px dark gray outline RGB(32,32,32) and a black uppercase letter. "
        "The video first holds the open broken line and the candidate markers, then draws two black 5 px closing segments from one visible endpoint to the correct candidate and from that candidate to the other visible endpoint so the full quadrilateral becomes a parallelogram. "
        "In the final state, only the correct missing-vertex marker changes to pale red fill RGB(255,220,220) with a dark red outline RGB(198,24,24), while the other four candidates remain white. In portrait. Static camera."
    )
    DEFAULT_VLM_PROMPT=(
        "A white canvas shows three black vertices connected as two adjacent sides of a parallelogram, plus five labeled "
        "candidate circles A-E near the missing corner. Determine which candidate is the fourth vertex that closes the "
        "parallelogram. Answer with one option from A-E."
    )
    DEFAULT_TI2I_PROMPT = PointTargetPuzzleGenerator.strip_video_instruction(DEFAULT_TI2V_PROMPT)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        target_point = self.pick_target_point()
        tries=0
        distance_threshold = self.canvas_short_side * 0.3
        while tries<9999:
            p1, p2 = self.pick_target_point(0.5), self.pick_target_point(0.5)
            # Calculate the fourth point of the parallelogram
            p3 = Point(
                x=p1.x + p2.x - target_point.x,
                y=p1.y + p2.y - target_point.y,
            )
            angle1,angle2=math.atan2(p1.y-target_point.y,p1.x-target_point.x), math.atan2(p2.y-target_point.y,p2.x-target_point.x)
            if not self.inside_canvas(p3) or self.distance(target_point, p1) < distance_threshold or self.distance(target_point, p2) < distance_threshold or abs(math.sin(angle2-angle1))<0.5:
                tries+=1
                continue
            break
        self.parallelogram_points = (p1, p2, p3, target_point)
        self.place_candidates(target_point)
        return self.save_puzzle()

    def build_record_extra(self) -> dict[str, object]:
        return {
            "known_vertices": [point.to_list() for point in self.parallelogram_points[:3]],
        }

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1,p2,p3,target=self.parallelogram_points
        self.draw_line(draw,[p1,p3,p2])
        if highlight_label:
            self.draw_line(draw,[p2,target,p1],)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ParallelogramGenerator.main(ParallelogramGenerator, argv)

if __name__ == "__main__":
    main()
