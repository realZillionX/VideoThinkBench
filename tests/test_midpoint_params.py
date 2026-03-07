import tempfile
import unittest
from pathlib import Path

from data.visioncentric.eyeballing.midpoint.generator import MidpointGenerator


class MidpointParameterTests(unittest.TestCase):
    def test_point_radius_and_line_width_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "midpoint"
            generator = MidpointGenerator(
                output_dir=output_dir,
                seed=123,
                point_radius=14,
                line_width=7,
            )

            record = generator.create_puzzle(puzzle_id="midpoint-params")

            self.assertEqual(generator.point_radius, 14)
            self.assertEqual(generator.line_width, 7)
            self.assertEqual(record.point_radius, 14)
            self.assertTrue((output_dir / record.image).exists())
            self.assertTrue((output_dir / record.solution_image_path).exists())


if __name__ == "__main__":
    unittest.main()
