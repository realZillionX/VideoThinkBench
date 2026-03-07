import tempfile
import unittest
from pathlib import Path

from core.io import write_json
from data.visioncentric.maze.maze_square.evaluator import MazeEvaluator
from data.visioncentric.maze.maze_square.generator import MazeGenerator


class MazeOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "maze_square"
        self.generator = MazeGenerator(output_dir=self.output_dir, rows=11, cols=11, cell_size=24, seed=42)
        self.record = self.generator.create_puzzle(puzzle_id="maze-square-test")
        self.metadata_path = self.output_dir / "data.json"
        write_json(self.metadata_path, [self.record.to_dict()])
        self.evaluator = MazeEvaluator(self.metadata_path, base_dir=self.output_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_solution_image_passes(self) -> None:
        candidate_path = self.output_dir / self.record.solution_image_path
        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertTrue(result.connected)
        self.assertTrue(result.touches_start)
        self.assertTrue(result.touches_goal)
        self.assertFalse(result.overlaps_walls)

    def test_puzzle_image_fails(self) -> None:
        candidate_path = self.output_dir / self.record.image
        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertFalse(result.connected)
        self.assertTrue(result.message)


if __name__ == "__main__":
    unittest.main()
