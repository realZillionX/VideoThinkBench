import json
import tempfile
import unittest
from pathlib import Path

from core.io import read_json, read_jsonl
from core.schemas import CanonicalAnswer, CanonicalAssets, CanonicalSample
from data.exporters.diffsynth_image import export_diffsynth_image
from data.exporters.diffsynth_video import export_diffsynth_video
from data.exporters.ms_swift import export_ms_swift


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            CanonicalSample(
                id="eye-1",
                task_group="eyeballing",
                task_type="midpoint",
                prompt_raw="raw",
                prompt_train="Find the midpoint.",
                assets=CanonicalAssets(
                    puzzle_image="/tmp/eye_puzzle.png",
                    solution_image="/tmp/eye_solution.png",
                    solution_video="/tmp/eye_solution.mp4",
                ),
                answer=CanonicalAnswer(correct_option="C"),
            ),
            CanonicalSample(
                id="maze-1",
                task_group="maze",
                task_type="maze_square",
                prompt_raw="raw",
                prompt_train="Draw a red path connecting two red dots without touching the black walls.",
                assets=CanonicalAssets(
                    puzzle_image="/tmp/maze_puzzle.png",
                    solution_image="/tmp/maze_solution.png",
                    solution_video="/tmp/maze_solution.mp4",
                ),
                answer=CanonicalAnswer(path_cell_ids=[1, 2, 3]),
            ),
            CanonicalSample(
                id="vp-1",
                task_group="visual_puzzle",
                task_type="color_grid",
                prompt_raw="raw",
                prompt_train="What is the missing color?",
                assets=CanonicalAssets(
                    puzzle_image="/tmp/vp_puzzle.png",
                    solution_image="/tmp/vp_solution.png",
                ),
                answer=CanonicalAnswer(correct_option="blue"),
            ),
        ]

    def test_export_ms_swift_preserves_solution_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            outputs = export_ms_swift(self.samples, output_dir=output_dir, modes=["sft", "grpo"], task_groups=["eyeballing", "maze", "visual_puzzle"])
            self.assertEqual(len(outputs), 2)

            sft_rows = read_jsonl(output_dir / "train_sft.jsonl")
            grpo_rows = read_jsonl(output_dir / "train_grpo.jsonl")

            by_id_sft = {row["id"]: row for row in sft_rows}
            by_id_grpo = {row["id"]: row for row in grpo_rows}

            self.assertEqual(by_id_sft["eye-1"]["solution"], "C")
            self.assertEqual(by_id_sft["maze-1"]["solution"], "[1, 2, 3]")
            self.assertEqual(by_id_sft["vp-1"]["solution"], "blue")
            self.assertIn("assistant", {msg["role"] for msg in by_id_sft["eye-1"]["messages"]})
            self.assertNotIn("assistant", {msg["role"] for msg in by_id_grpo["eye-1"]["messages"]})

    def test_export_diffsynth_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            image_out = output_dir / "image.json"
            video_out = output_dir / "video.csv"

            export_diffsynth_image(self.samples, output_path=image_out, task_groups=["eyeballing", "maze", "visual_puzzle"])
            export_diffsynth_video(self.samples, output_path=video_out, task_groups=["eyeballing", "maze", "visual_puzzle"])

            image_rows = read_json(image_out)
            video_text = video_out.read_text(encoding="utf-8")

            self.assertEqual(len(image_rows), 3)
            self.assertIn('"prompt"', json.dumps(image_rows, ensure_ascii=False))
            self.assertIn('"edit_image"', json.dumps(image_rows, ensure_ascii=False))
            self.assertIn('"image"', json.dumps(image_rows, ensure_ascii=False))
            self.assertIn('"vp-1"', json.dumps(image_rows, ensure_ascii=False))
            self.assertIn('"eye-1"', video_text)
            self.assertIn('"maze-1"', video_text)
            self.assertNotIn('"vp-1"', video_text)


if __name__ == "__main__":
    unittest.main()
