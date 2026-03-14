from __future__ import annotations

import math
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from PIL import Image

from data.registry import EYEBALLING_TASKS, TASK_SPECS


class EyeballingGenerationQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="eyeballing-quality-")
        self.output_root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _load_generator_cls(self, task_name: str):
        spec = TASK_SPECS[task_name]
        module = import_module(spec.module)
        return getattr(module, spec.class_name)

    def _generate(self, task_name: str, seed: int):
        generator_cls = self._load_generator_cls(task_name)
        out_dir = self.output_root / task_name / str(seed)
        generator = generator_cls(output_dir=out_dir, seed=seed)
        record = (
            generator.create_random_puzzle()
            if hasattr(generator, "create_random_puzzle")
            else generator.create_puzzle()
        )
        puzzle_path = out_dir / record.image
        solution_path = out_dir / record.solution_image_path
        return generator, puzzle_path, solution_path

    def _assert_blank_border(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("RGB")
        pixels = image.load()
        width, height = image.size
        for x in range(width):
            self.assertEqual(pixels[x, 0], (255, 255, 255), f"top border is dirty: {image_path}")
            self.assertEqual(pixels[x, height - 1], (255, 255, 255), f"bottom border is dirty: {image_path}")
        for y in range(height):
            self.assertEqual(pixels[0, y], (255, 255, 255), f"left border is dirty: {image_path}")
            self.assertEqual(pixels[width - 1, y], (255, 255, 255), f"right border is dirty: {image_path}")

    def _assert_candidate_layout(self, generator) -> None:
        self.assertTrue(
            generator.validate_candidate_layout(generator.candidates),
            "candidate labels or markers exceed the canvas or become too crowded",
        )

    def _triangle_angles(self, triangle_points) -> tuple[float, float, float]:
        p1, p2, p3 = triangle_points
        return (
            generator_angle(p2, p1, p3),
            generator_angle(p1, p2, p3),
            generator_angle(p1, p3, p2),
        )

    def _assert_triangle_quality(
        self,
        generator,
        *,
        min_side_ratio: float,
        min_area_ratio: float,
        min_angle_deg: float,
        max_angle_deg: float,
        forbid_near_right: bool = False,
    ) -> None:
        p1, p2, p3 = generator.triangle_points
        side_lengths = [
            math.hypot(a.x - b.x, a.y - b.y)
            for a, b in ((p1, p2), (p2, p3), (p3, p1))
        ]
        area = abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) / 2.0
        angles = self._triangle_angles(generator.triangle_points)
        short_side = generator.canvas_short_side
        self.assertGreaterEqual(min(side_lengths), short_side * min_side_ratio)
        self.assertGreaterEqual(area, (short_side ** 2) * min_area_ratio)
        self.assertGreaterEqual(min(angles), min_angle_deg)
        self.assertLessEqual(max(angles), max_angle_deg)
        if forbid_near_right:
            self.assertTrue(all(abs(angle - 90.0) >= 8.0 for angle in angles))

    def test_all_eyeballing_tasks_keep_borders_clean_and_candidates_inside(self) -> None:
        for task_name in EYEBALLING_TASKS:
            for seed in (0, 1):
                with self.subTest(task=task_name, seed=seed):
                    generator, puzzle_path, solution_path = self._generate(task_name, seed)
                    self._assert_blank_border(puzzle_path)
                    self._assert_blank_border(solution_path)
                    self._assert_candidate_layout(generator)

    def test_triangle_tasks_keep_minimum_drawable_area(self) -> None:
        cases = [
            ("triangle_center", (0, 7, 15), 0.20, 0.04, 31.0, 116.0, False),
            ("incenter", (0, 7, 15), 0.20, 0.04, 31.0, 116.0, False),
            ("circumcenter", (0, 7, 15), 0.20, 0.04, 31.0, 112.0, False),
            ("orthocenter", (0, 7, 15), 0.22, 0.045, 33.0, 110.0, True),
        ]
        for task_name, seeds, min_side_ratio, min_area_ratio, min_angle_deg, max_angle_deg, forbid_near_right in cases:
            for seed in seeds:
                with self.subTest(task=task_name, seed=seed):
                    generator, _, _ = self._generate(task_name, seed)
                    self._assert_triangle_quality(
                        generator,
                        min_side_ratio=min_side_ratio,
                        min_area_ratio=min_area_ratio,
                        min_angle_deg=min_angle_deg,
                        max_angle_deg=max_angle_deg,
                        forbid_near_right=forbid_near_right,
                    )

    def test_angle_and_polygon_tasks_avoid_degenerate_geometry(self) -> None:
        for seed in (0, 7, 15):
            with self.subTest(task="angle_bisector", seed=seed):
                generator, _, _ = self._generate("angle_bisector", seed)
                p1, vertex, p2 = generator.points
                opening = generator_angle(p1, vertex, p2)
                self.assertGreaterEqual(opening, 36.0)
                self.assertLessEqual(opening, 120.0)

            with self.subTest(task="parallelogram", seed=seed):
                generator, _, _ = self._generate("parallelogram", seed)
                p1, p2, p3, target = generator.parallelogram_points
                side_lengths = [
                    math.hypot(a.x - b.x, a.y - b.y)
                    for a, b in ((target, p1), (target, p2), (p1, p3), (p2, p3))
                ]
                area = polygon_area([p1, p3, p2, target])
                self.assertGreaterEqual(min(side_lengths), generator.canvas_short_side * 0.17)
                self.assertGreaterEqual(area, (generator.canvas_short_side ** 2) * 0.04)

            with self.subTest(task="isosceles_trapezoid", seed=seed):
                generator, _, _ = self._generate("isosceles_trapezoid", seed)
                p1, p2, p3, target = generator.trapezoid_points
                top_base = math.hypot(target.x - p3.x, target.y - p3.y)
                area = polygon_area([p3, p1, p2, target])
                self.assertGreaterEqual(top_base, generator.canvas_short_side * 0.18)
                self.assertGreaterEqual(area, (generator.canvas_short_side ** 2) * 0.05)

    def test_candidate_only_tasks_keep_legible_shapes(self) -> None:
        for seed in (0, 7, 15):
            with self.subTest(task="right_triangle", seed=seed):
                generator, _, _ = self._generate("right_triangle", seed)
                p_right, p1, p2 = generator.right_triangle_points
                leg_1 = math.hypot(p_right.x - p1.x, p_right.y - p1.y)
                leg_2 = math.hypot(p_right.x - p2.x, p_right.y - p2.y)
                ratio = max(leg_1, leg_2) / min(leg_1, leg_2)
                self.assertGreaterEqual(min(leg_1, leg_2), generator.canvas_short_side * 0.20)
                self.assertLessEqual(ratio, 1.35)

            with self.subTest(task="square_outlier", seed=seed):
                generator, _, _ = self._generate("square_outlier", seed)
                square_points = generator.square_points
                side_lengths = [
                    math.hypot(a.x - b.x, a.y - b.y)
                    for a, b in zip(square_points, square_points[1:] + square_points[:1])
                ]
                area = polygon_area(square_points)
                self.assertGreaterEqual(min(side_lengths), generator.canvas_short_side * 0.25)
                self.assertGreaterEqual(area, (generator.canvas_short_side ** 2) * 0.09)


def generator_angle(p1, vertex, p2) -> float:
    v1x = p1.x - vertex.x
    v1y = p1.y - vertex.y
    v2x = p2.x - vertex.x
    v2y = p2.y - vertex.y
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 <= 1e-6 or mag2 <= 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def polygon_area(points) -> float:
    area = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        area += point.x * other.y - point.y * other.x
    return abs(area) * 0.5


if __name__ == "__main__":
    unittest.main()
