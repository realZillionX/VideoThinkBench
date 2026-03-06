"""Unified dataset generation entrypoint for VideoThinkBench.

This wrapper keeps the historical `data.tools.generate_dataset` entry while
routing to the new VideoThinkBench pipeline.
"""

from __future__ import annotations

import argparse

from data.generate import run_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoThinkBench unified dataset generator")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or all")
    parser.add_argument(
        "--task_groups",
        nargs="+",
        default=["eyeballing", "maze", "visual_puzzle"],
        choices=["eyeballing", "maze", "visual_puzzle"],
    )
    parser.add_argument("--exclude_tasks", nargs="+", default=[])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--canvas_width", type=int, default=480)

    parser.add_argument("--point_radius", type=int, default=None)
    parser.add_argument("--line_width", type=int, default=None)

    parser.add_argument("--maze_rows", type=int, default=9)
    parser.add_argument("--maze_cols", type=int, default=9)
    parser.add_argument("--maze_cell_size", type=int, default=32)

    parser.add_argument("--hex_radius", type=int, default=4)
    parser.add_argument("--hex_cell_size", type=int, default=24)
    parser.add_argument("--hex_wall_thickness", type=int, default=None)

    parser.add_argument("--lab_rings", type=int, default=6)
    parser.add_argument("--lab_segments", type=int, default=18)
    parser.add_argument("--lab_cell_size", type=int, default=18)
    parser.add_argument("--lab_wall_thickness", type=int, default=None)

    parser.add_argument("--task_config_path", type=str, default=None)
    parser.add_argument("--task_config", type=str, default=None)

    args = parser.parse_args()
    translated = argparse.Namespace(
        output_root=args.output_dir,
        tasks=args.tasks,
        task_groups=args.task_groups,
        exclude_tasks=args.exclude_tasks,
        count=args.count,
        num_workers=args.num_workers,
        seed=args.seed,
        video=args.video,
        canvas_width=args.canvas_width,
        point_radius=args.point_radius,
        line_width=args.line_width,
        maze_rows=args.maze_rows,
        maze_cols=args.maze_cols,
        maze_cell_size=args.maze_cell_size,
        hex_radius=args.hex_radius,
        hex_cell_size=args.hex_cell_size,
        hex_wall_thickness=args.hex_wall_thickness,
        lab_rings=args.lab_rings,
        lab_segments=args.lab_segments,
        lab_cell_size=args.lab_cell_size,
        lab_wall_thickness=args.lab_wall_thickness,
        task_config_path=args.task_config_path,
        task_config=args.task_config,
    )

    result = run_generation(translated)
    print("=" * 60)
    print("Generation finished")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Report: {result['report_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
