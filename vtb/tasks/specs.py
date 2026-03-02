from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    name: str
    group: str
    module: str
    class_name: Optional[str] = None
    requires: Tuple[str, ...] = ()


EYEBALLING_TASKS: List[str] = [
    "angle_bisector",
    "arc_connect",
    "arc_connect_point_ver",
    "circle_center",
    "circle_tangent_line",
    "circle_tangent_point",
    "circumcenter",
    "fermat_point",
    "incenter",
    "isosceles_trapezoid",
    "midpoint",
    "orthocenter",
    "parallel",
    "parallelogram",
    "perpendicular",
    "perpendicular_bisector",
    "ray",
    "ray_intersection",
    "ray_reflect",
    "reflection",
    "right_triangle",
    "square_outlier",
    "triangle_center",
]

MAZE_TASKS: List[str] = [
    "maze_square",
    "maze_hexagon",
    "maze_labyrinth",
]


def _camel_task(task_name: str) -> str:
    return "".join(chunk.capitalize() for chunk in task_name.split("_"))


def build_task_specs() -> Dict[str, TaskSpec]:
    specs: Dict[str, TaskSpec] = {}
    for task in EYEBALLING_TASKS:
        specs[task] = TaskSpec(
            name=task,
            group="eyeballing",
            module=f"data.puzzle.eyeballing.{task}.generator",
            class_name=f"{_camel_task(task)}Generator",
        )

    specs["maze_square"] = TaskSpec(
        name="maze_square",
        group="maze",
        module="data.puzzle.maze.maze_square.generator",
        class_name="MazeGenerator",
    )
    specs["maze_hexagon"] = TaskSpec(
        name="maze_hexagon",
        group="maze",
        module="data.puzzle.maze.maze_hexagon.generator",
        class_name="MazeHexagonGenerator",
    )
    specs["maze_labyrinth"] = TaskSpec(
        name="maze_labyrinth",
        group="maze",
        module="data.puzzle.maze.maze_labyrinth.generator",
        class_name="MazeLabyrinthGenerator",
    )
    return specs


TASK_SPECS = build_task_specs()


def resolve_requested_tasks(
    tasks: List[str],
    task_groups: List[str],
    exclude_tasks: List[str],
) -> List[str]:
    requested = set(tasks)
    if "all" in requested:
        requested = set(TASK_SPECS.keys())
    requested = {task for task in requested if task in TASK_SPECS}
    if task_groups:
        requested = {task for task in requested if TASK_SPECS[task].group in task_groups}
    for task in exclude_tasks:
        requested.discard(task)
    return sorted(requested)
