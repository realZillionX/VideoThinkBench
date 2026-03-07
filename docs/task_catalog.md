# Task Catalog

## How to Use This Page

This page is a quick index for locating tasks in the current repository.

Use the group-specific documentation pages when you need generation parameters, data fields, or evaluation details.

## Group Guide

### `eyeballing`

Geometry precision tasks in which the model answers by placing points, drawing lines, or completing shapes.

### `maze`

Path-finding tasks in which the model must draw a valid route from the start position to the goal.

### `visual_puzzle`

Pattern completion tasks built around color, shape, size, symmetry, and compositional rules.

### `textcentric`

An independent pipeline that adapts text-heavy reasoning benchmarks into video generation evaluation.

The table below covers only the `36` registered `Vision-Centric` mainline tasks.

## Vision-Centric Unified Mainline

| Group | Task | Current Code Location | Origin |
| --- | --- | --- | --- |
| `eyeballing` | `angle_bisector` | `data/visioncentric/eyeballing/angle_bisector/` | Former `VisionCentric/puzzle/angle_bisector/` |
| `eyeballing` | `arc_connect` | `data/visioncentric/eyeballing/arc_connect/` | Former `VisionCentric/puzzle/arc_connect/` |
| `eyeballing` | `arc_connect_point_ver` | `data/visioncentric/eyeballing/arc_connect_point_ver/` | Former `VisionCentric/puzzle/arc_connect_point_ver/` |
| `eyeballing` | `circle_center` | `data/visioncentric/eyeballing/circle_center/` | Former `VisionCentric/puzzle/circle_center/` |
| `eyeballing` | `circle_tangent_line` | `data/visioncentric/eyeballing/circle_tangent_line/` | Former `VisionCentric/puzzle/circle_tangent_line/` |
| `eyeballing` | `circle_tangent_point` | `data/visioncentric/eyeballing/circle_tangent_point/` | Former `VisionCentric/puzzle/circle_tangent_point/` |
| `eyeballing` | `circumcenter` | `data/visioncentric/eyeballing/circumcenter/` | Former `VisionCentric/puzzle/circumcenter/` |
| `eyeballing` | `fermat_point` | `data/visioncentric/eyeballing/fermat_point/` | Former `VisionCentric/puzzle/fermat_point/` |
| `eyeballing` | `incenter` | `data/visioncentric/eyeballing/incenter/` | Former `VisionCentric/puzzle/incenter/` |
| `eyeballing` | `isosceles_trapezoid` | `data/visioncentric/eyeballing/isosceles_trapezoid/` | Former `VisionCentric/puzzle/isosceles_trapezoid/` |
| `eyeballing` | `midpoint` | `data/visioncentric/eyeballing/midpoint/` | Former `VisionCentric/puzzle/midpoint/` |
| `eyeballing` | `orthocenter` | `data/visioncentric/eyeballing/orthocenter/` | Former `VisionCentric/puzzle/orthocenter/` |
| `eyeballing` | `parallel` | `data/visioncentric/eyeballing/parallel/` | Former `VisionCentric/puzzle/parallel/` |
| `eyeballing` | `parallelogram` | `data/visioncentric/eyeballing/parallelogram/` | Former `VisionCentric/puzzle/parallelogram/` |
| `eyeballing` | `perpendicular` | `data/visioncentric/eyeballing/perpendicular/` | Former `VisionCentric/puzzle/perpendicular/` |
| `eyeballing` | `perpendicular_bisector` | `data/visioncentric/eyeballing/perpendicular_bisector/` | Former `VisionCentric/puzzle/perpendicular_bisector/` |
| `eyeballing` | `ray` | `data/visioncentric/eyeballing/ray/` | Former `VisionCentric/puzzle/ray/` |
| `eyeballing` | `ray_intersection` | `data/visioncentric/eyeballing/ray_intersection/` | Former `VisionCentric/puzzle/ray_intersection/` |
| `eyeballing` | `ray_reflect` | `data/visioncentric/eyeballing/ray_reflect/` | Former `VisionCentric/puzzle/ray_reflect/` |
| `eyeballing` | `reflection` | `data/visioncentric/eyeballing/reflection/` | Former `VisionCentric/puzzle/reflection/` |
| `eyeballing` | `right_triangle` | `data/visioncentric/eyeballing/right_triangle/` | Former `VisionCentric/puzzle/right_triangle/` |
| `eyeballing` | `square_outlier` | `data/visioncentric/eyeballing/square_outlier/` | Former `VisionCentric/puzzle/square_outlier/` |
| `eyeballing` | `triangle_center` | `data/visioncentric/eyeballing/triangle_center/` | Former `VisionCentric/puzzle/triangle_center/` |
| `maze` | `maze_square` | `data/visioncentric/maze/maze_square/` | Former `VisionCentric/puzzle/maze_square/` |
| `maze` | `maze_hexagon` | `data/visioncentric/maze/maze_hexagon/` | Former `VisionCentric/puzzle/maze_hexagon/` |
| `maze` | `maze_labyrinth` | `data/visioncentric/maze/maze_labyrinth/` | Former `VisionCentric/puzzle/maze_labyrinth/` |
| `visual_puzzle` | `color_grid` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `color_hexagon` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `color_overlap_squares` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `color_size` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `polygon_sides_color` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `rectangle_height_color` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `shape_reflect` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `shape_size_grid` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `size_cycle` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |
| `visual_puzzle` | `size_grid` | `data/visioncentric/visual_puzzles/data_generation.py` | Former `visual_puzzles/` |

## Related Documentation

- [Eyeballing Tasks and Parameters](tasks/eyeballing.md).
- [Maze Tasks and Parameters](tasks/maze.md).
- [Visual Puzzle Tasks and Parameters](tasks/visual_puzzle.md).
- [Text-Centric Independent Pipeline](tasks/textcentric.md).
- [ARC-AGI-2 Archived Tasks](tasks/arcagi2.md).
- [Benchmark Overview and Legacy Archive](benchmark_overview.md).
