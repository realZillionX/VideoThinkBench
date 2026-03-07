# Eyeballing Tasks and Parameters

## Visual Examples

`Eyeballing` tasks ask the model to solve geometry problems directly in the image and answer by placing points, drawing lines, drawing shapes, or highlighting one of the candidate options.

The unified mainline currently includes `23` tasks, organized into three subcategories.

### Point Tasks

|        Task        |                                        Puzzle                                         |                                        Solution                                         |
| :----------------: | :-----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: |
|  `circle_center`   |  <img src="../../assets/examples/eyeballing/circle_center_puzzle.png" width="220"/>   |  <img src="../../assets/examples/eyeballing/circle_center_solution.png" width="220"/>   |
|   `circumcenter`   |   <img src="../../assets/examples/eyeballing/circumcenter_puzzle.png" width="220"/>   |   <img src="../../assets/examples/eyeballing/circumcenter_solution.png" width="220"/>   |
|   `fermat_point`   |   <img src="../../assets/examples/eyeballing/fermat_point_puzzle.png" width="220"/>   |   <img src="../../assets/examples/eyeballing/fermat_point_solution.png" width="220"/>   |
|     `incenter`     |     <img src="../../assets/examples/eyeballing/incenter_puzzle.png" width="220"/>     |     <img src="../../assets/examples/eyeballing/incenter_solution.png" width="220"/>     |
|     `midpoint`     |     <img src="../../assets/examples/eyeballing/midpoint_puzzle.png" width="220"/>     |     <img src="../../assets/examples/eyeballing/midpoint_solution.png" width="220"/>     |
|   `orthocenter`    |   <img src="../../assets/examples/eyeballing/orthocenter_puzzle.png" width="220"/>    |   <img src="../../assets/examples/eyeballing/orthocenter_solution.png" width="220"/>    |
| `ray_intersection` | <img src="../../assets/examples/eyeballing/ray_intersection_puzzle.png" width="220"/> | <img src="../../assets/examples/eyeballing/ray_intersection_solution.png" width="220"/> |
| `triangle_center`  | <img src="../../assets/examples/eyeballing/triangle_center_puzzle.png" width="220"/>  | <img src="../../assets/examples/eyeballing/triangle_center_solution.png" width="220"/>  |

### Line Tasks

|           Task           |                                           Puzzle                                            |                                           Solution                                            |
| :----------------------: | :-----------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
|     `angle_bisector`     |     <img src="../../assets/examples/eyeballing/angle_bisector_puzzle.png" width="220"/>     |     <img src="../../assets/examples/eyeballing/angle_bisector_solution.png" width="220"/>     |
|      `arc_connect`       |      <img src="../../assets/examples/eyeballing/arc_connect_puzzle.png" width="220"/>       |      <img src="../../assets/examples/eyeballing/arc_connect_solution.png" width="220"/>       |
|  `circle_tangent_line`   |  <img src="../../assets/examples/eyeballing/circle_tangent_line_puzzle.png" width="220"/>   |  <img src="../../assets/examples/eyeballing/circle_tangent_line_solution.png" width="220"/>   |
|  `circle_tangent_point`  |  <img src="../../assets/examples/eyeballing/circle_tangent_point_puzzle.png" width="220"/>  |  <img src="../../assets/examples/eyeballing/circle_tangent_point_solution.png" width="220"/>  |
|        `parallel`        |        <img src="../../assets/examples/eyeballing/parallel_puzzle.png" width="220"/>        |        <img src="../../assets/examples/eyeballing/parallel_solution.png" width="220"/>        |
|     `perpendicular`      |     <img src="../../assets/examples/eyeballing/perpendicular_puzzle.png" width="220"/>      |     <img src="../../assets/examples/eyeballing/perpendicular_solution.png" width="220"/>      |
| `perpendicular_bisector` | <img src="../../assets/examples/eyeballing/perpendicular_bisector_puzzle.png" width="220"/> | <img src="../../assets/examples/eyeballing/perpendicular_bisector_solution.png" width="220"/> |
|      `ray_reflect`       |      <img src="../../assets/examples/eyeballing/ray_reflect_puzzle.png" width="220"/>       |      <img src="../../assets/examples/eyeballing/ray_reflect_solution.png" width="220"/>       |
|       `reflection`       |       <img src="../../assets/examples/eyeballing/reflection_puzzle.png" width="220"/>       |       <img src="../../assets/examples/eyeballing/reflection_solution.png" width="220"/>       |

### Shape Tasks

|         Task          |                                          Puzzle                                          |                                          Solution                                          |
| :-------------------: | :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| `isosceles_trapezoid` | <img src="../../assets/examples/eyeballing/isosceles_trapezoid_puzzle.png" width="220"/> | <img src="../../assets/examples/eyeballing/isosceles_trapezoid_solution.png" width="220"/> |
|    `parallelogram`    |    <img src="../../assets/examples/eyeballing/parallelogram_puzzle.png" width="220"/>    |    <img src="../../assets/examples/eyeballing/parallelogram_solution.png" width="220"/>    |
|   `right_triangle`    |   <img src="../../assets/examples/eyeballing/right_triangle_puzzle.png" width="220"/>    |   <img src="../../assets/examples/eyeballing/right_triangle_solution.png" width="220"/>    |
|   `square_outlier`    |   <img src="../../assets/examples/eyeballing/square_outlier_puzzle.png" width="220"/>    |   <img src="../../assets/examples/eyeballing/square_outlier_solution.png" width="220"/>    |

> **Note**: `arc_connect_point_ver` and `ray` share the same visual style as `arc_connect` and `ray_reflect` respectively.

## Data Records

Most `eyeballing` tasks output:

- the puzzle image `image`.
- the solution image `solution_image_path`.
- candidate-point or candidate-option metadata.
- the ground-truth answer `correct_option`.

Some tasks also output an optional solution video.

## Task-Local Evaluation

Task-local evaluators live alongside each task implementation.

Most of them reuse the shared candidate-point logic from `data/point_target_base.py`.

The batch offline evaluation entry point is `data/evaluation/offline/eyeballing.py`.

## Parameters Exposed by the Unified CLI

The following parameters can be adjusted directly through `python3 cli.py data generate ...`.

| Parameter        | Source      | Default | Description                                                                                                       |
| ---------------- | ----------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `--canvas-width` | Unified CLI | `480`   | Canvas width.                                                                                                     |
| `--seed`         | Unified CLI | `42`    | Random seed.                                                                                                      |
| `--video`        | Unified CLI | `False` | Whether to generate a solution video.                                                                             |
| `--point-radius` | Unified CLI | `None`  | Override the candidate-point radius. This is already wired into most tasks based on `PointTargetPuzzleGenerator`. |
| `--line-width`   | Unified CLI | `None`  | Override geometric line width. This is already wired into most tasks based on `PointTargetPuzzleGenerator`.       |

## Shared Constructor Parameters

Most `eyeballing` tasks inherit from `PointTargetPuzzleGenerator` and therefore share the following parameters.

Not all of them are exposed by the unified CLI yet, but they can be injected through `--task-config` or `--task-config-path`.

| Parameter       | Default                 | Description                           |
| --------------- | ----------------------- | ------------------------------------- |
| `output_dir`    | `DEFAULT_OUTPUT_DIR`    | Output directory.                     |
| `canvas_width`  | `480`                   | Canvas width.                         |
| `aspect`        | `None`                  | Aspect ratio.                         |
| `seed`          | `None`                  | Random seed.                          |
| `prompt`        | `None`                  | Override the default prompt.          |
| `option_labels` | `('A','B','C','D','E')` | Candidate option labels.              |
| `margin_ratio`  | `0.06`                  | Margin ratio.                         |
| `record_video`  | `False`                 | Whether to generate a solution video. |
| `point_radius`  | `None`                  | Candidate-point radius override.      |
| `line_width`    | `None`                  | Geometric line width override.        |

## Task-Specific Parameters

### `arc_connect`

| Parameter       | Default | Description                                             |
| --------------- | ------- | ------------------------------------------------------- |
| `mask_fraction` | `0.18`  | Occlusion band width as a fraction of the canvas width. |
| `arc_span_deg`  | `20.0`  | Visible angle of the arc segments on both sides.        |

### `arc_connect_point_ver`

The current constructor parameters are:

| Parameter      | Default | Description                         |
| -------------- | ------- | ----------------------------------- |
| `canvas_width` | `480`   | Canvas width.                       |
| `aspect`       | `None`  | Aspect ratio.                       |
| `seed`         | `None`  | Random seed.                        |
| `prompt`       | `None`  | Override the default prompt.        |
| `record_video` | `False` | Whether to output a solution video. |
| `point_radius` | `None`  | Candidate-point radius override.    |
| `line_width`   | `None`  | Line width override.                |

It also uses the following fixed internal settings:

| Internal Parameter | Fixed Value | Description                                 |
| ------------------ | ----------- | ------------------------------------------- |
| `mask_fraction`    | `0.35`      | Width ratio of the vertical occlusion band. |
| `arc_span_deg`     | `20.0`      | Visible angle of the arc segments.          |

### `ray`

| Parameter         | Default | Description                    |
| ----------------- | ------- | ------------------------------ |
| `canvas_size`     | `480`   | Base canvas size.              |
| `aspect`          | `None`  | Aspect ratio.                  |
| `mirror_count`    | `12`    | Number of mirrors.             |
| `min_reflections` | `2`     | Minimum number of reflections. |
| `prompt`          | `None`  | Override the default prompt.   |
| `seed`            | `None`  | Random seed.                   |

## Task Locations

| Task                     | Code Path                                               |
| ------------------------ | ------------------------------------------------------- |
| `angle_bisector`         | `data/visioncentric/eyeballing/angle_bisector/`         |
| `arc_connect`            | `data/visioncentric/eyeballing/arc_connect/`            |
| `arc_connect_point_ver`  | `data/visioncentric/eyeballing/arc_connect_point_ver/`  |
| `circle_center`          | `data/visioncentric/eyeballing/circle_center/`          |
| `circle_tangent_line`    | `data/visioncentric/eyeballing/circle_tangent_line/`    |
| `circle_tangent_point`   | `data/visioncentric/eyeballing/circle_tangent_point/`   |
| `circumcenter`           | `data/visioncentric/eyeballing/circumcenter/`           |
| `fermat_point`           | `data/visioncentric/eyeballing/fermat_point/`           |
| `incenter`               | `data/visioncentric/eyeballing/incenter/`               |
| `isosceles_trapezoid`    | `data/visioncentric/eyeballing/isosceles_trapezoid/`    |
| `midpoint`               | `data/visioncentric/eyeballing/midpoint/`               |
| `orthocenter`            | `data/visioncentric/eyeballing/orthocenter/`            |
| `parallel`               | `data/visioncentric/eyeballing/parallel/`               |
| `parallelogram`          | `data/visioncentric/eyeballing/parallelogram/`          |
| `perpendicular`          | `data/visioncentric/eyeballing/perpendicular/`          |
| `perpendicular_bisector` | `data/visioncentric/eyeballing/perpendicular_bisector/` |
| `ray`                    | `data/visioncentric/eyeballing/ray/`                    |
| `ray_intersection`       | `data/visioncentric/eyeballing/ray_intersection/`       |
| `ray_reflect`            | `data/visioncentric/eyeballing/ray_reflect/`            |
| `reflection`             | `data/visioncentric/eyeballing/reflection/`             |
| `right_triangle`         | `data/visioncentric/eyeballing/right_triangle/`         |
| `square_outlier`         | `data/visioncentric/eyeballing/square_outlier/`         |
| `triangle_center`        | `data/visioncentric/eyeballing/triangle_center/`        |

## Parameter Injection Example

```bash
python3 cli.py data generate \
  --tasks midpoint ray \
  --count 32 \
  --output-root ./outputs/eyeballing \
  --point-radius 14 \
  --line-width 6 \
  --task-config '{
    "ray": {
      "mirror_count": 16,
      "min_reflections": 3
    }
  }'
```
