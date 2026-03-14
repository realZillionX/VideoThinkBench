# Maze Tasks and Parameters

## Visual Examples

`Maze` tasks ask the model to draw a valid path from the start point to the end point inside a maze.

The unified mainline currently includes `3` tasks:

### All Maze Types

|       Type       |                                    Puzzle                                     |                                    Solution                                     |
| :--------------: | :---------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|  `maze_square`   |  <img src="../../assets/examples/maze/maze_square_puzzle.png" width="240"/>   |  <img src="../../assets/examples/maze/maze_square_solution.png" width="240"/>   |
|  `maze_hexagon`  |  <img src="../../assets/examples/maze/maze_hexagon_puzzle.png" width="240"/>  |  <img src="../../assets/examples/maze/maze_hexagon_solution.png" width="240"/>  |
| `maze_labyrinth` | <img src="../../assets/examples/maze/maze_labyrinth_puzzle.png" width="240"/> | <img src="../../assets/examples/maze/maze_labyrinth_solution.png" width="240"/> |

## Data Records

Each record typically contains:

- the puzzle image `image`.
- the solution image `solution_image_path`.
- an optional solution video.
- the start point and end point.
- the path cells in `solution_path_cell_ids`.
- bounding-box, grid, or ring metadata required for evaluation.

## Evaluation Logic

Task-local evaluators live inside each task directory.

The unified offline evaluation entry point is `data/evaluation/offline/maze.py`.

The current batch evaluation mainly checks:

- whether the route is connected.
- whether the path touches the start point.
- whether the path touches the end point.
- whether the path crosses walls.

## Parameters Exposed by the Unified CLI

The following parameters can be adjusted directly through `cli.py data generate`.

| Parameter              | Default | Applicable Tasks | Description                           |
| ---------------------- | ------- | ---------------- | ------------------------------------- |
| `--canvas-width`       | `480`   | All              | Output canvas width.                  |
| `--seed`               | `42`    | All              | Random seed.                          |
| `--video`              | `False` | All              | Whether to generate a solution video. |
| `--maze-rows`          | `9`     | `maze_square`    | Number of rows.                       |
| `--maze-cols`          | `9`     | `maze_square`    | Number of columns.                    |
| `--maze-cell-size`     | `32`    | `maze_square`    | Side length of one cell.              |
| `--hex-radius`         | `4`     | `maze_hexagon`   | Radius of the hexagonal grid.         |
| `--hex-cell-size`      | `24`    | `maze_hexagon`   | Radius of each hexagonal cell.        |
| `--hex-wall-thickness` | `None`  | `maze_hexagon`   | Override wall thickness.              |
| `--lab-rings`          | `6`     | `maze_labyrinth` | Number of rings.                      |
| `--lab-segments`       | `18`    | `maze_labyrinth` | Number of segments per ring.          |
| `--lab-cell-size`      | `18`    | `maze_labyrinth` | Ring-band width.                      |
| `--lab-wall-thickness` | `None`  | `maze_labyrinth` | Override wall thickness.              |

## Shared Constructor Parameters

All three maze families share the following base parameters, although some of them are currently available only through `task_config` or direct generator calls.

| Parameter      | Default              | Description                             |
| -------------- | -------------------- | --------------------------------------- |
| `output_dir`   | `DEFAULT_OUTPUT_DIR` | Output directory.                       |
| `canvas_width` | Task-dependent       | Output canvas width.                    |
| `aspect`       | `None`               | Canvas aspect ratio.                    |
| `seed`         | `None`               | Random seed.                            |
| `prompt`       | `None`               | Override the default prompt.            |
| `show_cell_id` | `False`              | Whether to print cell IDs on the image. |
| `video`        | `False`              | Whether to generate a solution video.   |

## `maze_square`

### Main Parameters

| Parameter      | Default | Description                                                                      |
| -------------- | ------- | -------------------------------------------------------------------------------- |
| `rows`         | `15`    | Number of rows. Even values are automatically promoted to the next odd value.    |
| `cols`         | `15`    | Number of columns. Even values are automatically promoted to the next odd value. |
| `cell_size`    | `32`    | Side length of one cell.                                                         |
| `size`         | `None`  | Compatibility alias for `cell_size`.                                             |
| `aspect_ratio` | `None`  | Final aspect ratio of the rendered puzzle image.                                 |

### Notes

The current default pipeline carves the maze with DFS and then finds the canonical path from `(1, 1)` to the lower-right corner with BFS.

## `maze_hexagon`

The hexagon renderer now draws blocked edges as short filled wall polygons instead of line segments plus oversized corner caps. This keeps wall joins smoother and avoids the small outward corner bumps that could otherwise appear at obstacle turns.

### Main Parameters

| Parameter        | Default | Description                            |
| ---------------- | ------- | -------------------------------------- |
| `radius`         | `4`     | Number of hexagonal layers.            |
| `cell_radius`    | `None`  | Radius of one hexagonal cell.          |
| `wall_thickness` | `None`  | Wall thickness.                        |
| `size`           | `None`  | Compatibility alias for `cell_radius`. |

## `maze_labyrinth`

### Main Parameters

| Parameter        | Default | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `rings`          | `6`     | Number of rings.                      |
| `segments`       | `18`    | Number of segments per ring.          |
| `ring_width`     | `None`  | Width of each ring.                   |
| `wall_thickness` | `None`  | Wall thickness.                       |
| `size`           | `None`  | Compatibility alias for `ring_width`. |

## Parameter Injection Example

```bash
python3 cli.py data generate \
  --tasks maze_square maze_hexagon \
  --count 64 \
  --output-root ./outputs/maze \
  --maze-rows 11 \
  --maze-cols 11 \
  --maze-cell-size 36 \
  --task-config '{
    "maze_square": {
      "aspect_ratio": 0.55
    },
    "maze_hexagon": {
      "show_cell_id": true
    }
  }'
```
