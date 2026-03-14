# VideoThinkBench — Developer Guide

## Repository Purpose

`VideoThinkBench` is the unified engineering repository for reasoning evaluation with video generation models, serving as the maintained successor to the original `Thinking-with-Video` codebase.

![VideoThinkBench task landscape](../assets/main_picture.png)

The benchmark combines two complementary evaluation tracks:

- **Vision-Centric** — dynamic visual reasoning (geometric construction, path planning, pattern completion).
- **Text-Centric** — established reasoning benchmarks adapted into video generation settings.

---

## Task Registry

The unified registry ([data/registry.py](../data/registry.py)) currently includes **33** mainline Vision-Centric tasks.

| Group           | Count | Code Location                        | Description                                    |
| --------------- | ----- | ------------------------------------ | ---------------------------------------------- |
| `eyeballing`    | 20    | `data/visioncentric/eyeballing/`     | Geometric point, line, and shape construction  |
| `maze`          | 3     | `data/visioncentric/maze/`           | Square, hexagon, and labyrinth path-finding    |
| `visual_puzzle` | 10    | `data/visioncentric/visual_puzzles/` | Color, shape, size, and compositional patterns |

<details>
<summary>Full 33-task listing</summary>

| Group           | Task                     | Code Location                                           |
| --------------- | ------------------------ | ------------------------------------------------------- |
| `eyeballing`    | `angle_bisector`         | `data/visioncentric/eyeballing/angle_bisector/`         |
| `eyeballing`    | `arc_connect`            | `data/visioncentric/eyeballing/arc_connect/`            |
| `eyeballing`    | `circle_center`          | `data/visioncentric/eyeballing/circle_center/`          |
| `eyeballing`    | `circle_tangent_line`    | `data/visioncentric/eyeballing/circle_tangent_line/`    |
| `eyeballing`    | `circle_tangent_point`   | `data/visioncentric/eyeballing/circle_tangent_point/`   |
| `eyeballing`    | `circumcenter`           | `data/visioncentric/eyeballing/circumcenter/`           |
| `eyeballing`    | `incenter`               | `data/visioncentric/eyeballing/incenter/`               |
| `eyeballing`    | `isosceles_trapezoid`    | `data/visioncentric/eyeballing/isosceles_trapezoid/`    |
| `eyeballing`    | `midpoint`               | `data/visioncentric/eyeballing/midpoint/`               |
| `eyeballing`    | `orthocenter`            | `data/visioncentric/eyeballing/orthocenter/`            |
| `eyeballing`    | `parallel`               | `data/visioncentric/eyeballing/parallel/`               |
| `eyeballing`    | `parallelogram`          | `data/visioncentric/eyeballing/parallelogram/`          |
| `eyeballing`    | `perpendicular`          | `data/visioncentric/eyeballing/perpendicular/`          |
| `eyeballing`    | `perpendicular_bisector` | `data/visioncentric/eyeballing/perpendicular_bisector/` |
| `eyeballing`    | `ray_intersection`       | `data/visioncentric/eyeballing/ray_intersection/`       |
| `eyeballing`    | `ray_reflect`            | `data/visioncentric/eyeballing/ray_reflect/`            |
| `eyeballing`    | `reflection`             | `data/visioncentric/eyeballing/reflection/`             |
| `eyeballing`    | `right_triangle`         | `data/visioncentric/eyeballing/right_triangle/`         |
| `eyeballing`    | `square_outlier`         | `data/visioncentric/eyeballing/square_outlier/`         |
| `eyeballing`    | `triangle_center`        | `data/visioncentric/eyeballing/triangle_center/`        |
| `maze`          | `maze_square`            | `data/visioncentric/maze/maze_square/`                  |
| `maze`          | `maze_hexagon`           | `data/visioncentric/maze/maze_hexagon/`                 |
| `maze`          | `maze_labyrinth`         | `data/visioncentric/maze/maze_labyrinth/`               |
| `visual_puzzle` | `color_grid`             | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `color_hexagon`          | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `color_overlap_squares`  | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `color_size`             | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `polygon_sides_color`    | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `rectangle_height_color` | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `shape_reflect`          | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `shape_size_grid`        | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `size_cycle`             | `data/visioncentric/visual_puzzles/`                    |
| `visual_puzzle` | `size_grid`              | `data/visioncentric/visual_puzzles/`                    |

</details>

### Text-Centric Pipeline

Text-Centric remains an independent pipeline under `data/textcentric/` and `data/evaluation/textcentric/`. It adapts 13 sub-benchmarks (GSM8K, MathVista, MATH-500, AIME24/25, MMLU, MMLU-Pro, MMMMU, MMBench, GPQA-diamond, SuperGPQA-easy, BBH, MathVision) into video generation evaluation. Unlike Vision-Centric tasks, it does not flow through the shared `CanonicalSample` manifest.

### Legacy Archive

Archived tasks under `data/visioncentric/legacy/` include: `arcagi`, `circle_count`, `jigsaw`, `mirror`, `rects`, `sudoku`. These are preserved for historical comparison but excluded from `cli.py data generate --tasks all`. See also [tasks/arcagi2.md](tasks/arcagi2.md).

---

## Data Pipeline

The maintained workflow has 4 layers:

1. **Task generators** produce `data.json`, puzzle images, solution images, and optional solution videos.
2. **[data/scan.py](../data/scan.py)** normalizes task records into `CanonicalSample`.
3. **[data/export.py](../data/export.py)** converts `CanonicalSample` into training or evaluation formats.
4. **`data/evaluation/`** handles inference, offline evaluation, frame matching, and result aggregation.

`CanonicalSample`, defined in [core/schemas.py](../core/schemas.py), is the key interface connecting generation, export, evaluation, and training.

The current `CanonicalSample` schema is modality-aware:

- `prompts.ti2v` / `prompts.ti2i` feed generation-style exports.
- `prompts.ti2t` / `prompts.ti2ti` feed reasoning-style exports.
- `assets.puzzle_image` is the generation-side input image.
- `assets.reasoning_image` is the reasoning-side input image. For `maze`, this is the cell-ID version; for `eyeballing` and `visual_puzzle`, it usually matches `puzzle_image`.

### Parameter Tuning

Generation parameters come from 3 sources:

1. Global options via `python3 cli.py data generate`.
2. Task-specific options via `--task-config` or `--task-config-path`.
3. Parameters requiring generator or CLI code changes.

---

## Evaluation Architecture

Task-local `evaluator.py` files handle single-answer judging; `data/evaluation/` handles benchmark-level workflows.

| Directory                             | Responsibility                                  |
| ------------------------------------- | ----------------------------------------------- |
| `data/visioncentric/.../evaluator.py` | Task-local rule-based evaluation                |
| `data/evaluation/offline/`            | Offline batch evaluation over `CanonicalSample` |
| `data/evaluation/infer/`              | Batch inference through external models         |
| `data/evaluation/textcentric/`        | Independent Text-Centric video evaluation       |
| `data/evaluation/frame_matching/`     | Video frame extraction and best-frame matching  |
| `data/evaluation/pipeline.py`         | Unified result writing and summary aggregation  |

### Offline Evaluation

| Group           | Entry Point                                | Key Metrics                                                                                  |
| --------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `maze`          | `data/evaluation/offline/maze.py`          | Red path detected, start/end touching, wall crossing, connectivity                           |
| `eyeballing`    | `data/evaluation/offline/eyeballing.py`    | Red-highlight locations, textual answers, video frame analysis, transcription → option match |
| `visual_puzzle` | `data/evaluation/offline/visual_puzzle.py` | Best-frame extraction → image-difference metrics against solution                            |

### Inference Backends

| Modality | Entry Point                      | Dependency         |
| -------- | -------------------------------- | ------------------ |
| `video`  | `data/evaluation/infer/video.py` | `DiffSynth-Studio` |
| `image`  | `data/evaluation/infer/image.py` | `DiffSynth-Studio` |
| `vlm`    | `data/evaluation/infer/vlm.py`   | `ms-swift`         |

---

## Training

The repository maintains 4 model-specific training branches:

| Branch | Entry Point                          | Target Model         | Framework          | Data Source                              |
| ------ | ------------------------------------ | -------------------- | ------------------ | ---------------------------------------- |
| `wan2.2-ti2v-5b` | `training/wan2.2-ti2v-5b/` | `Wan2.2-TI2V-5B`     | `DiffSynth-Studio` | `CanonicalSample → diffsynth-video CSV`  |
| `wan2.2-i2v-a14b` | `training/wan2.2-i2v-a14b/` | `Wan2.2-I2V-A14B`   | `DiffSynth-Studio` | `CanonicalSample → diffsynth-video CSV`  |
| `qwen-image-edit-2511` | `training/qwen-image-edit-2511/` | `Qwen-Image-Edit-2511` | `DiffSynth-Studio` | `CanonicalSample → diffsynth-image JSON` |
| `bagel` | `training/bagel/`                | `BAGEL-7B-MoT`       | `BAGEL`            | `CanonicalSample → bagel parquet+JSONL`  |

`training/` does not read task directories directly — it consumes unified exported intermediate data via: generate/scan → export → train.

Each model directory now owns its own `prepare_data.py` wrapper. The canonical export interface still remains `python3 cli.py data export`.

### Branch Details

- **`wan2.2-ti2v-5b`** — single-branch `LoRA` training with automatic resume for `LoRA` weights, optimizer, scheduler, and step counter.
- **`wan2.2-i2v-a14b`** — two-branch `LoRA` training where `high_noise` and `low_noise` checkpoints are saved and resumed independently.
- **`qwen-image-edit-2511`** — image-edit training only; supervision comes from `puzzle_image → solution_image`, and only samples with `ti2i_prompt` are exported.
- **`bagel`** — exports both editing parquet shards and VLM-style JSONL, but the two branches now route different inputs: `edit` uses `puzzle_image + ti2i_prompt`, while `vlm` uses `reasoning_image + ti2t_prompt`.

---

## Task-Specific Documentation

- [Eyeballing Tasks](tasks/eyeballing.md) — 23 geometric reasoning tasks with parameters & examples
- [Maze Tasks](tasks/maze.md) — 3 path-finding tasks with parameters & examples
- [Visual Puzzle Tasks](tasks/visual_puzzle.md) — 10 pattern completion tasks with parameters & examples
- [Text-Centric Pipeline](tasks/textcentric.md) — 13 sub-benchmarks adapted for video generation
- [ARC-AGI-2 (Archived)](tasks/arcagi2.md) — abstract reasoning, archived from mainline
