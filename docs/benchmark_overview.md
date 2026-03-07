# Benchmark Overview

## Repository Purpose

`VideoThinkBench` is the unified engineering repository for reasoning evaluation with video generation models.

It is the maintained successor to the original `Thinking-with-Video` codebase, and the public documentation in this repository should be treated as the authoritative source for current task definitions, data schemas, evaluation entry points, and training interfaces.

The original repository remains a useful historical reference for paper-era naming, early directory layouts, and archived examples, but it is no longer the primary engineering surface.

## Task Landscape Overview

![VideoThinkBench task landscape](../assets/main_picture.png)

`VideoThinkBench` combines two complementary evaluation tracks.

- `Vision-Centric` tasks focus on dynamic visual reasoning, such as geometric construction, path planning, and pattern completion.
- `Text-Centric` tasks adapt established reasoning benchmarks into video generation settings and judge the generated video content.

## Vision-Centric Unified Pipeline

The unified registry lives in [data/registry.py](../data/registry.py) and currently includes `36` mainline tasks.

| Group | Count | Current Code Location | Description |
| --- | --- | --- | --- |
| `eyeballing` | `23` | `data/visioncentric/eyeballing/` | Geometric point, line, and shape construction tasks. |
| `maze` | `3` | `data/visioncentric/maze/` | Square, hexagon, and labyrinth path-finding tasks. |
| `visual_puzzle` | `10` | `data/visioncentric/visual_puzzles/` | Color, shape, size, and compositional pattern matching tasks. |

These registered tasks share one engineering mainline for generation, scanning, export, and bench-level evaluation.

## Text-Centric Independent Pipeline

`Text-Centric` currently remains an independent pipeline, with code under `data/textcentric/` and `data/evaluation/textcentric/`.

Unlike the `Vision-Centric` tasks, this track does not yet flow through the shared `CanonicalSample` manifest. Its core objects are question text, generated videos, audio transcripts, and judge outputs, so it should be understood as a parallel evaluation path rather than a member of the unified `Vision-Centric` pipeline.

## Legacy Archive

Tasks inherited from the original `VisionCentric/puzzle/` tree but not currently included in the unified registry are archived under `data/visioncentric/legacy/`.

The current archive includes:

- `arcagi`.
- `circle_count`.
- `jigsaw`.
- `mirror`.
- `rects`.
- `sudoku`.

These tasks are preserved for historical comparison, evaluator reuse, and possible future restoration. They are not part of `cli.py data generate --tasks all`, and they are not included in the default `CanonicalSample` mainline.

`ARC-AGI-2` is documented separately in [tasks/arcagi2.md](tasks/arcagi2.md) because it remains useful for targeted abstract reasoning experiments even though it is archived.

## Data Pipeline Overview

The maintained data workflow has `4` layers:

1. Task generators produce raw `data.json`, puzzle images, solution images, and optional solution videos.
2. [data/scan.py](../data/scan.py) normalizes task records into `CanonicalSample`.
3. [data/export.py](../data/export.py) and `data/exporters/` convert `CanonicalSample` into training or evaluation formats for downstream frameworks.
4. `data/evaluation/` handles bench-level inference, offline evaluation, frame matching, and result aggregation.

`CanonicalSample`, defined in [core/schemas.py](../core/schemas.py), is the key engineering interface that connects generation, export, evaluation, and training.

## Parameter Tuning Strategy

When adjusting generation parameters, it is useful to separate them into `3` sources:

1. Global options exposed directly by `python3 cli.py data generate`.
2. Task-specific options injected through `--task-config` or `--task-config-path`.
3. Parameters that are not yet exposed and therefore require generator or CLI changes.

This overview and the task-specific pages call out those categories explicitly.
