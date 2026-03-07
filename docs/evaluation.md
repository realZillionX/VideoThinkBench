# Evaluation Architecture

## Why `data/evaluation/` Exists

Task-local `generator.py` and `evaluator.py` files define how one task produces data and how one candidate answer is judged.

`data/evaluation/` handles the benchmark-level workflow:running inference, matching frames, evaluating batches offline, and writing unified summaries.

These two layers serve different purposes, so they should remain separate.

The directory has been moved under `data/evaluation/` so that generation, scanning, export, and benchmark evaluation all live in the same top-level data workflow.

## Directory Responsibilities

| Directory | Responsibility |
| --- | --- |
| `data/visioncentric/.../evaluator.py` | Task-local rule-based evaluation logic. |
| `data/evaluation/offline/` | Offline batch evaluation over `CanonicalSample` records. |
| `data/evaluation/infer/` | Batch inference through external models. |
| `data/evaluation/textcentric/` | Independent `Text-Centric` video evaluation. |
| `data/evaluation/frame_matching/` | Video frame extraction and best-frame matching. |
| `data/evaluation/pipeline.py` | Unified result writing and summary aggregation. |

## Current Offline Evaluation Conventions

### `maze`

The entry point is `data/evaluation/offline/maze.py`.

The batch evaluator delegates core geometric checks to the local evaluators of each maze task and currently focuses on:

- whether a red path is detected.
- whether the path touches the start point.
- whether the path touches the end point.
- whether the path crosses walls.
- whether the predicted route is connected.

### `eyeballing`

The entry point is `data/evaluation/offline/eyeballing.py`.

The current pipeline aggregates several evidence sources:

- red-highlight locations in the final predicted image.
- textual answers.
- interpreted video frames.
- transcribed text.

It then compares the aggregated predicted option against `correct_option` to produce the final pass or fail decision.

### `visual_puzzle`

The entry point is `data/evaluation/offline/visual_puzzle.py`.

The current workflow is:

1. If a predicted video is available, extract the frame that best matches the reference solution.
2. Compare the best frame or predicted image against the solution image.
3. Report image-difference metrics and related outputs.

This group is still primarily difference-metric based and has not yet been fully standardized into one mandatory `pass / fail` threshold.

## Inference

`data/evaluation/infer/` is responsible only for sending inputs to models and saving predictions. It does not implement task-specific judging.

| Modality | Entry Point | External Dependency |
| --- | --- | --- |
| `video` | `data/evaluation/infer/video.py` | `DiffSynth-Studio` |
| `image` | `data/evaluation/infer/image.py` | `DiffSynth-Studio` |
| `vlm` | `data/evaluation/infer/vlm.py` | `ms-swift` |

## Text-Centric

`Text-Centric` keeps its own video-oriented evaluation flow under `data/evaluation/textcentric/` and does not currently use the same mainline as the `Vision-Centric` `cli.py eval` workflow.

See [tasks/textcentric.md](tasks/textcentric.md) for details.
