# Training Overview

## Training Branches

The repository currently maintains `3` training branches.

| Branch | Entry Point | Target Model | Training Framework | Training Data Source |
| --- | --- | --- | --- | --- |
| `video` | `training/video/` | `Wan2.2` | `DiffSynth-Studio` | `CanonicalSample -> diffsynth-video CSV` |
| `image` | `training/image/` | `Qwen-Image-Edit` | `DiffSynth-Studio` | `CanonicalSample -> diffsynth-image JSON` |
| `vlm` | `training/vlm/` | `Qwen3-VL` | `ms-swift` | `CanonicalSample -> ms-swift JSONL` |

## Relationship Between Training and the Data Mainline

`training/` does not read task directories directly. It primarily consumes the unified exported intermediate data.

In practice, that means:

1. Generate or scan task data under `data/`.
2. Export it into the format required by the target training framework.
3. Launch training from the corresponding directory under `training/`.

## Current Known Boundaries

### `video`

`training/video/train.py` already supports automatic resume for `LoRA`, `optimizer`, `scheduler`, and `step counter`.

This is currently the most important training path for follow-up maze-focused experiments.

### `image`

The image editing branch mainly serves experiments that evaluate only the final image instead of the full reasoning video.

It shares the same `CanonicalSample` source as the `video` branch, but uses only the solution image as supervision rather than the full solution video.

### `vlm`

`VLM SFT` data export already supports `eyeballing`, `maze`, and `visual_puzzle`.

The current `GRPO` reward functions support:

- path-list answers for `maze`.
- single-letter answers for `eyeballing`.
- textual answers for `visual_puzzle`.

For `visual_puzzle`, the current reward is normalized exact text matching.
