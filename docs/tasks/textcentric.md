# Text-Centric Independent Pipeline

## Example Outputs

Text-centric tasks evaluate whether video generation models can perform reasoning by embedding text and solutions within generated video frames. Below are example first frames (prompts) and last frames (solutions) from generated videos:

### Math Reasoning

| Benchmark |                            Prompt (First Frame)                            |                           Solution (Last Frame)                           |
| :-------: | :------------------------------------------------------------------------: | :-----------------------------------------------------------------------: |
| **GSM8K** | <img src="../../assets/examples/textcentric/gsm8k_first.png" width="280"/> | <img src="../../assets/examples/textcentric/gsm8k_last.png" width="280"/> |

### Multimodal Math

|   Benchmark   |                              Prompt (First Frame)                              |                             Solution (Last Frame)                             |
| :-----------: | :----------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
| **MathVista** | <img src="../../assets/examples/textcentric/mathvista_first.png" width="280"/> | <img src="../../assets/examples/textcentric/mathvista_last.png" width="280"/> |

### Multimodal Understanding

|  Benchmark  |                             Prompt (First Frame)                             |                            Solution (Last Frame)                            |
| :---------: | :--------------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| **MMBench** | <img src="../../assets/examples/textcentric/mmbench_first.png" width="280"/> | <img src="../../assets/examples/textcentric/mmbench_last.png" width="280"/> |
|  **MMMU**   |  <img src="../../assets/examples/textcentric/mmmu_first.png" width="280"/>   |  <img src="../../assets/examples/textcentric/mmmu_last.png" width="280"/>   |

## Current Status

`Text-Centric` is still an independent pipeline and has not yet been integrated into the unified `Canonical Manifest` mainline.

The relevant code currently lives in:

- `data/textcentric/request_videos.py`.
- `data/evaluation/textcentric/`.

## Sub-benchmarks

VideoThinkBench adapts the following text-centric sub-benchmarks into video generation evaluation.

### Math Reasoning

- `GSM8K`.
- `MATH-500`.
- `AIME24`.
- `AIME25`.

### General Knowledge

- `BBH`.
- `MMLU`.
- `MMLU-Pro`.
- `GPQA-diamond`.
- `SuperGPQA-easy`.

### Multimodal Math

- `MathVista`.
- `MathVision`.

### Multimodal Understanding

- `MMBench`.
- `MMMU`.

## Current Workflow

### Request Videos

The current entry script is `data/textcentric/request_videos.py`.

Inputs are typically question JSON files, and the outputs usually include:

- raw model responses.
- video download URLs.
- downloaded video files.
- `responses.json` and `questions.json`.

### Evaluate Videos

The current evaluation entry point is `data/evaluation/textcentric/evaluate_videos.py`.

The evaluation pipeline mainly checks:

- whether the last frame contains the correct answer.
- whether the audio transcript contains the correct answer.
- whether both modalities are correct.
- whether either modality is correct.

## Why It Is Not Yet in the Unified Mainline

This pipeline is centered on question text, generated videos, audio transcription, and LLM judging, which is materially different from the current `Vision-Centric` intermediate representation based on puzzle images, solution images, and optional solution videos.

If this track is integrated into the unified mainline in the future, the recommended first step is to design a dedicated `Text-Centric Canonical Schema` rather than forcing it into the existing `Vision-Centric` abstraction.
