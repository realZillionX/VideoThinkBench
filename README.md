<div align="center">

### Thinking with Video: Video Generation as a Promising Multimodal Reasoning Paradigm

**[CVPR 2026]**

</div>

<div align="center" style="font-size: 15pt">

<a href='https://arxiv.org/abs/2511.04570'><img src='https://img.shields.io/badge/Arxiv-2511.04570-purple'></a>
<a href='https://huggingface.co/papers/2511.04570'><img src='https://img.shields.io/badge/HF%20Paper-2511.04570-blue'></a>
<a href='https://thinking-with-video.github.io/'><img src='https://img.shields.io/badge/Project-Website-green'></a>
<a href='https://thinking-with-video.github.io/#leaderboard'><img src='https://img.shields.io/badge/Leaderboard-Table-E07A5F'></a>
<a href='https://huggingface.co/datasets/fnlp/VideoThinkBench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-yellow'></a>

</div>

<div align="center">
  <a href="https://huggingface.co/papers/month/2025-11">
    <img src="assets/badges/huggingface_paper_gold_month.svg"/>
  </a>
</div>

## 🎊 News <!-- omit in toc -->

- [2026.02] 🔥🔥 *Our work has been accepted by* **CVPR 2026**! 🎉🎉🎉
- [2025.12] 🔥 We release the VideoThinkBench [Leaderboard](https://thinking-with-video.github.io/#leaderboard) with results from multiple models.
- [2025.11] Our paper has been released on arXiv! 📄 [[Paper](https://arxiv.org/abs/2511.04570)] On HuggingFace, it achieved **#1 Paper of the Day** and **#1 Paper of the Month**!
- [2025.11] 🔥 We release *["minitest"](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench)* of VideoThinkBench — 500 vision-centric + 250 text-centric test samples.

## 📜 Brief Introduction <!-- omit in toc -->

Moving beyond the traditional paradigms of "Thinking with Text" (e.g., Chain-of-Thought) and "Thinking with Images", we propose **"Thinking with Video"** — a new paradigm that unifies visual and textual reasoning through video generation models. It naturally enables human-like dynamic reasoning through video generation, such as **drawing and imagination**.

💡 **A New Unified Reasoning Paradigm**
&nbsp;&nbsp;&nbsp;&nbsp;"Thinking with Video" leverages video generation models to visualize dynamic processes, represent temporal evolution, and embed text within video frames. This approach achieves unified multimodal understanding and generation, overcoming the static constraints of image-based reasoning and the modality separation in traditional approaches.

📊 **VideoThinkBench: A Comprehensive Benchmark**
&nbsp;&nbsp;&nbsp;&nbsp;We developed VideoThinkBench, the first reasoning benchmark specifically designed for evaluating video generation models. It comprises vision-centric tasks (eyeballing puzzles, visual puzzles, ARC-AGI-2, mazes) that leverage dynamic visual reasoning, and text-centric tasks adapted from established benchmarks (MATH, GSM8K, MMLU, MMMU, etc.) that test text-based reasoning capabilities within generated videos.

🚀 **Surpassing VLMs on Several Tasks**
&nbsp;&nbsp;&nbsp;&nbsp;Our evaluation shows that Sora-2 demonstrates competitive reasoning capabilities across both categories. Notably, Sora-2 **surpasses state-of-the-art vision-language models on several vision-centric tasks**, showcasing the unique advantages of dynamic visual reasoning. On text-centric tasks, Sora-2 achieves strong performance including 98.9% on GSM8K, 94.0% on MATH, and 75.5% on MMMU.

<div align="center">
<img src="assets/main_picture.png" width=100% />
</div>


## 📌 Contents <!-- omit in toc -->

- [📚 VideoThinkBench](#-videothinkbench)
- [📈 Benchmark Results](#-benchmark-results)
- [💡 Takeaways](#-takeaways)
- [🛠️ Installation and Quick Start](#-installation-and-quick-start)
- [📖 Documentation](#-documentation)
- [🏋️ Training](#-training)
- [🔎 Citation](#-citation)


## 📚 VideoThinkBench

VideoThinkBench is a comprehensive benchmark for evaluating video generation models' reasoning capabilities, consisting of two main categories:

### Vision-Centric Tasks

<details open>
<summary><b>Eyeballing Puzzles</b> — Spatial reasoning tasks requiring visual estimation and drawing</summary>

Eyeballing Puzzles evaluate a model's ability to perform geometric reasoning through visual estimation. They are divided into **Point Tasks** (e.g., finding midpoints, circle centers, incenters), **Line Tasks** (e.g., drawing parallels, perpendiculars, tangent lines), and **Shape Tasks** (e.g., completing parallelograms, right triangles). The model must identify or draw the correct geometric element among labeled candidates.

The current generators include geometry-quality guardrails so sampled puzzles stay legible: candidate circles plus their labels remain inside the drawable canvas, line-style candidate rows are reflowed instead of clipping at the border, and triangle / quadrilateral tasks reject tiny-area, near-degenerate, or overly ambiguous configurations before writing `data.json`.

|           Task            |                                    Puzzle                                    |                                    Solution                                    |
| :-----------------------: | :--------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| **Circle Center** (Point) | <img src="assets/examples/eyeballing/circle_center_puzzle.png" width="280"/> | <img src="assets/examples/eyeballing/circle_center_solution.png" width="280"/> |
|    **Parallel** (Line)    |   <img src="assets/examples/eyeballing/parallel_puzzle.png" width="280"/>    |   <img src="assets/examples/eyeballing/parallel_solution.png" width="280"/>    |
| **Ray Reflection** (Line) |  <img src="assets/examples/eyeballing/ray_reflect_puzzle.png" width="280"/>  |  <img src="assets/examples/eyeballing/ray_reflect_solution.png" width="280"/>  |
| **Parallelogram** (Shape) | <img src="assets/examples/eyeballing/parallelogram_puzzle.png" width="280"/> | <img src="assets/examples/eyeballing/parallelogram_solution.png" width="280"/> |

</details>

<details open>
<summary><b>Visual Puzzles</b> — Pattern recognition and visual logic problems</summary>

Visual Puzzles evaluate inductive reasoning over visual patterns involving color, shape, and size. They are organized into three categories:
- **Symmetry** tasks (color grid, color hexagon, size grid, shape reflection)
- **Gradient** tasks (color-size gradients, size cycles)
- **Compositionality** tasks (polygon-color mapping, rectangle-height-color, overlapping squares, shape-size grids)

The current generators also apply presentation guardrails for dataset stability: font lookup now falls back cleanly when the preferred OpenSans assets are unavailable, and pattern layouts such as `size_cycle` keep symbols and circles away from the outer canvas border instead of letting them graze or clip at the edge.

|                    Task                    |                                        Puzzle                                         |                                        Solution                                         |
| :----------------------------------------: | :-----------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------: |
|        **Color Hexagon** (Symmetry)        |    <img src="assets/examples/visual_puzzle/color_hexagon_puzzle.png" width="280"/>    |    <img src="assets/examples/visual_puzzle/color_hexagon_solution.png" width="280"/>    |
|         **Color Size** (Gradient)          |     <img src="assets/examples/visual_puzzle/color_size_puzzle.png" width="280"/>      |     <img src="assets/examples/visual_puzzle/color_size_solution.png" width="280"/>      |
| **Polygon Sides Color** (Compositionality) | <img src="assets/examples/visual_puzzle/polygon_sides_color_puzzle.png" width="280"/> | <img src="assets/examples/visual_puzzle/polygon_sides_color_solution.png" width="280"/> |

</details>

<details open>
<summary><b>ARC-AGI-2</b> — Abstract reasoning tasks requiring few-shot learning</summary>

ARC-AGI-2 tasks require models to discover input-output transformation rules from a few examples and apply them to new inputs. These tasks test few-shot learning and abstract pattern recognition capabilities.

|                               Puzzle                                |                               Solution                                |
| :-----------------------------------------------------------------: | :-------------------------------------------------------------------: |
| <img src="assets/examples/arcagi2/arcagi2_puzzle.png" width="350"/> | <img src="assets/examples/arcagi2/arcagi2_solution.png" width="350"/> |

</details>

<details open>
<summary><b>Mazes</b> — Path-finding and navigation challenges</summary>

Maze tasks require models to generate a video showing a valid path from start to finish. Three maze types are supported: **Square** (grid-based), **Hexagon** (hexagonal cells), and **Labyrinth** (concentric rings).

|     Type      |                                 Puzzle                                  |                                 Solution                                  |
| :-----------: | :---------------------------------------------------------------------: | :-----------------------------------------------------------------------: |
|  **Square**   |  <img src="assets/examples/maze/maze_square_puzzle.png" width="240"/>   |  <img src="assets/examples/maze/maze_square_solution.png" width="240"/>   |
|  **Hexagon**  |  <img src="assets/examples/maze/maze_hexagon_puzzle.png" width="240"/>  |  <img src="assets/examples/maze/maze_hexagon_solution.png" width="240"/>  |
| **Labyrinth** | <img src="assets/examples/maze/maze_labyrinth_puzzle.png" width="240"/> | <img src="assets/examples/maze/maze_labyrinth_solution.png" width="240"/> |

</details>

### Text-Centric Tasks

Text-centric tasks evaluate whether video generation models can perform reasoning by embedding text and solutions within generated video frames.

<details open>
<summary><b>Math & Knowledge Reasoning</b></summary>

Adapted from established benchmarks including:
- **Math Reasoning**: GSM8K, MATH-500, AIME24, AIME25
- **General Knowledge**: BBH, MMLU, MMLU-Pro, GPQA-diamond, SuperGPQA-easy
- **Multimodal Math**: MathVista, MathVision
- **Multimodal Understanding**: MMBench, MMMU

|     Task      |                           Prompt (First Frame)                           |                          Solution (Last Frame)                          |
| :-----------: | :----------------------------------------------------------------------: | :---------------------------------------------------------------------: |
|   **GSM8K**   |   <img src="assets/examples/textcentric/gsm8k_first.png" width="280"/>   |   <img src="assets/examples/textcentric/gsm8k_last.png" width="280"/>   |
| **MathVista** | <img src="assets/examples/textcentric/mathvista_first.png" width="280"/> | <img src="assets/examples/textcentric/mathvista_last.png" width="280"/> |
|  **MMBench**  |  <img src="assets/examples/textcentric/mmbench_first.png" width="280"/>  |  <img src="assets/examples/textcentric/mmbench_last.png" width="280"/>  |
|   **MMMU**    |   <img src="assets/examples/textcentric/mmmu_first.png" width="280"/>    |   <img src="assets/examples/textcentric/mmmu_last.png" width="280"/>    |

</details>

Dataset ("minitest" / full test version) is available on [Hugging Face](https://huggingface.co/datasets/fnlp/VideoThinkBench).


## 📈 Benchmark Results

### Performance Comparison Across All Tasks

The table below summarizes the accuracy (%) of Sora-2 compared with SOTA vision-language models across the tasks in VideoThinkBench (full test):

| **Category**        | **Task**                     | **Sora-2** | **Gemini 2.5 Pro** | **GPT5 high** | **Claude Sonnet 4.5** |
| ------------------- | ---------------------------- | :--------: | :----------------: | :-----------: | :-------------------: |
| **Vision-Centric**  | Eyeballing-Point             |    44.7    |        27.8        |     33.6      |         36.2          |
|                     | Eyeballing-Line              |    38.0    |        21.0        |     24.0      |         26.3          |
|                     | Eyeballing-Shape             |    34.5    |        34.5        |     32.5      |         50.5          |
|                     | Visual-Symmetry              |    81.9    |        94.9        |     98.5      |         80.1          |
|                     | Visual-Gradient              |    51.9    |        83.7        |     66.7      |         69.9          |
|                     | Visual-Compositionality      |    57.5    |        67.0        |     85.0      |         82.0          |
|                     | ARC-AGI-2                    |    1.3     |        1.9         |      0.5      |          5.3          |
|                     | Maze-Square                  |    40.0    |        0.0         |      0.0      |          0.0          |
|                     | Maze-Hexagon                 |    0.0     |        0.0         |      0.0      |          0.0          |
|                     | Maze-Labyrinth               |    0.0     |        0.0         |      0.0      |          0.0          |
|                     | **Average**                  |  **35.0**  |      **33.1**      |   **34.1**    |       **35.0**        |
| **Text-Centric**    | Text-Only Math               |    68.6    |        94.8        |     97.2      |         90.0          |
|                     | Text-Only General Knowledge  |    65.3    |        84.5        |     85.2      |         86.3          |
|                     | Multimodal Math              |    61.2    |        66.7        |     69.6      |         65.6          |
|                     | Multimodal General Knowledge |    79.1    |        83.0        |     80.6      |         82.3          |
|                     | **Average**                  |  **68.6**  |      **82.3**      |   **83.2**    |       **81.1**        |
| **Overall Average** |                              |  **44.6**  |      **47.1**      |   **48.1**    |       **48.2**        |

**Note**: For Sora-2: Eyeballing Puzzles use Major Frame evaluation; Text-Centric Reasoning tasks use Audio evaluation results.

### Leaderboard on VideoThinkBench (minitest)

Full interactive leaderboard: [HERE](https://thinking-with-video.github.io/#leaderboard)

**Video Generation Models**

|   #   |          Model          | Average | Eyeballing Point | Eyeballing Line | Eyeballing Shape | Visual Symmetry | Visual Gradient | Visual Compositionality | ARC AGI 2 | Maze-Square | Maze-Hexagon | Maze-Labyrinth |
| :---: | :---------------------: | :-----: | :--------------: | :-------------: | :--------------: | :-------------: | :-------------: | :---------------------: | :-------: | :---------: | :----------: | :------------: |
|   1   |         Sora 2          |  31.6   |        50        |       35        |        25        |       80        |       35        |           53            |    2.8    |    35.3     |     0.0      |      0.0       |
|   2   |         Veo 3.1         |  27.7   |        34        |       24        |        30        |       78        |       40        |           70            |    0.7    |     0.0     |     0.0      |      0.0       |
|   3   |   MiniMax Hailuo 2.3    |  26.0   |        37        |       34        |        28        |       73        |       45        |           43            |    0.0    |     0.0     |     0.0      |      0.0       |
|   4   | doubao-seedance-1-0-pro |  12.4   |        22        |       24        |        35        |       25        |       10        |            8            |    0.0    |     0.0     |     0.0      |      0.0       |
|   5   |     Wan2.2-TI2V-5B      |   7.5   |        18        |       10        |        20        |        8        |       10        |            8            |    0.7    |     0.0     |     0.0      |      0.0       |

**Image Generation Models**

|   #   |     Model     | Average | Eyeballing Point | Eyeballing Line | Eyeballing Shape | Visual Symmetry | Visual Gradient | Visual Compositionality | ARC-AGI-2 | Maze-Square | Maze-Hexagon | Maze-Labyrinth |
| :---: | :-----------: | :-----: | :--------------: | :-------------: | :--------------: | :-------------: | :-------------: | :---------------------: | :-------: | :---------: | :----------: | :------------: |
|   1   | Nano Banana 2 |  29.8   |        24        |       30        |        35        |       85        |       50        |           73            |   0.71    |     0.0     |     0.0      |      0.0       |
|   2   | Seedream 4.5  |  24.5   |        26        |       16        |        30        |       75        |       35        |           63            |     0     |     0.0     |     0.0      |      0.0       |
|   3   | GPT image 1.5 |  19.3   |        24        |       15        |        18        |       38        |       50        |           48            |     0     |     0.0     |     0.0      |      0.0       |

**Vision-Language Models**

|   #   |       Model        | Average | Eyeballing Point | Eyeballing Line | Eyeballing Shape | Visual Symmetry | Visual Gradient | Visual Compositionality | ARC AGI 2 | Maze-Square | Maze-Hexagon | Maze-Labyrinth |
| :---: | :----------------: | :-----: | :--------------: | :-------------: | :--------------: | :-------------: | :-------------: | :---------------------: | :-------: | :---------: | :----------: | :------------: |
|   1   | Claude Sonnet 4.5  |  37.3   |        40        |       34        |        60        |       75        |       75        |           83            |    5.7    |     0.0     |     0.0      |      0.0       |
|   2   |   Gemini 2.5 Pro   |  35.6   |        33        |       23        |        40        |       95        |       95        |           68            |    2.1    |     0.0     |     0.0      |      0.0       |
|   3   |     GPT5 high      |  35.5   |        39        |       30        |        23        |       98        |       80        |           85            |    0.0    |     0.0     |     0.0      |      0.0       |

**Note:**
* "Eyeballing Point/Line/Shape" refer to Point Tasks, Line Tasks and Shape Tasks in Eyeballing Puzzles. The results of video generation models are *Major Frame* evaluation results.
* "Visual Symmetry/Gradient/Compositionality" refer to Symmetry, Gradient and Compositionality tasks in Visual Puzzles.


## 💡 Takeaways

Our systematic evaluation on VideoThinkBench reveals seven key findings:

1. **Surpassing VLMs on Eyeballing Puzzles**: Sora-2 generally **surpasses SOTA VLMs** on eyeballing puzzles, exhibiting strong **geometric and physical reasoning** abilities. It can simulate the extension and reflection of rays and manipulate geometric elements to support spatial reasoning.

2. **Inductive Reasoning on Visual Puzzles**: Sora-2's performance is comparable to Claude Sonnet 4.5 on Shape-Drawing puzzles, demonstrating **inductive reasoning** capabilities. Sora-2 can recognize and apply **patterns of color, shape, and size**, solving visual puzzles involving symmetry, gradients, and compositionality.

3. **Few-Shot Learning Capabilities**: **Sora-2 is a few-shot learner**. On ARC-AGI-2, which requires finding patterns in input-output pairs, Sora-2 can often make **reasonable predictions**, although they do not strictly match dataset annotations.

4. **Unified Multimodal Reasoning**: On text-centric tasks, Sora-2 shows surprising performance on text and multimodal reasoning benchmarks. The video generation model can **embed text within video frames**, enabling unified multimodal understanding and generation. This demonstrates that "Thinking with Video" is potentially a **unified multimodal reasoning paradigm**.

5. **Improved In-Context Learning with More Examples**: Sora-2 achieves better in-context learning by providing more examples, revealing an underexplored direction for analyzing and improving the in-context learning abilities of video generation models.

6. **Test-Time Scaling with Self-Consistency**: **Self-consistency can improve** Sora-2's performance on verifiable video generation reasoning tasks, revealing an underexplored direction: **test-time scaling in video generation reasoning tasks**.

7. **Analysis of Capability Source**: Sora-2 maintains performance comparable to the original test set on adapted math problems, reducing the likelihood of test set leakage. Through comparative experiments with Wan 2.5, we speculate that Sora-2's text-centric reasoning ability originates from its **prompt rewriter** model.


## 🛠️ Installation and Quick Start

1. Clone and install:
   ```bash
   git clone https://github.com/tongjingqi/VideoThinkBench.git
   cd VideoThinkBench
   conda create -y -n videothinkbench python==3.12
   conda activate videothinkbench
   pip install -e .
   ```

2. Download benchmark datasets from Hugging Face:
   ```bash
   hf download --repo-type dataset OpenMOSS-Team/VideoThinkBench --local-dir VideoThinkBench-data
   cd VideoThinkBench-data
   bash unzip_dir.sh Vision-Centric_Reasoning
   bash unzip_dir.sh Text-Centric_Reasoning
   # Or use minitest for quick evaluation:
   # bash unzip_dir.sh minitest_Vision-Centric_Reasoning
   # bash unzip_dir.sh minitest_Text-Centric_Reasoning
   ```

3. Use the unified CLI:
   ```bash
   python3 cli.py --help
   ```

### Common Commands

```bash
# Generate unified dataset with canonical manifest
python3 cli.py data generate \
  --tasks all \
  --count 100 \
  --output-root /path/to/output \
  --num-workers 8

# Export to DiffSynth video CSV
python3 cli.py data export \
  --manifest /path/to/canonical_manifest.jsonl \
  --target diffsynth-video \
  --output /path/to/export/video.csv

# Export to BAGEL training format
python3 cli.py data export \
  --manifest /path/to/canonical_manifest.jsonl \
  --target bagel \
  --output-dir /path/to/export/bagel \
  --mode edit,vlm

# Offline rule-based evaluation
python3 cli.py eval offline \
  --manifest /path/to/canonical_manifest.jsonl \
  --task-group maze \
  --pred-root /path/to/preds \
  --output-dir /path/to/eval/maze
```

### Modal-Aware Assets

The generated `Vision-Centric` records now distinguish prompt / image routing by target modality:

- `ti2v_prompt` and `ti2i_prompt` use the generation-side `image`.
- `ti2t_prompt` and `ti2ti_prompt` use the reasoning-side `reasoning_image`.
- `maze` always saves both images, with `reasoning_image` carrying cell IDs.
- `visual_puzzle` defines `ti2v_prompt` and `ti2i_prompt`, but leaves `ti2t_prompt` empty, so it is excluded from `ms-swift` and `BAGEL` `vlm` exports.

### Repository Structure

```text
VideoThinkBench/
├── assets/                       # Images, badges, and task examples
├── docs/                         # Comprehensive documentation
├── data/                         # Data generation, task registry, export & evaluation
│   ├── visioncentric/
│   │   ├── eyeballing/           # 23 geometric precision tasks
│   │   ├── maze/                 # 3 maze-solving tasks
│   │   ├── visual_puzzles/       # 10 pattern matching tasks
│   │   └── legacy/               # Archived tasks from original repo
│   ├── textcentric/              # Text-Centric independent pipeline
│   ├── evaluation/               # Bench-level inference, offline eval & summary
│   │   ├── infer/                # Model inference (video/image/VLM)
│   │   ├── offline/              # Offline rule-based evaluation
│   │   ├── textcentric/          # Text-centric video evaluation
│   │   └── frame_matching/       # Video frame extraction & matching
│   ├── exporters/                # ms-swift / DiffSynth / BAGEL exporters
│   ├── registry.py               # Unified task registry
│   ├── generate.py               # Unified data generation entry
│   ├── export.py                 # Unified data export entry
│   └── scan.py                   # Canonical manifest builder
├── training/                     # Model-specific training scripts
├── core/                         # Shared schemas, paths, I/O & prompt utilities
├── cli.py                        # Unified CLI entry point
├── pyproject.toml
└── requirements.txt
```

### Migration from Thinking-with-Video

If you are migrating from the original [Thinking-with-Video](https://github.com/tongjingqi/Thinking-with-Video) repository:

| Old Path                       | New Path                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------- |
| `VisionCentric/puzzle/<task>/` | `data/visioncentric/eyeballing/<task>/` or `data/visioncentric/maze/<task>/` |
| `visual_puzzles/`              | `data/visioncentric/visual_puzzles/`                                         |
| `TextCentric/infer/`           | `data/textcentric/`                                                          |
| `TextCentric/eval/src/`        | `data/evaluation/textcentric/`                                               |
| Batch scripts                  | `cli.py data ...` / `cli.py eval ...`                                        |


## 📖 Documentation

- [Developer Guide](docs/guide.md) — benchmark overview, data pipeline, evaluation architecture, training
- [Eyeballing Puzzles](docs/tasks/eyeballing.md) — 23 geometric precision tasks (Point / Line / Shape)
- [Maze Tasks](docs/tasks/maze.md) — Square, Hexagon, and Labyrinth mazes
- [Visual Puzzles](docs/tasks/visual_puzzle.md) — 10 pattern matching tasks (Symmetry / Gradient / Compositionality)
- [ARC-AGI-2 Tasks](docs/tasks/arcagi2.md) — Abstract reasoning with few-shot learning
- [Text-Centric Tasks](docs/tasks/textcentric.md) — Math, knowledge, and multimodal reasoning


## 🏋️ Training

VideoThinkBench currently maintains 4 model-specific training branches:

| Branch | Target Model         | Framework        | Entry Point                                                           |
| ------ | -------------------- | ---------------- | --------------------------------------------------------------------- |
| Video  | Wan2.2-TI2V-5B       | DiffSynth-Studio | [training/wan2.2-ti2v-5b/](training/wan2.2-ti2v-5b/README.md)         |
| Video  | Wan2.2-I2V-A14B      | DiffSynth-Studio | [training/wan2.2-i2v-a14b/](training/wan2.2-i2v-a14b/README.md)       |
| Image  | Qwen-Image-Edit-2511 | DiffSynth-Studio | [training/qwen-image-edit-2511/](training/qwen-image-edit-2511/README.md) |
| BAGEL  | BAGEL-7B-MoT         | BAGEL            | [training/bagel/](training/bagel/README.md)                           |


## ⚖️ License <!-- omit in toc -->

[![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

This project is licensed under the MIT License - see the LICENSE file for details.


## 🔎 Citation

If you find our work helpful, please consider citing our paper 📝 and starring us ⭐!

```bibtex
@article{tong2025thinking,
  title={Thinking with video: Video generation as a promising multimodal reasoning paradigm},
  author={Tong, Jingqi and Mou, Yurong and Li, Hangcheng and Li, Mingzhe and Yang, Yongzhuo and Zhang, Ming and Chen, Qiguang and Liang, Tianyi and Hu, Xiaomeng and Zheng, Yining and others},
  journal={arXiv preprint arXiv:2511.04570},
  year={2025}
}
```

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tongjingqi/Thinking-with-Video&type=date&legend=top-left)](https://www.star-history.com/#tongjingqi/Thinking-with-Video&type=date&legend=top-left)

---

<div align="center">
Made with ❤️ for advancing multimodal reasoning research
</div>
