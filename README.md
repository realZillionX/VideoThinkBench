# VideoThinkBench

VideoThinkBench 是围绕 VideoThinkBench 数据工程与训练工程的一体化仓库。

## 仓库结构

```
VideoThinkBench/
├── data/puzzle/            # 数据生成器（eyeballing、maze、visual）
│   ├── eyeballing/         # 21 种 Eyeballing Puzzle（画线/画点精确度）
│   ├── maze/               # 3 种迷宫（方形、六角形、迷宫型）
│   └── visual/             # VLMPuzzle 视觉任务（arcagi、sudoku 等）
├── visual_puzzles/         # 10 种 Visual Puzzle（色彩/形状/大小模式匹配）
│   ├── gen_data/           # 数据生成脚本
│   ├── eval/               # 帧匹配评测
│   ├── infer/              # 推理脚本（视频生成 + VLM）
│   └── example_data/       # 示例数据
├── vtb/                    # 统一 CLI 与管线
│   ├── cli.py              # 入口
│   ├── data/               # 生成、导出（ms-swift / DiffSynth）
│   ├── eval/               # 评测（离线规则 / 模型推理）
│   ├── tasks/specs.py      # 任务注册表（36 个任务）
│   └── utils/              # 工具函数
├── training/               # 训练脚本
│   ├── video/              # Wan2.2 视频生成微调（DiffSynth-Studio）
│   ├── image/              # Qwen-Image 图像编辑微调
│   └── vlm/                # Qwen3-VL 微调（SFT + GRPO）
└── data/tools/             # 兼容入口（历史命令兼容层）
```

## 快速开始

### 安装

```bash
cd VideoThinkBench
pip install -e .
```

### CLI 入口

```bash
vtb --help
# 或
python -m vtb.cli --help
```

## 数据侧工作流

### 1. 生成数据（多核 CPU 并行）

```bash
vtb data generate \
  --output-root /path/to/output \
  --tasks all \
  --count 100 \
  --num-workers 8 \
  --seed 42 \
  --video
```

输出：
- `tasks/<task>/...`（puzzle + solution 图像/视频）
- `canonical_manifest.jsonl`（统一样本清单）
- `generation_report.json`（生成报告）

### 2. 导出训练数据

```bash
# ms-swift 格式（VLM SFT + GRPO）
vtb data export --manifest canonical_manifest.jsonl --target ms-swift --output-dir export/vlm --mode sft,grpo

# DiffSynth image 格式
vtb data export --manifest canonical_manifest.jsonl --target diffsynth-image --output export/image/metadata.json

# DiffSynth video 格式
vtb data export --manifest canonical_manifest.jsonl --target diffsynth-video --output export/video/train_video.csv
```

### 3. 评测

离线规则评测（maze / eyeballing / visual_puzzle）：

```bash
vtb eval offline --manifest canonical_manifest.jsonl --task-group maze --pred-root /path/to/predictions --output-dir eval/maze
vtb eval offline --manifest canonical_manifest.jsonl --task-group eyeballing --pred-root /path/to/predictions --output-dir eval/eyeballing
vtb eval offline --manifest canonical_manifest.jsonl --task-group visual_puzzle --pred-root /path/to/predictions --output-dir eval/visual_puzzle
```

推理评测（video / image / vlm）：

```bash
vtb eval infer --modality video --dataset /path/to/dataset --model-path /path/to/model --output-dir eval/infer
```

所有评测命令统一输出 `results.jsonl` 和 `summary.json`。

## 任务总览

| 组别          | 数量   | 说明                        |
| ------------- | ------ | --------------------------- |
| eyeballing    | 23     | VLMPuzzle 几何画线/画点任务 |
| maze          | 3      | 方形、六角形、迷宫型迷宫    |
| visual_puzzle | 10     | 色彩/形状/大小模式匹配      |
| **合计**      | **36** |                             |

## 兼容入口（data/tools）

为兼容历史命令，以下入口仍可用：

- `python -m data.tools.generate_dataset`
- `python -m data.tools.prepare_vlm_data`
- `python -m data.tools.prepare_image_data`
- `python -m data.tools.prepare_video_data`
- `python -m data.tools.eval_offline`
- `python -m data.tools.eval_infer`

## 训练

各训练模块有独立文档：

- [Video（Wan2.2 LoRA 微调）](training/video/README.md)
- [Image（Qwen-Image 编辑微调）](training/image/README.md)
- [VLM（Qwen3-VL SFT + GRPO）](training/vlm/README.md)
