# VideoThinkBench

VideoThinkBench 是围绕 VideoThinkBench 评测基准的数据工程与训练工程一体化仓库。

## 仓库结构

```
VideoThinkBench/
├── docs/                   # 项目文档（背景、资源、指南）
├── data/                   # 数据生成器 + 导出器
│   ├── eyeballing/         # 21 种几何精度任务（画线/画点）
│   ├── maze/               # 3 种迷宫（方形、六角形、迷宫型）
│   ├── visual/             # VLMPuzzle 视觉任务（arcagi、sudoku 等）
│   ├── visual_puzzles/     # 10 种模式匹配（色彩/形状/大小）
│   ├── textcentric/        # Text-Centric 推理（Sora-2 视频请求）
│   ├── exporters/          # 导出器（ms-swift / DiffSynth）
│   ├── generate.py         # 统一数据生成入口
│   ├── export.py           # 统一数据导出入口
│   └── scan.py             # Manifest 扫描
├── evaluators/             # 评测代码
│   ├── offline/            # 离线规则评测（maze / eyeballing / visual_puzzle）
│   ├── infer/              # 推理评测（video / image / vlm）
│   ├── textcentric/        # LLM-as-judge 文本推理评测
│   ├── frame_matching/     # 视频帧匹配评测
│   ├── commands.py         # CLI 子命令注册
│   └── pipeline.py         # 统一评测管线
├── training/               # 训练脚本
│   ├── video/              # Wan2.2 视频生成 LoRA 微调
│   ├── image/              # Qwen-Image 图像编辑微调
│   └── vlm/                # Qwen3-VL SFT + GRPO
├── tasks/                  # 任务注册表（36 个任务）
├── utils/                  # 通用工具
├── scripts/                # Shell + 兼容入口脚本
├── cli.py                  # 统一 CLI 入口
├── pyproject.toml
└── requirements.txt
```

## 快速开始

```bash
# 安装
pip install -e .

# CLI
python cli.py --help
```

## 数据生成

```bash
python cli.py data generate --tasks all --count 100 --output-root /path/to/output --num-workers 8
```

## 数据导出

```bash
# ms-swift（VLM SFT + GRPO）
python cli.py data export --manifest canonical_manifest.jsonl --target ms-swift --output-dir export/vlm

# DiffSynth video
python cli.py data export --manifest canonical_manifest.jsonl --target diffsynth-video --output export/video.csv
```

## 评测

```bash
# 离线规则评测
python cli.py eval offline --manifest canonical_manifest.jsonl --task-group maze --pred-root /path/to/preds --output-dir eval/maze

# 推理评测
python cli.py eval infer --modality video --dataset /path/to/dataset --model-path /path/to/model --output-dir eval/infer
```

## 任务总览

| 组别          | 数量   | 说明                      |
| ------------- | ------ | ------------------------- |
| eyeballing    | 23     | 几何精度任务（画线/画点） |
| maze          | 3      | 方形、六角形、迷宫型      |
| visual_puzzle | 10     | 色彩/形状/大小模式匹配    |
| **合计**      | **36** |                           |

## 训练

- [Video（Wan2.2 LoRA 微调）](training/video/README.md)
- [Image（Qwen-Image 编辑微调）](training/image/README.md)
- [VLM（Qwen3-VL SFT + GRPO）](training/vlm/README.md)

## 项目背景

详见 [docs/project_background.md](docs/project_background.md)。
