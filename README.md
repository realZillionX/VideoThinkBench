# VideoThinkBench

VideoThinkBench 是面向 `Thinking with Video` 后续实验的数据、评测与训练底座仓库。

它不再简单复刻旧仓库 `Thinking-with-Video/` 的目录结构，而是把公开基准任务、训练数据导出、整 Bench 级评测与后续微调流程统一到了同一套工程里。

## 仓库结构

```text
VideoThinkBench/
├── docs/                         # 面向公开使用者的说明文档
├── data/                         # 数据生成、任务注册、导出与 Manifest
│   ├── visioncentric/
│   │   ├── eyeballing/           # 23 个几何精度任务
│   │   ├── maze/                 # 3 个迷宫任务
│   │   ├── visual_puzzles/       # 10 个模式匹配任务
│   │   └── legacy/               # 旧仓库遗留任务，暂未接入统一主线
│   ├── textcentric/              # Text-Centric 独立流程
│   ├── exporters/                # ms-swift / DiffSynth 导出器
│   ├── registry.py               # 统一任务注册表
│   ├── generate.py               # 统一数据生成入口
│   ├── export.py                 # 统一数据导出入口
│   └── scan.py                   # Canonical Manifest 构建与扫描
├── evaluation/                   # 整 Bench 级推理、离线评测与汇总
│   ├── infer/
│   ├── offline/
│   ├── textcentric/
│   ├── frame_matching/
│   ├── commands.py
│   └── pipeline.py
├── training/                     # Video / Image / VLM 训练脚本
├── core/                         # 共享 Schema、路径、I/O 与 Prompt 工具
├── scripts/                      # 兼容旧命令路径的薄包装脚本
├── cli.py                        # 统一 CLI 入口
├── pyproject.toml
└── requirements.txt
```

## 当前支持状态

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| Vision-Centric 统一主线 | 稳定 | 已统一到 `Canonical Manifest` 与 `cli.py` |
| Text-Centric | 独立流程 | 仍保留请求视频与音视频评测链路，尚未并入统一 Manifest 主线 |
| Legacy 任务 | 保留但不主推 | 来源于旧 `VisionCentric`，用于归档与后续选择性恢复 |

## 快速开始

```bash
python3 -m pip install -e .
python3 cli.py --help
```

## 常用命令

```bash
# 生成统一数据集与 canonical manifest
python3 cli.py data generate \
  --tasks all \
  --count 100 \
  --output-root /path/to/output \
  --num-workers 8

# 导出为 ms-swift 数据
python3 cli.py data export \
  --manifest /path/to/canonical_manifest.jsonl \
  --target ms-swift \
  --output-dir /path/to/export/vlm

# 导出为 DiffSynth video CSV
python3 cli.py data export \
  --manifest /path/to/canonical_manifest.jsonl \
  --target diffsynth-video \
  --output /path/to/export/video.csv

# 离线规则评测
python3 cli.py eval offline \
  --manifest /path/to/canonical_manifest.jsonl \
  --task-group maze \
  --pred-root /path/to/preds \
  --output-dir /path/to/eval/maze
```

## 文档导航

- [基准总览](docs/benchmark_overview.md)。
- [任务目录总表](docs/task_catalog.md)。
- [评测结构说明](docs/evaluation.md)。
- [训练结构说明](docs/training.md)。
- [从旧仓库迁移说明](docs/migration_from_thinking_with_video.md)。
- [Eyeballing 任务细节与参数](docs/tasks/eyeballing.md)。
- [Maze 任务细节与参数](docs/tasks/maze.md)。
- [Visual Puzzle 任务细节与参数](docs/tasks/visual_puzzle.md)。
- [Text-Centric 独立流程说明](docs/tasks/textcentric.md)。
- [Legacy 任务说明](docs/tasks/legacy.md)。

## 训练入口

- [Video：Wan2.2 LoRA 微调](training/video/README.md)。
- [Image：Qwen-Image 编辑微调](training/image/README.md)。
- [VLM：Qwen3-VL SFT + GRPO](training/vlm/README.md)。
