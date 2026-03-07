# 从旧仓库迁移说明

## 背景

旧仓库 `Thinking-with-Video` 采用的是论文展示型目录：

- `VisionCentric/`。
- `visual_puzzles/`。
- `TextCentric/`。

这种结构对论文复现和网页展示很直观，但不适合继续承载统一数据导出、训练和整 Bench 级评测。

## 当前映射关系

| 旧路径 | 新路径 | 说明 |
| --- | --- | --- |
| `VisionCentric/puzzle/<task>/` | `data/visioncentric/eyeballing/<task>/` 或 `data/visioncentric/maze/<task>/` | 统一到 Vision-Centric 主线 |
| `visual_puzzles/` | `data/visioncentric/visual_puzzles/` | 保留单文件生成器实现 |
| `TextCentric/infer/` | `data/textcentric/` | 独立视频请求流程 |
| `TextCentric/eval/src/` | `evaluation/textcentric/` | 独立视频评测流程 |
| 旧批量脚本 | `evaluation/` 与 `cli.py` | 改成统一整 Bench 级入口 |

## 为什么把 `Vision-Centric` 任务收进 `data/visioncentric/`

因为当前工程重点已经不是“展示三块并列代码”，而是：

1. 统一生成 `CanonicalSample`。
2. 统一导出训练数据。
3. 统一做整 Bench 级评测。

从这个目标出发，把所有 `Vision-Centric` 任务收进同一个 `data/visioncentric/` 下更清楚。

## 为什么保留 `legacy`

旧 `VisionCentric` 中有一批任务没有进入当前统一主线，但仍然有参考价值。

为了避免它们与当前主线任务混淆，这些任务被集中归档到 `data/visioncentric/legacy/`。
