# 基准总览

## 仓库定位

`VideoThinkBench` 当前是一个面向后续实验的统一工程仓库，而不是旧 `Thinking-with-Video` 仓库的直接镜像。

公开可见的任务说明、数据格式、评测逻辑和训练入口都应该以本仓库文档为准。

旧仓库仍然是重要来源，尤其提供了论文原始目录结构、任务命名和部分测试样例，但不再是当前工程组织方式的唯一依据。

## 当前任务版图

### Vision-Centric 统一主线

当前统一注册表位于 [data/registry.py](../data/registry.py)，共收录 `36` 个任务。

| 组别 | 数量 | 当前代码位置 | 说明 |
| --- | --- | --- | --- |
| `eyeballing` | `23` | `data/visioncentric/eyeballing/` | 几何画点、画线、画形任务 |
| `maze` | `3` | `data/visioncentric/maze/` | 方形、六角形、迷宫环形 |
| `visual_puzzle` | `10` | `data/visioncentric/visual_puzzles/` | 颜色、形状、大小模式匹配 |

### Text-Centric 独立流程

`Text-Centric` 仍保留独立的视频请求与音视频评测链路，代码位于 `data/textcentric/` 和 `evaluation/textcentric/`。

它目前没有接入统一 `Canonical Manifest` 主线，因此在工程语义上应视为“并行流程”，而不是 `Vision-Centric` 那样的统一数据管线成员。

### Legacy 归档任务

来自旧 `VisionCentric/puzzle/`，但当前未进入统一主线的任务，被统一放到 `data/visioncentric/legacy/`。

这些任务包括：

- `arcagi`。
- `circle_count`。
- `jigsaw`。
- `mirror`。
- `rects`。
- `sudoku`。

## 数据主线

统一数据主线分成 `4` 层：

1. 单任务生成器输出原始 `data.json`、题图、解图与可选解题视频。
2. `data/scan.py` 将原始记录归一化为 `CanonicalSample`。
3. `data/export.py` 和 `data/exporters/` 将 `CanonicalSample` 导出到训练框架格式。
4. `evaluation/` 负责整 Bench 级推理、离线评测与汇总。

`CanonicalSample` 是本仓库最重要的工程中间层，定义见 [core/schemas.py](../core/schemas.py)。

## 参数调整原则

后续若要做数据生成参数调优，应优先区分 `3` 种参数来源：

1. `cli.py data generate` 直接暴露的统一参数。
2. 仅能通过 `--task-config` 或 `--task-config-path` 注入的任务专属参数。
3. 当前代码中尚未暴露、需要新增到生成器或统一 CLI 的参数。

本文档与各任务文档会明确标注这三类参数。
