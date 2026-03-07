# 评测结构说明

## 为什么保留 `evaluation/`

单任务目录中的 `generator.py` 和 `evaluator.py` 负责“某一个任务自己如何出题、如何判题”。

`evaluation/` 负责的是“整 Bench 级别如何把很多任务串起来跑完，并输出统一结果”。

这两层职责不同，因此不建议把它们合并成一个目录。

为了减少歧义，仓库已经将原来的 `evaluators/` 改名为 `evaluation/`。

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `data/visioncentric/.../evaluator.py` | 单任务规则评测逻辑 |
| `evaluation/offline/` | 针对整批 `CanonicalSample` 的离线批量评测 |
| `evaluation/infer/` | 调用外部模型进行批量推理 |
| `evaluation/textcentric/` | Text-Centric 独立视频评测 |
| `evaluation/frame_matching/` | 视频抽帧与最优帧匹配 |
| `evaluation/pipeline.py` | 统一结果落盘与 summary 汇总 |

## 当前离线评测口径

### `maze`

入口位于 `evaluation/offline/maze.py`。

核心判定依赖各迷宫任务自己的 evaluator，当前主要检查：

- 是否检测到红色路径。
- 是否碰到起点。
- 是否碰到终点。
- 是否穿墙。
- 路径是否连通。

### `eyeballing`

入口位于 `evaluation/offline/eyeballing.py`。

当前会综合以下信息源：

- 最终预测图像中的红色高亮位置。
- 文本回答。
- 视频帧判读。
- 转写文本。

最后通过聚合选项与正确选项对比得到通过与否。

### `visual_puzzle`

入口位于 `evaluation/offline/visual_puzzle.py`。

当前流程是：

1. 如果有预测视频，先抽取与标准解最接近的帧。
2. 将最佳帧或预测图像与标准解图比较差异。
3. 输出差异值等指标。

当前该组仍以差异值为主，尚未统一到强制 `pass / fail` 阈值，这是后续应继续完善的方向。

## 推理评测

`evaluation/infer/` 只负责“给模型喂数据并保存预测结果”，不负责单任务规则判断。

| 模态 | 入口 | 外部依赖 |
| --- | --- | --- |
| `video` | `evaluation/infer/video.py` | `DiffSynth-Studio` |
| `image` | `evaluation/infer/image.py` | `DiffSynth-Studio` |
| `vlm` | `evaluation/infer/vlm.py` | `ms-swift` |

## Text-Centric

`Text-Centric` 当前仍然沿用独立脚本式流程，不走统一 `cli.py eval` 主线。

详细见 [tasks/textcentric.md](tasks/textcentric.md)。
