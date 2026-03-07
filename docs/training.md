# 训练结构说明

## 训练分支

本仓库当前维护 `3` 条训练分支。

| 分支 | 入口 | 目标模型 | 训练框架 | 训练数据来源 |
| --- | --- | --- | --- | --- |
| `video` | `training/video/` | `Wan2.2` | `DiffSynth-Studio` | `CanonicalSample -> diffsynth-video CSV` |
| `image` | `training/image/` | `Qwen-Image-Edit` | `DiffSynth-Studio` | `CanonicalSample -> diffsynth-image JSON` |
| `vlm` | `training/vlm/` | `Qwen3-VL` | `ms-swift` | `CanonicalSample -> ms-swift JSONL` |

## 训练与数据主线的关系

`training/` 不直接读取单任务目录，而是优先读取统一导出后的中间数据。

这意味着：

1. 先在 `data/` 内生成或扫描任务数据。
2. 再导出成对应训练框架需要的格式。
3. 最后在 `training/` 下启动训练。

## 当前已知边界

### `video`

`training/video/train.py` 已支持自动恢复 `LoRA`、`optimizer`、`scheduler` 和 `step counter`。

这条链路当前是后续迷宫实验最关键的训练入口。

### `image`

图像编辑链路主要服务于“只看最终图像，不看完整视频”的实验。

它与 `video` 链路共享同一套 `CanonicalSample`，但监督信号只用解图，不用解题视频。

### `vlm`

`VLM SFT` 数据导出已经支持 `eyeballing`、`maze`、`visual_puzzle`。

`GRPO` 奖励函数当前兼容：

- `maze` 的路径列表答案。
- `eyeballing` 的单字母答案。
- `visual_puzzle` 的文本答案。

其中 `visual_puzzle` 目前采用的是“归一化后的精确文本匹配”。
