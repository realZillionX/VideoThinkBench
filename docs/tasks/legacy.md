# Legacy 任务说明

## 定义

`legacy` 指来自旧 `Thinking-with-Video / VisionCentric / puzzle /`，但当前没有进入统一注册表和统一主线的数据任务。

这些任务被保留，是为了：

- 方便对照旧实验。
- 保留已有 evaluator 与测试资产。
- 为未来选择性恢复做准备。

## 当前包含的任务

| 任务 | 当前路径 | 旧路径 |
| --- | --- | --- |
| `arcagi` | `data/visioncentric/legacy/arcagi/` | `VisionCentric/puzzle/arcagi/` |
| `circle_count` | `data/visioncentric/legacy/circle_count/` | `VisionCentric/puzzle/circle_count/` |
| `jigsaw` | `data/visioncentric/legacy/jigsaw/` | `VisionCentric/puzzle/jigsaw/` |
| `mirror` | `data/visioncentric/legacy/mirror/` | `VisionCentric/puzzle/mirror/` |
| `rects` | `data/visioncentric/legacy/rects/` | `VisionCentric/puzzle/rects/` |
| `sudoku` | `data/visioncentric/legacy/sudoku/` | `VisionCentric/puzzle/sudoku/` |

## 当前约束

- 它们不在 [data/registry.py](../../data/registry.py) 里。
- `cli.py data generate --tasks all` 不会覆盖它们。
- `Canonical Manifest` 主线默认不扫描它们。

## 何时考虑恢复

只有在满足以下条件时才建议恢复：

1. 明确需要该任务参与新实验。
2. 明确其答案格式与离线评测方式。
3. 有对应的训练导出需求。
4. 有最小回归测试。
