# Visual Puzzle 任务细节与参数

## 任务定位

`Visual Puzzle` 任务主要考察颜色、形状、大小和组合规律的模式匹配能力。

当前统一主线包含 `10` 个任务，代码集中在 `data/visioncentric/visual_puzzles/data_generation.py`。

## 任务列表

### Symmetry

- `color_hexagon`。
- `color_grid`。
- `size_grid`。
- `shape_reflect`。

### Gradient

- `color_size`。
- `size_cycle`。

### Compositionality

- `polygon_sides_color`。
- `rectangle_height_color`。
- `color_overlap_squares`。
- `shape_size_grid`。

## 数据记录

每个样本通常包含：

- `question`。
- `answer`。
- `options`。
- `caption`。
- `explanation`。
- `deduction`。
- 题图。
- 解图。

在统一主线中，这些字段会被转成 `CanonicalSample`，其中答案会落到 `correct_option` 字段。

## 评测逻辑

单任务不再各自实现独立 evaluator。

整批离线评测统一在 `evaluation/offline/visual_puzzle.py` 中完成，核心是：

- 若输入为视频，先找出与标准解最相近的帧。
- 再将最佳帧或预测图像与标准解图做差异比较。

## 统一 CLI 暴露参数

统一 CLI 当前只直接暴露少量参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | 随机种子 |
| `--task-config` / `--task-config-path` | `None` | 传入任务专属字段 |

除此之外，`data.generate` 内部还固定传入：

| 内部字段 | 固定值 | 说明 |
| --- | --- | --- |
| `target_size` | `(1280, 704)` | 最终输出分辨率 |
| `unique` | `True` | 按题图内容去重 |

## 各任务字段

这些任务类采用 `BaseModel` 风格构造，字段可以通过 `task_config` 直接覆盖。

### `color_grid`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `colors` | `blue / green / yellow / red / purple / orange` |

### `color_hexagon`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `colors` | `blue / green / yellow / red / purple / orange` |

### `color_overlap_squares`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `colors` | `blue / green / yellow / red / purple / orange` |
| `numbers` | `[1..9]` |
| `num_sides` | `4` |
| `rotate_range` | `(-45, 0)` |

### `color_size`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `colors` | 每种颜色对应 `4` 个深浅级别 |
| `shape_sides` | `circle / square / pentagon / hexagon` |

### `polygon_sides_color`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Light.ttf` |
| `colors` | `blue / green / yellow / red / purple / orange` |

### `rectangle_height_color`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Light.ttf` |
| `colors` | `blue / green / yellow / red / purple / orange` |

### `shape_reflect`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `color` | `#d9ead3` |
| `shapes` | `triangle / square / pentagon / hexagon` |

### `shape_size_grid`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `color` | `#d9ead3` |
| `shapes` | `triangle / square / pentagon / hexagon` |

### `size_cycle`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `color` | `#fff2cc` |

### `size_grid`

| 字段 | 默认值 |
| --- | --- |
| `image_size` | `512` |
| `scale_factor` | `4` |
| `path_font` | `fonts/OpenSans-Medium.ttf` |
| `color` | `#fff2cc` |

## 参数注入示例

```bash
python3 cli.py data generate \
  --tasks color_grid color_size \
  --count 20 \
  --output-root ./outputs/visual_puzzles \
  --task-config '{
    "color_grid": {
      "image_size": 640,
      "scale_factor": 3,
      "target_size": [1440, 810]
    },
    "color_size": {
      "image_size": 768,
      "scale_factor": 2
    }
  }'
```

## 当前训练边界

`visual_puzzle` 可以导出到 `ms-swift` 的 `SFT` 数据中。

但在 `GRPO` 链路中，当前奖励函数尚未完整兼容颜色单词这类答案格式，因此后续若要训练 `VLM`，需要先修奖励函数。
