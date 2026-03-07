# Eyeballing 任务细节与参数

## 任务定位

`Eyeballing` 任务要求模型在图像中完成几何判断，并通过“画点、画线、画形”或“高亮候选项”的方式给出答案。

当前统一主线共包含 `23` 个任务。

## 子类划分

### Point 类

- `circle_center`。
- `circumcenter`。
- `fermat_point`。
- `incenter`。
- `midpoint`。
- `orthocenter`。
- `ray_intersection`。
- `triangle_center`。
- `arc_connect_point_ver`。

### Line 类

- `angle_bisector`。
- `arc_connect`。
- `circle_tangent_line`。
- `circle_tangent_point`。
- `parallel`。
- `perpendicular`。
- `perpendicular_bisector`。
- `ray`。
- `ray_reflect`。
- `reflection`。

### Shape 类

- `isosceles_trapezoid`。
- `parallelogram`。
- `right_triangle`。
- `square_outlier`。

## 数据记录

绝大多数 `eyeballing` 任务会输出：

- 题图 `image`。
- 解图 `solution_image_path`。
- 候选点或候选项元数据。
- 正确选项 `correct_option`。

部分任务还会输出可选解题视频。

## 单任务评测

单任务评测器与任务代码放在一起。

其中大多数任务复用 `data/point_target_base.py` 中的通用候选点逻辑。

整批离线评测入口位于 `evaluation/offline/eyeballing.py`。

## 统一 CLI 暴露参数

下面这些参数可以直接通过 `python3 cli.py data generate ...` 调整。

| 参数 | 来源 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--canvas-width` | 统一 CLI | `480` | 画布宽度 |
| `--seed` | 统一 CLI | `42` | 随机种子 |
| `--video` | 统一 CLI | `False` | 是否生成解题视频 |
| `--point-radius` | 统一 CLI | `None` | 覆盖候选点半径，当前已接入多数基于 `PointTargetPuzzleGenerator` 的任务 |
| `--line-width` | 统一 CLI | `None` | 覆盖几何线条宽度，当前已接入多数基于 `PointTargetPuzzleGenerator` 的任务 |

## 共享构造参数

多数 `eyeballing` 任务继承 `PointTargetPuzzleGenerator`，共享以下参数。

这些参数当前不一定全部暴露在统一 CLI 上，但可以通过 `--task-config` 或 `--task-config-path` 注入。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `output_dir` | `DEFAULT_OUTPUT_DIR` | 输出目录 |
| `canvas_width` | `480` | 画布宽度 |
| `aspect` | `None` | 宽高比 |
| `seed` | `None` | 随机种子 |
| `prompt` | `None` | 覆盖默认提示词 |
| `option_labels` | `('A','B','C','D','E')` | 候选标签集合 |
| `margin_ratio` | `0.06` | 边距比例 |
| `record_video` | `False` | 是否生成解题视频 |
| `point_radius` | `None` | 候选点半径覆盖值 |
| `line_width` | `None` | 几何线条宽度覆盖值 |

## 特殊任务参数

### `arc_connect`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `mask_fraction` | `0.18` | 遮挡带宽度占画布宽度的比例 |
| `arc_span_deg` | `20.0` | 两侧弧段的可见角度 |

### `arc_connect_point_ver`

当前构造参数为：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `canvas_width` | `480` | 画布宽度 |
| `aspect` | `None` | 宽高比 |
| `seed` | `None` | 随机种子 |
| `prompt` | `None` | 覆盖默认提示词 |
| `record_video` | `False` | 是否输出解题视频 |
| `point_radius` | `None` | 候选点半径覆盖值 |
| `line_width` | `None` | 线宽覆盖值 |

其内部还固定使用：

| 内部参数 | 固定值 | 说明 |
| --- | --- | --- |
| `mask_fraction` | `0.35` | 垂直遮挡带宽度比例 |
| `arc_span_deg` | `20.0` | 弧段可见角度 |

### `ray`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `canvas_size` | `480` | 画布基准尺寸 |
| `aspect` | `None` | 宽高比 |
| `mirror_count` | `12` | 镜面数量 |
| `min_reflections` | `2` | 最少反射次数 |
| `prompt` | `None` | 覆盖默认提示词 |
| `seed` | `None` | 随机种子 |

## 任务与代码位置

| 任务 | 代码路径 |
| --- | --- |
| `angle_bisector` | `data/visioncentric/eyeballing/angle_bisector/` |
| `arc_connect` | `data/visioncentric/eyeballing/arc_connect/` |
| `arc_connect_point_ver` | `data/visioncentric/eyeballing/arc_connect_point_ver/` |
| `circle_center` | `data/visioncentric/eyeballing/circle_center/` |
| `circle_tangent_line` | `data/visioncentric/eyeballing/circle_tangent_line/` |
| `circle_tangent_point` | `data/visioncentric/eyeballing/circle_tangent_point/` |
| `circumcenter` | `data/visioncentric/eyeballing/circumcenter/` |
| `fermat_point` | `data/visioncentric/eyeballing/fermat_point/` |
| `incenter` | `data/visioncentric/eyeballing/incenter/` |
| `isosceles_trapezoid` | `data/visioncentric/eyeballing/isosceles_trapezoid/` |
| `midpoint` | `data/visioncentric/eyeballing/midpoint/` |
| `orthocenter` | `data/visioncentric/eyeballing/orthocenter/` |
| `parallel` | `data/visioncentric/eyeballing/parallel/` |
| `parallelogram` | `data/visioncentric/eyeballing/parallelogram/` |
| `perpendicular` | `data/visioncentric/eyeballing/perpendicular/` |
| `perpendicular_bisector` | `data/visioncentric/eyeballing/perpendicular_bisector/` |
| `ray` | `data/visioncentric/eyeballing/ray/` |
| `ray_intersection` | `data/visioncentric/eyeballing/ray_intersection/` |
| `ray_reflect` | `data/visioncentric/eyeballing/ray_reflect/` |
| `reflection` | `data/visioncentric/eyeballing/reflection/` |
| `right_triangle` | `data/visioncentric/eyeballing/right_triangle/` |
| `square_outlier` | `data/visioncentric/eyeballing/square_outlier/` |
| `triangle_center` | `data/visioncentric/eyeballing/triangle_center/` |

## 参数注入示例

```bash
python3 cli.py data generate \
  --tasks midpoint ray \
  --count 32 \
  --output-root ./outputs/eyeballing \
  --point-radius 14 \
  --line-width 6 \
  --task-config '{
    "ray": {
      "mirror_count": 16,
      "min_reflections": 3
    }
  }'
```
