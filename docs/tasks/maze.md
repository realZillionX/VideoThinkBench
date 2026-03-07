# Maze 任务细节与参数

## 任务定位

`Maze` 任务要求模型在给定迷宫中绘制一条从起点到终点的合法路径。

当前统一主线包含 `3` 个任务：

- `maze_square`。
- `maze_hexagon`。
- `maze_labyrinth`。

## 数据记录

每条记录通常包含：

- 题图 `image`。
- 解图 `solution_image_path`。
- 可选解题视频。
- 起点、终点。
- 路径对应的单元格 `solution_path_cell_ids`。
- 评测所需的边界框、网格或环带元数据。

## 评测逻辑

单任务评测器在各任务目录下。

统一离线评测入口位于 `evaluation/offline/maze.py`。

当前整批离线评测主要判定：

- 是否连通。
- 是否碰到起点。
- 是否碰到终点。
- 是否越过墙体。

## 统一 CLI 暴露参数

以下参数可以直接通过 `cli.py data generate` 调整。

| 参数 | 默认值 | 适用任务 | 说明 |
| --- | --- | --- | --- |
| `--canvas-width` | `480` | 全部 | 输出画布宽度 |
| `--seed` | `42` | 全部 | 随机种子 |
| `--video` | `False` | 全部 | 是否生成解题视频 |
| `--maze-rows` | `9` | `maze_square` | 行数 |
| `--maze-cols` | `9` | `maze_square` | 列数 |
| `--maze-cell-size` | `32` | `maze_square` | 单元格边长 |
| `--hex-radius` | `4` | `maze_hexagon` | 六角网格半径 |
| `--hex-cell-size` | `24` | `maze_hexagon` | 六角单元半径 |
| `--hex-wall-thickness` | `None` | `maze_hexagon` | 墙体厚度覆盖值 |
| `--lab-rings` | `6` | `maze_labyrinth` | 环数 |
| `--lab-segments` | `18` | `maze_labyrinth` | 每环切分段数 |
| `--lab-cell-size` | `18` | `maze_labyrinth` | 环带宽度 |
| `--lab-wall-thickness` | `None` | `maze_labyrinth` | 墙体厚度覆盖值 |

## 共享构造参数

三类迷宫都共享以下基础参数，但其中部分目前只通过 `task_config` 或直接调用生成器时可用。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `output_dir` | `DEFAULT_OUTPUT_DIR` | 输出目录 |
| `canvas_width` | 任务相关 | 输出画布宽度 |
| `aspect` | `None` | 画布纵横比 |
| `seed` | `None` | 随机种子 |
| `prompt` | `None` | 覆盖默认提示词 |
| `show_cell_id` | `False` | 是否在图上打印单元编号 |
| `video` | `False` | 是否生成解题视频 |

## `maze_square`

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `rows` | `15` | 行数，若传偶数会自动补到奇数 |
| `cols` | `15` | 列数，若传偶数会自动补到奇数 |
| `cell_size` | `32` | 单元格边长 |
| `size` | `None` | `cell_size` 的兼容别名 |
| `aspect_ratio` | `None` | 题图最终纵横比 |

### 说明

当前默认使用 DFS 开迷宫，再用 BFS 找到从 `(1, 1)` 到右下角的标准路径。

## `maze_hexagon`

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `radius` | `4` | 六角网格层数 |
| `cell_radius` | `None` | 六角单元半径 |
| `wall_thickness` | `None` | 墙体厚度 |
| `size` | `None` | `cell_radius` 的兼容别名 |

## `maze_labyrinth`

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `rings` | `6` | 环数 |
| `segments` | `18` | 每环切分段数 |
| `ring_width` | `None` | 每环宽度 |
| `wall_thickness` | `None` | 墙体厚度 |
| `size` | `None` | `ring_width` 的兼容别名 |

## 参数注入示例

```bash
python3 cli.py data generate \
  --tasks maze_square maze_hexagon \
  --count 64 \
  --output-root ./outputs/maze \
  --maze-rows 11 \
  --maze-cols 11 \
  --maze-cell-size 36 \
  --task-config '{
    "maze_square": {
      "aspect_ratio": 0.55
    },
    "maze_hexagon": {
      "show_cell_id": true
    }
  }'
```
