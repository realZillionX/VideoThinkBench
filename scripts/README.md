# Scripts 说明

## 定位

`scripts/` 目录现在只承担两类职责：

1. 兼容旧命令路径的薄包装脚本。
2. 一些仍然适合手工运行的示例脚本。

主入口已经迁移到：

- `python3 cli.py data ...`。
- `python3 cli.py eval ...`。

## 建议优先使用的入口

| 目标 | 推荐入口 |
| --- | --- |
| 生成统一数据 | `cli.py data generate` |
| 导出训练数据 | `cli.py data export` |
| 批量离线评测 | `cli.py eval offline` |
| 批量推理评测 | `cli.py eval infer` 或 `cli.py eval run` |

## 兼容包装脚本

这些脚本仍然保留，但本质上只是对统一主线的薄包装：

- `prepare_image_data.py`。
- `prepare_video_data.py`。
- `prepare_vlm_data.py`。
- `eval_infer.py`。
- `eval_offline.py`。
- `generate_dataset.py`。

## 示例脚本

这些脚本更接近“论文期 / 人工实验期”的运行示例，不是统一主线的一部分：

- `run_visual_puzzles.sh`。
- `run_VLM.sh`。
- `extract_best_frame.sh`。
- `run_textcentric.sh`。
- `eval_textcentric.sh`。
- `generate_data.sh`。

使用这些脚本前，建议先阅读对应文档页，确认它们是否仍然适合当前实验目标。
