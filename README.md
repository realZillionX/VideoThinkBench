# VideoThinkBench

`VideoThinkBench` 是围绕 VideoThinkBench 数据工程与训练工程的一体化仓库。

## 仓库职责

- `data/`：负责数据生成、评测数据处理与评测流程（Maze、Eyeballing）。
- `training/`：负责与 `DiffSynth-Studio`、`ms-swift` 对齐的训练入口与脚本。
- `evaluators/`：已移除，评测逻辑统一并入 `data/` 与统一 CLI。
- `tests/`：定位为本地临时验证目录，不作为远程同步资产。

## 统一 CLI

安装：

```bash
cd /Users/zillionx/Desktop/code/VideoThinkBench
python3 -m pip install -e .
```

入口：

```bash
videothinkbench --help
# 或
vtb --help
# 或
python3 -m vtb.cli --help
```

## 数据侧工作流

### 1. 生成数据（多核 CPU 并行）

```bash
vtb data generate \
  --output-root /path/to/output \
  --tasks all \
  --count 100 \
  --num-workers 8 \
  --seed 42 \
  --video
```

输出：

- `/path/to/output/tasks/<task>/...`
- `/path/to/output/canonical_manifest.jsonl`
- `/path/to/output/generation_report.json`

说明：并行中间文件写入 `tasks/<task>/.tmp_workers/`，合并后自动清理，避免重复采样。

### 2. 导出训练数据

ms-swift：

```bash
vtb data export \
  --manifest /path/to/output/canonical_manifest.jsonl \
  --target ms-swift \
  --output-dir /path/to/export/vlm \
  --mode sft,grpo
```

DiffSynth image：

```bash
vtb data export \
  --manifest /path/to/output/canonical_manifest.jsonl \
  --target diffsynth-image \
  --output /path/to/export/image/metadata.json
```

DiffSynth video：

```bash
vtb data export \
  --manifest /path/to/output/canonical_manifest.jsonl \
  --target diffsynth-video \
  --output /path/to/export/video/train_video.csv
```

### 3. 评测（离线规则 + 模型推理）

离线规则评测：

```bash
vtb eval offline \
  --manifest /path/to/output/canonical_manifest.jsonl \
  --task-group maze \
  --pred-root /path/to/predictions \
  --output-dir /path/to/eval/maze
```

```bash
vtb eval offline \
  --manifest /path/to/output/canonical_manifest.jsonl \
  --task-group eyeballing \
  --pred-root /path/to/predictions \
  --output-dir /path/to/eval/eyeballing
```

推理评测：

```bash
vtb eval infer \
  --modality vlm|image|video \
  --dataset /path/to/dataset \
  --model-path /path/to/model \
  --output-dir /path/to/eval/infer
```

所有评测命令统一输出：`results.jsonl` 与 `summary.json`。

## 兼容入口（data/tools）

为兼容历史命令，以下入口仍可用，且已接入统一核心：

- `python3 -m data.tools.generate_dataset`
- `python3 -m data.tools.prepare_vlm_data`
- `python3 -m data.tools.prepare_image_data`
- `python3 -m data.tools.prepare_video_data`
- `python3 -m data.tools.eval_offline`
- `python3 -m data.tools.eval_infer`

## 启智平台注意事项

- 启智域名链路使用 `127.0.0.1:8888`（HTTP）与 `127.0.0.1:1080`（SOCKS5）。
- 外网链路使用 `127.0.0.1:7897`。
- `DiffSynth` 推理评测需显式设置 `DIFFSYNTH_PATH`。

