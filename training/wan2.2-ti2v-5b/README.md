# Wan2.2-TI2V-5B Training

使用 `DiffSynth-Studio` 框架对 `Wan2.2-TI2V-5B` 做 `LoRA SFT` 训练。

以下命令默认在仓库根目录执行。

## 数据准备

推荐直接从 `canonical_manifest.jsonl` 导出：

```bash
python3 cli.py data export \
    --manifest /path/to/canonical_manifest.jsonl \
    --target diffsynth-video \
    --task-groups eyeballing maze \
    --output ./dataset/train_video.csv
```

如果你手头只有 `cli.py data generate --output-root ...` 生成出来的数据目录，也可以使用模型目录内的兼容包装脚本：

```bash
python3 training/wan2.2-ti2v-5b/prepare_data.py \
    --dataset_root /path/to/output_root \
    --output_path ./dataset/train_video.csv
```

## 模型权重布局

`MODEL_BASE_PATH` 下应包含：

- `diffusion_pytorch_model-*.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.2_VAE.pth`
- `google/umt5-xxl/`

## 训练

```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

bash training/wan2.2-ti2v-5b/train_sft.sh \
    --dataset ./dataset/train_video.csv \
    --output_dir ./output/wan2.2-ti2v-5b
```

### 仅检查参数拼装

```bash
bash training/wan2.2-ti2v-5b/train_sft.sh \
    --dataset ./dataset/train_video.csv \
    --dry_run
```

## 自动续训

脚本会自动扫描 `output_dir` 中最新且带 `training_state_*` 目录的 checkpoint，并恢复：

- LoRA 权重
- optimizer / scheduler
- step counter

正常情况下，启智任务被打断后直接重跑同一条命令即可。

## 推理预检

```bash
python3 cli.py eval infer \
    --modality video \
    --dataset ./dataset/train_video.csv \
    --model-path /path/to/Wan2.2-TI2V-5B \
    --lora ./output/wan2.2-ti2v-5b \
    --mode precheck \
    --num-samples 5 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/precheck_ti2v
```
