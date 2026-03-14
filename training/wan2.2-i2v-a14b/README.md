# Wan2.2-I2V-A14B Training

使用 `DiffSynth-Studio` 框架对 `Wan2.2-I2V-A14B` 做 `LoRA SFT` 训练。

这个模型的训练按 `high_noise`／`low_noise` 两路独立进行，输出目录也分开维护。

## 数据准备

```bash
python3 cli.py data export \
    --manifest /path/to/canonical_manifest.jsonl \
    --target diffsynth-video \
    --task-groups eyeballing maze \
    --output ./dataset/train_video.csv
```

兼容包装脚本：

```bash
python3 training/wan2.2-i2v-a14b/prepare_data.py \
    --dataset_root /path/to/output_root \
    --output_path ./dataset/train_video.csv
```

## 模型权重布局

`MODEL_BASE_PATH` 下应包含：

- `high_noise_model/diffusion_pytorch_model*.safetensors`
- `low_noise_model/diffusion_pytorch_model*.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.1_VAE.pth`
- `google/umt5-xxl/`

## 训练

```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-I2V-A14B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

bash training/wan2.2-i2v-a14b/train_sft.sh \
    --dataset ./dataset/train_video.csv \
    --output_dir ./output/wan2.2-i2v-a14b
```

### 只训一个分支

```bash
bash training/wan2.2-i2v-a14b/train_sft.sh \
    --dataset ./dataset/train_video.csv \
    --branch high_noise
```

### 仅检查参数拼装

```bash
bash training/wan2.2-i2v-a14b/train_sft.sh \
    --dataset ./dataset/train_video.csv \
    --dry_run
```

## 自动续训

恢复逻辑按分支独立执行：

- `output_dir/high_noise/`
- `output_dir/low_noise/`

每个分支都会自动恢复对应目录下最新的 checkpoint 与训练状态。

## 推理预检

```bash
python3 cli.py eval infer \
    --modality video \
    --dataset ./dataset/train_video.csv \
    --model-path /path/to/Wan2.2-I2V-A14B \
    --lora ./output/wan2.2-i2v-a14b \
    --mode precheck \
    --num-samples 5 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/precheck_i2v
```
