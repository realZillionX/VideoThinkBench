# Wan2.2 Training (Video Generation Model)

使用 DiffSynth-Studio 框架对 Wan2.2-TI2V-5B 模型进行 LoRA SFT 训练。

## 环境要求

```bash
# 安装 DiffSynth-Studio
cd /path/to/DiffSynth-Studio
pip install -e .

pip install accelerate deepspeed pandas
```

## 模型权重

需要预先下载以下文件到 `MODEL_BASE_PATH`:
- `diffusion_pytorch_model-*.safetensors` (DiT 模型，3个文件)
- `models_t5_umt5-xxl-enc-bf16.pth` (T5 编码器)
- `Wan2.2_VAE.pth` (VAE)
- `google/umt5-xxl/` (Tokenizer)

## 数据准备

训练脚本需要 CSV 文件，包含两列：
- `video`: 视频文件绝对路径
- `prompt`: 文本描述

```csv
video,prompt
\"/path/to/video1.mp4\",\"Draw a red path...\"
\"/path/to/video2.mp4\",\"Connect the dots...\"
```

**重要**: 使用 `QUOTE_ALL` 格式化 CSV 以避免解析错误。

## 训练

### 统一训练脚本（单机 / 多机）

```bash
export MODEL_BASE_PATH=/path/to/Wan2.2-TI2V-5B
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

# 先转换 VideoThinkBench 数据为 CSV
python -m data.tools.prepare_video_data \
    --dataset_root /path/to/VideoThinkBench/dataset \
    --output_path ./dataset/train_video.csv

# 单机
bash train_sft.sh --dataset ./dataset/train_video.csv --dataset_root /path/to/VideoThinkBench/dataset --num_nodes 1

# 可选参数:
#   --output_dir ./output/wan_lora
#   --num_frames 81
#   --height 896
#   --width 480
#   --lora_rank 32
#   --num_epochs 3
```

```bash
# 多机（示例：15 节点 × 8 GPU）
bash train_sft.sh --dataset ./dataset/train_video.csv --dataset_root /path/to/VideoThinkBench/dataset --num_nodes 15 --gpus_per_node 8 --machine_rank 0
```

**可用参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_nodes` | 1 | 节点数 |
| `--gpus_per_node` | 8 | 每节点 GPU 数 |
| `--machine_rank` | 0 | 机器 rank（多机必填） |
| `--dataset` | `./dataset.csv` | 数据集 CSV 路径 |
| `--dataset_root` | - | 数据集根目录（相对路径时需要） |
| `--output_dir` | `./output/wan_lora` | 输出目录 |
| `--lora_rank` | 32 | LoRA Rank |
| `--num_epochs` | 3 | 训练轮数 |
| `--learning_rate` | 1e-4 | 学习率 |
| `--num_frames` | 81 | 视频帧数 |
| `--height` | 896 | 视频高度 |
| `--width` | 480 | 视频宽度 |
| `--save_steps` | 250 | 保存间隔 |
| `--lora_checkpoint` | - | 续训 checkpoint 路径 |

> **Note**: 多机模式会自动生成临时 accelerate 配置文件。

## 视频配置约束 (CRITICAL)

| 参数 | 约束 | 示例 |
|------|------|------|
| num_frames | `(n-1) % 4 == 0` | 81, 49, 25 |
| height/width | 被 32 整除 | 896, 480 |

**违反约束会导致静默失败或崩溃！**

## 自动续训

脚本支持自动从中断处恢复，包括**完整的训练状态保存**：

### 保存内容
每个 checkpoint 保存时，除 LoRA 权重外，还会自动保存一个 `training_state_*.pt` 文件，包含：
- **Optimizer 状态**（AdamW 的一阶/二阶动量估计）
- **Scheduler 状态**
- **Step counter**

### 恢复行为
- 自动检测 `output_path` 下最新的 checkpoint
- 恢复 LoRA 权重 + Optimizer/Scheduler 状态 + Step counter
- 以 **epoch 为粒度**恢复：如果在 epoch 中间中断，该 epoch 会从头重训（但 optimizer 状态已恢复，训练质量不受影响）
- DataLoader 使用固定 seed，保证相同 epoch 的数据顺序可复现
- 直接重新运行脚本即可

### 输出文件结构
```
output/wan_lora/
├── epoch-0.safetensors          # LoRA 权重
├── training_state_epoch-0.pt    # 训练状态
├── step-250.safetensors         # 步间 checkpoint
├── training_state_step-250.pt   # 对应训练状态
└── wan_train_logs/              # TensorBoard 日志
```

## 推理预检与验证

```bash
# 预检（建议先跑 1-5 条）
vtb eval infer \
    --modality video \
    --dataset ./dataset/train_video.csv \
    --dataset-root /path/to/VideoThinkBench/dataset \
    --model-path /path/to/Wan2.2-TI2V-5B \
    --lora ./output/wan_lora/epoch-2.safetensors \
    --mode precheck \
    --num-samples 5 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/precheck_video

# 验证（批量样本）
vtb eval infer \
    --modality video \
    --dataset ./dataset/train_video.csv \
    --dataset-root /path/to/VideoThinkBench/dataset \
    --model-path /path/to/Wan2.2-TI2V-5B \
    --lora ./output/wan_lora/epoch-2.safetensors \
    --mode validate \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/validate_video
```

## 离线规则评测（Maze／Eyeballing）

```bash
vtb eval offline \
    --manifest /abs/path/canonical_manifest.jsonl \
    --task-group maze \
    --pred-root ./outputs/validate_video/samples \
    --output-dir ./outputs/offline_maze

vtb eval offline \
    --manifest /abs/path/canonical_manifest.jsonl \
    --task-group eyeballing \
    --pred-root ./outputs/validate_video/samples \
    --output-dir ./outputs/offline_eyeballing
```
