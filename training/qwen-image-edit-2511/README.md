# Qwen-Image Training (Image Editing Model)

使用 DiffSynth-Studio 框架对 Qwen-Image-Edit-2511 模型进行 LoRA SFT 训练。

以下命令默认在仓库根目录执行。

## 环境要求

```bash
# 安装 DiffSynth-Studio
cd /path/to/DiffSynth-Studio
pip install -e .

pip install accelerate deepspeed
```

## 数据准备

```bash
# 推荐：从 canonical_manifest.jsonl 直接导出
python3 cli.py data export \
    --manifest /path/to/canonical_manifest.jsonl \
    --target diffsynth-image \
    --output ./data/metadata.json

# 兼容：从 cli.py data generate 的 output_root 重新扫描导出
python3 training/qwen-image-edit-2511/prepare_data.py \
    --dataset_root /path/to/VideoThinkBench/output_root \
    --output_path ./data/metadata.json

# 输出: data/metadata.json
```

## 训练

```bash
# 设置 DiffSynth-Studio 路径
export DIFFSYNTH_PATH=/path/to/DiffSynth-Studio

# 启动训练
bash training/qwen-image-edit-2511/train_sft.sh --metadata_path ./data/metadata.json

# 可选参数:
#   --output_dir ./outputs/train
#   --learning_rate 1e-4
#   --num_epochs 5
#   --lora_rank 32
#   --num_nodes 2
#   --gpus_per_node 8
#   --machine_rank 0
#   --main_process_ip 10.0.0.1
#   --main_process_port 29500
#   --accelerate_config /path/to/accelerate_config.yaml
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-4 | 学习率 |
| num_epochs | 5 | 训练轮数 |
| lora_rank | 32 | LoRA Rank |
| max_pixels | 1048576 | 最大像素数 (1024×1024) |

## 推理预检与验证

```bash
# 预检（少量样本）
python3 cli.py eval infer \
    --modality image \
    --dataset ./data/metadata.json \
    --model-path /path/to/model_base \
    --lora ./outputs/train/Qwen-Image-Edit-2511_lora/epoch-4.safetensors \
    --mode precheck \
    --num-samples 5 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/precheck

# 验证（批量样本）
python3 cli.py eval infer \
    --modality image \
    --dataset ./data/metadata.json \
    --model-path /path/to/model_base \
    --lora ./outputs/train/Qwen-Image-Edit-2511_lora/epoch-4.safetensors \
    --mode validate \
    --num-samples 10 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/validate
```

如需同时输出离线规则评测结果，可使用以下命令。

```bash
python3 cli.py eval run \
    --modality image \
    --dataset ./data/metadata.json \
    --model-path /path/to/model_base \
    --lora ./outputs/train/Qwen-Image-Edit-2511_lora/epoch-4.safetensors \
    --mode validate \
    --num-samples 10 \
    --diffsynth-path "${DIFFSYNTH_PATH}" \
    --output-dir ./outputs/validate \
    --with-offline \
    --manifest /abs/path/canonical_manifest.jsonl
```

## 输出格式

训练数据格式（metadata.json）:
```json
[
  {
    "id": "maze_square-00001",
    "prompt": "Draw a red path connecting two red dots...",
    "image": "/abs/path/to/solution.png",
    "edit_image": "/abs/path/to/puzzle.png",
    "task_type": "maze_square",
    "task_group": "maze"
  }
]
```
