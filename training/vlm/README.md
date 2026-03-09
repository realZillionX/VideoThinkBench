# Qwen3-VL Training (VLM)

使用 ms-swift 框架对 Qwen3-VL 模型进行 SFT 和 GRPO 训练。

以下命令默认在仓库根目录执行。

## 环境要求

```bash
pip install ms-swift peft vllm datasets
```

## 数据准备

```bash
# 推荐：从 canonical_manifest.jsonl 直接导出
python3 cli.py data export \
    --manifest /path/to/canonical_manifest.jsonl \
    --target ms-swift \
    --output-dir ./data \
    --mode sft,grpo

# 兼容：从 cli.py data generate 的 output_root 重新扫描导出
python3 scripts/prepare_vlm_data.py \
    --data_root /path/to/VideoThinkBench/output_root \
    --output_dir ./data

# 输出:
#   data/train_sft.jsonl   - SFT 训练数据（默认使用绝对图片路径）
#   data/train_grpo.jsonl  - GRPO 训练数据
```

## Phase 1: SFT 监督微调

```bash
bash training/vlm/train_sft.sh --model_path /path/to/Qwen3-VL-32B-Thinking

# 可选参数:
#   --dataset data/train_sft.jsonl
#   --output_dir output/sft_qwen3_vl
#   --num_gpus 8
```

## Phase 2: GRPO 强化学习

在 SFT 完成后，加载 checkpoint 继续 GRPO 训练：

```bash
bash training/vlm/train_grpo.sh \
    --model_path output/sft_qwen3_vl/checkpoint-100 \
    --data_path data/train_grpo.jsonl \
    --output_dir output/grpo_qwen3_vl

# 可选参数:
#   --learning_rate 1e-6
#   --num_generations 8
#   --lora_rank 16
```

## 奖励函数

GRPO 使用自定义奖励函数（`training/vlm/rewards/vlm_rewards.py`）：

| 任务类型 | 评分规则 |
|---------|---------|
| Eyeballing | 1.0=正确, 0.0=错误, -1.0=格式错误 |
| Maze | 0.0~1.0=部分匹配, -1.0=格式错误 |
| Visual Puzzle | 1.0=文本精确匹配, 0.0=错误, -1.0=空答案 |

当前 `GRPO` 已支持：

- `maze` 的路径列表答案。
- `eyeballing` 的单字母答案。
- `visual_puzzle` 的文本答案，例如颜色、形状或大小词。

## 推理预检与验证

```bash
# 预检（少量样本）
python3 cli.py eval infer \
    --modality vlm \
    --dataset data/train_sft.jsonl \
    --model-path /path/to/Qwen3-VL-32B \
    --mode precheck \
    --num-samples 5 \
    --output-dir output/precheck_vlm

# 验证（批量样本）
python3 cli.py eval infer \
    --modality vlm \
    --dataset data/train_sft.jsonl \
    --model-path /path/to/Qwen3-VL-32B \
    --mode validate \
    --num-samples 50 \
    --output-dir output/validate_vlm
```
