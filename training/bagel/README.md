# BAGEL Training

使用上游 `BAGEL` 仓库训练统一多模态模型，并通过 `VideoThinkBench` 的导出器把 `CanonicalSample` 转成可直接消费的 `edit + VLM` 数据。

以下命令默认在 `VideoThinkBench` 仓库根目录执行。

## 目录与前置条件

- `BAGEL_PATH`：本地克隆的 `BAGEL` 仓库目录，例如 `/Users/zillionx/Desktop/video-reason/Bagel`
- `BAGEL_MODEL_PATH`：`BAGEL-7B-MoT` 权重目录
- 训练数据优先通过 `python3 cli.py data export --target bagel` 生成

## 数据导出

推荐直接从 `canonical_manifest.jsonl` 导出：

```bash
python3 cli.py data export \
    --manifest /path/to/canonical_manifest.jsonl \
    --target bagel \
    --output-dir ./dataset/bagel \
    --task-groups eyeballing maze visual_puzzle \
    --mode edit,vlm
```

如果手头只有 `cli.py data generate --output-root ...` 的数据目录，也可以使用兼容包装脚本：

```bash
python3 training/bagel/prepare_data.py \
    --dataset_root /path/to/output_root \
    --output_dir ./dataset/bagel
```

导出目录会生成这些关键文件：

- `dataset_info.json`：供启动包装器动态注入到上游 `BAGEL` 数据注册表
- `config_unified.yaml`：联合训练配置，包含 `unified_edit + vlm_sft`
- `config_edit.yaml`：仅编辑训练
- `config_vlm.yaml`：仅图文问答训练
- `editing/parquet/*.parquet`：`Text+Image -> Image` 训练数据
- `vlm/train_sft.jsonl`：`Image -> Text` 监督微调数据

## 训练

### 联合训练

```bash
export BAGEL_PATH=/path/to/Bagel
export BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT

bash training/bagel/train_sft.sh \
    --dataset_dir ./dataset/bagel \
    --mode unified \
    --output_dir ./output/bagel_unified
```

### 仅编辑

```bash
bash training/bagel/train_sft.sh \
    --dataset_dir ./dataset/bagel \
    --mode edit \
    --output_dir ./output/bagel_edit
```

### 仅 VLM

```bash
bash training/bagel/train_sft.sh \
    --dataset_dir ./dataset/bagel \
    --mode vlm \
    --output_dir ./output/bagel_vlm
```

### 仅做参数检查

```bash
bash training/bagel/train_sft.sh \
    --dataset_dir ./dataset/bagel \
    --mode unified \
    --dry_run
```

## 自动续训

`training/bagel/train_sft.sh` 默认把上游 `BAGEL` 的恢复机制打开：

- `--auto_resume True`
- `--resume_model_only True`
- `--finetune_from_ema True`

第一次启动时，如果 `checkpoint_dir` 为空，会从 `BAGEL_MODEL_PATH` 加载基础权重。

后续同一路径重启时，如果 `output_dir/checkpoints/` 下已有上游格式的 step 目录，包装器会优先恢复最新 checkpoint，包括：

- model
- EMA model
- optimizer
- scheduler
- data status

这意味着启智低优或不稳定分布式任务被打断后，直接重新执行同一条命令即可继续训练。

## 常用参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--mode` | `unified` | `unified`、`edit`、`vlm` |
| `--num_nodes` | `1` | 节点数 |
| `--gpus_per_node` | `8` | 每节点 GPU 数 |
| `--num_workers` | `1` | DataLoader worker 数。样例数据量较小时建议保持 `1` |
| `--save_every` | `2000` | checkpoint 间隔 |
| `--total_steps` | `500000` | 总训练 step |
| `--learning_rate` | `2e-5` | 学习率 |
| `--wandb_offline` | `True` | 默认关闭在线同步，适配离线环境 |

## 注意事项

- `BAGEL` 上游训练脚本强依赖 `torchrun + FSDP`，推荐在启智分布式训练空间运行。
- `mode=edit` 仍然需要 `visual_und=True`，因为编辑分支会读取输入图的 `ViT` 特征。
- `mode=vlm` 会自动关闭 `visual_gen`，只保留图像理解和文本生成路径。
- 如果导出目录里的 `dataset_info.json` 或 `config_*.yaml` 缺失，脚本会报错而不是静默回退。
