#!/bin/bash
# ============================================================
# BAGEL Unified Multimodal Fine-Tuning Script
# Modes:
#   - unified: edit + vlm
#   - edit:    text+image -> image
#   - vlm:     image -> text
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_DIR="${SCRIPT_DIR}/dataset_bagel"
DATASET_BASE_PATH=""
OUTPUT_PATH="${SCRIPT_DIR}/output/bagel"
MODE="unified"

NUM_NODES=1
GPUS_PER_NODE=8
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_WORKERS=1
SAVE_EVERY=2000
TOTAL_STEPS=500000
LOG_EVERY=10
LEARNING_RATE=2e-5
EXPECTED_NUM_TOKENS=10240
MAX_NUM_TOKENS=11520
MAX_NUM_TOKENS_PER_SAMPLE=10240
GLOBAL_SEED=4396

AUTO_RESUME="True"
RESUME_FROM=""
RESUME_MODEL_ONLY="True"
FINETUNE_FROM_EMA="True"
FINETUNE_FROM_HF="True"
WANDB_OFFLINE="True"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Usage:
  BAGEL_PATH=/path/to/Bagel \
  BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT \
  bash training/bagel/train_sft.sh --dataset_dir /path/to/bagel_export [options]

Options:
  --dataset_dir PATH
  --dataset_root PATH
  --output_dir PATH
  --mode unified|edit|vlm
  --num_nodes N
  --gpus_per_node N
  --node_rank N
  --master_addr HOST
  --master_port PORT
  --num_workers N
  --save_every N
  --total_steps N
  --log_every N
  --learning_rate FLOAT
  --expected_num_tokens N
  --max_num_tokens N
  --max_num_tokens_per_sample N
  --global_seed N
  --resume_from PATH
  --auto_resume True|False
  --wandb_offline True|False
  --dry_run
  --help
EOF
}

render_command() {
    local rendered=()
    for arg in "$@"; do
        rendered+=("$(printf '%q' "${arg}")")
    done
    printf '%s\n' "${rendered[*]}"
}

run_or_echo() {
    echo "[BAGEL] launch"
    render_command "$@"
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir)
            DATASET_DIR="$2"; shift 2;;
        --dataset_root)
            DATASET_BASE_PATH="$2"; shift 2;;
        --output_dir)
            OUTPUT_PATH="$2"; shift 2;;
        --mode)
            MODE="$2"; shift 2;;
        --num_nodes)
            NUM_NODES="$2"; shift 2;;
        --gpus_per_node)
            GPUS_PER_NODE="$2"; shift 2;;
        --node_rank)
            NODE_RANK="$2"; shift 2;;
        --master_addr)
            MASTER_ADDR="$2"; shift 2;;
        --master_port)
            MASTER_PORT="$2"; shift 2;;
        --num_workers)
            NUM_WORKERS="$2"; shift 2;;
        --save_every)
            SAVE_EVERY="$2"; shift 2;;
        --total_steps)
            TOTAL_STEPS="$2"; shift 2;;
        --log_every)
            LOG_EVERY="$2"; shift 2;;
        --learning_rate)
            LEARNING_RATE="$2"; shift 2;;
        --expected_num_tokens)
            EXPECTED_NUM_TOKENS="$2"; shift 2;;
        --max_num_tokens)
            MAX_NUM_TOKENS="$2"; shift 2;;
        --max_num_tokens_per_sample)
            MAX_NUM_TOKENS_PER_SAMPLE="$2"; shift 2;;
        --global_seed)
            GLOBAL_SEED="$2"; shift 2;;
        --resume_from)
            RESUME_FROM="$2"; shift 2;;
        --auto_resume)
            AUTO_RESUME="$2"; shift 2;;
        --wandb_offline)
            WANDB_OFFLINE="$2"; shift 2;;
        --dry_run)
            DRY_RUN="true"; shift;;
        --help|-h)
            usage; exit 0;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1;;
    esac
done

if [[ -z "${BAGEL_PATH:-}" ]]; then
    echo "Error: BAGEL_PATH not set" >&2
    exit 1
fi
if [[ -z "${BAGEL_MODEL_PATH:-}" ]]; then
    echo "Error: BAGEL_MODEL_PATH not set" >&2
    exit 1
fi
if [[ "${MODE}" != "unified" && "${MODE}" != "edit" && "${MODE}" != "vlm" ]]; then
    echo "Error: --mode must be one of unified, edit, vlm" >&2
    exit 1
fi

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
    if [[ -n "${DATASET_BASE_PATH}" && -d "${DATASET_BASE_PATH}" ]]; then
        echo "Bagel dataset export not found. Generating with VideoThinkBench compatibility wrapper..."
        python3 "${SCRIPT_DIR}/prepare_data.py" \
            --dataset_root "${DATASET_BASE_PATH}" \
            --output_dir "${DATASET_DIR}"
    else
        echo "Error: Bagel export not found under ${DATASET_DIR}" >&2
        exit 1
    fi
fi

CONFIG_PATH="${DATASET_DIR}/config_${MODE}.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

CHECKPOINT_DIR="${OUTPUT_PATH}/checkpoints"
RESULTS_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

if [[ -z "${RESUME_FROM}" ]]; then
    RESUME_FROM="${BAGEL_MODEL_PATH}"
fi

VISUAL_GEN="True"
VISUAL_UND="True"
if [[ "${MODE}" == "vlm" ]]; then
    VISUAL_GEN="False"
fi

CMD=(
    torchrun
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --nproc_per_node "${GPUS_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    "${SCRIPT_DIR}/launch.py"
    --bagel-path "${BAGEL_PATH}"
    --dataset-info-json "${DATASET_DIR}/dataset_info.json"
    --dataset_config_file "${CONFIG_PATH}"
    --model_path "${BAGEL_MODEL_PATH}"
    --layer_module Qwen2MoTDecoderLayer
    --max_latent_size 64
    --resume_from "${RESUME_FROM}"
    --finetune_from_hf "${FINETUNE_FROM_HF}"
    --auto_resume "${AUTO_RESUME}"
    --resume_model_only "${RESUME_MODEL_ONLY}"
    --finetune_from_ema "${FINETUNE_FROM_EMA}"
    --results_dir "${RESULTS_DIR}"
    --checkpoint_dir "${CHECKPOINT_DIR}"
    --log_every "${LOG_EVERY}"
    --save_every "${SAVE_EVERY}"
    --lr "${LEARNING_RATE}"
    --num_workers "${NUM_WORKERS}"
    --expected_num_tokens "${EXPECTED_NUM_TOKENS}"
    --max_num_tokens "${MAX_NUM_TOKENS}"
    --max_num_tokens_per_sample "${MAX_NUM_TOKENS_PER_SAMPLE}"
    --global_seed "${GLOBAL_SEED}"
    --total_steps "${TOTAL_STEPS}"
    --visual_gen "${VISUAL_GEN}"
    --visual_und "${VISUAL_UND}"
    --wandb_offline "${WANDB_OFFLINE}"
)

run_or_echo "${CMD[@]}"
