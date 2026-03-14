#!/bin/bash
# ============================================================
# Wan2.2-I2V-A14B LoRA Training Script
# high_noise / low_noise are trained separately.
# ============================================================

set -euo pipefail

export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_PATH="${SCRIPT_DIR}/dataset.csv"
DATASET_BASE_PATH=""
OUTPUT_PATH="${SCRIPT_DIR}/output/wan_lora"

NUM_NODES=1
GPUS_PER_NODE=8
MACHINE_RANK=0

NUM_FRAMES=81
HEIGHT=896
WIDTH=480

LEARNING_RATE=1e-4
NUM_EPOCHS=3
LORA_RANK=32
GRADIENT_ACCUMULATION=1
DATASET_REPEAT=1
SAVE_STEPS=250
LORA_CHECKPOINT=""
BRANCH="all"
DRY_RUN="false"

usage() {
    cat <<'EOF'
Usage:
  MODEL_BASE_PATH=/path/to/Wan2.2-I2V-A14B \
  DIFFSYNTH_PATH=/path/to/DiffSynth-Studio \
  bash training/wan2.2-i2v-a14b/train_sft.sh --dataset /path/to/dataset.csv [options]

Options:
  --dataset PATH
  --dataset_root PATH
  --output_dir PATH
  --num_nodes N
  --gpus_per_node N
  --machine_rank N
  --num_frames N
  --height N
  --width N
  --lora_rank N
  --num_epochs N
  --learning_rate FLOAT
  --save_steps N
  --lora_checkpoint PATH
  --branch all|high_noise|low_noise
  --dry_run
  --help
EOF
}

resolve_checkpoint_file() {
    local candidate="$1"
    if [[ -z "${candidate}" ]]; then
        return 0
    fi
    if [[ -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    if [[ ! -d "${candidate}" ]]; then
        return 0
    fi
    python3 - "${candidate}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"^(epoch|step)-(\d+)\.safetensors$")
candidates = []
for path in root.glob("*.safetensors"):
    match = pattern.match(path.name)
    if match is None:
        continue
    kind = match.group(1)
    index = int(match.group(2))
    candidates.append((index, 1 if kind == "step" else 0, path.stat().st_mtime, path))
if not candidates:
    raise SystemExit(0)
candidates.sort()
print(candidates[-1][3].as_posix())
PY
}

render_command() {
    local rendered=()
    for arg in "$@"; do
        rendered+=("$(printf '%q' "${arg}")")
    done
    printf '%s\n' "${rendered[*]}"
}

run_or_echo() {
    local label="$1"
    shift
    echo "[Wan2.2-I2V-A14B] ${label}"
    render_command "$@"
    if [[ "${DRY_RUN}" == "true" ]]; then
        return 0
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_PATH="$2"; shift 2;;
        --dataset_root)
            DATASET_BASE_PATH="$2"; shift 2;;
        --output_dir)
            OUTPUT_PATH="$2"; shift 2;;
        --num_nodes)
            NUM_NODES="$2"; shift 2;;
        --gpus_per_node)
            GPUS_PER_NODE="$2"; shift 2;;
        --machine_rank)
            MACHINE_RANK="$2"; shift 2;;
        --num_frames)
            NUM_FRAMES="$2"; shift 2;;
        --height)
            HEIGHT="$2"; shift 2;;
        --width)
            WIDTH="$2"; shift 2;;
        --lora_rank)
            LORA_RANK="$2"; shift 2;;
        --num_epochs)
            NUM_EPOCHS="$2"; shift 2;;
        --learning_rate)
            LEARNING_RATE="$2"; shift 2;;
        --save_steps)
            SAVE_STEPS="$2"; shift 2;;
        --lora_checkpoint)
            LORA_CHECKPOINT="$2"; shift 2;;
        --branch)
            BRANCH="$2"; shift 2;;
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

if [[ -z "${MODEL_BASE_PATH:-}" ]]; then
    echo "Error: MODEL_BASE_PATH not set" >&2
    exit 1
fi
if [[ -z "${DIFFSYNTH_PATH:-}" ]]; then
    echo "Error: DIFFSYNTH_PATH not set" >&2
    exit 1
fi
if [[ "${BRANCH}" != "all" && "${BRANCH}" != "high_noise" && "${BRANCH}" != "low_noise" ]]; then
    echo "Error: --branch must be one of all, high_noise, low_noise" >&2
    exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    if [[ -n "${DATASET_BASE_PATH}" && -d "${DATASET_BASE_PATH}" ]]; then
        python3 "${SCRIPT_DIR}/prepare_data.py" \
            --dataset_root "${DATASET_BASE_PATH}" \
            --output_path "${DATASET_PATH}"
    else
        echo "Error: Dataset CSV not found: ${DATASET_PATH}" >&2
        exit 1
    fi
fi

if [[ $(( (NUM_FRAMES - 1) % 4 )) -ne 0 ]]; then
    echo "Error: NUM_FRAMES must satisfy (n-1) % 4 == 0" >&2
    exit 1
fi
if [[ $((HEIGHT % 32)) -ne 0 ]] || [[ $((WIDTH % 32)) -ne 0 ]]; then
    echo "Error: HEIGHT and WIDTH must be divisible by 32" >&2
    exit 1
fi

if [[ "${NUM_NODES}" -le 1 ]]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
fi

TMP_CONFIG=""
trap 'if [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]]; then rm -f "${TMP_CONFIG}"; fi' EXIT

build_accelerate_prefix() {
    ACCELERATE_PREFIX=(accelerate launch)
    if [[ "${NUM_NODES}" -gt 1 ]]; then
        if [[ -n "${TMP_CONFIG}" && -f "${TMP_CONFIG}" ]]; then
            rm -f "${TMP_CONFIG}"
        fi
        local num_processes=$((NUM_NODES * GPUS_PER_NODE))
        TMP_CONFIG="$(mktemp /tmp/accelerate_config_XXXX.yaml)"
        cat > "${TMP_CONFIG}" <<CONFIG
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: ${GRADIENT_ACCUMULATION}
  offload_optimizer_device: 'none'
  offload_param_device: 'none'
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: ${MACHINE_RANK}
main_training_function: main
mixed_precision: bf16
num_machines: ${NUM_NODES}
num_processes: ${num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
CONFIG
        ACCELERATE_PREFIX+=(--config_file "${TMP_CONFIG}")
    fi
}

run_branch() {
    local branch_name="$1"
    local model_id_with_origin_paths="$2"
    local min_boundary="$3"
    local max_boundary="$4"

    build_accelerate_prefix
    local output_dir="${OUTPUT_PATH}/${branch_name}"
    mkdir -p "${output_dir}"

    local checkpoint_source=""
    if [[ -n "${LORA_CHECKPOINT}" ]]; then
        if [[ -d "${LORA_CHECKPOINT}" && -d "${LORA_CHECKPOINT}/${branch_name}" ]]; then
            checkpoint_source="${LORA_CHECKPOINT}/${branch_name}"
        elif [[ "${BRANCH}" != "all" ]]; then
            checkpoint_source="${LORA_CHECKPOINT}"
        fi
    fi
    local resolved_checkpoint
    resolved_checkpoint="$(resolve_checkpoint_file "${checkpoint_source}")"

    CMD=(
        "${ACCELERATE_PREFIX[@]}"
        "${SCRIPT_DIR}/train.py"
        --dataset_base_path "${DATASET_BASE_PATH}"
        --dataset_metadata_path "${DATASET_PATH}"
        --height "${HEIGHT}"
        --width "${WIDTH}"
        --num_frames "${NUM_FRAMES}"
        --dataset_repeat "${DATASET_REPEAT}"
        --model_id_with_origin_paths "${model_id_with_origin_paths}"
        --learning_rate "${LEARNING_RATE}"
        --num_epochs "${NUM_EPOCHS}"
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION}"
        --remove_prefix_in_ckpt "pipe.dit."
        --output_path "${output_dir}"
        --lora_base_model "dit"
        --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
        --lora_rank "${LORA_RANK}"
        --extra_inputs "input_image"
        --use_gradient_checkpointing
        --save_steps "${SAVE_STEPS}"
        --min_timestep_boundary "${min_boundary}"
        --max_timestep_boundary "${max_boundary}"
    )
    if [[ -n "${resolved_checkpoint}" ]]; then
        CMD+=(--lora_checkpoint "${resolved_checkpoint}")
    fi
    run_or_echo "${branch_name}" "${CMD[@]}"
}

HIGH_MODEL_ID_WITH_ORIGIN_PATHS="Wan-AI/Wan2.2-I2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"
LOW_MODEL_ID_WITH_ORIGIN_PATHS="Wan-AI/Wan2.2-I2V-A14B:low_noise_model/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-I2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-I2V-A14B:Wan2.1_VAE.pth"

cd "${DIFFSYNTH_PATH}"
if [[ "${BRANCH}" == "all" || "${BRANCH}" == "high_noise" ]]; then
    run_branch "high_noise" "${HIGH_MODEL_ID_WITH_ORIGIN_PATHS}" "0.0" "0.358"
fi
if [[ "${BRANCH}" == "all" || "${BRANCH}" == "low_noise" ]]; then
    run_branch "low_noise" "${LOW_MODEL_ID_WITH_ORIGIN_PATHS}" "0.358" "1.0"
fi
