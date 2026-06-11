#!/usr/bin/env bash
# End-to-end fine-tuning example.
#
# Pass options on the command line to point at your own data and output
# locations. Defaults assume the standard `./data/` and `./outputs/` layout in
# a cloned checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_PATH_IMAGE="./data/imagenet/"
DATA_PATH_NEURAL="./data/neural"
DATA_NEURAL_FILENAME="SachiMajajHong2015.h5"
DATA_NEURAL_REGIONS="V4,IT"
LAYER_COMMITMENTS="configs/evaluation/layer_commitment/layer_commitments.json"
LAYER_COMMITMENT_DATASET="bs_mh"
CONFIG_ENCODER="configs/training/encoders/finetune/deit/deit_small.yaml"
OUTPUT_DIR="./outputs/train_example"
PRETRAINED_MODEL_ID="deit_small_imagenet_full_seed-0"
RUN_NAME="deit_small_finetune_example"
LINEAR_PROBES_DIR=""
FROZEN_DECODERS="true"

usage() {
    cat <<'EOF'
Usage: scripts/train_example.sh [OPTIONS]

Run the fine-tuning example.

Options:
  --data-path-image PATH            Image training data root.
  --data-path-neural PATH           Neural data directory.
  --data-neural-filename FILE       Neural HDF5 filename.
  --data-neural-regions REGIONS     Comma-separated neural regions.
  --layer-commitments PATH          Layer commitment JSON path.
  --layer-commitment-dataset NAME   Dataset key in the commitment file.
  --config-encoder PATH             Encoder config YAML path.
  --output-dir PATH                 Directory for training checkpoints.
  --pretrained-model-id ID          Pretrained SPVVS model id.
  --run-name NAME                   Run name for logging/checkpoints.
  --linear-probes-dir PATH          Optional frozen probe directory.
  --frozen-decoders BOOL            Whether loaded decoders stay frozen.
  -h, --help                        Show this help message.
EOF
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "Missing value for ${flag}" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-path-image) require_value "$1" "${2:-}"; DATA_PATH_IMAGE="$2"; shift 2 ;;
        --data-path-neural) require_value "$1" "${2:-}"; DATA_PATH_NEURAL="$2"; shift 2 ;;
        --data-neural-filename) require_value "$1" "${2:-}"; DATA_NEURAL_FILENAME="$2"; shift 2 ;;
        --data-neural-regions) require_value "$1" "${2:-}"; DATA_NEURAL_REGIONS="$2"; shift 2 ;;
        --layer-commitments) require_value "$1" "${2:-}"; LAYER_COMMITMENTS="$2"; shift 2 ;;
        --layer-commitment-dataset) require_value "$1" "${2:-}"; LAYER_COMMITMENT_DATASET="$2"; shift 2 ;;
        --config-encoder) require_value "$1" "${2:-}"; CONFIG_ENCODER="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --pretrained-model-id) require_value "$1" "${2:-}"; PRETRAINED_MODEL_ID="$2"; shift 2 ;;
        --run-name) require_value "$1" "${2:-}"; RUN_NAME="$2"; shift 2 ;;
        --linear-probes-dir) require_value "$1" "${2:-}"; LINEAR_PROBES_DIR="$2"; shift 2 ;;
        --frozen-decoders) require_value "$1" "${2:-}"; FROZEN_DECODERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

MBS_EXTRAS="training evaluation analysis"
MBS_FORCE_SYNC="false"
source "${SCRIPT_DIR}/env.sh"

linear_probe_args=()
if [[ -n "${LINEAR_PROBES_DIR}" ]]; then
    linear_probe_args+=(--linear-probes-dir "${LINEAR_PROBES_DIR}")
fi

mbs_run mbs-train \
    --config-encoder "$CONFIG_ENCODER" \
    --save-dir "$OUTPUT_DIR" \
    --data-path-image "$DATA_PATH_IMAGE" \
    --data-path-neural "$DATA_PATH_NEURAL" \
    --data-neural-filename "$DATA_NEURAL_FILENAME" \
    --data-neural-regions "$DATA_NEURAL_REGIONS" \
    --layer-commitments "$LAYER_COMMITMENTS" \
    --layer-commitment-dataset "$LAYER_COMMITMENT_DATASET" \
    --pretrained-model-id "$PRETRAINED_MODEL_ID" \
    --run-name "$RUN_NAME" \
    "${linear_probe_args[@]}" \
    --frozen-decoders "$FROZEN_DECODERS" \
    --disable-wandb \
    --max-epochs 100 \
    --opt sgd \
    --lr-encoder 1e-3 \
    --lr-decoder 1e-2 \
    --wd-encoder 5e-2 \
    --wd-decoder 1e-0
