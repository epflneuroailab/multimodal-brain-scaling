#!/usr/bin/env bash
# End-to-end feature evaluation example (committed layers).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="deit_small_imagenet_full_seed-0"
FEATURES_DIR=""
DATA_HDF5_PATH="./data/neural/things_fmri.h5"
OUTPUT_DIR=""
LAYER_COMMITMENTS="configs/evaluation/layer_commitment/layer_commitments.json"

usage() {
    cat <<'EOF'
Usage: scripts/evaluate_example.sh [OPTIONS]

Run the committed-layer evaluation example.

Options:
  --model-id ID                  Model identifier.
  --features-dir PATH            Extracted feature directory.
  --data-hdf5-path PATH          Neural benchmark HDF5 path.
  --output-dir PATH              Directory for evaluation results.
  --layer-commitments PATH       Layer commitment JSON path.
  -h, --help                     Show this help message.
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
        --model-id) require_value "$1" "${2:-}"; MODEL_ID="$2"; shift 2 ;;
        --features-dir) require_value "$1" "${2:-}"; FEATURES_DIR="$2"; shift 2 ;;
        --data-hdf5-path) require_value "$1" "${2:-}"; DATA_HDF5_PATH="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --layer-commitments) require_value "$1" "${2:-}"; LAYER_COMMITMENTS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

FEATURES_DIR="${FEATURES_DIR:-./outputs/features/${MODEL_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/evaluation/${MODEL_ID}}"
MBS_EXTRAS="training evaluation analysis"
MBS_FORCE_SYNC="false"
source "${SCRIPT_DIR}/env.sh"

mbs_run mbs-evaluate-committed-layers \
    --model_id "$MODEL_ID" \
    --features_dir "$FEATURES_DIR" \
    --data_hdf5_path "$DATA_HDF5_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --layer_commitments "$LAYER_COMMITMENTS" \
    --use_gpu true
