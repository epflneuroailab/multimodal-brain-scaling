#!/usr/bin/env bash
# End-to-end feature extraction example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="deit_small_imagenet_full_seed-0"
DATA_ROOT="./data/stimuli/object_images"
DATASET_TYPE="h5"
STIMULUS_SET_ID="object_images"
OUTPUT_DIR=""
COMMITTED_EXTRACTION_LAYERS="configs/evaluation/layer_commitment/committed_extraction_layers.json"

usage() {
    cat <<'EOF'
Usage: scripts/extract_example.sh [OPTIONS]

Run the feature extraction example.

Options:
  --model-id ID                         Model identifier.
  --data-root PATH                      Dataset directory, HDF5 path, or stimulus set id.
  --dataset-type TYPE                   Dataset type: h5, things, or brain_score.
  --stimulus-set-id ID                  Stimulus-set key for committed extraction layers.
  --output-dir PATH                     Directory for extracted features.
  --committed-extraction-layers PATH    Committed extraction layers JSON path.
  -h, --help                            Show this help message.
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
        --data-root) require_value "$1" "${2:-}"; DATA_ROOT="$2"; shift 2 ;;
        --dataset-type) require_value "$1" "${2:-}"; DATASET_TYPE="$2"; shift 2 ;;
        --stimulus-set-id) require_value "$1" "${2:-}"; STIMULUS_SET_ID="$2"; shift 2 ;;
        --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR="$2"; shift 2 ;;
        --committed-extraction-layers) require_value "$1" "${2:-}"; COMMITTED_EXTRACTION_LAYERS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-./outputs/features/${MODEL_ID}}"
MBS_EXTRAS="training evaluation analysis"
MBS_FORCE_SYNC="false"
source "${SCRIPT_DIR}/env.sh"

mbs_run mbs-extract-features \
    --model_id "$MODEL_ID" \
    --backbone_source spvvs \
    --data_root "$DATA_ROOT" \
    --dataset_type "$DATASET_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --committed_extraction_layers "$COMMITTED_EXTRACTION_LAYERS" \
    --stimulus_set_id "$STIMULUS_SET_ID" \
    --batch_size 32 \
    --num_workers 4
