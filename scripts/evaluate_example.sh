#!/usr/bin/env bash
# End-to-end feature evaluation example (committed layers).

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODEL_ID="${MODEL_ID:-deit_small_imagenet_full_seed-0}"
FEATURES_DIR="${FEATURES_DIR:-./outputs/features/${MODEL_ID}}"
DATA_HDF5_PATH="${DATA_HDF5_PATH:-./data/neural/things_fmri.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/evaluation/${MODEL_ID}}"
LAYER_COMMITMENTS="${LAYER_COMMITMENTS:-configs/evaluation/layer_commitment/layer_commitments.json}"

mbs_run mbs-evaluate-committed-layers \
    --model_id "$MODEL_ID" \
    --features_dir "$FEATURES_DIR" \
    --data_hdf5_path "$DATA_HDF5_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --layer_commitments "$LAYER_COMMITMENTS" \
    --use_gpu true
