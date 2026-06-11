#!/usr/bin/env bash
# End-to-end fine-tuning example.
#
# Override these variables on the command line or in your environment to
# point at your own data and output locations. Defaults assume the standard
# `./data/` and `./outputs/` layout in a cloned checkout.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

DATA_PATH_IMAGE="${DATA_PATH_IMAGE:-./data/imagenet/}"
DATA_PATH_NEURAL="${DATA_PATH_NEURAL:-./data/neural}"
DATA_NEURAL_FILENAME="${DATA_NEURAL_FILENAME:-SachiMajajHong2015.h5}"
DATA_NEURAL_REGIONS="${DATA_NEURAL_REGIONS:-V4,IT}"
LAYER_COMMITMENTS="${LAYER_COMMITMENTS:-configs/evaluation/layer_commitment/layer_commitments.json}"
LAYER_COMMITMENT_DATASET="${LAYER_COMMITMENT_DATASET:-bs_mh}"
CONFIG_ENCODER="${CONFIG_ENCODER:-configs/training/encoders/finetune/deit/deit_small.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/train_example}"
PRETRAINED_MODEL_ID="${PRETRAINED_MODEL_ID:-deit_small_imagenet_full_seed-0}"
RUN_NAME="${RUN_NAME:-deit_small_finetune_example}"
LINEAR_PROBES_DIR="${LINEAR_PROBES_DIR:-}"
FROZEN_DECODERS="${FROZEN_DECODERS:-true}"

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
