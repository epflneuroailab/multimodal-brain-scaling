#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBJECT=1
DATA_SPACE="nativesurface" # func1pt8mm nativesurface
ROI_MAPPING="combined" # combined  individual
DS_DIR="${MBS_NSD_DIR:?Set MBS_NSD_DIR to the root NSD dataset directory.}"
OUTPUT_DIR="${MBS_DATA_PREP_OUTPUT_DIR:-${REPO_ROOT}/data_prep_outputs}"
STIM_CSV="${MBS_NSD_STIM_CSV:-${OUTPUT_DIR}/nsd_stim_mapping.csv}"

mkdir -p "${OUTPUT_DIR}"

python "${SCRIPT_DIR}/preprocess_nsd.py" \
  --subject "${SUBJECT}" \
  --data-space "${DATA_SPACE}" \
  --roi-mapping "${ROI_MAPPING}" \
  --ds-dir "${DS_DIR}" \
  --stim-csv-path "${STIM_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --keep-reps
# Add --debug-mode above for a smaller dry-run output.
