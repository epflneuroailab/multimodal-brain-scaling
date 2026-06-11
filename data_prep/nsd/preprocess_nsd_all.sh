#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBJECTS=(1 2 3 4 5 6 7 8)
# SUBJECTS=(2)
# SUBJECTS=(6 8)
DATA_SPACES=("func1pt8mm" "nativesurface" "fsaverage")
# DATA_SPACES=("nativesurface")
# DATA_SPACES=("fsaverage")
ROI_MAPPING=("combined" "individual")
ROI_MAPPING=("combined")
DS_DIR="${MBS_NSD_DIR:?Set MBS_NSD_DIR to the root NSD dataset directory.}"
OUTPUT_DIR="${MBS_DATA_PREP_OUTPUT_DIR:-${REPO_ROOT}/data_prep_outputs}"
STIM_CSV="${MBS_NSD_STIM_CSV:-${OUTPUT_DIR}/nsd_stim_mapping.csv}"

# NC_THRESHOLD=10

mkdir -p "${OUTPUT_DIR}"

for roi_mapping in "${ROI_MAPPING[@]}"; do
    for data_space in "${DATA_SPACES[@]}"; do
        for subject in "${SUBJECTS[@]}"; do

            python "${SCRIPT_DIR}/preprocess_nsd.py" \
              --subject "${subject}" \
              --data-space "${data_space}" \
              --roi-mapping "${roi_mapping}" \
              --ds-dir "${DS_DIR}" \
              --stim-csv-path "${STIM_CSV}" \
              --output-dir "${OUTPUT_DIR}" \
              --apply-nsdgeneral-mask
            #   --nc-threshold $NC_THRESHOLD \
              # --debug-mode

        done

    done

done
