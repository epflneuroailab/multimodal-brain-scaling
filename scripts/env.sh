#!/usr/bin/env bash
# Shared uv environment handling for example scripts.
#
# The examples intentionally install all workflow extras together once, then
# run commands with `uv run --no-sync`. This avoids uv pruning packages from an
# earlier workflow when a later command mentions only one extra.

set -euo pipefail

MBS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MBS_REPO_ROOT="$(cd "${MBS_SCRIPT_DIR}/.." && pwd)"
MBS_EXTRAS="${MBS_EXTRAS:-training evaluation analysis}"
MBS_ENV_DIR="${UV_PROJECT_ENVIRONMENT:-${MBS_REPO_ROOT}/.venv}"
MBS_ENV_READY="${MBS_ENV_READY:-${MBS_ENV_DIR}/.mbs_env_ready}"

cd "${MBS_REPO_ROOT}"

mbs_env_is_ready() {
    [[ -x "${MBS_ENV_DIR}/bin/python" && -f "${MBS_ENV_READY}" ]] || return 1

    local installed
    installed="$(<"${MBS_ENV_READY}")"
    [[ "${installed}" == "all-extras" ]] && return 0
    [[ "${installed}" == "${MBS_EXTRAS}" ]]
}

mbs_sync_env() {
    echo "Preparing uv environment with extras: ${MBS_EXTRAS}" >&2

    if [[ "${MBS_EXTRAS}" == "all" || "${MBS_EXTRAS}" == "all-extras" ]]; then
        uv sync --all-extras
        mkdir -p "$(dirname "${MBS_ENV_READY}")"
        printf '%s\n' "all-extras" > "${MBS_ENV_READY}"
        return
    fi

    local extras=()
    read -r -a extras <<< "${MBS_EXTRAS}"
    local sync_args=(sync)
    for extra in "${extras[@]}"; do
        sync_args+=(--extra "${extra}")
    done
    uv "${sync_args[@]}"
    mkdir -p "$(dirname "${MBS_ENV_READY}")"
    printf '%s\n' "${MBS_EXTRAS}" > "${MBS_ENV_READY}"
}

mbs_prepare_env() {
    if [[ "${MBS_FORCE_SYNC:-false}" == "true" ]] || ! mbs_env_is_ready; then
        mbs_sync_env
    fi
}

mbs_run() {
    mbs_prepare_env
    uv run --no-sync "$@"
}
