#!/usr/bin/env bash
set -euo pipefail

export QVC_CODE_DIR="${QVC_CODE_DIR:-/opt/qvc}"
export QVC_WORKDIR="${QVC_WORKDIR:-/work/qvc-demo}"
export QVC_DATA_DIR="${QVC_DATA_DIR:-${QVC_WORKDIR}/data}"
export QVC_RESULT_DIR="${QVC_RESULT_DIR:-${QVC_WORKDIR}/results}"
export QVC_PLOTS_DIR="${QVC_PLOTS_DIR:-${QVC_WORKDIR}/plots}"
export QVC_DUSTMAPS_DIR="${QVC_DUSTMAPS_DIR:-${QVC_WORKDIR}/.dustmaps}"

if [[ $# -eq 0 ]]; then
  exec "${QVC_CODE_DIR}/scripts/run_demo.sh" all
fi

exec "${QVC_CODE_DIR}/scripts/run_demo.sh" "$@"
