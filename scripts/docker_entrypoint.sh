#!/usr/bin/env bash
set -euo pipefail

QVC_CODE_DIR="${QVC_CODE_DIR:-/opt/qvc}"

if [[ $# -eq 0 ]]; then
  exec "${QVC_CODE_DIR}/scripts/run_demo.sh" all
fi

exec "${QVC_CODE_DIR}/scripts/run_demo.sh" "$@"
