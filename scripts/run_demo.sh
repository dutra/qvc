#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-all}"

QVC_CODE_DIR="${QVC_CODE_DIR:-/opt/qvc}"
QVC_WORKDIR="${QVC_WORKDIR:-/work/qvc-demo}"
QVC_DATA_DIR="${QVC_DATA_DIR:-${QVC_WORKDIR}/data}"
QVC_RESULT_DIR="${QVC_RESULT_DIR:-${QVC_WORKDIR}/results}"
QVC_PLOTS_DIR="${QVC_PLOTS_DIR:-${QVC_WORKDIR}/plots}"
QVC_DUSTMAPS_DIR="${QVC_DUSTMAPS_DIR:-${QVC_WORKDIR}/.dustmaps}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export NUM_CORES="${NUM_CORES:-4}"
export QVC_DATA_DIR
export QVC_RESULT_DIR
export QVC_DUSTMAPS_DIR
export PYTHONPATH="${QVC_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DEMO_PREFIX="${DEMO_PREFIX:-demo}"
DEMO_OBJECT_ID="${DEMO_OBJECT_ID:-1465126}"
DEMO_COSMO_MODEL="${DEMO_COSMO_MODEL:-FlatLambdaCDM}"

LC_H5_REL="results/data/${DEMO_PREFIX}/${DEMO_OBJECT_ID}.h5"
LC_H5="${QVC_WORKDIR}/${LC_H5_REL}"
SPECTRA_CSV_REL="results/data/${DEMO_PREFIX}/spectra_${DEMO_OBJECT_ID}.csv"
SPECTRA_CSV="${QVC_WORKDIR}/${SPECTRA_CSV_REL}"
AGN_DATA_REL="results/data/light_curves.h5"
AGN_DATA="${QVC_WORKDIR}/${AGN_DATA_REL}"

log() {
  printf '[qvc-demo] %s\n' "$*"
}

ensure_workdirs() {
  mkdir -p \
    "${QVC_WORKDIR}" \
    "${QVC_DATA_DIR}" \
    "${QVC_RESULT_DIR}" \
    "${QVC_PLOTS_DIR}" \
    "${QVC_DUSTMAPS_DIR}"
}

fetch_dustmaps() {
  local sentinel="${QVC_DUSTMAPS_DIR}/sfd/SFD_dust_4096_ngp.fits"
  if [[ -f "${sentinel}" ]]; then
    log "dustmaps already present at ${sentinel}; skipping fetch"
    return
  fi

  log "fetching dustmaps into ${QVC_DUSTMAPS_DIR}"
  python -c "from dustmaps.config import config; config['data_dir'] = r'${QVC_DUSTMAPS_DIR}'; import dustmaps.sfd; dustmaps.sfd.fetch()"
}

run_setup() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  log "running qvc.setup_data in ${QVC_WORKDIR}"
  python -m qvc.setup_data
  fetch_dustmaps

  local required=(
    "${QVC_WORKDIR}/data/dr16q_prop_May01_2024.fits"
    "${QVC_WORKDIR}/data/ssp_data_fsps_v3.2_lgmet_age.h5"
    "${QVC_WORKDIR}/data/spectra_cache/spec-9180-57693-0463.fits"
    "${AGN_DATA}"
    "${QVC_WORKDIR}/results/data/spectra.csv"
    "${QVC_DUSTMAPS_DIR}/sfd/SFD_dust_4096_ngp.fits"
  )

  local path
  for path in "${required[@]}"; do
    if [[ ! -e "${path}" ]]; then
      log "missing required setup artifact: ${path}"
      exit 1
    fi
  done
}

run_light_curve() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  mkdir -p "${QVC_WORKDIR}/results/data/${DEMO_PREFIX}" "${QVC_WORKDIR}/plots/multiband/${DEMO_PREFIX}"
  export PREFIX="${DEMO_PREFIX}"
  export SUFFIX="${DEMO_PREFIX}"

  if [[ -f "${LC_H5}" ]]; then
    log "light-curve output already exists at ${LC_H5}; reusing it"
    return
  fi

  log "running light-curve fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.light_curve.fit_light_curves \
    --plot \
    --progress \
    --nwarm 200 \
    --nsamp 100 \
    --nchains 4 \
    --max_tree_depth 8 \
    --filter_object_id "${DEMO_OBJECT_ID}"

  if [[ ! -f "${LC_H5}" ]]; then
    log "expected light-curve output not found: ${LC_H5}"
    exit 1
  fi
}

run_spectra() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  mkdir -p "${QVC_WORKDIR}/results/data/${DEMO_PREFIX}" "${QVC_WORKDIR}/results/jaxqsofit/${DEMO_PREFIX}" "${QVC_WORKDIR}/plots/jaxqsofit/${DEMO_PREFIX}"

  if [[ ! -f "${LC_H5}" ]]; then
    log "missing light-curve input for spectra stage: ${LC_H5}"
    exit 1
  fi

  if [[ -f "${SPECTRA_CSV}" ]]; then
    log "spectra output already exists at ${SPECTRA_CSV}; reusing it"
    return
  fi

  log "running spectra fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.spectra.fit_spectra \
    --mode fit \
    --cache-dir "data/spectra_cache" \
    --output-dir "results/jaxqsofit/${DEMO_PREFIX}" \
    --fig-dir "plots/jaxqsofit/${DEMO_PREFIX}" \
    --verbose \
    --save-fig \
    --nuts-warmup 200 \
    --nuts-samples 100 \
    --nuts-chains 1 \
    --filter_object_id "${DEMO_OBJECT_ID}" \
    "${LC_H5_REL}" \
    "${SPECTRA_CSV_REL}"

  if [[ ! -f "${SPECTRA_CSV}" ]]; then
    log "expected spectra output not found: ${SPECTRA_CSV}"
    exit 1
  fi
}

run_hubble() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  if [[ ! -f "${AGN_DATA}" ]]; then
    log "missing downloaded AGN input for hubble stage: ${AGN_DATA}"
    exit 1
  fi
  if [[ ! -f "${SPECTRA_CSV}" ]]; then
    log "missing spectra summary for hubble stage: ${SPECTRA_CSV}"
    exit 1
  fi

  log "running hubble fit using downloaded data"
  python -m qvc.hubble.hubble_fit \
    --cosmo_models "${DEMO_COSMO_MODEL}" \
    --run single \
    --speed fast \
    --disable_completeness \
    --spectra_fit_csv "${SPECTRA_CSV_REL}" \
    --z_range 0.44 3.16 \
    --prefix "${DEMO_PREFIX}" \
    "${AGN_DATA_REL}"
}

case "${COMMAND}" in
  setup)
    run_setup
    ;;
  light-curve)
    run_setup
    run_light_curve
    ;;
  spectra)
    run_setup
    run_light_curve
    run_spectra
    ;;
  hubble)
    run_setup
    run_light_curve
    run_spectra
    run_hubble
    ;;
  all)
    run_setup
    run_light_curve
    run_spectra
    run_hubble
    ;;
  *)
    log "unknown command: ${COMMAND}"
    log "expected one of: all, setup, light-curve, spectra, hubble"
    exit 2
    ;;
esac
