#!/usr/bin/env bash
set -euo pipefail

COMMAND="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

QVC_CODE_DIR="${QVC_CODE_DIR:-${REPO_ROOT}}"
QVC_WORKDIR="${QVC_WORKDIR:-${REPO_ROOT}}"
QVC_DATA_DIR="${QVC_DATA_DIR:-${QVC_WORKDIR}/data}"
QVC_RESULT_DIR="${QVC_RESULT_DIR:-${QVC_WORKDIR}/results}"
QVC_PLOTS_DIR="${QVC_PLOTS_DIR:-${QVC_WORKDIR}/plots}"
QVC_DUSTMAPS_DIR="${QVC_DUSTMAPS_DIR:-${QVC_WORKDIR}/results/dustmaps}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
export NUM_CORES="${NUM_CORES:-4}"
export QVC_CODE_DIR
export QVC_WORKDIR
export QVC_DATA_DIR
export QVC_RESULT_DIR
export QVC_PLOTS_DIR
export QVC_DUSTMAPS_DIR
export PYTHONPATH="${QVC_CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DEMO_PREFIX="${DEMO_PREFIX:-lc_paper}"
DEMO_OBJECT_ID="${DEMO_OBJECT_ID:-1452887}"

LC_H5_REL="results/data/${DEMO_PREFIX}/${DEMO_OBJECT_ID}.h5"
LC_H5="${QVC_WORKDIR}/${LC_H5_REL}"
SPECTRA_INPUT_H5_REL="results/data/lc_data_all.h5"
SPECTRA_INPUT_H5="${QVC_WORKDIR}/${SPECTRA_INPUT_H5_REL}"
SPECTRA_CSV_REL="results/data/jaxqsofit/test.csv"
SPECTRA_CSV="${QVC_WORKDIR}/${SPECTRA_CSV_REL}"
HUBBLE_LC_H5_REL="results/data/lc_data_all.h5"
HUBBLE_LC_H5="${QVC_WORKDIR}/${HUBBLE_LC_H5_REL}"
HUBBLE_SPECTRA_CSV_REL="results/data/spectra_data_all.csv"
HUBBLE_SPECTRA_CSV="${QVC_WORKDIR}/${HUBBLE_SPECTRA_CSV_REL}"
HUBBLE_COMPLETENESS_SIM_REL="results/data/mock_completeness_catalog_fresh.h5"
HUBBLE_COMPLETENESS_SIM="${QVC_WORKDIR}/${HUBBLE_COMPLETENESS_SIM_REL}"
HUBBLE_FIDUCIAL_PREFIX="${HUBBLE_FIDUCIAL_PREFIX:-paper_hubble_final_production}"
HUBBLE_RESTRICTED_PREFIX="${HUBBLE_RESTRICTED_PREFIX:-paper_hubble_final_production_restricted}"
HUBBLE_FIDUCIAL_POSTERIORS="${QVC_WORKDIR}/results/hubble_posteriors/${HUBBLE_FIDUCIAL_PREFIX}"
HUBBLE_RESTRICTED_POSTERIORS="${QVC_WORKDIR}/results/hubble_posteriors/${HUBBLE_RESTRICTED_PREFIX}"

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
    "${SPECTRA_INPUT_H5}"
    "${HUBBLE_SPECTRA_CSV}"
    "${HUBBLE_COMPLETENESS_SIM}"
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

  log "running light-curve fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.light_curve.fit_light_curves \
    --filter_object_id "${DEMO_OBJECT_ID}" \
    --svi_steps 100 \
    --nwarm 100 \
    --nsamp 100 \
    --nchains 1 \
    --progress \
    --plot \
    --fit_method "svi+nuts" \
    --corner_plot_mode "full"

  # if [[ ! -f "${LC_H5}" ]]; then
  #   log "expected light-curve output not found: ${LC_H5}"
  #   exit 1
  # fi
}

run_spectra() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  mkdir -p "${QVC_WORKDIR}/results/data/jaxqsofit" "${QVC_WORKDIR}/results/jaxqsofit/test" "${QVC_WORKDIR}/plots/jaxqsofit/test"

  if [[ ! -f "${SPECTRA_INPUT_H5}" ]]; then
    log "missing H5 input for spectra stage: ${SPECTRA_INPUT_H5}"
    exit 1
  fi
  if [[ ! -d "${QVC_WORKDIR}/data/spectra_cache_all" ]]; then
    log "missing spectra cache for spectra stage: ${QVC_WORKDIR}/data/spectra_cache_all"
    exit 1
  fi


  log "running spectra fit for object ${DEMO_OBJECT_ID}"
  python -m qvc.spectra.fit_spectra \
    --mode fit \
    --fpath-in "${SPECTRA_INPUT_H5_REL}" \
    "${SPECTRA_CSV_REL}" \
    --cache-dir "data/spectra_cache_all" \
    --output-dir "results/jaxqsofit/test" \
    --fig-dir "plots/jaxqsofit/test" \
    --verbose \
    --save-fig \
    --nuts-warmup 100 \
    --nuts-samples 100 \
    --nuts-chains 1 \
    --filter_object_id "${DEMO_OBJECT_ID}" \
    --plot_mcmc_diagnostics \
    --nproc 1

  # if [[ ! -f "${SPECTRA_CSV}" ]]; then
  #   log "expected spectra output not found: ${SPECTRA_CSV}"
  #   exit 1
  # fi
}

run_hubble() {
  ensure_workdirs
  cd "${QVC_WORKDIR}"

  if [[ ! -f "${HUBBLE_LC_H5}" ]]; then
    log "missing light-curve input for hubble stage: ${HUBBLE_LC_H5}"
    exit 1
  fi
  if [[ ! -f "${HUBBLE_SPECTRA_CSV}" ]]; then
    log "missing spectra summary for hubble stage: ${HUBBLE_SPECTRA_CSV}"
    exit 1
  fi
  if [[ ! -f "${HUBBLE_COMPLETENESS_SIM}" ]]; then
    log "missing completeness mock catalog for hubble stage: ${HUBBLE_COMPLETENESS_SIM}"
    exit 1
  fi
  if [[ ! -d "${HUBBLE_FIDUCIAL_POSTERIORS}" ]]; then
    log "missing fiducial resume checkpoints: ${HUBBLE_FIDUCIAL_POSTERIORS}"
    exit 1
  fi
  if [[ ! -d "${HUBBLE_RESTRICTED_POSTERIORS}" ]]; then
    log "missing restricted resume checkpoints: ${HUBBLE_RESTRICTED_POSTERIORS}"
    exit 1
  fi

  log "running fiducial hubble fit from resume checkpoints"
  python -m qvc.hubble.hubble_fit --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "${HUBBLE_SPECTRA_CSV_REL}" \
    --completeness_sim_file "${HUBBLE_COMPLETENESS_SIM_REL}" \
    --z_range 0.44 3.16 \
    --result_prefix "fiducial" \
    --prefix "${HUBBLE_FIDUCIAL_PREFIX}" \
    --sigma_clip_threshold 3.0 \
    "${HUBBLE_LC_H5_REL}"

  log "running restricted hubble fit from resume checkpoints"
  python -m qvc.hubble.hubble_fit --resume \
    --cosmo_models FlatLambdaCDM FlatwCDM Flatw0waCDM \
    --run full \
    --speed production \
    --spectra_fit_csv "${HUBBLE_SPECTRA_CSV_REL}" \
    --completeness_sim_file "${HUBBLE_COMPLETENESS_SIM_REL}" \
    --z_range 1.0 3.16 \
    --result_prefix "restricted" \
    --prefix "${HUBBLE_RESTRICTED_PREFIX}" \
    --sigma_clip_threshold 3.0 \
    "${HUBBLE_LC_H5_REL}"
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
    #run_light_curve
    run_spectra
    ;;
  hubble)
    run_setup
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
