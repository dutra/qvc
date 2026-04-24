#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/copy_paper_assets.sh \
    --speed fast \
    --fiducial-dir plots/hubble/apr2a_apr1a_fast_paper \
    --restricted-dir plots/hubble/apr2a_apr1a_fast_paper_restricted \
    [--only hubble|spectra|light-curve|appendix] \
    [--draft path/to/draft.tex]

Description:
  Copy paper-ready plots and TeX parameter files into plots/paper/.

Notes:
  - Run this from the repository root.
  - --speed must match the hubble run tag used inside the source directories.
  - Hubble paper assets are expected under completeness-tagged `_2d` run directories.
  - The manifest is hardcoded for the current paper draft.
  - --draft is accepted for logging only and is not parsed.
  - --only filters the copy to a single asset group.
EOF
}

require_repo_root() {
  local required=(
    "plots"
    "src"
  )
  local item
  for item in "${required[@]}"; do
    if [[ ! -e "$item" ]]; then
      echo "error: expected to run from the repo root; missing '$item' in $(pwd)" >&2
      exit 1
    fi
  done
}

copy_file() {
  local source_path="$1"
  local dest_subdir="$2"
  local dest_name="$3"
  local dest_dir="$DEST_ROOT/$dest_subdir"

  mkdir -p "$dest_dir"

  if [[ ! -f "$source_path" ]]; then
    MISSING+=("$dest_subdir/$dest_name <= $source_path")
    return
  fi

  cp "$source_path" "$dest_dir/$dest_name"
  COPIED+=("$dest_subdir/$dest_name <= $source_path")
}

copy_from_root() {
  local relative_source="$1"
  local dest_subdir="$2"
  local dest_name="$3"
  copy_file "$REPO_ROOT/$relative_source" "$dest_subdir" "$dest_name"
}

copy_from_dir() {
  local base_dir="$1"
  local relative_source="$2"
  local dest_subdir="$3"
  local dest_name="$4"
  copy_file "$base_dir/$relative_source" "$dest_subdir" "$dest_name"
}

REPO_ROOT="$(pwd)"
DEST_ROOT="$REPO_ROOT/plots/paper"
SPEED=""
FIDUCIAL_DIR=""
RESTRICTED_DIR=""
DRAFT_PATH=""
ONLY_GROUP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --speed)
      SPEED="${2:-}"
      shift 2
      ;;
    --fiducial-dir)
      FIDUCIAL_DIR="${2:-}"
      shift 2
      ;;
    --restricted-dir)
      RESTRICTED_DIR="${2:-}"
      shift 2
      ;;
    --draft)
      DRAFT_PATH="${2:-}"
      shift 2
      ;;
    --only)
      ONLY_GROUP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SPEED" || -z "$FIDUCIAL_DIR" || -z "$RESTRICTED_DIR" ]]; then
  echo "error: --speed, --fiducial-dir, and --restricted-dir are required" >&2
  usage >&2
  exit 1
fi

case "$SPEED" in
  production|standard|quick|fastest)
    ;;
  *)
    echo "error: --speed must be one of: production, standard, quick, fastest" >&2
    exit 1
    ;;
esac

if [[ -n "$ONLY_GROUP" ]]; then
  case "$ONLY_GROUP" in
    hubble|spectra|light-curve|appendix)
      ;;
    *)
      echo "error: --only must be one of: hubble, spectra, light-curve, appendix" >&2
      exit 1
      ;;
  esac
fi

require_repo_root

if [[ "$FIDUCIAL_DIR" != /* ]]; then
  FIDUCIAL_DIR="$REPO_ROOT/$FIDUCIAL_DIR"
fi
if [[ "$RESTRICTED_DIR" != /* ]]; then
  RESTRICTED_DIR="$REPO_ROOT/$RESTRICTED_DIR"
fi
if [[ -n "$DRAFT_PATH" && "$DRAFT_PATH" != /* ]]; then
  DRAFT_PATH="$REPO_ROOT/$DRAFT_PATH"
fi

if [[ ! -d "$FIDUCIAL_DIR" ]]; then
  echo "error: fiducial directory not found: $FIDUCIAL_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESTRICTED_DIR" ]]; then
  echo "error: restricted directory not found: $RESTRICTED_DIR" >&2
  exit 1
fi
if [[ -n "$DRAFT_PATH" && ! -f "$DRAFT_PATH" ]]; then
  echo "error: draft path not found: $DRAFT_PATH" >&2
  exit 1
fi

mkdir -p "$DEST_ROOT"

declare -a COPIED=()
declare -a MISSING=()

FIDUCIAL_RUN_DIR="Flatw0waCDM_joint_${SPEED}_all_z0p44_3p16_2d"
FIDUCIAL_MODEL_COMPARE_DIR="model_compare_${SPEED}_all_z0p44_3p16_2d"
RESTRICTED_MODEL_COMPARE_DIR="model_compare_${SPEED}_all_z1p00_3p16_2d"

copy_hubble_assets() {
  copy_from_root "src/plots/appendix/N_vs_logZ_grid.pdf" "hubble" "N_vs_logZ_grid.pdf"
  copy_from_root "src/plots/appendix/N_vs_cosmo_corner_grid.pdf" "hubble" "N_vs_cosmo_corner_grid.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/predicted_L2500_vs_fullcorr_band_debiased.pdf" "hubble" "predicted_L2500_vs_fullcorr_band_debiased.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/redshift_histograms.pdf" "hubble" "redshift_histograms.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/hubble_diagram_debiased.pdf" "hubble" "hubble_diagram_debiased.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/hubble_diagram.pdf" "hubble" "hubble_diagram.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/predicted_vs_actual_M2500_debias.pdf" "hubble" "predicted_vs_actual_M2500_debias.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/alphaOX_residuals.pdf" "hubble" "alphaOx_residuals.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/delta_alphaOX_residuals.pdf" "hubble" "dalphaOx_int_residuals.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/completeness/completeness_map.pdf" "hubble" "completeness_map.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "diagnostics/spectral_fraction_vs_redshift_cuts.pdf" "hubble" "spectral_fraction_vs_redshift_cuts.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_Flatw0waCDM_alphabeta.pdf" "hubble" "cosmo_corner_Flatw0waCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatwCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatwCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatLambdaCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatLambdaCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_Flatw0waCDM_alphabeta.pdf" "hubble" "cosmo_corner_Flatw0waCDM_noalphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatwCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatwCDM_noalphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatLambdaCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatLambdaCDM_noalphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/param_results_fiducial.tex" "hubble" "param_results_fiducial.tex"
  copy_from_dir "$RESTRICTED_DIR" "$RESTRICTED_MODEL_COMPARE_DIR/param_results_restricted.tex" "hubble" "param_results_restricted.tex"
}

copy_spectra_assets() {
  copy_from_root "plots/jaxqsofit/z0.907_212805.25-005145.7.pdf" "spectra" "z0.907_212805.25-005145.7.pdf"
}

copy_light_curve_assets() {
  copy_from_root "src/plots/multiband/test/light_curves_fits/0.9_1465126_light_curve_job4709.pdf" "light_curve" "0.9_1465126_light_curve_job4709.pdf"
}

copy_appendix_assets() {
  copy_from_root "src/plots/appendix/N_vs_logZ_grid.pdf" "appendix" "N_vs_logZ_grid.pdf"
  copy_from_root "src/plots/appendix/N_vs_cosmo_corner_grid.pdf" "appendix" "N_vs_cosmo_corner_grid.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/hubble_diagram.pdf" "appendix" "hubble_diagram.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/completeness/completeness_map.pdf" "appendix" "completeness_map.pdf"
}

case "${ONLY_GROUP:-all}" in
  all)
    copy_hubble_assets
    copy_spectra_assets
    copy_light_curve_assets
    copy_appendix_assets
    ;;
  hubble)
    copy_hubble_assets
    ;;
  spectra)
    copy_spectra_assets
    ;;
  light-curve)
    copy_light_curve_assets
    ;;
  appendix)
    copy_appendix_assets
    ;;
esac

if [[ ${#MISSING[@]} -gt 0 ]]; then
  printf 'error: missing required paper assets:\n' >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  exit 1
fi

printf 'Copied %d assets into %s\n' "${#COPIED[@]}" "$DEST_ROOT"
printf 'Speed: %s\n' "$SPEED"
if [[ -n "$DRAFT_PATH" ]]; then
  printf 'Draft reference: %s\n' "$DRAFT_PATH"
fi
if [[ -n "$ONLY_GROUP" ]]; then
  printf 'Asset group: %s\n' "$ONLY_GROUP"
fi
printf 'Fiducial source: %s\n' "$FIDUCIAL_DIR"
printf 'Restricted source: %s\n' "$RESTRICTED_DIR"
printf 'Files:\n'
printf '  - %s\n' "${COPIED[@]}"
