#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/copy_paper_assets.sh \
    --speed quicker \
    --fiducial-dir plots/hubble/fiducial_run \
    [--restricted-dir plots/hubble/restricted_run] \
    [--only hubble|spectra|light-curve|appendix] \
    [--draft path/to/draft.tex] [--dry-run] [--dest-dir path/to/assets]

Description:
  Copy paper-ready plots, tables and TeX parameter files into plots/paper/.

Notes:
  - Run this from the repository root.
  - --speed must match the hubble run tag used inside the source directories.
  - Supports current short directory names and unique legacy completeness-tagged names.
  - Missing or ambiguous sources abort before any destination files are changed.
  - Restricted parameters are copied only when --restricted-dir is supplied.
  - --dry-run validates and prints the manifest without copying files.
  - All files are copied directly into one flat destination directory.
  - --dest-dir overrides the default <repo-root>/plots/paper destination.
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
  # The second argument identifies the asset group; destinations are flat.
  local dest_name="$3"
  local dest_dir="$DEST_ROOT"

  if [[ ! -f "$source_path" ]]; then
    MISSING+=("$dest_name <= $source_path")
    return
  fi

  SOURCES+=("$source_path")
  DESTINATIONS+=("$dest_dir/$dest_name")
  COPIED+=("$dest_name <= $source_path")
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
DEST_ROOT=""
SPEED=""
FIDUCIAL_DIR=""
RESTRICTED_DIR=""
DRAFT_PATH=""
ONLY_GROUP=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --dest-dir)
      DEST_ROOT="${2:-}"
      [[ -n "$DEST_ROOT" ]] || { echo "error: --dest-dir needs a path" >&2; exit 1; }
      shift 2
      ;;
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

if [[ -z "$SPEED" || -z "$FIDUCIAL_DIR" ]]; then
  echo "error: --speed and --fiducial-dir are required" >&2
  usage >&2
  exit 1
fi

case "$SPEED" in
  production|standard|quick|quicker|fastest)
    ;;
  *)
    echo "error: --speed must be one of: production, standard, quick, quicker, fastest" >&2
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
if [[ -z "$DEST_ROOT" ]]; then
  DEST_ROOT="$REPO_ROOT/plots/paper"
fi
if [[ -n "$RESTRICTED_DIR" && "$RESTRICTED_DIR" != /* ]]; then
  RESTRICTED_DIR="$REPO_ROOT/$RESTRICTED_DIR"
fi
if [[ -n "$DRAFT_PATH" && "$DRAFT_PATH" != /* ]]; then
  DRAFT_PATH="$REPO_ROOT/$DRAFT_PATH"
fi

if [[ ! -d "$FIDUCIAL_DIR" ]]; then
  echo "error: fiducial directory not found: $FIDUCIAL_DIR" >&2
  exit 1
fi
if [[ -n "$RESTRICTED_DIR" && ! -d "$RESTRICTED_DIR" ]]; then
  echo "error: restricted directory not found: $RESTRICTED_DIR" >&2
  exit 1
fi
if [[ -n "$DRAFT_PATH" && ! -f "$DRAFT_PATH" ]]; then
  echo "error: draft path not found: $DRAFT_PATH" >&2
  exit 1
fi

# Resolve all inputs before creating or changing any destination files.
resolve_run_dir() {
  local base="$1" short="$2" legacy="$3"
  local candidates=() path
  [[ ! -d "$base/$short" ]] || candidates+=("$short")
  for path in "$base"/"$legacy"*; do
    [[ ! -d "$path" ]] || candidates+=("${path##*/}")
  done
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "error: expected exactly one $short source in $base; found ${#candidates[@]}" >&2
    printf '  %s\n' "${candidates[@]:-none}" >&2
    return 1
  fi
  printf '%s\n' "${candidates[0]}"
}

declare -a COPIED=()
declare -a MISSING=()
declare -a SOURCES=()
declare -a DESTINATIONS=()

if [[ -z "$ONLY_GROUP" || "$ONLY_GROUP" == hubble ]]; then
  FIDUCIAL_RUN_DIR="$(resolve_run_dir "$FIDUCIAL_DIR" Flatw0waCDM_joint "Flatw0waCDM_joint_${SPEED}_all_z0p44_3p16_2d")"
  FIDUCIAL_MODEL_COMPARE_DIR="$(resolve_run_dir "$FIDUCIAL_DIR" model_compare "model_compare_joint_${SPEED}_all_z0p44_3p16_2d")"
  if [[ -n "$RESTRICTED_DIR" ]]; then
    RESTRICTED_MODEL_COMPARE_DIR="$(resolve_run_dir "$RESTRICTED_DIR" model_compare "model_compare_joint_${SPEED}_all_z1p00_3p16_2d")"
  fi
fi

copy_hubble_assets() {
  #copy_from_root "src/plots/appendix/N_vs_logZ_grid.pdf" "hubble" "N_vs_logZ_grid.pdf"
  #copy_from_root "src/plots/appendix/N_vs_cosmo_corner_grid.pdf" "hubble" "N_vs_cosmo_corner_grid.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/predicted_L2500_vs_fullcorr_band_debiased.pdf" "hubble" "predicted_L2500_vs_fullcorr_band_debiased.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/redshift_histograms.pdf" "hubble" "redshift_histograms.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/hubble_diagram_debiased.pdf" "hubble" "hubble_diagram_debiased.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/hubble_diagram.pdf" "hubble" "hubble_diagram.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/agn_table.csv" "hubble" "agn_table.csv"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/agn_table.tex" "hubble" "agn_table.tex"
  copy_from_dir "$FIDUCIAL_DIR" "diagnostics/sigma_tau_psd_fixed_postcut.pdf" "hubble" "sigma_tau_psd_fixed_postcut.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/predicted_vs_actual_M2500_debias.pdf" "hubble" "predicted_vs_actual_M2500_debias.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/alphaOX_residuals.pdf" "hubble" "alphaOx_residuals.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/delta_alphaOX_residuals.pdf" "hubble" "delta_alphaOX_residuals.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_RUN_DIR/completeness/completeness_map_with_absolute_percent_contours.pdf" "hubble" "completeness_map_with_absolute_percent_contours.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "diagnostics/blr_postcut.pdf" "hubble" "blr_postcut.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "diagnostics/sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf" "hubble" "sigma_tau_vs_lambda_broken_pl_fit_postcut.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_Flatw0waCDM_alphabeta.pdf" "hubble" "cosmo_corner_Flatw0waCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatwCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatwCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/cosmo_corner_FlatLambdaCDM_alphabeta.pdf" "hubble" "cosmo_corner_FlatLambdaCDM_alphabeta.pdf"
  copy_from_dir "$FIDUCIAL_DIR" "$FIDUCIAL_MODEL_COMPARE_DIR/param_results_fiducial.tex" "hubble" "param_results_fiducial.tex"
  if [[ -n "$RESTRICTED_DIR" ]]; then
    copy_from_dir "$RESTRICTED_DIR" "$RESTRICTED_MODEL_COMPARE_DIR/param_results_restricted.tex" "hubble" "param_results_restricted.tex"
  fi
}

copy_spectra_assets() {
  :
  #copy_from_root "plots/jaxqsofit/z0.907_212805.25-005145.7.pdf" "spectra" "z0.907_212805.25-005145.7.pdf"
}

copy_light_curve_assets() {
  :
  #copy_from_root "src/plots/multiband/test/light_curves_fits/0.9_1465126_light_curve_job4709.pdf" "light_curve" "0.9_1465126_light_curve_job4709.pdf"
}

copy_appendix_assets() {
  :
  #copy_from_root "src/plots/appendix/N_vs_logZ_grid.pdf" "appendix" "N_vs_logZ_grid.pdf"
  #copy_from_root "src/plots/appendix/N_vs_cosmo_corner_grid.pdf" "appendix" "N_vs_cosmo_corner_grid.pdf"
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

if [[ -z "$RESTRICTED_DIR" && ( -z "$ONLY_GROUP" || "$ONLY_GROUP" == hubble ) ]]; then
  echo "Note: param_results_restricted.tex is not updated; supply --restricted-dir for the draft's restricted results." >&2
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
  printf 'error: missing required paper assets:\n' >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  exit 1
fi

if [[ "$DRY_RUN" == false ]]; then
  for ((i=0; i<${#SOURCES[@]}; i++)); do
    mkdir -p "$(dirname "${DESTINATIONS[$i]}")"
    cp "${SOURCES[$i]}" "${DESTINATIONS[$i]}"
  done
else
  printf 'Dry run: no files copied.\n'
fi
printf 'Validated %d assets for %s\n' "${#COPIED[@]}" "$DEST_ROOT"
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
