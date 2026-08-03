from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "hpc_scripts" / "sfitspectra.xsh"


def test_sfitspectra_uses_csv_object_ids_without_h5_membership_filtering():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'chisq_csv = "results/data/variability_chi_sq_red_g_gt_20.csv"' in source
    assert "submit_object_ids = requested_object_ids" in source
    assert "--filter_object_id" in source
    for legacy_text in (
        "read_quasars_from_hdf5_flat",
        "h5_file",
        "load_h5_object_ids",
        "missing_from_h5",
        "H5_FILE",
        "USE_H5",
    ):
        assert legacy_text not in source


def test_sfitspectra_supports_both_fit_backends():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'fit_script = "fit_spectra.py"' in source
    assert '"fit_spectra.py": "qvc.spectra.fit_spectra"' in source
    assert (
        '"fit_spectra_jaxsedfit_joint.py": '
        '"qvc.spectra.fit_spectra_jaxsedfit_joint"'
    ) in source
    assert '"-m", fit_module' in source
    assert "Unsupported fit_script" in source


def test_sfitspectra_uses_backend_specific_arguments():
    source = SCRIPT.read_text(encoding="utf-8")

    assert (
        'sed_photometry_path = '
        '"data/jul14_master_input_file_chisqgt20_bandwagon_photometry.csv"'
    ) in source
    assert 'if fit_script == "fit_spectra.py":' in source
    assert '"--plot_mcmc_diagnostics"' in source
    assert '"--save-fig"' in source
    assert 'elif fit_script == "fit_spectra_jaxsedfit_joint.py":' in source
    assert '"--sed-photometry-path", sed_photometry_path' in source
    assert '"--progress"' in source
    assert "SED photometry input not found" in source


def test_sfitspectra_accepts_cli_overrides_and_builds_timestamped_git_job_name():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument(\n    "--description",' in source
    assert 'parser.add_argument(\n    "--fit-script",' in source
    assert "fit_script = cli_args.fit_script" in source
    assert 'parser.add_argument(\n    "--sed-photometry-path",' not in source
    assert 'datetime.now().strftime("%b%d_%H%M").lower()' in source
    assert '["git", "rev-parse", "--short", "HEAD"]' in source
    assert 'r"[^A-Za-z0-9.-]+", "_", cli_args.description' in source
    assert 'job_name = "_".join(job_name_parts)' in source
    assert "prefix = job_name" in source
    assert 'output_dir = f"results/data/jaxqsofit/{prefix}"' in source
    assert 'fig_dir = f"plots/jaxqsofit/{prefix}"' in source
    assert "#SBATCH --job-name={job_name}" in source


def test_sfitspectra_named_resume_keeps_current_csv_selection_and_separate_outputs():
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--resume",\n    metavar="OLD_RUN_NAME"' in source
    assert 'submit_object_ids = requested_object_ids' in source
    assert 'f"results/data/jaxqsofit/{resume_run_name}/all"' in source
    assert 'f"results/data/jaxqsofit/{prefix}"' in source
    assert 'resume_path.resolve() == (output_path / "all").resolve()' in source
    assert '"--resume is supported only with fit_spectra_jaxsedfit_joint.py"' in source
    assert 'export RESUME_DIR="{resume_dir}"' in source
    assert '"--resume", resume_dir' in source
    assert '"--resume-run-name", resume_run_name' in source
