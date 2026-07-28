#!/usr/bin/env python3
"""Compare the QVC mock M_2500 LF with published optical/UV QLFs."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM


ALPHA_NU = -0.5
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
LOG10_LSUN_ERG_S = np.log10(3.9e33)
NU_2500_HZ = 2.99792458e18 / 2500.0
AB_ABSOLUTE_MAG_ZEROPOINT = 51.59477721004232

# Shen et al. (2020), Table 3, local "polished" bolometric DPL fits.
# Columns are z, gamma1, gamma2, log10(phi* / Mpc^-3 dex^-1), and
# log10(L* / L_sun).  These local fits are an internal check on the global
# Model A/B curves used by QVC, rather than an independent data comparison.
SHEN_TABLE3_POLISHED = np.array(
    [
        (0.2, 0.787, 1.713, -4.240, 11.275),
        (0.4, 0.561, 2.108, -4.151, 11.650),
        (0.8, 0.599, 2.199, -4.412, 12.223),
        (1.2, 0.504, 2.423, -4.530, 12.622),
        (1.6, 0.484, 2.546, -4.668, 12.919),
        (2.0, 0.411, 2.487, -4.679, 13.011),
        (3.0, 0.424, 1.878, -4.698, 12.708),
        (4.0, 0.213, 1.885, -5.034, 12.562),
        (5.0, 0.245, 1.912, -5.243, 12.308),
        (6.0, 1.509, 1.509, -5.452, 11.978),
    ],
    dtype=[
        ("z", float),
        ("gamma1", float),
        ("gamma2", float),
        ("log_phi_star", float),
        ("log_lstar_lsun", float),
    ],
)


def m1450_to_m2500(m1450, alpha_nu=ALPHA_NU):
    """Convert AB magnitudes for f_nu proportional to nu**alpha_nu."""
    return np.asarray(m1450) - 2.5 * alpha_nu * np.log10(1450.0 / 2500.0)


def log_nu_lnu_to_ab_absolute_magnitude(log_nu_lnu, frequency_hz):
    log_lnu = np.asarray(log_nu_lnu) - np.log10(frequency_hz)
    return AB_ABSOLUTE_MAG_ZEROPOINT - 2.5 * log_lnu


def shen_bolometric_dpl(log_lbol_erg_s, row):
    """Evaluate Shen et al. (2020) equation 9 from one Table 3 row."""
    log_l_lsun = np.asarray(log_lbol_erg_s) - LOG10_LSUN_ERG_S
    ratio = 10.0 ** (log_l_lsun - row["log_lstar_lsun"])
    return row["log_phi_star"] - np.log10(
        ratio ** row["gamma1"] + ratio ** row["gamma2"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shen-pubtools", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pubtools = args.shen_pubtools.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(pubtools))
    old_cwd = Path.cwd()
    os.chdir(pubtools)
    try:
        from utilities import (
            bolometric_correction,
            return_bolometric_qlf,
            return_qlf_in_band,
        )
        from obdata_copy.new_load_palanque16_lf_data import load_palanque16_lf_data
        from obdata_copy import new_load_ross13_lf_data as ross13
        from obdata_copy.new_load_kk18_lf_shape import return_kk18_lf_fitted
        # The archived pubtools config concatenates ``pubtools`` and ``data``
        # without a path separator in this loader. Point it explicitly at the
        # data directory shipped with the same Shen checkout.
        ross13.datapath = str(pubtools / "data") + os.sep

        log_lbol_erg_s = np.linspace(43.0, 49.0, 241)
        log_lbol_lsun = log_lbol_erg_s - LOG10_LSUN_ERG_S
        log_nu_lnu_2500 = (
            np.asarray(
                [bolometric_correction(value, NU_2500_HZ) for value in log_lbol_lsun]
            )
            + LOG10_LSUN_ERG_S
        )
        m2500_shen_sed = log_nu_lnu_to_ab_absolute_magnitude(
            log_nu_lnu_2500, NU_2500_HZ
        )
        m2500_fixed_bc = 91.0 - 2.5 * log_lbol_erg_s
        delta_m2500 = m2500_fixed_bc - m2500_shen_sed
        bc2500 = 10.0 ** (log_lbol_erg_s - log_nu_lnu_2500)
        sed_check = pd.DataFrame(
            {
                "log_lbol_erg_s": log_lbol_erg_s,
                "log_nu_lnu_2500_erg_s": log_nu_lnu_2500,
                "bc2500_shen": bc2500,
                "M2500_shen_sed": m2500_shen_sed,
                "M2500_fixed_bc_4p82": m2500_fixed_bc,
                "delta_M_fixed_minus_shen": delta_m2500,
            }
        )
        sed_check.to_csv(output_dir / "m2500_fixed_bc_vs_shen_sed.csv", index=False)

        fig, (ax_mag, ax_delta) = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        ax_mag.plot(log_lbol_erg_s, m2500_shen_sed, label="Shen mean SED at 2500 A")
        ax_mag.plot(
            log_lbol_erg_s,
            m2500_fixed_bc,
            ls="--",
            label=r"$91-2.5\log L_{\rm bol}$ (fixed BC=4.82)",
        )
        ax_mag.set_ylabel(r"$M_{2500,\rm AB}$")
        ax_mag.invert_yaxis()
        ax_mag.legend(frameon=False)
        ax_mag.grid(alpha=0.2)
        ax_delta.axhline(0.0, color="0.3", lw=1)
        ax_delta.plot(log_lbol_erg_s, delta_m2500, color="tab:red")
        ax_delta.set_xlabel(r"$\log_{10}(L_{\rm bol}/{\rm erg\,s^{-1}})$")
        ax_delta.set_ylabel(r"$M_{\rm fixed}-M_{\rm Shen}$")
        ax_delta.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "m2500_fixed_bc_vs_shen_sed.pdf")
        fig.savefig(output_dir / "m2500_fixed_bc_vs_shen_sed.png", dpi=180)
        plt.close(fig)

        qvc_luminosity_range = (log_lbol_erg_s >= 44.0) & (log_lbol_erg_s <= 48.0)
        print("\nFixed-BC M2500 shortcut versus Shen mean SED (44 <= log Lbol <= 48):")
        print(
            f"delta M range={delta_m2500[qvc_luminosity_range].min():+.3f} to "
            f"{delta_m2500[qvc_luminosity_range].max():+.3f} mag; "
            f"median={np.median(delta_m2500[qvc_luminosity_range]):+.3f} mag"
        )
        print(
            f"Shen BC2500 range={bc2500[qvc_luminosity_range].min():.3f} to "
            f"{bc2500[qvc_luminosity_range].max():.3f}; "
            f"median={np.median(bc2500[qvc_luminosity_range]):.3f}"
        )

        redshifts = [0.87, 1.25, 1.63, 2.01, 2.40, 2.80, 3.25]
        loaders = {
            "Palanque-Delabrouille+16": load_palanque16_lf_data,
            "Ross+13 BOSS": ross13.load_ross13_lf_data,
            "Ross+13 Stripe 82": ross13.load_ross13_s82_lf_data,
        }
        rows = []
        uv_conversion_rows = []

        fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
        for ax, redshift in zip(axes.flat, redshifts):
            log_nu_lnu, log_phi_dex = return_qlf_in_band(
                redshift, NU_2500_HZ, model="B"
            )
            m2500_model = log_nu_lnu_to_ab_absolute_magnitude(
                log_nu_lnu, NU_2500_HZ
            )
            log_phi_mag = np.asarray(log_phi_dex) + np.log10(0.4)
            order = np.argsort(m2500_model)
            m2500_model = m2500_model[order]
            log_phi_mag = log_phi_mag[order]

            ax.plot(
                m2500_model,
                log_phi_mag,
                color="black",
                lw=2,
                label=r"QVC / Shen observed 2500 $\AA$",
            )

            m1450, log_phi_1450_mag = return_qlf_in_band(
                redshift, -5, model="B"
            )
            m2500_from_1450 = m1450_to_m2500(m1450)
            converted_1450 = np.interp(
                m2500_model,
                np.asarray(m2500_from_1450)[::-1],
                np.asarray(log_phi_1450_mag)[::-1],
                left=np.nan,
                right=np.nan,
            )
            conversion_delta = log_phi_mag - converted_1450
            conversion_valid = np.isfinite(conversion_delta)
            uv_conversion_rows.append(
                {
                    "z": redshift,
                    "median_delta_log_phi_direct_2500_minus_converted_1450": np.median(
                        conversion_delta[conversion_valid]
                    ),
                    "max_abs_delta_log_phi_direct_2500_minus_converted_1450": np.max(
                        np.abs(conversion_delta[conversion_valid])
                    ),
                }
            )

            dm = float(COSMO.distmod(redshift).value)
            relevant_lo = 18.5 - dm
            relevant_hi = 24.0 - dm
            ax.axvspan(relevant_lo, relevant_hi, color="0.8", alpha=0.35)

            for label, loader in loaders.items():
                loaded = loader(redshift)
                if loaded is False:
                    continue
                m1450, log_phi_obs, log_phi_err = loaded
                m2500 = m1450_to_m2500(m1450)
                predicted = np.interp(m2500, m2500_model, log_phi_mag, left=np.nan, right=np.nan)
                residual = predicted - log_phi_obs
                relevant = (m2500 >= relevant_lo) & (m2500 <= relevant_hi)
                for values in zip(m2500, log_phi_obs, log_phi_err, predicted, residual, relevant):
                    rows.append(
                        {
                            "reference": label,
                            "z": redshift,
                            "M2500": values[0],
                            "log_phi_observed": values[1],
                            "log_phi_error": values[2],
                            "log_phi_qvc": values[3],
                            "delta_log_phi_qvc_minus_observed": values[4],
                            "in_completeness_magnitude_window": bool(values[5]),
                        }
                    )
                ax.errorbar(m2500, log_phi_obs, yerr=log_phi_err, fmt="o", ms=3, capsize=2, label=label)

            m_eval = np.linspace(max(relevant_lo, -30.0), min(relevant_hi, -19.0), 80)
            kk18 = return_kk18_lf_fitted(
                m_eval + 2.5 * ALPHA_NU * np.log10(1450.0 / 2500.0),
                redshift,
            )
            qvc_eval = np.interp(m_eval, m2500_model, log_phi_mag)
            ax.plot(m_eval, kk18, color="tab:purple", ls="--", lw=1.5, label="Kulkarni+19 Model 2")
            for m_value, observed, predicted in zip(m_eval, kk18, qvc_eval):
                rows.append(
                    {
                        "reference": "Kulkarni+19 Model 2",
                        "z": redshift,
                        "M2500": m_value,
                        "log_phi_observed": observed,
                        "log_phi_error": np.nan,
                        "log_phi_qvc": predicted,
                        "delta_log_phi_qvc_minus_observed": predicted - observed,
                        "in_completeness_magnitude_window": True,
                    }
                )

            ax.set_title(f"z = {redshift:.2f}")
            ax.set_xlim(-30, -19)
            ax.set_ylim(-10, -4)
            ax.grid(alpha=0.2)

        for ax in axes.flat[len(redshifts):]:
            ax.set_visible(False)
        axes[1, 0].set_ylabel(r"$\log_{10}\Phi\ [{\rm Mpc}^{-3}\,{\rm mag}^{-1}]$")
        axes[-1, 1].set_xlabel(r"$M_{2500}$")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_dir / "mock_lf_literature_comparison.pdf")
        fig.savefig(output_dir / "mock_lf_literature_comparison.png", dpi=180)
        plt.close(fig)

        comparisons = pd.DataFrame(rows)
        comparisons.to_csv(output_dir / "mock_lf_literature_residuals.csv", index=False)
        summary = (
            comparisons.groupby(["reference", "in_completeness_magnitude_window"])
            ["delta_log_phi_qvc_minus_observed"]
            .agg(["count", "median", "mean", "std"])
            .reset_index()
        )
        summary.to_csv(output_dir / "mock_lf_literature_summary.csv", index=False)
        print(summary.to_string(index=False))
        uv_conversion_summary = pd.DataFrame(uv_conversion_rows)
        uv_conversion_summary.to_csv(
            output_dir / "mock_lf_2500_vs_converted_1450.csv", index=False
        )
        print("\nDirect Shen 2500 A versus alpha_nu=-0.5 conversion from 1450 A:")
        print(uv_conversion_summary.to_string(index=False))

        table3_rows = []
        fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True)
        for ax, row in zip(axes.flat, SHEN_TABLE3_POLISHED):
            z = float(row["z"])
            for model, color in (("A", "tab:blue"), ("B", "tab:orange")):
                log_lbol, log_phi_global = return_bolometric_qlf(z, model=model)
                log_lbol = np.asarray(log_lbol)
                log_phi_global = np.asarray(log_phi_global)
                log_phi_local = shen_bolometric_dpl(log_lbol, row)
                delta = log_phi_global - log_phi_local
                qvc_range = (log_lbol >= 44.0) & (log_lbol <= 48.0)
                table3_rows.append(
                    {
                        "z": z,
                        "model": model,
                        "median_delta_log_phi_global_minus_table3": np.median(delta[qvc_range]),
                        "max_abs_delta_log_phi_global_minus_table3": np.max(
                            np.abs(delta[qvc_range])
                        ),
                    }
                )
                ax.plot(log_lbol, log_phi_global, color=color, label=f"Global Model {model}")
            ax.plot(
                log_lbol,
                log_phi_local,
                color="black",
                ls="--",
                label='Table 3 local "polished"',
            )
            ax.axvspan(44.0, 48.0, color="0.8", alpha=0.25)
            ax.set_title(f"z = {z:g}")
            ax.grid(alpha=0.2)
        axes[0, 0].set_xlim(43.0, 49.0)
        axes[0, 0].set_ylim(-11.0, -3.0)
        fig.supxlabel(r"$\log_{10}(L_{\rm bol}/{\rm erg\,s^{-1}})$")
        fig.supylabel(r"$\log_{10}\Phi\ [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(output_dir / "mock_lf_shen_table3_self_validation.pdf")
        fig.savefig(output_dir / "mock_lf_shen_table3_self_validation.png", dpi=180)
        plt.close(fig)

        table3_summary = pd.DataFrame(table3_rows)
        table3_summary.to_csv(output_dir / "mock_lf_shen_table3_self_validation.csv", index=False)
        print("\nShen Table 3 self-validation over 44 <= log Lbol <= 48:")
        print(table3_summary.to_string(index=False))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
