import math
import os

import numpy as np
import pandas as pd

from qvc.hubble.hubble_completeness_refactored import (
    COMPLETENESS_FHOST_COL,
    evaluate_dm_interp,
    resolve_completeness_magnitude_column,
)
from qvc.hubble.completeness_strata import COMPLETENESS_STRATUM_COL


REQUIRED_AGN_TABLE_COLUMNS = (
    "sdss_name",
    "ra",
    "dec",
    "z",
    "z_err",
    "apparent_mag_2500",
    "apparent_mag_2500_err",
    "pl_slope",
    "pl_slope_err",
    "fracAGN_5100_fit",
    "fracAGN_5100_fit_err",
    "ebv_agn",
    "ebv_agn_err",
    "ebv_gal",
    "ebv_gal_err",
    "log_tau_uv_rf",
    "log_tau_uv_rf_std_psd",
    "log_sigma_uv",
    "log_sigma_uv_std_psd",
    "log_sigma_uv_log_tau_uv_rf_cov_psd",
)

AGN_TABLE_PLAIN_COLUMNS = {
    "SDSS Name": "sdss_name",
    "RA": "ra",
    "Dec": "dec",
    "z": "z",
    "z_err": "z_err",
    "m_2500": "apparent_mag_2500_corr",
    "m_2500_err": "apparent_mag_2500_corr_err",
    "m_2500_uncorr": "apparent_mag_2500",
    "m_2500_uncorr_err": "apparent_mag_2500_err",
    "pl_slope": "pl_slope",
    "pl_slope_err": "pl_slope_err",
    "fracAGN_5100_fit": "fracAGN_5100_fit",
    "fracAGN_5100_fit_err": "fracAGN_5100_fit_err",
    "ebv_agn": "ebv_agn",
    "ebv_agn_err": "ebv_agn_err",
    "ebv_gal": "ebv_gal",
    "ebv_gal_err": "ebv_gal_err",
    "mu": "mu",
    "mu_err": "mu_err",
    "log_tau_UV_RF": "log_tau_uv_rf",
    "log_tau_UV_RF_err": "log_tau_uv_rf_std_psd",
    "log_sigma_UV": "log_sigma_uv",
    "log_sigma_UV_err": "log_sigma_uv_std_psd",
    "cov_log_sigma_UV_log_tau_UV_RF": "log_sigma_uv_log_tau_uv_rf_cov_psd",
}


def _resolve_table_debias_values(agn_df, *, dm_interp=None, dmi_values=None):
    dmi = None
    if dmi_values is not None:
        dmi = np.asarray(dmi_values, dtype=float)
        if dmi.shape != (len(agn_df),):
            raise ValueError(
                f"dmi_values has shape {dmi.shape}, but expected {(len(agn_df),)}."
            )

    if dm_interp is None:
        if dmi is None:
            raise ValueError("AGN table requires either dm_interp or dmi_values.")
        return dmi

    magnitude_col = resolve_completeness_magnitude_column(agn_df)
    dmi_interp = evaluate_dm_interp(
        dm_interp,
        agn_df["z"],
        agn_df[magnitude_col],
        completeness_stratum=(
            agn_df[COMPLETENESS_STRATUM_COL]
            if COMPLETENESS_STRATUM_COL in agn_df.columns
            else None
        ),
    )
    if dmi_interp.shape != (len(agn_df),):
        raise ValueError(
            f"dm_interp returned shape {dmi_interp.shape}, expected {(len(agn_df),)}."
        )
    if dmi is None:
        return dmi_interp
    return np.where(np.isfinite(dmi), dmi, dmi_interp)


def _prepare_agn_table_dataframe(agn_df, mu, mu_err, dm_interp=None, dmi_values=None):
    missing_cols = [col for col in REQUIRED_AGN_TABLE_COLUMNS if col not in agn_df.columns]
    if missing_cols:
        raise KeyError(f"AGN table requires columns: {missing_cols}")

    df = agn_df.copy()
    df["mu"] = np.asarray(mu, dtype=float)
    df["mu_err"] = np.asarray(mu_err, dtype=float)

    dm_values = _resolve_table_debias_values(
        df,
        dm_interp=dm_interp,
        dmi_values=dmi_values,
    )
    df["apparent_mag_2500_corr"] = np.asarray(df["apparent_mag_2500"], dtype=float) - dm_values
    df["apparent_mag_2500_corr_err"] = np.asarray(df["apparent_mag_2500_err"], dtype=float)
    return df


def make_agn_csv_table(
    agn_df,
    mu,
    mu_err,
    dm_interp=None,
    *,
    dmi_values=None,
    sort_by,
    ascending,
    write_path,
) -> pd.DataFrame:
    df = _prepare_agn_table_dataframe(
        agn_df,
        mu,
        mu_err,
        dm_interp=dm_interp,
        dmi_values=dmi_values,
    )

    if sort_by is not None:
        if sort_by not in df.columns:
            raise KeyError(f"sort_by column {sort_by!r} is not present in AGN table data.")
        df = df.sort_values(sort_by, ascending=ascending)

    os.makedirs(write_path, exist_ok=True)
    out_path = os.path.join(write_path, "agn_table_all_fields.csv")
    df.to_csv(out_path, index=False)

    missing_plain_cols = [
        input_col
        for input_col in AGN_TABLE_PLAIN_COLUMNS.values()
        if input_col not in df.columns
    ]
    if missing_plain_cols:
        raise KeyError(f"Plain AGN table requires columns: {missing_plain_cols}")
    plain_df = pd.DataFrame(
        {
            output_col: df[input_col].to_numpy()
            for output_col, input_col in AGN_TABLE_PLAIN_COLUMNS.items()
        }
    )
    plain_path = os.path.join(write_path, "agn_table.csv")
    plain_df.to_csv(plain_path, index=False)
    return df


def make_agn_latex_table(
    agn_df,
    mu,
    mu_err,
    dm_interp=None,
    *,
    dmi_values=None,
    sort_by,
    ascending,
    max_rows,
    write_path,
) -> str:
    def _is_bad(x):
        return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

    def _fmt_num(x, nd):
        return r"$\dots$" if _is_bad(x) else rf"${float(x):.{nd}f}$"

    def _fmt_signed_num(x, nd):
        return r"$\dots$" if _is_bad(x) else rf"${float(x):+.{nd}f}$"

    def _name_to_bold(name):
        safe_name = str(name).replace("-", "$-$")
        return rf"\textbf{{J{safe_name}}}"

    def _fmt_with_sym_err(row, base_col, nd_val, nd_err, *, err_col=None):
        value = row[base_col]
        if _is_bad(value):
            return r"$\dots$"
        value = float(value)
        if err_col is None:
            err_col = f"{base_col}_err"
        err_value = row[err_col]
        if _is_bad(err_value):
            return rf"${value:.{nd_val}f}$"
        err_value = abs(float(err_value))
        return rf"${value:.{nd_val}f} \pm {err_value:.{nd_err}f}$"

    df = _prepare_agn_table_dataframe(
        agn_df,
        mu,
        mu_err,
        dm_interp=dm_interp,
        dmi_values=dmi_values,
    )

    if max_rows is not None:
        df = df.sample(n=min(int(max_rows), len(df)), random_state=42)
    if sort_by is not None:
        if sort_by not in df.columns:
            raise KeyError(f"sort_by column {sort_by!r} is not present in AGN table data.")
        df = df.sort_values(sort_by, ascending=ascending)

    lines = [
        r"\begin{tabular}{@{}lccccccccccccc@{}}",
        r"\hline\hline",
        r"\textbf{SDSS Name} & RA & Dec & $z$ & $m_{2500}$ & $m_{2500}^{\mathrm{uncorr}}$ & $\alpha_\lambda$ & $f_{\rm AGN,5100}$ & $E(B-V)_{\rm AGN}$ & $E(B-V)_{\rm Gal}$ & $\mu$ & $\log\tau_{\mathrm{UV,RF}}$ & $\log\sigma_{\mathrm{UV}}$ & $\mathrm{Cov}(\log\sigma_{\mathrm{UV}},\,\log\tau_{\mathrm{UV,RF}})$ \\",
        r"& (deg) & (deg) &  & (mag) & (mag) &  &  &  &  & (mag) & (days) & (mag) &  \\",
        r"\hline",
    ]

    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    _name_to_bold(row["sdss_name"]),
                    _fmt_num(row["ra"], 4),
                    _fmt_signed_num(row["dec"], 4),
                    _fmt_with_sym_err(row, "z", 4, 4),
                    _fmt_with_sym_err(row, "apparent_mag_2500_corr", 2, 2),
                    _fmt_with_sym_err(row, "apparent_mag_2500", 2, 2),
                    _fmt_with_sym_err(row, "pl_slope", 2, 2),
                    _fmt_with_sym_err(row, "fracAGN_5100_fit", 2, 2),
                    _fmt_with_sym_err(row, "ebv_agn", 3, 3),
                    _fmt_with_sym_err(row, "ebv_gal", 3, 3),
                    _fmt_with_sym_err(row, "mu", 2, 2),
                    _fmt_with_sym_err(row, "log_tau_uv_rf", 2, 2, err_col="log_tau_uv_rf_std_psd"),
                    _fmt_with_sym_err(row, "log_sigma_uv", 2, 2, err_col="log_sigma_uv_std_psd"),
                    _fmt_num(row["log_sigma_uv_log_tau_uv_rf_cov_psd"], 3),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
        ]
    )

    latex_str = "\n".join(lines)
    os.makedirs(write_path, exist_ok=True)
    out_path = os.path.join(write_path, "agn_table.tex")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(latex_str)
    return latex_str
