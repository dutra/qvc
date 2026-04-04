import math
import os

import numpy as np

from qvc.hubble.hubble_completeness_refactored import evaluate_dm_interp


REQUIRED_AGN_TABLE_COLUMNS = (
    "sdss_name",
    "ra",
    "dec",
    "z",
    "z_err",
    "apparent_mag_2500",
    "apparent_mag_2500_err",
    "PL_slope",
    "PL_slope_err",
    "log_tau_uv_rf",
    "log_tau_uv_rf_std_psd",
    "log_sigma_uv",
    "log_sigma_uv_std_psd",
    "log_sigma_uv_log_tau_uv_rf_cov_psd",
    "f_host_2500",
    "f_host_2500_err",
    "f_bc_3000",
    "f_bc_3000_err",
    "f_fe_uv_3000",
    "f_fe_uv_3000_err",
    "f_na",
    "f_na_err",
    "f_br",
    "f_br_err",
)


def make_agn_latex_table(
    agn_df,
    mu,
    mu_err,
    dm_interp,
    *,
    sort_by,
    ascending,
    max_rows,
    write_path,
) -> str:
    missing_cols = [col for col in REQUIRED_AGN_TABLE_COLUMNS if col not in agn_df.columns]
    if missing_cols:
        raise KeyError(f"AGN LaTeX table requires columns: {missing_cols}")
    if dm_interp is None:
        raise ValueError("AGN LaTeX table requires a non-None dm_interp.")

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

    df = agn_df.copy()
    df["mu"] = np.asarray(mu, dtype=float)
    df["mu_err"] = np.asarray(mu_err, dtype=float)

    dm_values = evaluate_dm_interp(
        dm_interp,
        df["z"],
        df["apparent_mag_2500"],
        f_host_2500=df["f_host_2500"] if "f_host_2500" in df.columns else None,
        alpha_lambda=df["alpha_lambda"] if "alpha_lambda" in df.columns else None,
    )
    if dm_values.shape != (len(df),):
        raise ValueError(
            f"dm_interp returned shape {dm_values.shape}, expected {(len(df),)}."
        )
    df["apparent_mag_2500_corr"] = np.asarray(df["apparent_mag_2500"], dtype=float) - dm_values
    df["apparent_mag_2500_corr_err"] = np.asarray(df["apparent_mag_2500_err"], dtype=float)
    df["f_lines"] = np.asarray(df["f_na"], dtype=float) + np.asarray(df["f_br"], dtype=float)
    df["f_lines_err"] = np.hypot(
        np.asarray(df["f_na_err"], dtype=float),
        np.asarray(df["f_br_err"], dtype=float),
    )

    if max_rows is not None:
        df = df.sample(n=min(int(max_rows), len(df)), random_state=42)
    if sort_by is not None:
        if sort_by not in df.columns:
            raise KeyError(f"sort_by column {sort_by!r} is not present in AGN table data.")
        df = df.sort_values(sort_by, ascending=ascending)

    lines = [
        r"\begin{tabular}{@{}lcccccccccccccc@{}}",
        r"\hline\hline",
        r"\textbf{SDSS Name} & RA & Dec & $z$ & $m_{2500}$ & $m_{2500}^{\mathrm{uncorr}}$ & \texttt{PL\_slope} & $\mu$ & $\log\tau_{\mathrm{UV,RF}}$ & $\log\sigma_{\mathrm{UV}}$ & $\mathrm{Cov}(\log\sigma_{\mathrm{UV}},\,\log\tau_{\mathrm{UV,RF}})$ & $f_{\rm{host,\,2500\,\text{\AA}}}$ & $f_{\rm{BC}}$ & $f_{\rm{lines}}$ & $f_{\rm{Fe\,II}}$ \\",
        r"& (deg) & (deg) &  & (mag) & (mag) &  & (mag) & (days) & (mag) &  &  &  &  &  \\",
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
                    _fmt_with_sym_err(row, "PL_slope", 2, 2, err_col="PL_slope_err"),
                    _fmt_with_sym_err(row, "mu", 2, 2),
                    _fmt_with_sym_err(row, "log_tau_uv_rf", 2, 2, err_col="log_tau_uv_rf_std_psd"),
                    _fmt_with_sym_err(row, "log_sigma_uv", 2, 2, err_col="log_sigma_uv_std_psd"),
                    _fmt_num(row["log_sigma_uv_log_tau_uv_rf_cov_psd"], 3),
                    _fmt_with_sym_err(row, "f_host_2500", 2, 2),
                    _fmt_with_sym_err(row, "f_bc_3000", 2, 2),
                    _fmt_with_sym_err(row, "f_lines", 2, 2),
                    _fmt_with_sym_err(row, "f_fe_uv_3000", 2, 2),
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
