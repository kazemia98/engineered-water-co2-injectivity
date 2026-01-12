# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:37:55 2026

@author: 96898
"""

# postprocess_early_injectivity.py
# Computes early-time (0–10 y) injectivity KPIs and a Sustainability Index (SI_SUS)
# from existing PHREEQC selected output (*.sel) files. No PHREEQC reruns needed.

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User settings
# -----------------------------
EARLY_YEARS = 10.0              # early-time window for injectivity risk
SI_THRESHOLD = 0.0              # supersaturation threshold
EPS_STORAGE = 0.01              # allowable storage drop vs baseline for "ready" (1%)

# weights for risk aggregation (edit if desired)
W_CHAL = 1.0
W_CALC = 0.5
W_ANH = 0.25   # only used if si_anhydrite exists

# weights for Sustainability Index (after normalization)
ALPHA_RISK = 1.0
BETA_COST = 1.0


# -----------------------------
# Helpers
# -----------------------------
def read_sel(sel_path: Path) -> pd.DataFrame:
    """Read PHREEQC selected output (.sel) written with whitespace delimiter."""
    df = pd.read_csv(sel_path, delim_whitespace=True, comment="#")
    df.columns = df.columns.str.strip()

    # detect time column
    time_col = None
    for c in df.columns:
        if "time" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column found in {sel_path.name}. Columns: {list(df.columns)}")

    df = df.rename(columns={time_col: "time_s"})
    df["time_years"] = df["time_s"] / (3600 * 24 * 365.25)
    return df


def area_over_threshold(df: pd.DataFrame, col: str, years_max: float, thr: float = 0.0) :

    """
    Compute integral over time of max(0, value-thr) from t=0 to t=years_max.
    Uses trapezoidal rule in YEARS.
    """
    if col not in df.columns:
        return np.nan

    sub = df[df["time_years"] <= years_max].copy()
    if sub.empty:
        return np.nan

    t = sub["time_years"].to_numpy()
    y = sub[col].to_numpy()
    y_excess = np.maximum(0.0, y - thr)

    # if only one point, integral is 0
    if len(t) < 2:
        return float(0.0)

    return float(np.trapz(y_excess, t))


def time_above_threshold(df: pd.DataFrame, col: str, years_max: float, thr: float = 0.0) :
    """
    Approximate duration (years) where col > thr between 0 and years_max.
    Uses discrete time steps; assumes piecewise constant over each interval.
    """
    if col not in df.columns:
        return np.nan

    sub = df[df["time_years"] <= years_max].copy()
    if sub.empty or len(sub) < 2:
        return np.nan

    t = sub["time_years"].to_numpy()
    y = sub[col].to_numpy()

    dt = np.diff(t)
    above = (y[:-1] > thr).astype(float)
    return float(np.sum(dt * above))


def max_in_window(df: pd.DataFrame, col: str, years_max: float) :
    if col not in df.columns:
        return np.nan
    sub = df[df["time_years"] <= years_max]
    if sub.empty:
        return np.nan
    return float(sub[col].max())


def min_in_window(df: pd.DataFrame, col: str, years_max: float) :
    if col not in df.columns:
        return np.nan
    sub = df[df["time_years"] <= years_max]
    if sub.empty:
        return np.nan
    return float(sub[col].min())


def value_near_time(df: pd.DataFrame, col: str, years: float):
    if col not in df.columns:
        return np.nan
    idx = (df["time_years"] - years).abs().idxmin()
    return float(df.loc[idx, col])


def resolve_col(df: pd.DataFrame, candidates: list[str]) :
    """Return the first matching column name (case-insensitive), else None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0,1]. If constant, return zeros."""
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# -----------------------------
# Main
# -----------------------------
def main():
    root = Path(__file__).resolve().parent
    master_csv = root / "master_kpis.csv"
    cases_root = root / "cases"

    if not master_csv.exists():
        raise FileNotFoundError(f"master_kpis.csv not found in: {root}")
    if not cases_root.exists():
        raise FileNotFoundError(f"cases folder not found: {cases_root}")

    df_master = pd.read_csv(master_csv)
    df_master.columns = df_master.columns.str.strip()

    # Resolve storage column name
    storage_col = None
    for c in df_master.columns:
        if c.strip().lower() in ("dic_1000y", "tot_c4_1000y", "tot_c(4)_1000y"):
            storage_col = c
            break
    if storage_col is None:
        # fallback: use any column containing 'dic' and '1000'
        for c in df_master.columns:
            if "dic" in c.lower() and "1000" in c.lower():
                storage_col = c
                break

    if storage_col is None:
        raise KeyError(f"Could not find DIC_1000y column in master_kpis.csv. Columns: {df_master.columns.tolist()}")

    if "CostIndex" not in df_master.columns:
        raise KeyError("CostIndex not found in master_kpis.csv")

    # Baseline DIC: choose the minimum CostIndex case as baseline proxy (often baseline has CostIndex=0)
    baseline_row = df_master.sort_values("CostIndex", ascending=True).iloc[0]
    DIC_base = float(baseline_row[storage_col])

    records = []

    # Iterate over cases listed in master
    for _, row in df_master.iterrows():
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue

        sel_path = cases_root / case_id / "Fully_Kinetic.sel"
        if not sel_path.exists():
            # if missing, skip but keep a placeholder row
            rec = dict(case_id=case_id, sel_exists=False)
            records.append(rec)
            continue

        df_sel = read_sel(sel_path)

        # Resolve SI columns (your .sel uses these names typically)
        col_chal = resolve_col(df_sel, ["si_chalcedony"])
        col_calc = resolve_col(df_sel, ["si_calcite"])
        col_anh  = resolve_col(df_sel, ["si_anhydrite"])
        col_ph   = resolve_col(df_sel, ["pH"])

        # Compute early-window KPIs (0–10y)
        chal_max_10 = max_in_window(df_sel, col_chal, EARLY_YEARS) if col_chal else np.nan
        calc_max_10 = max_in_window(df_sel, col_calc, EARLY_YEARS) if col_calc else np.nan
        anh_max_10  = max_in_window(df_sel, col_anh,  EARLY_YEARS) if col_anh  else np.nan

        chal_Tpos_10 = time_above_threshold(df_sel, col_chal, EARLY_YEARS, SI_THRESHOLD) if col_chal else np.nan
        calc_Tpos_10 = time_above_threshold(df_sel, col_calc, EARLY_YEARS, SI_THRESHOLD) if col_calc else np.nan
        anh_Tpos_10  = time_above_threshold(df_sel, col_anh,  EARLY_YEARS, SI_THRESHOLD) if col_anh  else np.nan

        chal_Apos_10 = area_over_threshold(df_sel, col_chal, EARLY_YEARS, SI_THRESHOLD) if col_chal else np.nan
        calc_Apos_10 = area_over_threshold(df_sel, col_calc, EARLY_YEARS, SI_THRESHOLD) if col_calc else np.nan
        anh_Apos_10  = area_over_threshold(df_sel, col_anh,  EARLY_YEARS, SI_THRESHOLD) if col_anh  else np.nan

        ph_min_10 = min_in_window(df_sel, col_ph, EARLY_YEARS) if col_ph else np.nan
        ph_10 = value_near_time(df_sel, col_ph, EARLY_YEARS) if col_ph else np.nan

        rec = dict(
            case_id=case_id,
            sel_exists=True,
            SI_chal_max_0_10y=chal_max_10,
            SI_calc_max_0_10y=calc_max_10,
            SI_anh_max_0_10y=anh_max_10,
            T_SI_chal_pos_0_10y=chal_Tpos_10,
            T_SI_calc_pos_0_10y=calc_Tpos_10,
            T_SI_anh_pos_0_10y=anh_Tpos_10,
            A_SI_chal_pos_0_10y=chal_Apos_10,
            A_SI_calc_pos_0_10y=calc_Apos_10,
            A_SI_anh_pos_0_10y=anh_Apos_10,
            pH_min_0_10y=ph_min_10,
            pH_10y=ph_10,
        )
        records.append(rec)

    df_early = pd.DataFrame(records)

    # Merge early KPIs back to master
    df_all = df_master.merge(df_early, on="case_id", how="left")

    # Build injectivity risk metric using area-over-threshold (preferred)
    # If calcite is missing, its term becomes 0; same for anhydrite.
    df_all["RiskInj_A_0_10y"] = (
        W_CHAL * df_all["A_SI_chal_pos_0_10y"].fillna(0.0) +
        W_CALC * df_all["A_SI_calc_pos_0_10y"].fillna(0.0) +
        W_ANH  * df_all["A_SI_anh_pos_0_10y"].fillna(0.0)
    )

    # Storage readiness term (ratio)
    df_all["StorageReadyRatio"] = df_all[storage_col] / DIC_base

    # Readiness flag (constraint)
    df_all["StorageReady_OK"] = df_all["StorageReadyRatio"] >= (1.0 - EPS_STORAGE)

    # Normalize risk and cost for index building
    df_all["RiskInj_norm"] = normalize_series(df_all["RiskInj_A_0_10y"].fillna(df_all["RiskInj_A_0_10y"].max()))
    df_all["Cost_norm"] = normalize_series(df_all["CostIndex"].astype(float))

    # Sustainability Index (higher is better)
    # If storage readiness fails, apply a penalty multiplier (hard constraint style)
    penalty = np.where(df_all["StorageReady_OK"], 1.0, 0.25)  # strong penalty if storage drops too much
    df_all["SI_SUS"] = penalty * (df_all["StorageReadyRatio"]) / (1.0 + ALPHA_RISK * df_all["RiskInj_norm"] + BETA_COST * df_all["Cost_norm"])

    # Export results
    out_csv = root / "master_kpis_with_early10y_and_SI.csv"
    df_all.to_csv(out_csv, index=False)

    # Plots
    plots_dir = root / "plots_postprocess"
    plots_dir.mkdir(exist_ok=True)

    # 1) Cost vs early injectivity risk (area-based)
    plt.figure()
    plt.scatter(df_all["CostIndex"], df_all["RiskInj_A_0_10y"], s=18)
    plt.xlabel("CostIndex (proxy) — lower is better")
    plt.ylabel("Injectivity risk (0–10y) — A_SI>0 area metric")
    plt.title("Cost vs Early Injectivity Risk (0–10 years)")
    plt.tight_layout()
    plt.savefig(plots_dir / "cost_vs_riskinj_A_0_10y.png", dpi=200)
    plt.close()

    # 2) SI_SUS ranking scatter (Cost vs SI_SUS)
    plt.figure()
    plt.scatter(df_all["CostIndex"], df_all["SI_SUS"], s=18)
    plt.xlabel("CostIndex (proxy)")
    plt.ylabel("Sustainability Index (SI_SUS) — higher is better")
    plt.title("Sustainability Index vs Conditioning Cost")
    plt.tight_layout()
    plt.savefig(plots_dir / "SI_SUS_vs_cost.png", dpi=200)
    plt.close()

    # 3) Top 15 cases by SI_SUS
    top = df_all.sort_values("SI_SUS", ascending=False).head(15)
    top_csv = root / "top15_by_SI_SUS.csv"
    top.to_csv(top_csv, index=False)

    print("DONE.")
    print("Input master:", master_csv)
    print("Output master + early KPIs:", out_csv)
    print("Top 15 by SI_SUS:", top_csv)
    print("Plots folder:", plots_dir)
    print(f"Baseline DIC used ({storage_col}) =", DIC_base)


if __name__ == "__main__":
    main()
