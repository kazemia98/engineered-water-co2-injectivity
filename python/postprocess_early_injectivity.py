# postprocess_early_injectivity.py
# Computes early-time (0–10 y) injectivity KPIs and Sustainability Index (SI_SUS)
# NO FIGURE GENERATION (data-only postprocessing)

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# User settings
# -----------------------------
EARLY_YEARS = 10.0
SI_THRESHOLD = 0.0
EPS_STORAGE = 0.01

W_CHAL = 1.0
W_CALC = 0.5
W_ANH = 0.25

ALPHA_RISK = 1.0
BETA_COST = 1.0

# -----------------------------
# Helpers
# -----------------------------
def read_sel(sel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sel_path, delim_whitespace=True, comment="#")
    df.columns = df.columns.str.strip()

    time_col = next((c for c in df.columns if "time" in c.lower()), None)
    if time_col is None:
        raise ValueError(f"No time column in {sel_path.name}")

    df = df.rename(columns={time_col: "time_s"})
    df["time_years"] = df["time_s"] / (3600 * 24 * 365.25)
    return df


def area_over_threshold(df, col, years_max, thr=0.0):
    if col not in df.columns:
        return np.nan
    sub = df[df["time_years"] <= years_max]
    if len(sub) < 2:
        return 0.0
    y = np.maximum(0.0, sub[col] - thr)
    return float(np.trapz(y, sub["time_years"]))


def time_above_threshold(df, col, years_max, thr=0.0):
    if col not in df.columns:
        return np.nan
    sub = df[df["time_years"] <= years_max]
    if len(sub) < 2:
        return np.nan
    dt = np.diff(sub["time_years"])
    above = (sub[col].values[:-1] > thr).astype(float)
    return float(np.sum(dt * above))


def max_in_window(df, col, years_max):
    return float(df[df["time_years"] <= years_max][col].max()) if col in df.columns else np.nan


def min_in_window(df, col, years_max):
    return float(df[df["time_years"] <= years_max][col].min()) if col in df.columns else np.nan


def value_near_time(df, col, years):
    if col not in df.columns:
        return np.nan
    return float(df.loc[(df["time_years"] - years).abs().idxmin(), col])


def resolve_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def normalize_series(s):
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# -----------------------------
# Main
# -----------------------------
def main():
    root = Path(__file__).resolve().parent
    master_csv = root / "master_kpis.csv"
    cases_root = root / "cases"

    df_master = pd.read_csv(master_csv)
    df_master.columns = df_master.columns.str.strip()

    storage_col = next(c for c in df_master.columns if "dic" in c.lower() and "1000" in c.lower())
    baseline = df_master.sort_values("CostIndex").iloc[0]
    DIC_base = baseline[storage_col]

    records = []

    for _, row in df_master.iterrows():
        case_id = row["case_id"]
        sel_path = cases_root / case_id / "Fully_Kinetic.sel"

        if not sel_path.exists():
            records.append({"case_id": case_id, "sel_exists": False})
            continue

        df = read_sel(sel_path)

        c_chal = resolve_col(df, ["si_chalcedony"])
        c_calc = resolve_col(df, ["si_calcite"])
        c_anh = resolve_col(df, ["si_anhydrite"])
        c_ph = resolve_col(df, ["ph"])

        records.append({
            "case_id": case_id,
            "sel_exists": True,
            "A_SI_chal_pos_0_10y": area_over_threshold(df, c_chal, EARLY_YEARS),
            "A_SI_calc_pos_0_10y": area_over_threshold(df, c_calc, EARLY_YEARS),
            "A_SI_anh_pos_0_10y": area_over_threshold(df, c_anh, EARLY_YEARS),
            "pH_min_0_10y": min_in_window(df, c_ph, EARLY_YEARS),
            "pH_10y": value_near_time(df, c_ph, EARLY_YEARS),
        })

    df_early = pd.DataFrame(records)
    df = df_master.merge(df_early, on="case_id", how="left")

    df["RiskInj_A_0_10y"] = (
        W_CHAL * df["A_SI_chal_pos_0_10y"].fillna(0) +
        W_CALC * df["A_SI_calc_pos_0_10y"].fillna(0) +
        W_ANH * df["A_SI_anh_pos_0_10y"].fillna(0)
    )

    df["StorageReadyRatio"] = df[storage_col] / DIC_base
    df["StorageReady_OK"] = df["StorageReadyRatio"] >= (1 - EPS_STORAGE)

    df["Risk_norm"] = normalize_series(df["RiskInj_A_0_10y"])
    df["Cost_norm"] = normalize_series(df["CostIndex"])

    penalty = np.where(df["StorageReady_OK"], 1.0, 0.25)
    df["SI_SUS"] = penalty * df["StorageReadyRatio"] / (
        1 + ALPHA_RISK * df["Risk_norm"] + BETA_COST * df["Cost_norm"]
    )

    df.to_csv(root / "master_kpis_with_early10y_and_SI.csv", index=False)
    df.sort_values("SI_SUS", ascending=False).head(15).to_csv(
        root / "top15_by_SI_SUS.csv", index=False
    )

    print("Postprocessing completed (no figures generated).")


if __name__ == "__main__":
    main()
