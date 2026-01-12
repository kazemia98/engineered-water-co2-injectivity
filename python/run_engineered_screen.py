# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 09:22:48 2026

@author: 96898
"""

"""
Batch PHREEQC runner for engineered-water screening (constant T and log pCO2).

Inputs required in the SAME folder:
  - baseline_template.phr   (template with {{...}} placeholders)
  - (optional) runs/ folder (created automatically if missing)

Outputs:
  runs/engineered_screen/
    - cases/<case_id>/...  (PHREEQC inputs/outputs per case)
    - master_kpis.csv
    - pareto_storage_vs_risk.png
    - top_cases.csv

Edit only:
  PHREEQC_EXE, DATABASE
  scenario grid in make_scenarios()
"""


import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# EDIT THESE TWO PATHS
# =========================
PHREEQC_EXE = r"C:\Program Files\USGS\phreeqc-3.7.3-15968-x64\bin\Release\phreeqc.exe"
DATABASE    = r"C:\Program Files\USGS\phreeqc-3.7.3-15968-x64\database\llnl.dat"
# =========================


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "baseline_template.phr"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)


# ---------- Fixed constants for this study ----------
FIXED_TEMP_C = 75
FIXED_LOG_PCO2 = 1.91

# Main screening horizon (years)
TOTAL_YEARS = 1000
N_STEPS = 1000  # 1 year/step; fast and stable for screening


# ---------- Base case chemistry (mg/L) ----------
BASE = dict(
    Al_mgL=0.002,
    Alkalinity_mgL_asCaCO3=427.0,
    Ca_mgL=148.0,
    Cl_mgL=2485.0,
    Fe_mgL=1.0,
    K_mgL=35.2,
    Mg_mgL=55.9,
    Na_mgL=1794.0,
    Si_mgL_asSiO2=41.4,
    S6_mgL_asSO4=633.6,
    S_2_mgL_asHS=11.9,
)

# ---------- Cost weights (dimensionless proxy) ----------
COST_W = dict(
    alk=3.0,    # alkalinity chemicals are costly
    si=2.0,     # lowering silica via blending/pretreatment is costly
    sal=2.0,    # lowering salinity via blending is costly
    ca=1.0,
    mg=1.0,
    so4=0.7
)


def preflight() -> None:
    print("BASE_DIR:", BASE_DIR)
    print("Template exists:", TEMPLATE_FILE.exists(), "-", TEMPLATE_FILE)
    print("PHREEQC_EXE exists:", Path(PHREEQC_EXE).exists())
    print("DATABASE exists:", Path(DATABASE).exists())
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE_FILE}")
    if not Path(PHREEQC_EXE).exists():
        raise FileNotFoundError(f"PHREEQC executable not found: {PHREEQC_EXE}")
    if not Path(DATABASE).exists():
        raise FileNotFoundError(f"Database not found: {DATABASE}")


def render_template(text: str, params: dict) -> str:
    for k, v in params.items():
        text = text.replace("{{" + k + "}}", str(v))
    if "{{" in text or "}}" in text:
        leftovers = [ln for ln in text.splitlines() if "{{" in ln or "}}" in ln]
        raise ValueError("Unreplaced placeholders remain:\n" + "\n".join(leftovers[:40]))
    return text


# Run PHREEQC (skip if already completed)
def run_phreeqc(input_path: Path, out_path: Path) -> None:
    cmd = [PHREEQC_EXE, str(input_path), str(out_path), str(DATABASE)]
    subprocess.run(cmd, check=True)

    
def read_sel(sel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sel_path, delim_whitespace=True, comment="#")
    # detect time column
    time_col = None
    for c in df.columns:
        if "time" in c.strip().lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No time column found in {sel_path.name}. Columns: {list(df.columns)}")

    df = df.rename(columns={time_col: "time_s"})
    df["time_years"] = df["time_s"] / (3600 * 24 * 365.25)
    return df


def value_at_time(df: pd.DataFrame, col: str, years: float):
    if col not in df.columns:
        return None
    # nearest time row
    idx = (df["time_years"] - years).abs().idxmin()
    return float(df.loc[idx, col])


def max_over_window(df: pd.DataFrame, col: str, years_max: float):
    if col not in df.columns:
        return None
    sub = df[df["time_years"] <= years_max]
    if sub.empty:
        return None
    return float(sub[col].max())


def compute_cost_index(m_alk, m_si, m_sal, m_ca, m_mg, m_so4) -> float:
    w = COST_W
    return (
        w["alk"] * (m_alk - 1.0) ** 2 +
        w["si"]  * (1.0 - m_si) ** 2 +
        w["sal"] * (1.0 - m_sal) ** 2 +
        w["ca"]  * (m_ca - 1.0) ** 2 +
        w["mg"]  * (m_mg - 1.0) ** 2 +
        w["so4"] * (m_so4 - 1.0) ** 2
    )


def compute_objective(dic_1000y, risk_si_early, cost_index,
                      w_storage=1.0, w_risk=2.0, w_cost=0.5) -> float:
    """
    Higher objective is better:
      maximize storage proxy (DIC at 1000y),
      minimize early scaling risk (max SI chalcedony/calcite/etc.),
      minimize cost proxy.
    """
    if dic_1000y is None:
        return float("-inf")
    if risk_si_early is None:
        risk_si_early = 0.0
    return w_storage * dic_1000y - w_risk * max(0.0, risk_si_early) - w_cost * cost_index


def make_scenarios() -> list[dict]:
    """
    Define engineered-water design space.
    Keep constant T and log pCO2; only modify brine chemistry multipliers.

    You can expand levels freely; PHREEQC runs fast.
    """
    m_alk_levels = [1.0, 1.5, 2.0, 2.5, 3.0]
    m_si_levels  = [1.0, 0.75, 0.5, 0.25]
    m_sal_levels = [1.0, 0.85, 0.7, 0.55]     # applies to Na and Cl
    m_ca_levels  = [1.0, 1.25, 1.5]
    m_mg_levels  = [1.0, 0.85, 0.7]
    m_so4_levels = [1.0, 1.5]                 # optional; remove if you want fewer cases

    scenarios = []
    case_id = 0
    for m_alk in m_alk_levels:
        for m_si in m_si_levels:
            for m_sal in m_sal_levels:
                for m_ca in m_ca_levels:
                    for m_mg in m_mg_levels:
                        for m_so4 in m_so4_levels:
                            case_id += 1
                            scenarios.append(dict(
                                case_id=f"case_{case_id:04d}",
                                m_alk=m_alk,
                                m_si=m_si,
                                m_sal=m_sal,
                                m_ca=m_ca,
                                m_mg=m_mg,
                                m_so4=m_so4,
                            ))
    return scenarios


def build_params_for_case(sc: dict, case_dir: Path) -> dict:
    """
    Convert scenario multipliers into PHREEQC template parameters.
    Note: We keep K, Fe, S(-2), Al constant in this first campaign for stability.
    """
    m_alk = sc["m_alk"]
    m_si  = sc["m_si"]
    m_sal = sc["m_sal"]
    m_ca  = sc["m_ca"]
    m_mg  = sc["m_mg"]
    m_so4 = sc["m_so4"]

    sel_path = case_dir / "Fully_Kinetic.sel"

    params = dict(
        # base chemistry with multipliers
        Al_mgL=BASE["Al_mgL"],
        Alkalinity_mgL_asCaCO3=BASE["Alkalinity_mgL_asCaCO3"] * m_alk,
        Ca_mgL=BASE["Ca_mgL"] * m_ca,
        Mg_mgL=BASE["Mg_mgL"] * m_mg,
        Na_mgL=BASE["Na_mgL"] * m_sal,
        Cl_mgL=BASE["Cl_mgL"] * m_sal,
        Fe_mgL=BASE["Fe_mgL"],
        K_mgL=BASE["K_mgL"],
        Si_mgL_asSiO2=BASE["Si_mgL_asSiO2"] * m_si,
        S6_mgL_asSO4=BASE["S6_mgL_asSO4"] * m_so4,
        S_2_mgL_asHS=BASE["S_2_mgL_asHS"],

        # constants
        LOG_PCO2=FIXED_LOG_PCO2,
        TOTAL_SECONDS=int(TOTAL_YEARS * 365.25 * 24 * 3600),
        N_STEPS=int(N_STEPS),

        # selected output file
        SEL_FILE=str(sel_path),
    )
    return params


def pareto_plot(df: pd.DataFrame, out_png: Path) -> None:
    """
    Storage proxy vs early risk proxy.
    """
    plt.figure()
    plt.scatter(df["risk_SI_chalcedony_0_100y"], df["DIC_1000y"], s=15)
    plt.xlabel("Early scaling risk (max SI_chalcedony, 0–100 y)")
    plt.ylabel("Storage proxy (Tot_C(4) at 1000 y)")
    plt.title("Pareto view: storage vs early scaling risk")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    preflight()

    root = RUNS_DIR / "engineered_screen"
    cases_root = root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    template_text = TEMPLATE_FILE.read_text(encoding="utf-8")

    scenarios = make_scenarios()
    print(f"Total scenarios: {len(scenarios)}")

    records = []

    for i, sc in enumerate(scenarios, start=1):
        case_id = sc["case_id"]
        case_dir = cases_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        input_path = case_dir / "input.phr"
        out_path = case_dir / "out.out"
        sel_path = case_dir / "Fully_Kinetic.sel"

        params = build_params_for_case(sc, case_dir)
        phr_text = render_template(template_text, params)
        input_path.write_text(phr_text, encoding="utf-8")

        # Run PHREEQC
        run_phreeqc(input_path, out_path)

        # Read selected output
        df = read_sel(sel_path)

        # --- KPIs ---
        # Storage proxy
        dic_1000 = None
        for c in df.columns:
            if c.strip().lower() in ("tot_c(4)", "c(4)"):
                dic_1000 = value_at_time(df, c, 1000.0)
                break

        # Early scaling risk: chalcedony
        risk_chal = None
        for c in df.columns:
            if c.strip().lower() == "si_chalcedony":
                risk_chal = max_over_window(df, c, 100.0)
                break

        # Optional: early calcite SI
        risk_calc = None
        for c in df.columns:
            if c.strip().lower() == "si_calcite":
                risk_calc = max_over_window(df, c, 100.0)
                break

        # pH stability
        ph_1000 = value_at_time(df, "pH", 1000.0) if "pH" in df.columns else None
        ph_min_0_100 = None
        if "pH" in df.columns:
            sub = df[df["time_years"] <= 100.0]
            if not sub.empty:
                ph_min_0_100 = float(sub["pH"].min())

        # Cost index
        cost_index = compute_cost_index(
            sc["m_alk"], sc["m_si"], sc["m_sal"], sc["m_ca"], sc["m_mg"], sc["m_so4"]
        )

        # Define a single “risk” metric for objective (you can extend it)
        # Here: risk = max(0, SI_chalcedony early) + 0.5*max(0, SI_calcite early)
        risk_combined = 0.0
        if risk_chal is not None:
            risk_combined += max(0.0, risk_chal)
        if risk_calc is not None:
            risk_combined += 0.5 * max(0.0, risk_calc)

        # Objective (higher is better)
        obj = compute_objective(dic_1000, risk_combined, cost_index)

        rec = dict(
            case_id=case_id,
            m_alk=sc["m_alk"],
            m_si=sc["m_si"],
            m_sal=sc["m_sal"],
            m_ca=sc["m_ca"],
            m_mg=sc["m_mg"],
            m_so4=sc["m_so4"],
            CostIndex=cost_index,
            DIC_1000y=dic_1000,
            risk_SI_chalcedony_0_100y=risk_chal,
            risk_SI_calcite_0_100y=risk_calc,
            risk_combined=risk_combined,
            pH_1000y=ph_1000,
            pH_min_0_100y=ph_min_0_100,
            Objective=obj,
        )
        records.append(rec)

        if i % 20 == 0:
            print(f"Completed {i}/{len(scenarios)} cases...")

    # Master KPI table
    kpi_df = pd.DataFrame(records)
    root.mkdir(parents=True, exist_ok=True)
    master_csv = root / "master_kpis.csv"
    kpi_df.to_csv(master_csv, index=False)

    # Filter out failed cases (if any)
    valid = kpi_df.dropna(subset=["DIC_1000y"]).copy()

    # Pareto-style plot
    pareto_png = root / "pareto_storage_vs_risk.png"
    if not valid.empty:
        pareto_plot(valid, pareto_png)

    # Export top cases by objective
    top = valid.sort_values("Objective", ascending=False).head(25)
    top_csv = root / "top_cases.csv"
    top.to_csv(top_csv, index=False)

    print("\nDONE.")
    print("Master KPIs:", master_csv)
    print("Top cases:", top_csv)
    print("Pareto plot:", pareto_png)
    print("Cases folder:", cases_root)


if __name__ == "__main__":
    main()
