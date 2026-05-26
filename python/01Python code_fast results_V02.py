# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PHREEQC_EXE = r"C:\Program Files\USGS\phreeqc-3.7.3-15968-x64\bin\Release\phreeqc.exe"
DATABASE    = r"C:\Program Files\USGS\phreeqc-3.7.3-15968-x64\database\llnl.dat"

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "baseline_template.phr"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

FIXED_TEMP_C = 75
FIXED_LOG_PCO2 = 1.91

TOTAL_YEARS = 1000
N_STEPS = 300

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

COST_W = dict(
    alk=3.0,
    si=2.0,
    sal=2.0,
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


def run_phreeqc(input_path: Path, out_path: Path) -> None:
    cmd = [PHREEQC_EXE, str(input_path), str(out_path), str(DATABASE)]
    subprocess.run(cmd, check=True)


def read_sel(sel_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sel_path, delim_whitespace=True, comment="#")

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


def normalize_series(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to the range [0,1].
    """
    if s.max() == s.min():
        return s * 0.0

    return (s - s.min()) / (s.max() - s.min())


def make_scenarios():
    """
    Fast screening design: 72 cases total.

    Total cases = 3 × 3 × 2 × 2 × 2 × 1 = 72
    """
    m_alk_levels = [1.0, 2.0, 3.0]
    m_si_levels  = [1.0, 0.5, 0.25]
    m_sal_levels = [1.0, 0.7]
    m_ca_levels  = [1.0, 1.5]
    m_mg_levels  = [1.0, 0.7]
    m_so4_levels = [1.0]

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
    m_alk = sc["m_alk"]
    m_si  = sc["m_si"]
    m_sal = sc["m_sal"]
    m_ca  = sc["m_ca"]
    m_mg  = sc["m_mg"]
    m_so4 = sc["m_so4"]

    sel_path = case_dir / "Fully_Kinetic.sel"

    params = dict(
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

        LOG_PCO2=FIXED_LOG_PCO2,
        TOTAL_SECONDS=int(TOTAL_YEARS * 365.25 * 24 * 3600),
        N_STEPS=int(N_STEPS),

        SEL_FILE=str(sel_path),
    )

    return params


def pareto_plot(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure()
    plt.scatter(df["risk_SI_chalcedony_0_100y"], df["DIC_1000y"], s=15)
    plt.xlabel("Early scaling risk: max SI_chalcedony, 0–100 years")
    plt.ylabel("Storage proxy: DIC at 1000 years")
    plt.title("Pareto view: storage readiness versus early scaling risk")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
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

        run_phreeqc(input_path, out_path)

        df = read_sel(sel_path)

        # Storage proxy: DIC at 1000 years
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

        # Early scaling risk: calcite
        risk_calc = None
        for c in df.columns:
            if c.strip().lower() == "si_calcite":
                risk_calc = max_over_window(df, c, 100.0)
                break

        # pH indicators
        ph_1000 = value_at_time(df, "pH", 1000.0) if "pH" in df.columns else None

        ph_min_0_100 = None
        if "pH" in df.columns:
            sub = df[df["time_years"] <= 100.0]
            if not sub.empty:
                ph_min_0_100 = float(sub["pH"].min())

        # Conditioning effort
        cost_index = compute_cost_index(
            sc["m_alk"],
            sc["m_si"],
            sc["m_sal"],
            sc["m_ca"],
            sc["m_mg"],
            sc["m_so4"]
        )

        # Combined early scaling risk
        risk_combined = 0.0

        if risk_chal is not None:
            risk_combined += max(0.0, risk_chal)

        if risk_calc is not None:
            risk_combined += 0.5 * max(0.0, risk_calc)

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
        )

        records.append(rec)

        if i % 20 == 0:
            print(f"Completed {i}/{len(scenarios)} cases...")

    # Master KPI table
    kpi_df = pd.DataFrame(records)
    root.mkdir(parents=True, exist_ok=True)

    master_csv = root / "master_kpis.csv"
    kpi_df.to_csv(master_csv, index=False)

    # Filter valid cases
    valid = kpi_df.dropna(subset=["DIC_1000y"]).copy()

    # =========================
    # Sustainability Index, SISUS
    # =========================
    valid["DIC_norm"] = normalize_series(valid["DIC_1000y"])
    valid["Risk_norm"] = normalize_series(valid["risk_combined"])
    valid["Cost_norm"] = normalize_series(valid["CostIndex"])

    epsilon = 1e-9

    valid["SISUS"] = (
        valid["DIC_norm"] /
        (valid["Risk_norm"] + valid["Cost_norm"] + epsilon)
    )

    # Save updated KPI table with normalized quantities and SISUS
    master_csv_sisus = root / "master_kpis_with_SISUS.csv"
    valid.to_csv(master_csv_sisus, index=False)

    # Pareto-style plot
    pareto_png = root / "pareto_storage_vs_risk.png"
    if not valid.empty:
        pareto_plot(valid, pareto_png)

    # Export top cases by SISUS
    top = valid.sort_values("SISUS", ascending=False).head(25)
    top_csv = root / "top_cases_by_SISUS.csv"
    top.to_csv(top_csv, index=False)

    print("\nDONE.")
    print("Master KPIs:", master_csv)
    print("Master KPIs with SISUS:", master_csv_sisus)
    print("Top cases by SISUS:", top_csv)
    print("Pareto plot:", pareto_png)
    print("Cases folder:", cases_root)


if __name__ == "__main__":
    main()