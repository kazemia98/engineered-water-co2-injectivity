from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def pareto_front_min2(df, x_col, y_col):
    d = df[[x_col, y_col]].dropna().copy()
    d = d.sort_values(x_col, ascending=True).reset_index(drop=True)

    is_eff = []
    best_y = float("inf")

    for _, row in d.iterrows():
        y = float(row[y_col])
        if y < best_y:
            is_eff.append(True)
            best_y = y
        else:
            is_eff.append(False)

    d["is_pareto"] = is_eff
    return d[d["is_pareto"]].copy()


def plot_cost_risk_pareto(df_all, df_pareto, x_col, y_col, out_png):
    plt.figure(figsize=(8, 5.5))

    plt.scatter(
        df_all[x_col],
        df_all[y_col],
        s=35,
        alpha=0.35,
        label="All scenarios"
    )

    plt.scatter(
        df_pareto[x_col],
        df_pareto[y_col],
        s=80,
        color="orange",
        edgecolor="black",
        linewidth=1.0,
        label="Pareto front",
        zorder=3
    )

    plt.annotate(
        "Lower cost\nHigher scaling tendency",
        xy=(0.45, 0.755),
        xytext=(1.8, 0.84),
        arrowprops=dict(arrowstyle="->", linewidth=1.0),
        fontsize=10
    )

    plt.xlabel("Conditioning CostIndex (lower is better)")
    plt.ylabel("Composite early-time scaling tendency")
    plt.title("Pareto front: conditioning cost vs early-time scaling tendency")

    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    base_dir = Path(__file__).resolve().parent
    root = base_dir / "runs" / "engineered_screen"

    master_csv = root / "master_kpis.csv"

    if not master_csv.exists():
        raise FileNotFoundError(f"master_kpis.csv not found in: {root}")

    df = pd.read_csv(master_csv)
    df.columns = df.columns.str.strip()

    if "risk_combined" in df.columns:
        risk_col = "risk_combined"
    elif "risk_comb" in df.columns:
        risk_col = "risk_comb"
    else:
        raise KeyError(
            "No risk column found. Expected 'risk_combined' or 'risk_comb'. "
            f"Available columns: {df.columns.tolist()}"
        )

    df_valid = df.dropna(subset=["CostIndex", risk_col]).copy()

    pareto_xy = pareto_front_min2(df_valid, "CostIndex", risk_col)

    pareto_full = df_valid.merge(
        pareto_xy[["CostIndex", risk_col]],
        on=["CostIndex", risk_col],
        how="inner"
    ).drop_duplicates()

    pareto_csv = root / "pareto_cost_risk.csv"
    pareto_full.to_csv(pareto_csv, index=False)

    out_png = root / "pareto_cost_vs_risk.png"
    plot_cost_risk_pareto(df_valid, pareto_full, "CostIndex", risk_col, out_png)

    print("DONE.")
    print("Master:", master_csv)
    print("Pareto CSV:", pareto_csv)
    print("Pareto plot:", out_png)
    print("Risk column used:", risk_col)
    print(f"Pareto points: {len(pareto_full)} out of {len(df_valid)} cases")


if __name__ == "__main__":
    main()