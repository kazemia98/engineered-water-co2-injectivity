import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your output folder
root = Path("runs/engineered_screen")

# Use the file that includes SISUS
df = pd.read_csv(root / "master_kpis_with_SISUS.csv")

# Design variables actually varied
variables = ["m_alk", "m_si", "m_sal", "m_ca", "m_mg"]

# KPIs including SISUS
kpis = [
    "DIC_1000y",
    "risk_SI_chalcedony_0_100y",
    "risk_SI_calcite_0_100y",
    "risk_combined",
    "CostIndex",
    "SISUS"
]

# Clean labels for the paper figure
variable_labels = {
    "m_alk": "Alkalinity",
    "m_si": "Silica",
    "m_sal": "Salinity",
    "m_ca": "Calcium",
    "m_mg": "Magnesium"
}

kpi_labels = {
    "DIC_1000y": "DIC (1000 y)",
    "risk_SI_chalcedony_0_100y": "Chalcedony risk",
    "risk_SI_calcite_0_100y": "Calcite risk",
    "risk_combined": "Combined risk",
    "CostIndex": "CostIndex",
    "SISUS": "Sustainability Index"
}

# Correlation-based sensitivity
sens = df[variables + kpis].corr().loc[variables, kpis]

# Plot heatmap
plt.figure(figsize=(10, 5))
plt.imshow(sens, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation coefficient")

plt.xticks(
    range(len(kpis)),
    [kpi_labels[k] for k in kpis],
    rotation=35,
    ha="right"
)

plt.yticks(
    range(len(variables)),
    [variable_labels[v] for v in variables]
)

plt.title("Correlation-based sensitivity analysis of engineered-water screening KPIs")

# Add values inside cells
for i in range(len(variables)):
    for j in range(len(kpis)):
        value = sens.iloc[i, j]
        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(root / "sensitivity_heatmap_SISUS.png", dpi=300, bbox_inches="tight")
plt.show()