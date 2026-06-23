import sys, warnings
from pathlib import Path
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import shap

ROOT  = Path(__file__).resolve().parent
df    = pd.read_parquet(ROOT / "data" / "processed" / "dataset_modelado.parquet")
META  = ["split", "season", "date", "jornada", "home_team", "away_team", "fthg", "ftag"]
FEATS = [c for c in df.columns if c not in META + ["ftr"]]
X     = df[df.split == "test"].dropna(subset=FEATS)[FEATS]
model = joblib.load(ROOT / "models" / "xgboost_simple.pkl")

FIG = ROOT.parent / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT.parent / "reports" / "tables";  TAB.mkdir(parents=True, exist_ok=True)

exp  = shap.TreeExplainer(model)(X)
vals = exp.values
imp  = (np.abs(vals).mean(axis=0).mean(axis=1) if vals.ndim == 3
        else np.abs(vals).mean(axis=0))
imp_s = pd.Series(imp, index=FEATS).sort_values(ascending=False)
imp_s.to_csv(TAB / "t13_shap_importancia.csv", header=["importancia_shap"])
print("Top-12 variables por importancia SHAP (XGBoost sin pesos):")
print(imp_s.head(12).round(4).to_string())

top = imp_s.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color="#4477aa")
plt.xlabel("Importancia media |SHAP|"); plt.title("Importancia de variables (SHAP) - XGBoost sin pesos")
plt.tight_layout(); plt.savefig(FIG / "f13_shap_importancia.png", dpi=150); plt.close()

try:
    cls = ["A", "D", "H"].index("H")
    shap.plots.beeswarm(exp[:, :, cls], max_display=12, show=False)
    plt.title("Efecto SHAP sobre la probabilidad de victoria local (H)")
    plt.tight_layout(); plt.savefig(FIG / "f13_shap_beeswarm_H.png", dpi=150, bbox_inches="tight"); plt.close()
    print("\nbeeswarm clase H guardado")
except Exception as e:
    print("beeswarm omitido:", e)

print("\nFiguras en reports/figures/, tabla en reports/tables/t13_shap_importancia.csv")
