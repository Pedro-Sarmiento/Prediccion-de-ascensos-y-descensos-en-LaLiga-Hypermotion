from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

LABELS    = ["A", "D", "H"]
MAX_GOALS = 8
PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "figures"

df = pd.read_parquet(DATA_PROCESSED / "dataset_modelado.parquet")
META_COLS    = ["split", "season", "date", "jornada", "home_team", "away_team", "fthg", "ftag"]
TARGET_COLS  = ["ftr"]
FEATURE_COLS = [c for c in df.columns if c not in META_COLS + TARGET_COLS]

train = df[df["split"] == "train"].copy()
test  = df[df["split"] == "test"].copy().dropna(subset=FEATURE_COLS).reset_index(drop=True)

X_test = test[FEATURE_COLS].astype(float)
y_test = test["ftr"].astype(str).values

scaler = StandardScaler().fit(train[FEATURE_COLS].dropna().astype(float))

modelos_sklearn = {
    "Doble Poisson":            None,
    "LogReg sin pesos":        (joblib.load(MODELS_DIR / "logistic_regression_simple.pkl"),    True),
    "LogReg balanced":         (joblib.load(MODELS_DIR / "logistic_regression_balanced.pkl"),  True),
    "Random Forest sin pesos": (joblib.load(MODELS_DIR / "random_forest_simple.pkl"),          False),
    "Random Forest balanced":  (joblib.load(MODELS_DIR / "random_forest_balanced.pkl"),        False),
    "XGBoost sin pesos":       (joblib.load(MODELS_DIR / "xgboost_simple.pkl"),                False),
    "XGBoost balanced":        (joblib.load(MODELS_DIR / "xgboost_balanced.pkl"),              False),
    "LightGBM sin pesos":      (joblib.load(MODELS_DIR / "lgb_simple.pkl"),                    False),
    "LightGBM balanced":       (joblib.load(MODELS_DIR / "lgb_balanced.pkl"),                  False),
    "MLP":                     (joblib.load(MODELS_DIR / "mlp_final.pkl"),                     True),
}

glm_home_dp = joblib.load(MODELS_DIR / "double_poisson" / "glm_home.joblib")
glm_away_dp = joblib.load(MODELS_DIR / "double_poisson" / "glm_away.joblib")
dp_scaler   = joblib.load(MODELS_DIR / "double_poisson" / "scaler.joblib")


def predict_dp(X: pd.DataFrame) -> np.ndarray:
    Xs = dp_scaler.transform(X)
    Xc = np.hstack([np.ones((len(Xs), 1)), Xs])
    lam_h = np.asarray(glm_home_dp.predict(Xc))
    lam_a = np.asarray(glm_away_dp.predict(Xc))
    goles  = np.arange(MAX_GOALS + 1)
    p_h    = poisson.pmf(goles[None, :], lam_h[:, None])
    p_a    = poisson.pmf(goles[None, :], lam_a[:, None])
    matriz = p_h[:, :, None] * p_a[:, None, :]
    i_idx, j_idx = np.meshgrid(goles, goles, indexing="ij")
    pH = (matriz * (i_idx > j_idx)).sum(axis=(1, 2))
    pD = (matriz * (i_idx == j_idx)).sum(axis=(1, 2))
    pA = (matriz * (i_idx < j_idx)).sum(axis=(1, 2))
    tot = pH + pD + pA
    return np.column_stack([pA / tot, pD / tot, pH / tot])     # orden A, D, H


def predicciones_modelo(nombre: str) -> np.ndarray:
    if nombre == "Doble Poisson":
        proba = predict_dp(X_test)
    else:
        mdl, scale = modelos_sklearn[nombre]
        Xv = scaler.transform(X_test) if scale else X_test.values
        proba = mdl.predict_proba(Xv)
    idx = proba.argmax(axis=1)
    return np.array(LABELS)[idx]

n_modelos = len(modelos_sklearn)
n_cols    = 5
n_rows    = int(np.ceil(n_modelos / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows))
axes = axes.flatten()

for i, nombre in enumerate(modelos_sklearn.keys()):
    y_pred = predicciones_modelo(nombre)
    cm     = confusion_matrix(y_test, y_pred, labels=LABELS)
    ax     = axes[i]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=LABELS, yticklabels=LABELS,
        annot_kws={"size": 11},
    )
    ax.set_title(nombre, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel("Predicho",   fontsize=10)
    ax.set_ylabel("Verdadero",  fontsize=10)
    ax.tick_params(axis="both", labelsize=10)

for j in range(n_modelos, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Figura 6.2 — Matrices de confusión sobre el conjunto de prueba "
    f"(n = {len(test):,}; orden A/D/H)",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()

out = OUTPUT_DIR / "figura_6_2_matrices_confusion.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
