from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
NOTEBOOKS      = PROJECT_ROOT / "notebooks"
DATA_PROCESSED = NOTEBOOKS / "data" / "processed"
DATA_INTERIM   = NOTEBOOKS / "data" / "interim"
MODELS_DIR     = NOTEBOOKS / "models"
REPORTS_DIR    = PROJECT_ROOT / "reports"

LABELS    = ["A", "D", "H"]
MAX_GOALS = 8

META_COLS   = ["split", "season", "date", "jornada", "home_team", "away_team", "fthg", "ftag"]
TARGET_COLS = ["ftr"]

MODELOS_SKLEARN = {
    "LogReg sin pesos":        ("logistic_regression_simple.pkl",   True),
    "LogReg balanced":         ("logistic_regression_balanced.pkl", True),
    "Random Forest sin pesos": ("random_forest_simple.pkl",         False),
    "Random Forest balanced":  ("random_forest_balanced.pkl",       False),
    "XGBoost sin pesos":       ("xgboost_simple.pkl",               False),
    "XGBoost balanced":        ("xgboost_balanced.pkl",             False),
    "LightGBM sin pesos":      ("lgb_simple.pkl",                   False),
    "LightGBM balanced":       ("lgb_balanced.pkl",                 False),
    "MLP":                     ("mlp_final.pkl",                    True),
}


def _proba_en_orden_adh(model, Xv: np.ndarray) -> np.ndarray:
    P = model.predict_proba(Xv)
    clases = list(getattr(model, "classes_", LABELS))
    if all(isinstance(c, str) for c in clases):
        idx = [clases.index(c) for c in LABELS]
    else:  # LabelEncoder numérico: 0/1/2 ya es A/D/H
        idx = [clases.index(i) for i in range(len(LABELS))]
    return P[:, idx]


def _predict_double_poisson(X: pd.DataFrame) -> np.ndarray:
    glm_home = joblib.load(MODELS_DIR / "double_poisson" / "glm_home.joblib")
    glm_away = joblib.load(MODELS_DIR / "double_poisson" / "glm_away.joblib")
    dp_scaler = joblib.load(MODELS_DIR / "double_poisson" / "scaler.joblib")

    Xs = dp_scaler.transform(X)
    Xc = np.hstack([np.ones((len(Xs), 1)), Xs])
    lam_h = np.asarray(glm_home.predict(Xc))
    lam_a = np.asarray(glm_away.predict(Xc))

    goles = np.arange(MAX_GOALS + 1)
    p_h = poisson.pmf(goles[None, :], lam_h[:, None])
    p_a = poisson.pmf(goles[None, :], lam_a[:, None])
    matriz = p_h[:, :, None] * p_a[:, None, :]
    i_idx, j_idx = np.meshgrid(goles, goles, indexing="ij")
    pH = (matriz * (i_idx > j_idx)).sum(axis=(1, 2))
    pD = (matriz * (i_idx == j_idx)).sum(axis=(1, 2))
    pA = (matriz * (i_idx < j_idx)).sum(axis=(1, 2))
    total = pH + pD + pA
    return np.column_stack([pA / total, pD / total, pH / total])


def generar_predicciones_test(splits=("test",)) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "dataset_modelado.parquet")
    feature_cols = [c for c in df.columns if c not in META_COLS + TARGET_COLS]

    train = df[df["split"] == "train"]
    scaler = StandardScaler().fit(train[feature_cols].dropna().astype(float))

    sub = df[df["split"].isin(splits)].dropna(subset=feature_cols).copy()
    out = sub[["split", "season", "date", "jornada", "home_team", "away_team", "ftr"]].copy()
    X = sub[feature_cols]

    for nombre, (fichero, escala) in MODELOS_SKLEARN.items():
        model = joblib.load(MODELS_DIR / fichero)
        Xv = scaler.transform(X) if escala else X.values
        P = _proba_en_orden_adh(model, Xv)
        slug = nombre.lower().replace(" ", "_")
        for j, c in enumerate(LABELS):
            out[f"{slug}__p{c}"] = P[:, j]

    P_dp = _predict_double_poisson(X)
    for j, c in enumerate(LABELS):
        out[f"doble_poisson__p{c}"] = P_dp[:, j]

    m = pd.read_parquet(DATA_INTERIM / "matches_clean.parquet")
    m = m[["season", "date", "home_team", "away_team",
           "prob_h", "prob_d", "prob_a",
           "odds_b365_h", "odds_b365_d", "odds_b365_a"]]
    antes = len(out)
    out = out.merge(m, on=["season", "date", "home_team", "away_team"],
                    how="left", validate="one_to_one")
    assert len(out) == antes, "el join con matches_clean duplicó filas"

    total = out["prob_a"] + out["prob_d"] + out["prob_h"]
    out["mercado__pA"] = out.pop("prob_a") / total
    out["mercado__pD"] = out.pop("prob_d") / total
    out["mercado__pH"] = out.pop("prob_h") / total

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = REPORTS_DIR / f"predictions_{'_'.join(splits)}.parquet"
    out.to_parquet(ruta, index=False)
    print(f"Guardado {ruta} — {len(out)} partidos, {out.shape[1]} columnas")
    return out


MODELOS_TODOS = list(MODELOS_SKLEARN) + ["Doble Poisson"]


def slug_de(nombre: str) -> str:
    return nombre.lower().replace(" ", "_").replace("ó", "o") \
        if nombre == "Doble Poisson" else nombre.lower().replace(" ", "_")


def cols_proba(nombre: str) -> list[str]:
    s = slug_de(nombre)
    return [f"{s}__p{c}" for c in LABELS]


if __name__ == "__main__":
    generar_predicciones_test()
