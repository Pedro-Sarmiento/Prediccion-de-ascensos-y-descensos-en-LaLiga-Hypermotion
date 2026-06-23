from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ECON_DIR     = PROJECT_ROOT / "data" / "economia"
CORTES_DIR   = PROJECT_ROOT / "notebooks" / "data" / "simulations" / "cortes"

VALOR_ASCENSO_EUR  = 42_000_000
VALOR_DESCENSO_EUR = 1_500_000

MODELO_REFERENCIA_PESOS = "XGBoost sin pesos"
CORTES_DISPONIBLES = [5, 10, 15, 20, 25, 30, 35, 40]


def cargar_reparto_tv() -> pd.DataFrame:
    return pd.read_csv(ECON_DIR / "reparto_audiovisual_segunda_2025_26.csv")


def _slug(nombre: str) -> str:
    return nombre.lower().replace(" ", "_").replace("ó", "o")


def cargar_cortes(modelo: str, season: int) -> dict[int, pd.DataFrame]:
    out = {}
    for j in CORTES_DISPONIBLES:
        f = CORTES_DIR / f"mc_corte_j{j:02d}_{_slug(modelo)}_t{season}.csv"
        if f.exists():
            out[j] = pd.read_csv(f).set_index("equipo")
    if not out:
        raise FileNotFoundError(f"No hay cortes para {modelo} t{season} en {CORTES_DIR}")
    return out


def valor_esperado_equipo(df_corte: pd.DataFrame,
                          v_asc: float = VALOR_ASCENSO_EUR,
                          v_desc: float = VALOR_DESCENSO_EUR) -> pd.Series:
    return df_corte["P(asc_total)"] * v_asc - df_corte["P(descenso)"] * v_desc


def dinero_en_juego_equipo(df_corte: pd.DataFrame,
                           v_asc: float = VALOR_ASCENSO_EUR,
                           v_desc: float = VALOR_DESCENSO_EUR) -> pd.Series:
    p_a = df_corte["P(asc_total)"].clip(0, 1)
    p_d = df_corte["P(descenso)"].clip(0, 1)
    return 4 * p_a * (1 - p_a) * v_asc + 4 * p_d * (1 - p_d) * v_desc


def _corte_anterior(jornada: float) -> int | None:
    previos = [c for c in CORTES_DISPONIBLES if c < jornada]
    return max(previos) if previos else None


def pesos_economicos(partidos: pd.DataFrame,
                     modelo_ref: str = MODELO_REFERENCIA_PESOS,
                     v_asc: float = VALOR_ASCENSO_EUR,
                     v_desc: float = VALOR_DESCENSO_EUR) -> pd.Series:

    cache: dict[tuple[int, int], pd.Series] = {}

    def juego(season: int, corte: int) -> pd.Series:
        if (season, corte) not in cache:
            cortes = cargar_cortes(modelo_ref, season)
            c = corte if corte in cortes else min(cortes)
            cache[(season, corte)] = dinero_en_juego_equipo(cortes[c], v_asc, v_desc)
        return cache[(season, corte)]

    w = np.zeros(len(partidos))
    for k, (_, r) in enumerate(partidos.iterrows()):
        corte = _corte_anterior(r["jornada"]) or min(CORTES_DISPONIBLES)
        s = juego(int(r["season"]), corte)
        for eq in (r["home_team"], r["away_team"]):
            if eq in s.index:
                w[k] += float(s[eq])
    return pd.Series(w, index=partidos.index, name="peso_eur")


def pee(y_true, proba: np.ndarray, pesos: pd.Series) -> float:
    from sklearn.metrics import log_loss
    labels = ["A", "D", "H"]
    Y = np.column_stack([(np.asarray(y_true) == c).astype(float) for c in labels])
    P = np.clip(np.asarray(proba, dtype=float), 1e-15, 1 - 1e-15)
    ll_i = -np.sum(Y * np.log(P), axis=1)
    w = np.asarray(pesos, dtype=float)
    w_norm = w / w.mean()
    return float(np.mean(w_norm * ll_i))
