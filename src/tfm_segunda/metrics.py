from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

LABELS = ["A", "D", "H"]


def _one_hot(y_true) -> np.ndarray:
    y = np.asarray(y_true)
    return np.column_stack([(y == c).astype(float) for c in LABELS])


def brier_multiclase(y_true, proba: np.ndarray) -> float:
    Y = _one_hot(y_true)
    P = np.asarray(proba, dtype=float)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def rps(y_true, proba: np.ndarray) -> float:
    Y = _one_hot(y_true)
    P = np.asarray(proba, dtype=float)
    cdf_p = np.cumsum(P, axis=1)[:, :-1]
    cdf_y = np.cumsum(Y, axis=1)[:, :-1]
    return float(np.mean(np.sum((cdf_p - cdf_y) ** 2, axis=1) / (len(LABELS) - 1)))


def ece(y_true, proba: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true)
    P = np.asarray(proba, dtype=float)
    conf = P.max(axis=1)
    pred = np.array(LABELS)[P.argmax(axis=1)]
    correcto = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        out += (m.sum() / total) * abs(correcto[m].mean() - conf[m].mean())
    return float(out)


def reliability_data(y_true, proba: np.ndarray, clase: str = "H",
                     n_bins: int = 10) -> pd.DataFrame:
    idx = LABELS.index(clase)
    y = (np.asarray(y_true) == clase).astype(float)
    p = np.asarray(proba, dtype=float)[:, idx]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin_centro": (lo + hi) / 2, "p_media": p[m].mean(),
                     "frec_obs": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def evaluar_ampliado(y_true, proba: np.ndarray, nombre: str,
                     tiempo_entrenamiento_s: float | None = None) -> dict:
    P = np.asarray(proba, dtype=float)
    pred = np.array(LABELS)[P.argmax(axis=1)]
    f1_clases = f1_score(y_true, pred, labels=LABELS, average=None)
    return {
        "modelo":    nombre,
        "log_loss":  float(log_loss(y_true, P, labels=LABELS)),
        "brier":     brier_multiclase(y_true, P),
        "rps":       rps(y_true, P),
        "ece":       ece(y_true, P),
        "accuracy":  float(accuracy_score(y_true, pred)),
        "f1_macro":  float(f1_score(y_true, pred, labels=LABELS, average="macro")),
        "f1_A":      float(f1_clases[0]),
        "f1_D":      float(f1_clases[1]),
        "f1_H":      float(f1_clases[2]),
        "t_train_s": tiempo_entrenamiento_s,
    }
