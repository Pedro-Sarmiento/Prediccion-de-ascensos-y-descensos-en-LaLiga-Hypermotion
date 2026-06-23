from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import os
import time

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

import warnings
warnings.filterwarnings("ignore")


SEED        = 42
N_SIM       = int(os.environ.get("N_SIM", 100_000))
MAX_GOALS   = 8
LABELS      = ["A", "D", "H"]
LABELS_ARR  = np.array(LABELS)
PUNTOS      = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}

TEMPORADAS = [
    (2023, "Espanyol"),
    (2024, "Oviedo"),
]

PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
OUTPUT_DIR      = PROJECT_ROOT / "data" / "simulations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Cargando dataset…")
df = pd.read_parquet(DATA_PROCESSED / "dataset_modelado.parquet")

META_COLS    = ["split", "season", "date", "jornada", "home_team", "away_team", "fthg", "ftag"]
TARGET_COLS  = ["ftr"]
FEATURE_COLS = [c for c in df.columns if c not in META_COLS + TARGET_COLS]

train  = df[df["split"] == "train"].copy()
scaler = StandardScaler().fit(train[FEATURE_COLS].dropna().astype(float))

print("Cargando modelos persistidos…")
modelos_sklearn = {
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
    total = pH + pD + pA
    return np.column_stack([pA / total, pD / total, pH / total])


def predict_sklearn(model, X: pd.DataFrame, scale: bool) -> np.ndarray:
    Xv = scaler.transform(X) if scale else X.values
    return model.predict_proba(Xv)


def predecir_todos(X: pd.DataFrame) -> dict:
    out = {}
    for nombre, (mdl, scale) in modelos_sklearn.items():
        out[nombre] = predict_sklearn(mdl, X, scale)
    out["Doble Poisson"] = predict_dp(X)
    return out


def ejecutar_temporada(season_year: int, asc_playoff_real: str) -> dict:

    etiqueta = f"{season_year}-{(season_year + 1) % 100:02d}"
    print(f"\n{'='*78}\n  Temporada {etiqueta}  (season={season_year})\n{'='*78}")
    season      = df[df["season"] == season_year].copy().sort_values(
        ["date", "jornada"]
    ).reset_index(drop=True)
    season_full = season.copy()
    season_ok   = season.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)

    EQUIPOS = sorted(season_full["home_team"].unique())
    N_EQ    = len(EQUIPOS)
    assert N_EQ == 22, f"Se esperaban 22 equipos, hay {N_EQ}"

    print(f"  partidos: {len(season_full)} (con features completas: {len(season_ok)} / "
          f"con resultado real conservado: {len(season_full) - len(season_ok)})")

    probs_reg = predecir_todos(season_ok[FEATURE_COLS])

    home_cols    = [c for c in FEATURE_COLS if c.startswith("home_")]
    away_cols    = [c for c in FEATURE_COLS if c.startswith("away_")]
    neutral_cols = [c for c in FEATURE_COLS if not c.startswith(("home_", "away_"))]

    last_as_home, last_as_away = {}, {}
    for eq in EQUIPOS:
        h_rows = season_ok[season_ok["home_team"] == eq].sort_values("date")
        a_rows = season_ok[season_ok["away_team"] == eq].sort_values("date")
        last_as_home[eq] = h_rows.iloc[-1][home_cols]
        last_as_away[eq] = a_rows.iloc[-1][away_cols]
    neutral_means = season_ok[neutral_cols].mean()

    def feats(home: str, away: str) -> dict:
        v = {c: float(last_as_home[home][c]) for c in home_cols}
        v.update({c: float(last_as_away[away][c]) for c in away_cols})
        v.update({c: float(neutral_means[c])      for c in neutral_cols})
        return v

    playoff_pairs   = [(h, a) for h in EQUIPOS for a in EQUIPOS if h != a]
    X_playoff       = pd.DataFrame([feats(h, a) for h, a in playoff_pairs])[FEATURE_COLS]
    playoff_idx_map = {pair: i for i, pair in enumerate(playoff_pairs)}
    probs_playoff   = predecir_todos(X_playoff)

    HOME_IDX = np.array([EQUIPOS.index(h) for h in season_ok["home_team"]])
    AWAY_IDX = np.array([EQUIPOS.index(a) for a in season_ok["away_team"]])
    descartados = season_full.drop(season_ok.index, errors="ignore")

    PTS_BASE = np.zeros(N_EQ, dtype=np.int32)
    GD_BASE  = np.zeros(N_EQ, dtype=np.int32)
    GF_BASE  = np.zeros(N_EQ, dtype=np.int32)
    for _, r in descartados.iterrows():
        if pd.isna(r["ftr"]):
            continue
        h, a = EQUIPOS.index(r["home_team"]), EQUIPOS.index(r["away_team"])
        gh, ga = int(r["fthg"]), int(r["ftag"])
        ph, pa = PUNTOS[r["ftr"]]
        PTS_BASE[h] += ph; PTS_BASE[a] += pa
        GD_BASE[h]  += gh - ga; GD_BASE[a] += ga - gh
        GF_BASE[h]  += gh; GF_BASE[a]  += ga

    pts_real, gd_real, gf_real = PTS_BASE.copy(), GD_BASE.copy(), GF_BASE.copy()
    GH_REG = season_ok["fthg"].astype(int).values
    GA_REG = season_ok["ftag"].astype(int).values
    for k in range(len(season_ok)):
        h, a, gh, ga = HOME_IDX[k], AWAY_IDX[k], GH_REG[k], GA_REG[k]
        if gh > ga:
            pts_real[h] += 3
        elif ga > gh:
            pts_real[a] += 3
        else:
            pts_real[h] += 1; pts_real[a] += 1
        gd_real[h] += gh - ga; gd_real[a] += ga - gh
        gf_real[h] += gh; gf_real[a] += ga

    key_real   = pts_real * 10_000 + gd_real * 100 + gf_real
    order_real = np.argsort(-key_real, kind="stable")
    TABLA_REAL = [EQUIPOS[i] for i in order_real]

    def sample_resultados(proba, rng_local):
        u   = rng_local.random(proba.shape[0])
        cdf = np.cumsum(proba, axis=1)
        return LABELS_ARR[(u[:, None] >= cdf[:, :-1]).sum(axis=1)]

    def eliminatoria(mejor, peor, probs_pl, rng_local):
        p_ida = probs_pl[playoff_idx_map[(peor, mejor)]]
        p_vue = probs_pl[playoff_idx_map[(mejor, peor)]]
        r_ida = LABELS[int((rng_local.random() >= np.cumsum(p_ida)[:-1]).sum())]
        r_vue = LABELS[int((rng_local.random() >= np.cumsum(p_vue)[:-1]).sum())]
        sm = (r_ida == "A") + (r_vue == "H")
        sp = (r_ida == "H") + (r_vue == "A")
        if sm > sp: return mejor
        if sp > sm: return peor
        return mejor

    def una_temporada(probs_reg_arr, probs_pl, rng_local):
        res = sample_resultados(probs_reg_arr, rng_local)
        pts, gd, gf = PTS_BASE.copy(), GD_BASE.copy(), GF_BASE.copy()
        mH, mA, mD = res == "H", res == "A", res == "D"
        np.add.at(pts, HOME_IDX[mH], 3)
        np.add.at(pts, AWAY_IDX[mA], 3)
        np.add.at(pts, HOME_IDX[mD], 1)
        np.add.at(pts, AWAY_IDX[mD], 1)
        np.add.at(gd, HOME_IDX[mH], 1); np.add.at(gd, AWAY_IDX[mH], -1)
        np.add.at(gd, AWAY_IDX[mA], 1); np.add.at(gd, HOME_IDX[mA], -1)
        np.add.at(gf, HOME_IDX[mH], 1)
        np.add.at(gf, AWAY_IDX[mA], 1)
        order = np.argsort(-(pts * 10_000 + gd * 100 + gf), kind="stable")
        tabla = [EQUIPOS[i] for i in order]
        sf1 = eliminatoria(tabla[2], tabla[5], probs_pl, rng_local)
        sf2 = eliminatoria(tabla[3], tabla[4], probs_pl, rng_local)
        fin = sorted([sf1, sf2], key=lambda e: tabla.index(e))
        campeon_pl = eliminatoria(fin[0], fin[1], probs_pl, rng_local)
        return tabla, tabla[:2], campeon_pl, tabla[-4:]

    def correr_modelo(nombre, probs_reg_arr, probs_pl):
        rng_local = np.random.default_rng(SEED)
        c_dir, c_pl, c_desc = Counter(), Counter(), Counter()
        c_pos = defaultdict(lambda: np.zeros(N_EQ, dtype=np.int64))
        t0 = time.time()
        for _ in range(N_SIM):
            tabla, asc_d, campeon, desc = una_temporada(probs_reg_arr, probs_pl, rng_local)
            for e in asc_d:  c_dir[e]  += 1
            c_pl[campeon] += 1
            for e in desc:   c_desc[e] += 1
            for pos, e in enumerate(tabla):
                c_pos[e][pos] += 1
        dur = time.time() - t0
        print(f"    {nombre}: {N_SIM:,} sim. en {dur:.1f} s ({1000*dur/N_SIM:.2f} ms/it)")
        rows = []
        for e in EQUIPOS:
            rows.append({
                "equipo":         e,
                "P(asc_directo)": c_dir[e] / N_SIM,
                "P(asc_playoff)": c_pl[e]  / N_SIM,
                "P(asc_total)":   (c_dir[e] + c_pl[e]) / N_SIM,
                "P(descenso)":    c_desc[e] / N_SIM,
                "pos_media":      float(np.average(np.arange(1, N_EQ + 1), weights=c_pos[e])),
                "pos_real":       TABLA_REAL.index(e) + 1,
            })
        return (pd.DataFrame(rows)
                .sort_values("pos_real")
                .reset_index(drop=True))

    print(f"  Monte Carlo {N_SIM:,} iteraciones por modelo:")
    resultados = {n: correr_modelo(n, probs_reg[n], probs_playoff[n]) for n in probs_reg}

    ASC_DIR_REAL = {TABLA_REAL[0], TABLA_REAL[1]}
    DESC_REAL    = set(TABLA_REAL[-4:])
    print(f"\n  --- Realidad temporada {etiqueta} ---")
    print(f"    Ascensos directos: {sorted(ASC_DIR_REAL)}")
    print(f"    Asciende vía play-off: {asc_playoff_real}")
    print(f"    Descensos: {sorted(DESC_REAL)}")

    filas = []
    for nombre, dfres in resultados.items():
        d = dfres.set_index("equipo")
        filas.append({
            "modelo":                                     nombre,
            f"P({TABLA_REAL[0]} asc.dir)":                d.loc[TABLA_REAL[0], "P(asc_directo)"],
            f"P({TABLA_REAL[1]} asc.dir)":                d.loc[TABLA_REAL[1], "P(asc_directo)"],
            f"P({asc_playoff_real} asc.PO)":              d.loc[asc_playoff_real, "P(asc_playoff)"],
            "P(media descensos reales)":                  sum(d.loc[e, "P(descenso)"] for e in DESC_REAL) / 4,
        })
    resumen = pd.DataFrame(filas)
    print()
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for nombre, dfres in resultados.items():
        slug = nombre.lower().replace(" ", "_").replace("ó", "o")
        dfres.to_csv(OUTPUT_DIR / f"montecarlo_{slug}_t{season_year}.csv", index=False)
    resumen.to_csv(OUTPUT_DIR / f"montecarlo_resumen_calibracion_t{season_year}.csv",
                   index=False)

    COLOR_ASC_DIR = "#1b8a4a"       # verde
    COLOR_ASC_PO  = "#e07b1a"       # naranja
    COLOR_DESC    = "#c41e1e"       # rojo

    def heatmap(metric, titulo, fname):
        M = pd.DataFrame({n: dfr.set_index("equipo")[metric] for n, dfr in resultados.items()})
        M = M.loc[TABLA_REAL]
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(M, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": metric})
        ax.set_title(titulo)
        ax.set_ylabel(f"Equipo (orden real de la temporada {etiqueta})")

        for label in ax.get_yticklabels():
            eq = label.get_text()
            if eq in ASC_DIR_REAL:
                label.set_color(COLOR_ASC_DIR); label.set_fontweight("bold")
            elif eq == asc_playoff_real:
                label.set_color(COLOR_ASC_PO);  label.set_fontweight("bold")
            elif eq in DESC_REAL:
                label.set_color(COLOR_DESC);    label.set_fontweight("bold")

        from matplotlib.patches import Patch
        leyenda = [
            Patch(facecolor=COLOR_ASC_DIR, label="Ascenso directo"),
            Patch(facecolor=COLOR_ASC_PO,  label="Ascenso vía play-off"),
            Patch(facecolor=COLOR_DESC,    label="Descenso"),
        ]
        ax.legend(handles=leyenda, loc="lower right", bbox_to_anchor=(1.32, 0.0),
                  frameon=True, fontsize=9, title="Desenlace real")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)

    heatmap("P(asc_total)", f"Probabilidad de ascenso – 100k Monte Carlo ({etiqueta})",
            f"heatmap_ascenso_t{season_year}.png")
    heatmap("P(descenso)",  f"Probabilidad de descenso – 100k Monte Carlo ({etiqueta})",
            f"heatmap_descenso_t{season_year}.png")

    ref = "XGBoost sin pesos"
    print(f"\n  === Resultados según {ref} ({etiqueta}) ===")
    print(resultados[ref].to_string(
        index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    ))

    return resultados

todos = {}
for season_year, asc_pl in TEMPORADAS:
    todos[season_year] = ejecutar_temporada(season_year, asc_pl)

print(f"\nCSVs y heatmaps guardados en: {OUTPUT_DIR}")
