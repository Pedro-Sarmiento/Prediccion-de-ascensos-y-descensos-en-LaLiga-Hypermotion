from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from ..config import config as global_config
from ..config import data_config, resolve_path
from .team_normalization import canonical


COL_RENAME: dict[str, str] = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "ftr",
    "B365H": "odds_b365_h",
    "B365D": "odds_b365_d",
    "B365A": "odds_b365_a",
    "BWH": "odds_bw_h",
    "BWD": "odds_bw_d",
    "BWA": "odds_bw_a",
    "WHH": "odds_wh_h",
    "WHD": "odds_wh_d",
    "WHA": "odds_wh_a",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

REQUIRED_COLS: list[str] = ["date", "home_team", "away_team", "fthg", "ftag", "ftr"]

ODDS_1X2_COLS: list[str] = [
    "odds_b365_h", "odds_b365_d", "odds_b365_a",
    "odds_bw_h",   "odds_bw_d",   "odds_bw_a",
    "odds_wh_h",   "odds_wh_d",   "odds_wh_a",
]

STATS_COLS: list[str] = [
    "home_shots",        "away_shots",
    "home_shots_target", "away_shots_target",
    "home_fouls",        "away_fouls",
    "home_corners",      "away_corners",
    "home_yellow",       "away_yellow",
    "home_red",          "away_red",
]

AH_RAW_COLS: list[str] = ["ah_line", "ah_odds_home", "ah_odds_away"]

FINAL_SCHEMA: list[str] = (
    ["season", "season_label", "date", "home_team", "away_team", "fthg", "ftag", "ftr"]
    + ODDS_1X2_COLS
    + ["odds_avg_h", "odds_avg_d", "odds_avg_a"]
    + ["prob_h", "prob_d", "prob_a"]
    + STATS_COLS
    + AH_RAW_COLS
    + ["ah_prob_home"]
)

def _season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _resolve_ah_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if "AHh" in df.columns:
        mapping["ah_line"] = "AHh"
    elif "BbAHh" in df.columns:
        mapping["ah_line"] = "BbAHh"

    if "AvgAHH" in df.columns and "AvgAHA" in df.columns:
        mapping["ah_odds_home"] = "AvgAHH"
        mapping["ah_odds_away"] = "AvgAHA"
    elif "BbAvAHH" in df.columns and "BbAvAHA" in df.columns:
        mapping["ah_odds_home"] = "BbAvAHH"
        mapping["ah_odds_away"] = "BbAvAHA"
    return mapping


def _implied_probs_no_vig(odds: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / odds.to_numpy()
    total = inv.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        norm = inv / total
    return pd.DataFrame(
        norm, columns=["prob_h", "prob_d", "prob_a"], index=odds.index
    )

def load_one_season(season: int) -> pd.DataFrame:
    raw_root = resolve_path(global_config()["paths"]["data"]["raw"])
    sub = data_config()["sources"]["footballdata"]["raw_subdir"]
    csv_path = raw_root / sub / f"SP2_{_season_label(season)}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV para temporada {season}: {csv_path}")

    logger.debug(f"Cargando {csv_path.name}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    n_raw = len(df)

    df["Date"] = pd.to_datetime(
        df["Date"], format="mixed", dayfirst=True, errors="coerce"
    )

    present = {k: v for k, v in COL_RENAME.items() if k in df.columns}
    df = df.rename(columns=present)

    ah_map = _resolve_ah_columns(df)
    for canon_name, orig_name in ah_map.items():
        df[canon_name] = pd.to_numeric(df[orig_name], errors="coerce")

    for c in ODDS_1X2_COLS + STATS_COLS + AH_RAW_COLS:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["fthg", "ftag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ODDS_1X2_COLS + STATS_COLS + AH_RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["ftr"].isin(["H", "D", "A"])].copy()
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    n_dropped = n_raw - len(df)
    if n_dropped > 0:
        logger.warning(
            f"  {season}: descartadas {n_dropped} filas invalidas "
            f"({n_raw} -> {len(df)})"
        )

    df.insert(0, "season", season)
    df.insert(1, "season_label", _season_label(season))

    if data_config().get("team_handling", {}).get("normalize_team_names", True):
        df["home_team"] = df["home_team"].astype(str).map(canonical)
        df["away_team"] = df["away_team"].astype(str).map(canonical)

    df["ftr"] = pd.Categorical(df["ftr"], categories=["H", "D", "A"], ordered=False)

    return df


def load_seasons(seasons: Iterable[int]) -> pd.DataFrame:
    seasons_list = list(seasons)
    if not seasons_list:
        raise ValueError("Debes pasar al menos una temporada")

    parts: list[pd.DataFrame] = []
    for s in seasons_list:
        try:
            parts.append(load_one_season(s))
        except FileNotFoundError as exc:
            logger.warning(f"  saltando temporada {s}: {exc}")

    if not parts:
        raise RuntimeError("No se pudo cargar ninguna temporada.")

    df = pd.concat(parts, ignore_index=True)

    df["odds_avg_h"] = df[["odds_b365_h", "odds_bw_h", "odds_wh_h"]].mean(axis=1)
    df["odds_avg_d"] = df[["odds_b365_d", "odds_bw_d", "odds_wh_d"]].mean(axis=1)
    df["odds_avg_a"] = df[["odds_b365_a", "odds_bw_a", "odds_wh_a"]].mean(axis=1)

    probs = _implied_probs_no_vig(df[["odds_avg_h", "odds_avg_d", "odds_avg_a"]])
    df = pd.concat([df, probs], axis=1)

    inv_h = 1.0 / df["ah_odds_home"]
    inv_a = 1.0 / df["ah_odds_away"]
    with np.errstate(invalid="ignore"):
        df["ah_prob_home"] = inv_h / (inv_h + inv_a)

    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    df = df[FINAL_SCHEMA]

    logger.info(
        f"Cargadas {len(parts)} temporadas | "
        f"{len(df)} partidos | "
        f"rango {df['date'].min().date()} -> {df['date'].max().date()}"
    )
    return df


def load_all_seasons() -> pd.DataFrame:
    """Carga TODAS las temporadas en alcance segun `config/data.yaml`."""
    cfg = data_config()["seasons"]
    seasons = range(int(cfg["scope_start"]), int(cfg["scope_end"]) + 1)
    return load_seasons(seasons)
