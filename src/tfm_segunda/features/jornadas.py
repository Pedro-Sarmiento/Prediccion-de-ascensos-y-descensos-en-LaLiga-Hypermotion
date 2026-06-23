from __future__ import annotations

from collections import defaultdict

import pandas as pd
from loguru import logger


def assign_jornadas(df: pd.DataFrame) -> pd.DataFrame:

    required = {"season", "date", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")

    out = df.copy().sort_values(["season", "date"]).reset_index(drop=True)

    home_n = pd.Series(0, index=out.index, dtype="int64")
    away_n = pd.Series(0, index=out.index, dtype="int64")

    for season, idxs in out.groupby("season", sort=True).groups.items():
        counters: dict[str, int] = defaultdict(int)
        for i in idxs:
            h = out.at[i, "home_team"]
            a = out.at[i, "away_team"]
            counters[h] += 1
            counters[a] += 1
            home_n.at[i] = counters[h]
            away_n.at[i] = counters[a]

    out["home_match_n"] = home_n.astype("int64")
    out["away_match_n"] = away_n.astype("int64")
    out["jornada"] = out[["home_match_n", "away_match_n"]].max(axis=1).astype("int64")

    n_seasons = out["season"].nunique()
    n_misalign = (out["home_match_n"] != out["away_match_n"]).sum()
    logger.info(
        f"Jornadas asignadas: {n_seasons} temporadas | "
        f"{n_misalign} partidos con desfase home/away (aplazados o recuperados)"
    )

    return out


def verify_jornadas(
    df: pd.DataFrame,
    expected_per_jornada: int = 11,
    expected_jornadas_per_season: int = 42,
) -> pd.DataFrame:
    if "jornada" not in df.columns:
        raise ValueError("df debe haber pasado por assign_jornadas() primero")

    counts = (
        df.groupby(["season", "jornada"]).size().rename("n_partidos").reset_index()
    )

    rows = []
    for season, grp in counts.groupby("season"):
        n_jornadas = grp["jornada"].nunique()
        completas = (grp["n_partidos"] == expected_per_jornada).sum()
        incompletas = n_jornadas - completas
        rows.append(
            {
                "season": int(season),
                "jornadas_distintas": int(n_jornadas),
                "jornadas_completas": int(completas),
                "jornadas_incompletas": int(incompletas),
                "total_partidos": int(grp["n_partidos"].sum()),
                "esperados": expected_per_jornada * expected_jornadas_per_season,
            }
        )

    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
