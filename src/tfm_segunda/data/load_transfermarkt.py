from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from ..config import config as global_config
from ..config import data_config, resolve_path
from .team_normalization import TEAM_ALIASES, canonical

def _root_dirs() -> tuple[Path, Path, Path]:
    raw_root = resolve_path(global_config()["paths"]["data"]["raw"])
    sub = data_config()["sources"]["transfermarkt"]
    return (
        raw_root / sub["raw_clubs_subdir"],
        raw_root / sub["raw_standings_subdir"],
        raw_root / sub["raw_squads_subdir"],
    )


def _available_seasons_clubs() -> list[int]:
    folder, _, _ = _root_dirs()
    if not folder.exists():
        return []
    return sorted(
        {int(f.stem) for f in folder.glob("*.csv") if f.stem.isdigit() and len(f.stem) == 4}
    )


def _available_seasons_standings() -> list[int]:
    _, folder, _ = _root_dirs()
    if not folder.exists():
        return []
    pat = re.compile(r"^(\d{4})_J\d+$")
    out = set()
    for f in folder.glob("*.csv"):
        m = pat.match(f.stem)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _available_seasons_squads() -> list[int]:
    _, _, folder = _root_dirs()
    if not folder.exists():
        return []
    seasons: set[int] = set()
    for f in folder.glob("*.csv"):
        if f.stem == ".gitkeep":
            continue
        head = f.stem.split("_", 1)[0]
        if head.isdigit() and len(head) == 4:
            seasons.add(int(head))
    return sorted(seasons)

def load_clubs(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    clubs_dir, _, _ = _root_dirs()
    if seasons is None:
        seasons = _available_seasons_clubs()
    seasons = list(seasons)

    parts: list[pd.DataFrame] = []
    for s in seasons:
        path = clubs_dir / f"{s}.csv"
        if not path.exists():
            logger.warning(f"  clubs: falta {path.name}")
            continue
        df = pd.read_csv(path, dtype={"club_id": "int64"})
        df.insert(0, "season", s)
        parts.append(df)

    if not parts:
        raise RuntimeError("No hay archivos de clubs disponibles.")

    out = pd.concat(parts, ignore_index=True)
    out["club_canonical"] = out["club_name"].astype(str).map(canonical)

    no_alias = sorted(set(out["club_name"]) - set(TEAM_ALIASES.keys()))
    if no_alias:
        logger.debug(
            f"  {len(no_alias)} club_name sin entrada en TEAM_ALIASES "
            f"(se usan tal cual): {no_alias[:5]}..."
        )

    logger.info(f"Clubs: {out['season'].nunique()} temporadas, {len(out)} filas")
    return out

def _parse_goles_column(s: pd.Series) -> pd.DataFrame:
    parts = s.astype(str).str.split(":", n=1, expand=True)
    parts.columns = ["gf", "gc"]
    for c in ["gf", "gc"]:
        parts[c] = pd.to_numeric(parts[c], errors="coerce").astype("Int64")
    return parts


def load_standings(
    seasons: Iterable[int] | None = None,
    matchdays: Iterable[int] | None = None,
) -> pd.DataFrame:
    _, standings_dir, _ = _root_dirs()
    if seasons is None:
        seasons = _available_seasons_standings()
    seasons = list(seasons)
    md_filter = set(matchdays) if matchdays is not None else None

    parts: list[pd.DataFrame] = []
    for s in seasons:
        for f in sorted(standings_dir.glob(f"{s}_J*.csv")):
            try:
                md = int(f.stem.split("_J")[1])
            except (IndexError, ValueError):
                continue
            if md_filter is not None and md not in md_filter:
                continue
            df = pd.read_csv(f)
            df.insert(0, "season", s)
            df.insert(1, "jornada", md)
            parts.append(df)

    if not parts:
        raise RuntimeError("No hay archivos de standings disponibles.")

    out = pd.concat(parts, ignore_index=True)
    out = out.rename(
        columns={
            "Pos": "pos",
            "Equipo": "equipo",
            "G": "g",
            "E": "e",
            "P": "p",
            "+/-": "dif_goles",
            "Pts": "pts",
        }
    )

    goles_split = _parse_goles_column(out["Goles"])
    out = pd.concat([out.drop(columns=["Goles"]), goles_split], axis=1)

    out["equipo_canonical"] = out["equipo"].astype(str).map(canonical)
    out["dif_goles"] = pd.to_numeric(out["dif_goles"], errors="coerce").astype("Int64")

    final_cols = [
        "season", "jornada", "pos", "equipo", "equipo_canonical",
        "g", "e", "p", "gf", "gc", "dif_goles", "pts",
    ]
    out = out[final_cols].sort_values(["season", "jornada", "pos"]).reset_index(drop=True)

    n_classif = out[["season", "jornada"]].drop_duplicates().shape[0]
    logger.info(
        f"Standings: {out['season'].nunique()} temporadas, "
        f"{n_classif} clasificaciones (season x jornada), {len(out)} filas"
    )
    return out


def load_squads(
    seasons: Iterable[int] | None = None,
    drop_no_value: bool = False,
) -> pd.DataFrame:
    _, _, squads_dir = _root_dirs()
    if seasons is None:
        seasons = _available_seasons_squads()
    seasons = list(seasons)

    parts: list[pd.DataFrame] = []
    for s in seasons:
        files = sorted(squads_dir.glob(f"{s}_*.csv"))
        for f in files:
            df = pd.read_csv(f)
            parts.append(df)

    if not parts:
        raise RuntimeError("No hay plantillas disponibles.")

    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={"saison_id": "season"})

    clubs = load_clubs(seasons)[["season", "club_id", "club_name", "club_canonical"]]
    out = out.merge(clubs, on=["season", "club_id"], how="left")

    out["valor_mercado_eur"] = pd.to_numeric(out["valor_mercado_eur"], errors="coerce").astype("Int64")
    out["edad"] = pd.to_numeric(out["edad"], errors="coerce").astype("Int64")
    out["dorsal"] = pd.to_numeric(out["dorsal"], errors="coerce").astype("Int64")

    if drop_no_value:
        before = len(out)
        out = out.dropna(subset=["valor_mercado_eur"]).reset_index(drop=True)
        logger.info(f"  Filtrados {before - len(out)} jugadores sin valor")

    final_cols = [
        "season", "club_id", "club_slug", "club_name", "club_canonical",
        "dorsal", "jugador", "posicion", "edad", "nacionalidades",
        "altura", "pie", "en_club_desde", "contrato_hasta",
        "valor_mercado_eur", "valor_mercado_texto",
    ]
    cols_present = [c for c in final_cols if c in out.columns]
    out = out[cols_present].reset_index(drop=True)

    logger.info(
        f"Squads: {out['season'].nunique()} temporadas, "
        f"{out['club_id'].nunique()} equipos distintos, "
        f"{len(out)} filas (jugadores-temporada), "
        f"{out['valor_mercado_eur'].isna().sum()} sin valor"
    )
    return out
