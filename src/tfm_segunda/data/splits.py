from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger

from ..config import data_config


SplitName = Literal["train", "validation", "test", "demo"]
SPLIT_ORDER: tuple[SplitName, ...] = ("train", "validation", "test", "demo")
SEQUENTIAL: tuple[SplitName, ...] = ("train", "validation", "test")


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "season" not in df.columns or "date" not in df.columns:
        raise ValueError("df debe contener columnas 'season' y 'date'")

    splits_cfg = data_config()["splits"]
    out: dict[str, pd.DataFrame] = {}

    for name in SPLIT_ORDER:
        rng = splits_cfg[name]
        s, e = int(rng["start"]), int(rng["end"])
        mask = (df["season"] >= s) & (df["season"] <= e)
        sub = df[mask].copy().sort_values(["season", "date"]).reset_index(drop=True)
        out[name] = sub

    _enforce_no_temporal_leak(out)

    for name, sub in out.items():
        if len(sub) > 0:
            logger.info(
                f"Split {name:<10}: {len(sub):>5} partidos | "
                f"{sub['date'].min().date()} -> {sub['date'].max().date()}"
            )
        else:
            logger.warning(f"Split {name}: VACIO")

    return out


def _enforce_no_temporal_leak(splits: dict[str, pd.DataFrame]) -> None:
    for prev, nxt in zip(SEQUENTIAL, SEQUENTIAL[1:]):
        prev_df, nxt_df = splits[prev], splits[nxt]
        if prev_df.empty or nxt_df.empty:
            continue
        max_prev = prev_df["date"].max()
        min_nxt = nxt_df["date"].min()
        if max_prev >= min_nxt:
            raise ValueError(
                f"Fuga temporal detectada: max({prev}) = {max_prev.date()} "
                f">= min({nxt}) = {min_nxt.date()}"
            )


def get_split(df: pd.DataFrame, name: SplitName) -> pd.DataFrame:
    return temporal_split(df)[name]


def split_info(df: pd.DataFrame) -> pd.DataFrame:
    splits = temporal_split(df)
    rows = []
    for name in SPLIT_ORDER:
        sub = splits[name]
        if sub.empty:
            rows.append(
                {
                    "split": name,
                    "seasons": "",
                    "n_partidos": 0,
                    "fecha_inicio": pd.NaT,
                    "fecha_fin": pd.NaT,
                }
            )
        else:
            rows.append(
                {
                    "split": name,
                    "seasons": f"{sub['season'].min()}-{sub['season'].max()}",
                    "n_partidos": len(sub),
                    "fecha_inicio": sub["date"].min().date(),
                    "fecha_fin": sub["date"].max().date(),
                }
            )
    return pd.DataFrame(rows)
