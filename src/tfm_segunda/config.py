from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = ROOT / "config"


@lru_cache(maxsize=16)
def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config() -> dict[str, Any]:
    """Configuracion global (config/config.yaml)."""
    return load_yaml("config.yaml")


def data_config() -> dict[str, Any]:
    """Configuracion de datos y splits (config/data.yaml)."""
    return load_yaml("data.yaml")


def model_config(name: str) -> dict[str, Any]:
    """Configuracion de un modelo concreto (config/models/<name>.yaml)."""
    return load_yaml(f"models/{name}.yaml")


def resolve_path(relative: str | Path) -> Path:
    """Convierte una ruta relativa al repo en absoluta. Idempotente con absolutas."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Crea y devuelve el directorio absoluto."""
    abs_path = resolve_path(path)
    abs_path.mkdir(parents=True, exist_ok=True)
    return abs_path


def seed() -> int:
    """Semilla aleatoria global declarada en config.yaml."""
    return int(config().get("random", {}).get("seed", 42))
