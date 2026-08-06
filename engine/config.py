from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


def load_config(path, required=()):
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"{p} missing required keys: {missing}")
    return cfg
