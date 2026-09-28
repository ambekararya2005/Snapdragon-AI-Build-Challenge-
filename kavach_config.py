"""Kavach config loader.

Loads config.yaml once (cached) and returns a nested dict with attribute access:

    from kavach_config import get_config
    cfg = get_config()
    cfg.runtime.provider        # "dml"
    cfg["runtime"]["provider"]  # same

Env overrides:
    KAVACH_CONFIG   - alternate path to the YAML file
    KAVACH_PROVIDER - overrides runtime.provider (qnn | cuda | dml | cpu)

Self-test / print resolved config:  python -m kavach_config
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
VALID_PROVIDERS = ("qnn", "cuda", "dml", "cpu")
VALID_OCR_BACKENDS = ("rapidocr", "native")
REQUIRED_SECTIONS = ("runtime", "privacy", "processes", "screen", "ocr", "audio", "fusion", "logging")


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed or violates a project rule."""


class AttrDict(dict):
    """dict with recursive attribute access (cfg.runtime.provider)."""

    def __init__(self, data: Mapping[str, Any] | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"config has no key {name!r}") from None

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def resolve_config_path(path: str | os.PathLike | None = None, env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if path is None:
        path = env.get("KAVACH_CONFIG") or DEFAULT_CONFIG_PATH
    return Path(path).expanduser().resolve()


def validate(cfg: AttrDict) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config missing sections: {', '.join(missing)}")

    provider = cfg.runtime.get("provider")
    if provider not in VALID_PROVIDERS:
        raise ConfigError(f"runtime.provider must be one of {VALID_PROVIDERS}, got {provider!r}")

    if cfg.privacy.get("write_raw_to_disk", False) is not False:
        raise ConfigError("privacy.write_raw_to_disk must be false: Kavach never writes raw screen/audio/text to disk")
    if not isinstance(cfg.privacy.get("debug_show_text", False), bool):
        raise ConfigError("privacy.debug_show_text must be true or false")

    backend = cfg.ocr.get("backend")
    if backend not in VALID_OCR_BACKENDS:
        raise ConfigError(f"ocr.backend must be one of {VALID_OCR_BACKENDS}, got {backend!r}")

    bands = cfg.fusion.get("bands", {})
    if not bands.get("caution", 0) < bands.get("alert", 0):
        raise ConfigError("fusion.bands.caution must be lower than fusion.bands.alert")


def load_config(path: str | os.PathLike | None = None, env: Mapping[str, str] | None = None) -> AttrDict:
    """Load, apply env overrides and validate. Uncached; most code should call get_config()."""
    env = os.environ if env is None else env
    cfg_path = resolve_config_path(path, env)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {cfg_path}: {e}") from e
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{cfg_path} must contain a mapping at top level")

    cfg = AttrDict(raw)
    if "runtime" in cfg and env.get("KAVACH_PROVIDER"):
        cfg.runtime["provider"] = env["KAVACH_PROVIDER"].strip().lower()
    validate(cfg)
    cfg["config_path"] = str(cfg_path)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> AttrDict:
    """Cached config for the whole process (config.yaml or KAVACH_CONFIG, plus overrides)."""
    return load_config()


def reload_config() -> AttrDict:
    get_config.cache_clear()
    return get_config()


if __name__ == "__main__":
    cfg = get_config()
    print(f"# resolved config from {cfg.config_path}")
    for var in ("KAVACH_CONFIG", "KAVACH_PROVIDER"):
        if os.environ.get(var):
            print(f"# override {var}={os.environ[var]}")
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=None), end="")
