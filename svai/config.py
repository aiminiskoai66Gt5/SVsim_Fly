"""Load config.toml and resolve every output path the AI side may write to."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - 3.8-3.10 fallback
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.toml"

PATH_KEYS = ("data_root", "datasets", "cache", "checkpoints", "logs", "tmp")


class ForbiddenPathError(RuntimeError):
    """A configured output path points at a location the owner forbids."""


def _forbidden_roots() -> list:
    roots = []
    if sys.platform.startswith("win"):
        roots.append(Path("C:\\"))
    for var in ("USERPROFILE", "LOCALAPPDATA"):
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    if not sys.platform.startswith("win"):
        # Under WSL the Windows C: drive is /mnt/c.
        roots.append(Path("/mnt/c"))
    return roots


def assert_allowed_path(path: Path) -> Path:
    """Raise ForbiddenPathError if ``path`` is under C:\\, %USERPROFILE% or %LOCALAPPDATA%."""
    resolved = Path(path).resolve()
    for root in _forbidden_roots():
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ForbiddenPathError(f"{resolved} is under forbidden root {root}; outputs must stay on E:")
    return resolved


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read config.toml (or ``path``) and return it as a dict with resolved paths.

    ``config["paths"][key]`` become absolute ``Path`` objects; directories are
    created on first use by ``ensure_dir``. Nothing is created at load time.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    paths = cfg.setdefault("paths", {})
    for key in PATH_KEYS:
        raw = paths.get(key, f"data/{key}" if key != "data_root" else "data")
        p = Path(raw)
        if not p.is_absolute():
            p = REPO_ROOT / p
        paths[key] = assert_allowed_path(p)
    env = cfg.setdefault("env", {})
    env.setdefault("max_turns", 40)
    env.setdefault("opponent_actions_per_turn", 30)
    env.setdefault("deck_mode", "random")
    env.setdefault("deck_files", [])
    env.setdefault("exclude_unparsed_cards", False)
    env.setdefault("card_database", "card_database/3_parsed_database/card_database_parsed.json")
    disp = cfg.setdefault("display", {})
    disp.setdefault("card_name_language", "en")
    disp.setdefault("zh_tw_names_file", "")
    return cfg


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (after the forbidden-root check) and return it."""
    path = assert_allowed_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_path(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()
