"""Loads the Isaac-free serve modules as package ``_serve`` (the real ``serve/__init__`` imports Isaac)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/serve"


def load(*names: str) -> dict:
    pkg = sys.modules.get("_serve")
    if pkg is None:
        pkg = types.ModuleType("_serve")
        pkg.__path__ = [str(SERVE)]
        sys.modules["_serve"] = pkg
    out = {}
    for name in names:
        key = f"_serve.{name}"
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, SERVE / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
        out[name] = sys.modules[key]
    return out
