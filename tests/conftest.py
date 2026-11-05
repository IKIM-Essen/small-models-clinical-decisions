"""Shared test helpers. Scripts live under scripts/ as plain files, not a package,
so tests load them by path."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(relpath: str):
    path = REPO / relpath
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def repo_root() -> Path:
    return REPO
