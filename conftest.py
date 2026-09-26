"""
Root pytest configuration: decide which package modules doctest collection skips.

``pytest --doctest-modules`` imports every module under ``vd/``. Each backend
adapter raises :class:`ImportError` at import time when its optional client
SDK is absent (``pip install vd[<backend>]`` installs it), which would abort
collection. Those modules are skipped here, so the rest of the package's
doctests always run and a backend's doctests run whenever its SDK is present.

``misc/`` holds demo scripts and design notes, not tests.
"""

import importlib
import pathlib

_HERE = pathlib.Path(__file__).parent


def _unimportable_backend_modules() -> list[str]:
    """Return the paths of backend modules whose optional SDK is not installed."""
    skipped = []
    for path in sorted((_HERE / "vd" / "backends").glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"vd.backends.{path.stem}")
        except ImportError:
            skipped.append(str(path.relative_to(_HERE)))
    return skipped


collect_ignore = ["misc", *_unimportable_backend_modules()]
