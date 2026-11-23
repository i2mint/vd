"""Backends register lazily: ``import vd`` must not import the client SDKs (vd#58)."""

import subprocess
import sys

import pytest

SDKS = ("chromadb", "lancedb", "qdrant_client", "faiss", "duckdb", "pymilvus", "weaviate", "redis")


def _run(code: str) -> str:
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return done.stdout.strip()


def test_import_vd_does_not_import_any_backend_sdk():
    out = _run(f"import sys, vd; print([m for m in {SDKS!r} if m in sys.modules])")
    assert out == "[]"


def test_only_the_memory_adapter_is_registered_at_import():
    assert _run("import vd.util as u; print(sorted(u._backends))") == "['memory']"


def test_resolving_a_backend_by_name_imports_just_that_adapter():
    out = _run(
        "import sys, vd; vd.util.get_backend('memory'); "
        "print(sorted(m for m in sys.modules if m.startswith('vd.backends.') and m != 'vd.backends._helpers'))"
    )
    assert out == "['vd.backends.memory']"


def test_list_backends_still_lists_every_installed_adapter():
    import vd
    from vd.backends import BACKEND_MODULES, load

    expected = sorted(n for n in BACKEND_MODULES if load(n))
    assert vd.list_backends() == expected and "memory" in expected


def test_a_known_backend_whose_sdk_is_missing_keeps_its_install_hint(monkeypatch):
    import vd
    import vd.util as util
    from vd.backends import BACKEND_MODULES

    missing = next((n for n in BACKEND_MODULES if n != "memory" and not __import__("vd.backends", fromlist=["load"]).load(n)), None)
    if missing is None:
        pytest.skip("every backend SDK is installed here")
    with pytest.raises(vd.BackendNotInstalledError):
        util.get_backend(missing)


def test_an_unknown_name_is_still_a_value_error():
    import vd.util as util

    with pytest.raises(ValueError, match="Unknown backend"):
        util.get_backend("no-such-backend")


def test_connect_async_resolves_a_backend_by_name_in_a_fresh_process():
    out = _run(
        "import asyncio, vd\n"
        "async def go():\n"
        "    c = await vd.connect_async('memory')\n"
        "    return type(c).__name__\n"
        "print(asyncio.run(go()))"
    )
    assert out
