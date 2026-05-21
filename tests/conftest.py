"""
Pytest fixtures shared across the vd test suite.

The headline fixture is :func:`client` — parametrized over every backend that
can actually run in a plain CI environment (no servers, no cloud accounts).
A test that takes ``client`` runs once per backend, which is how the test
suite proves the facade contract holds uniformly.
"""

import hashlib

import pytest

import vd

#: Embedding dimension used by the test embedder.
EMBED_DIM = 16

#: Backends exercised end-to-end here: embedded / pip-only, no server needed.
#: (sqlite_vec is intentionally excluded — some Python builds ship a sqlite3
#: with extension loading disabled, which the backend needs.)
TESTABLE_BACKENDS = ["memory", "chroma", "faiss", "duckdb", "lancedb", "qdrant"]


def make_embedder():
    """Return a deterministic ``text -> 16-dim vector`` embedder for tests."""

    def embed(text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        vector = [(b / 128.0) - 1.0 for b in digest]
        while len(vector) < EMBED_DIM:
            vector += vector
        return vector[:EMBED_DIM]

    return embed


@pytest.fixture
def embedder():
    """A deterministic test embedder."""
    return make_embedder()


@pytest.fixture(params=TESTABLE_BACKENDS)
def backend_name(request):
    """Each installed testable backend, one at a time."""
    name = request.param
    if name not in vd.list_backends():
        pytest.skip(f"backend {name!r} is not installed")
    return name


@pytest.fixture
def client(backend_name, embedder):
    """A fresh, connected client for each testable backend (with an embedder)."""
    connection = vd.connect(backend_name, embedder=embedder)
    yield connection
    if hasattr(connection, "close"):
        try:
            connection.close()
        except Exception:
            pass
