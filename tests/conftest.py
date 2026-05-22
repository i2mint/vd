"""
Pytest fixtures shared across the vd test suite.

The headline fixture is :func:`client` — parametrized over every backend vd
can reach in the current environment. A test that takes ``client`` runs once
per backend, which is how the suite proves the facade contract holds uniformly.

Two kinds of backend run:

- **Embedded backends** (``memory``, ``chroma``, ``faiss``, ``duckdb``,
  ``lancedb``, ``qdrant``, ``sqlite_vec``, ``milvus``) need no server. Each
  test gets a fresh client. ``sqlite_vec`` is skipped on a Python whose
  ``sqlite3`` lacks loadable-extension support; ``milvus`` runs against the
  embedded Milvus Lite engine and is skipped if ``milvus-lite`` is absent.
- **Server backends** (``pgvector``, ``redis``, ``elasticsearch``,
  ``weaviate``, ``mongodb``) need a running container — see
  ``tests/docker-compose.yml``. Each is TCP-probed and **skipped** when its
  container is down, so the suite stays green in a plain CI environment.

Connection settings for the server backends are environment-overridable
(``VD_PGVECTOR_DSN``, ``VD_REDIS_HOST``/``VD_REDIS_PORT``,
``VD_ELASTICSEARCH_URL``, ``VD_WEAVIATE_HOST``, ``VD_MONGODB_URI``).
"""

import hashlib
import importlib.util
import os
import socket
import sqlite3

import pytest

import vd

#: Embedding dimension used by the test embedder.
EMBED_DIM = 16

#: Backends that need no server. Each test gets a fresh client.
EMBEDDED_BACKENDS = [
    "memory",
    "chroma",
    "faiss",
    "duckdb",
    "lancedb",
    "qdrant",
    "sqlite_vec",
    "milvus",  # verified against the embedded Milvus Lite engine
]

#: Server backends — each needs a container (``tests/docker-compose.yml``).
#: ``probe`` is TCP-probed; the backend is skipped when the port is closed.
#: ``connect_kwargs`` builds the :func:`vd.connect` arguments (env-overridable).
SERVER_BACKENDS = {
    "pgvector": {
        "probe": ("localhost", 5432),
        "connect_kwargs": lambda: {
            "dsn": os.environ.get(
                "VD_PGVECTOR_DSN", "postgresql://vd:vd@localhost:5432/vd"
            )
        },
    },
    "redis": {
        "probe": ("localhost", 6379),
        "connect_kwargs": lambda: {
            "host": os.environ.get("VD_REDIS_HOST", "localhost"),
            "port": int(os.environ.get("VD_REDIS_PORT", "6379")),
        },
    },
    "elasticsearch": {
        "probe": ("localhost", 9200),
        "connect_kwargs": lambda: {
            "url": os.environ.get("VD_ELASTICSEARCH_URL", "http://localhost:9200")
        },
    },
    "weaviate": {
        "probe": ("localhost", 8080),
        "connect_kwargs": lambda: {
            "host": os.environ.get("VD_WEAVIATE_HOST", "localhost")
        },
    },
    "mongodb": {
        # Host port 27018 — see tests/docker-compose.yml (avoids colliding
        # with a developer's native mongod on the default 27017).
        "probe": ("localhost", 27018),
        "connect_kwargs": lambda: {
            "uri": os.environ.get(
                "VD_MONGODB_URI", "mongodb://localhost:27018/?directConnection=true"
            )
        },
    },
}

#: Every backend the parametrized ``client`` fixture sweeps over.
ALL_BACKENDS = EMBEDDED_BACKENDS + list(SERVER_BACKENDS)


# --------------------------------------------------------------------------- #
# Availability probes
# --------------------------------------------------------------------------- #


def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return ``True`` if a TCP connection to ``host:port`` succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _sqlite_ext_supported() -> bool:
    """Return ``True`` if this Python's sqlite3 supports loadable extensions."""
    try:
        conn = sqlite3.connect(":memory:")
        ok = hasattr(conn, "enable_load_extension")
        conn.close()
        return ok
    except Exception:
        return False


def _unavailable_reason(name: str) -> str | None:
    """Return a skip reason for backend ``name``, or ``None`` if it can run."""
    if name == "sqlite_vec" and not _sqlite_ext_supported():
        return "sqlite3 was built without loadable-extension support"
    if name == "milvus" and importlib.util.find_spec("milvus_lite") is None:
        return "milvus-lite not installed (embedded Milvus engine unavailable)"
    if name in SERVER_BACKENDS:
        host, port = SERVER_BACKENDS[name]["probe"]
        if not _tcp_open(host, port):
            return (
                f"{name!r} server unreachable at {host}:{port} "
                f"— start it with tests/docker-compose.yml"
            )
    return None


def _connect_kwargs(name: str) -> dict:
    """Return the :func:`vd.connect` kwargs for backend ``name``."""
    if name in SERVER_BACKENDS:
        return SERVER_BACKENDS[name]["connect_kwargs"]()
    return {}


def _drop_all_collections(client) -> None:
    """
    Delete every collection on ``client`` — best-effort.

    Server backends keep state across runs; embedded backends are fresh each
    time. Dropping everything before and after each test makes a run against a
    live server idempotent (a re-run does not trip "already exists").
    """
    try:
        names = list(client.list_collections())
    except Exception:
        return
    for name in names:
        try:
            client.delete_collection(name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Test embedder
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Backend fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(params=ALL_BACKENDS)
def backend_name(request):
    """Each reachable backend, one at a time; unreachable ones are skipped."""
    name = request.param
    if name not in vd.list_backends():
        pytest.skip(f"backend {name!r} is not installed")
    reason = _unavailable_reason(name)
    if reason:
        pytest.skip(reason)
    return name


@pytest.fixture
def client(backend_name, embedder):
    """A fresh, connected client for each backend (with an embedder)."""
    connection = vd.connect(
        backend_name, embedder=embedder, **_connect_kwargs(backend_name)
    )
    _drop_all_collections(connection)
    yield connection
    _drop_all_collections(connection)
    if hasattr(connection, "close"):
        try:
            connection.close()
        except Exception:
            pass
