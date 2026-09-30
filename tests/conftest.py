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
  ``weaviate``, ``mongodb``, ``pinecone`` via Pinecone Local, and
  ``qdrant_server``: the qdrant adapter with ``url=``) need a running
  container — see ``tests/docker-compose.yml``; none needs an account. Each is TCP-probed and **skipped** when its
  container is down, so the suite stays green in a plain CI environment.

Connection settings for the server backends are environment-overridable
(``VD_PGVECTOR_DSN``, ``VD_REDIS_HOST``/``VD_REDIS_PORT``,
``VD_ELASTICSEARCH_URL``, ``VD_WEAVIATE_HOST``, ``VD_MONGODB_URI``,
``VD_PINECONE_HOST``/``VD_PINECONE_API_KEY``, ``VD_QDRANT_URL``). An entry may
set ``"backend"`` to test another entry's adapter under a different setup.
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
#: The server address (from ``connect_kwargs``, falling back to ``probe``) is
#: TCP-probed; the backend is skipped when the port is closed. Only local
#: servers are used unless ``VD_ALLOW_REMOTE_TESTS=1``: the ``client`` fixture
#: deletes every collection it can see, which would wipe a real account.
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
    "pinecone": {
        # Pinecone Local, the official in-memory emulator (no account):
        # ghcr.io/pinecone-io/pinecone-local. It speaks the pre-2026-07 API, so
        # these tests need the pinecone SDK < 10 until the emulator catches up.
        "probe": ("localhost", 5080),
        "connect_kwargs": lambda: {
            "api_key": os.environ.get("VD_PINECONE_API_KEY", "pclocal"),
            "host": os.environ.get("VD_PINECONE_HOST", "http://localhost:5080"),
        },
    },
    "qdrant_server": {
        # The qdrant adapter against a real Qdrant server (url=), next to the
        # embedded "qdrant" entry above: exercises the network client paths.
        "backend": "qdrant",
        "probe": ("localhost", 6333),
        "connect_kwargs": lambda: {
            "url": os.environ.get("VD_QDRANT_URL", "http://localhost:6333"),
            "check_compatibility": False,
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


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def server_address(name: str) -> tuple:
    """
    The ``(host, port)`` a server entry will connect to.

    ``host`` is ``None`` when it can't be determined (a keyword-style or
    multi-host DSN, ...); callers must then treat the target as remote.
    """
    import re
    from urllib.parse import parse_qs, urlsplit

    entry = SERVER_BACKENDS[name]
    default_host, default_port = entry["probe"]
    kwargs = entry["connect_kwargs"]()
    for key in ("url", "dsn", "uri", "host"):
        value = kwargs.get(key)
        if not isinstance(value, str):
            continue
        if "://" in value:
            parts = urlsplit(value)
            if "," in parts.netloc:  # multi-host URI
                return None, default_port
            query_host = parse_qs(parts.query).get("host")
            try:
                port = parts.port or default_port
            except ValueError:
                return None, default_port
            if query_host:
                return None, port  # host given in the query string
            return parts.hostname or default_host, port
        if key == "dsn":  # keyword DSN, e.g. "host=db port=5432 dbname=vd"
            found = re.search(r"\bhost(?:addr)?\s*=\s*'?([^\s']+)", value)
            port = re.search(r"\bport\s*=\s*'?(\d+)", value)
            if not found or "," in found.group(1):
                return None, default_port
            return found.group(1), int(port.group(1)) if port else default_port
        if key == "host":
            return value, int(kwargs.get("port", default_port))
    return default_host, default_port


def _unavailable_reason(name: str) -> str | None:
    """Return a skip reason for backend ``name``, or ``None`` if it can run."""
    if name == "sqlite_vec" and not _sqlite_ext_supported():
        return "sqlite3 was built without loadable-extension support"
    if name == "milvus" and importlib.util.find_spec("milvus_lite") is None:
        return "milvus-lite not installed (embedded Milvus engine unavailable)"
    if name == "pinecone":
        import pinecone

        major = str(getattr(pinecone, "__version__", "0")).split(".")[0]
        if major.isdigit() and int(major) >= 10:
            return (
                "Pinecone Local speaks the pre-2026-07 API; the pinecone SDK "
                ">= 10 cannot drive it. Install 'pinecone<10' to run these."
            )
    if name in SERVER_BACKENDS:
        host, port = server_address(name)
        if host not in _LOCAL_HOSTS and os.environ.get("VD_ALLOW_REMOTE_TESTS") != "1":
            return (
                f"{name!r} points at {host or 'a host that could not be parsed'}, "
                f"not a local server; the tests "
                f"delete every collection they see. Set VD_ALLOW_REMOTE_TESTS=1 "
                f"to allow it."
            )
        if host is None or not _tcp_open(host, port):
            return (
                f"{name!r} server unreachable at {host}:{port} "
                f"— start it with tests/docker-compose.yml"
            )
    return None


def backend_of(name: str) -> str:
    """The ``vd.connect`` backend for a fixture entry (entries may alias one)."""
    return SERVER_BACKENDS.get(name, {}).get("backend", name)


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
    if backend_of(name) not in vd.list_backends():
        pytest.skip(f"backend {name!r} is not installed")
    reason = _unavailable_reason(name)
    if reason:
        pytest.skip(reason)
    return name


@pytest.fixture
def client(backend_name, embedder):
    """A fresh, connected client for each backend (with an embedder)."""
    connection = vd.connect(
        backend_of(backend_name), embedder=embedder, **_connect_kwargs(backend_name)
    )
    _drop_all_collections(connection)
    yield connection
    _drop_all_collections(connection)
    if hasattr(connection, "close"):
        try:
            connection.close()
        except Exception:
            pass
