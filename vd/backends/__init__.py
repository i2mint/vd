"""
Backend adapters for the vector databases ``vd`` supports.

Adapters are registered **lazily**: ``import vd`` imports only the always-available
``memory`` adapter, because importing every installed client SDK (chromadb, lancedb,
...) costs seconds. An adapter module is imported the first time its backend is
resolved by name (:func:`vd.connect`, :func:`vd.util.get_backend`,
:func:`vd.connect_async`), or all at once by :func:`load_all`, which
:func:`vd.list_backends` and :func:`vd.list_async_backends` call. Each adapter lives
in its own module and registers itself with the :func:`~vd.util.register_backend`
decorator; a module whose third-party client is not installed fails to import and
is skipped silently, so that backend does not appear in :func:`vd.list_backends`.
"""

import threading

#: Every backend module vd ships; the module's name is the backend's name.
BACKEND_MODULES = (
    "memory",  # always available — no third-party dependency
    "chroma",
    "faiss",
    "sqlite_vec",
    "duckdb",
    "lancedb",
    "qdrant",
    "pgvector",
    "pinecone",
    "weaviate",
    "milvus",
    "redis",
    "elasticsearch",
    "mongodb",
    "turbopuffer",
)
#: Kept for callers of the old name.
_BACKEND_MODULES = BACKEND_MODULES

#: One adapter import at a time: client SDKs imported concurrently from several
#: threads can leave a shared dependency (numpy) half-initialised for good (vd#62).
_LOAD_LOCK = threading.RLock()


def load(name: str) -> bool:
    """Import the adapter module for backend ``name``; whether it is now available.

    ``False`` for a name that is not a shipped adapter and for one whose third-party
    client is not installed (the module then fails to import and is skipped). A
    name someone already registered (``register_backend``) is left alone, so a
    user's own adapter is never replaced by the shipped one (vd#62). Thread-safe:
    imports are serialised.
    """
    import importlib

    from vd.util import _backends

    if name not in BACKEND_MODULES:
        return False
    with _LOAD_LOCK:
        if name in _backends:
            return True
        try:
            importlib.import_module(f"vd.backends.{name}")
        except ImportError:
            return False  # third-party client not installed — backend unavailable
    return True


def load_all() -> None:
    """Import every adapter module whose client library is installed (the old import-time behaviour)."""
    for name in BACKEND_MODULES:
        load(name)


load("memory")  # the core adapter: no third-party import, so it stays eager
