"""
Backend adapters for the vector databases ``vd`` supports.

Importing this package registers every adapter whose client library is
installed. Each adapter lives in its own module and registers itself with the
:func:`~vd.util.register_backend` decorator; modules whose third-party client
is not installed fail to import and are skipped silently — that backend simply
will not appear in :func:`vd.list_backends`.
"""

#: Every backend module vd ships. Each is imported defensively below.
_BACKEND_MODULES = (
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


def _register_all() -> None:
    """Import each backend module, skipping any whose client is not installed."""
    import importlib

    for name in _BACKEND_MODULES:
        try:
            importlib.import_module(f"vd.backends.{name}")
        except ImportError:
            pass  # third-party client not installed — backend unavailable


_register_all()
