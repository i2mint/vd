"""
The backend registry, the :func:`connect` factory, and small shared utilities.

This module is intentionally thin. It owns three things:

- the **registry** mapping a backend name (``"chroma"``, ``"qdrant"``, ...) to
  its adapter :class:`~vd.base.Client` class, via the :func:`register_backend`
  decorator;
- :func:`connect`, the one entry point users call to get a client;
- backend-agnostic helpers: document-input normalization, search-result
  ``egress`` functions, and vector math.

Everything *descriptive* about a backend (pip package, license, docs URLs,
deployment archetype, setup checks) lives in :mod:`vd.providers` and its
data file, not here.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Callable, Iterable, Optional

from vd.base import (
    BackendNotInstalledError,
    Client,
    Document,
    DocumentInput,
    SearchResult,
    Vector,
)

# --------------------------------------------------------------------------- #
# Backend registry
# --------------------------------------------------------------------------- #

#: name -> adapter Client class. Populated by the @register_backend decorator
#: as each backend module is imported (see vd/backends/__init__.py).
_backends: dict[str, type] = {}


def register_backend(name: str) -> Callable[[type], type]:
    """
    Class decorator: register an adapter :class:`~vd.base.Client` under ``name``.

    Examples
    --------
    >>> from vd.base import AbstractClient
    >>> @register_backend('example')           # doctest: +SKIP
    ... class ExampleClient(AbstractClient):
    ...     ...
    """

    def decorator(client_class: type) -> type:
        client_class.backend_name = name
        _backends[name] = client_class
        return client_class

    return decorator


def get_backend(name: str) -> type:
    """
    Return the adapter :class:`~vd.base.Client` class registered under ``name``.

    Raises
    ------
    BackendNotInstalledError
        If ``name`` is a known backend whose client library is not installed.
    ValueError
        If ``name`` is not a known backend at all.
    """
    if name in _backends:
        return _backends[name]

    # Not registered: distinguish "known but unavailable" from "unknown".
    try:
        from vd.providers import install_command, provider

        meta = provider(name)
    except Exception:  # pragma: no cover - providers data missing
        meta = None

    if meta is not None:
        cmd = ""
        try:
            cmd = install_command(name)
        except Exception:  # pragma: no cover
            cmd = f"pip install vd[{name}]"
        raise BackendNotInstalledError(
            f"Backend {name!r} ({meta.get('display_name', name)}) is not "
            f"available — its client library is probably not installed.\n"
            f"  {cmd}\n"
            f"Then retry. Run  vd.check_requirements({name!r})  for full "
            f"setup guidance."
        )

    registered = ", ".join(sorted(_backends)) or "none"
    raise ValueError(
        f"Unknown backend {name!r}. Registered backends: {registered}. "
        f"Run  vd.print_backends_table()  to see every backend vd knows about."
    )


def list_backends() -> list[str]:
    """Return the names of all backends with a registered (importable) adapter."""
    return sorted(_backends)


# --------------------------------------------------------------------------- #
# connect — the entry point
# --------------------------------------------------------------------------- #


def connect(
    backend: str,
    *,
    embedder: Optional[Callable[[str], Vector]] = None,
    **backend_kwargs,
) -> Client:
    """
    Connect to a vector database backend and return its :class:`~vd.base.Client`.

    This is the single entry point of ``vd``. Switching vector databases is a
    one-argument change here.

    Parameters
    ----------
    backend : str
        Backend name: ``"memory"``, ``"chroma"``, ``"qdrant"``, ``"faiss"``,
        ``"lancedb"``, ``"sqlite_vec"``, ``"duckdb"``, ``"pgvector"``,
        ``"pinecone"``, ... Run :func:`vd.list_backends` for what is installed.
    embedder : callable, optional
        A ``text -> vector`` function. Supply it only if you want the
        *convenience* of passing raw text to ``collection[key] = "text"`` and
        ``collection.search("query text")``. ``vd`` never embeds on its own —
        with no embedder, pass :class:`~vd.base.Document` objects with vectors
        and pre-computed query vectors.
    **backend_kwargs
        Backend-specific connection options (``persist_directory``, ``url``,
        ``api_key``, ``path``, ...). See each adapter's docstring.

    Returns
    -------
    Client
        A connected client — a ``Mapping`` of collection name to collection.

    Examples
    --------
    >>> client = connect('memory')                                  # doctest: +SKIP
    >>> client = connect('chroma', persist_directory='./db')        # doctest: +SKIP
    >>> client = connect('qdrant', url='http://localhost:6333')     # doctest: +SKIP
    """
    client_class = get_backend(backend)
    return client_class(embedder=embedder, **backend_kwargs)


# --------------------------------------------------------------------------- #
# Document-input normalization
# --------------------------------------------------------------------------- #


def _generate_id(text: str, *, prefix: str = "doc") -> str:
    """
    Generate a unique, mostly-deterministic id for a document.

    Examples
    --------
    >>> _generate_id("Hello world").startswith('doc_')
    True
    """
    text_hash = hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:8]
    return f"{prefix}_{text_hash}_{uuid.uuid4().hex[:8]}"


def normalize_document_input(
    doc_input: DocumentInput,
    *,
    auto_id: bool = True,
) -> Document:
    """
    Normalize a flexible document input to a :class:`~vd.base.Document`.

    Accepted shapes: a :class:`~vd.base.Document`; a ``str`` (just text); a
    tuple ``(text, id)``, ``(text, metadata)``, or ``(text, id, metadata)``.

    Parameters
    ----------
    doc_input : DocumentInput
        The input to normalize.
    auto_id : bool
        When the input carries no id, generate one (vs. leaving it empty).

    Examples
    --------
    >>> normalize_document_input(("Hello", "doc1")).id
    'doc1'
    >>> normalize_document_input(("Hello", {"k": "v"})).metadata
    {'k': 'v'}
    >>> normalize_document_input("Hello world").id.startswith('doc_')
    True
    """
    if isinstance(doc_input, Document):
        if not doc_input.id and auto_id:
            doc_input.id = _generate_id(doc_input.text)
        return doc_input

    if isinstance(doc_input, str):
        doc_id = _generate_id(doc_input) if auto_id else ""
        return Document(id=doc_id, text=doc_input)

    if isinstance(doc_input, tuple):
        if len(doc_input) == 2:
            text, second = doc_input
            if isinstance(second, dict):
                doc_id = _generate_id(text) if auto_id else ""
                return Document(id=doc_id, text=text, metadata=second)
            return Document(id=second, text=text)
        if len(doc_input) == 3:
            text, doc_id, metadata = doc_input
            return Document(id=doc_id, text=text, metadata=metadata or {})

    raise TypeError(
        f"Cannot interpret {type(doc_input).__name__} as a document input. "
        f"Use a str, a (text, id)/(text, metadata)/(text, id, metadata) tuple, "
        f"or a vd.Document."
    )


# --------------------------------------------------------------------------- #
# Egress functions — transform a search result on the way out
# --------------------------------------------------------------------------- #


def text_only(result: SearchResult) -> str:
    """Egress: keep only the text. ``>>> text_only({'text': 'hi'})`` -> ``'hi'``."""
    return result["text"]


def id_only(result: SearchResult) -> str:
    """Egress: keep only the document id."""
    return result["id"]


def id_and_score(result: SearchResult) -> tuple[str, float]:
    """Egress: keep ``(id, score)``."""
    return result["id"], result["score"]


def id_text_score(result: SearchResult) -> tuple[str, str, float]:
    """Egress: keep ``(id, text, score)``."""
    return result["id"], result["text"], result["score"]


# --------------------------------------------------------------------------- #
# Vector math
# --------------------------------------------------------------------------- #


def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
    """
    Cosine similarity of two vectors (1.0 identical, 0.0 orthogonal).

    Examples
    --------
    >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
    1.0
    >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
    0.0
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def euclidean_distance(vec1: Vector, vec2: Vector) -> float:
    """
    Euclidean (L2) distance between two vectors.

    Examples
    --------
    >>> euclidean_distance([1.0, 0.0], [1.0, 0.0])
    0.0
    """
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5


def mean_vector(vectors: Iterable[Vector]) -> Vector:
    """
    Component-wise mean of a non-empty iterable of equal-length vectors.

    Examples
    --------
    >>> mean_vector([[0.0, 2.0], [2.0, 4.0]])
    [1.0, 3.0]
    """
    vectors = list(vectors)
    if not vectors:
        raise ValueError("mean_vector requires at least one vector")
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]
