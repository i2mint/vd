"""
Shared helpers for backend adapters.

Small, adapter-agnostic utilities used across several backends: turning a raw
distance into a "higher-is-better" score, and applying the canonical metadata
filter client-side for backends whose native filtering is absent or weaker
than ``vd``'s filter language.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from vd.base import Filter, SearchResult
from vd.filters import matches_filter

#: When a backend must filter client-side, fetch this many times ``limit``
#: candidates from the index first, so post-filtering still returns a full page.
OVERFETCH_FACTOR = 10


def score_from_distance(distance: float, metric: str) -> float:
    """
    Convert a raw backend distance to ``vd``'s canonical similarity score.

    Reference implementation of the cross-backend score contract documented
    in :mod:`vd.base` ("Score semantics"): every ``SearchResult`` ``score``
    is **higher-is-better** on a per-metric canonical scale, so fusion /
    dedup / threshold logic works the same way across adapters.

    Per-metric output:

    ============  ===============================  ======================
    metric        formula                          range
    ============  ===============================  ======================
    ``cosine``    ``1 - distance``                 ``[-1, 1]``
    ``dot``       ``-distance``                    ``(-inf, +inf)``
    ``l2``        ``1 / (1 + distance)``           ``(0, 1]``
    ============  ===============================  ======================

    Adapters whose backend already returns a higher-is-better number on a
    *different* per-metric scale (Elasticsearch kNN ``_score``, MongoDB
    Atlas ``vectorSearchScore``, Pinecone ``match.score``) **do not** route
    through this helper — they pass the native score through and document
    the deviation. Adapters whose backend returns a lower-is-better distance
    (Chroma, DuckDB, FAISS L2, LanceDB, Milvus L2, pgvector, Redis,
    sqlite-vec, Turbopuffer, Weaviate) call this helper to canonicalize.

    Parameters
    ----------
    distance : float
        The backend's raw distance (lower = closer). For ``dot``, this is
        the convention some backends use of negating the inner product so
        that "distance" sorts the same way; for ``cosine``, the cosine
        distance in ``[0, 2]``; for ``l2``, the Euclidean (or squared
        Euclidean) distance in ``[0, +inf)``.
    metric : str
        ``"cosine"``, ``"dot"``, or ``"l2"``. Unknown metrics fall through
        to the ``l2`` formula so the result is at least bounded.

    Examples
    --------
    >>> score_from_distance(0.0, 'cosine')
    1.0
    >>> round(score_from_distance(1.0, 'l2'), 3)
    0.5
    >>> score_from_distance(-0.7, 'dot')
    0.7
    """
    if metric == "cosine":
        # Cosine distance is in [0, 2]; similarity = 1 - distance ∈ [-1, 1].
        return 1.0 - distance
    if metric == "dot":
        # Many backends report negative inner product as the "distance" so
        # the smallest "distance" is the most similar; un-negate to recover
        # the raw inner product (vd's canonical dot score).
        return -distance
    # l2 (and any unknown): squash a non-negative distance into (0, 1].
    return 1.0 / (1.0 + distance)


def overfetch_limit(limit: int, filter: Optional[Filter]) -> int:
    """Return how many candidates to fetch: ``limit``, or more if filtering."""
    return limit * OVERFETCH_FACTOR if filter else limit


def apply_client_filter(
    results: Iterable[SearchResult],
    filter: Optional[Filter],
    *,
    limit: int,
) -> list[SearchResult]:
    """
    Keep the first ``limit`` results whose metadata satisfies ``filter``.

    For backends that cannot filter natively (FAISS) or whose native filtering
    does not cover the full ``vd`` filter language: over-fetch, then filter
    here against the canonical evaluator so semantics match every other backend.
    """
    out: list[SearchResult] = []
    for result in results:
        if filter and not matches_filter(result.get("metadata") or {}, filter):
            continue
        out.append(result)
        if len(out) >= limit:
            break
    return out


def coerce_metadata(metadata: Any) -> dict:
    """Return a plain ``dict`` for storage, treating ``None`` as ``{}``."""
    return dict(metadata) if metadata else {}
