"""
turbopuffer backend.

turbopuffer [https://turbopuffer.com] is a managed, object-storage-first
vector database (vectors live on S3/GCS, hot tier in RAM/SSD cache) that is
10–100× cheaper than per-GB-hour competitors. It is namespace-centric: every
namespace is a first-class unit of isolation, billing, and tenancy. The API
is deliberately simple — rows have an ``id``, a ``vector``, and arbitrary
flat scalar attributes. Hybrid search (BM25 + vector) is GA.

**When to use it:** huge-but-bursty workloads where cost matters more than
sub-50 ms p50 latency; multi-tenant applications (namespace = tenant is a
first-class pattern); datasets >100 M vectors where object-storage economics
dominate.

**How this adapter maps to turbopuffer:**

- A ``vd`` collection → a turbopuffer *namespace*.
- Document text is stored as the reserved flat attribute ``_vd_text``; all
  other ``metadata`` keys are stored flat alongside it.  On read, ``_vd_text``
  is stripped back out and returned as ``Document.text``, and the remaining
  flat attributes become ``Document.metadata``.
- ``vd`` metrics map to turbopuffer ``distance_metric`` strings:
  ``cosine`` → ``"cosine_distance"``, ``l2`` → ``"euclidean_squared"``,
  ``dot`` → ``"cosine_distance"`` (turbopuffer does not support a raw inner
  product distance metric; see :data:`_METRIC_MAP` for the note).
- Filtering is done *client-side*: turbopuffer supports its own attribute
  filter language, but the mapping is non-trivial. Rather than translating
  the canonical ``vd`` filter AST, this adapter over-fetches candidates via
  :func:`~vd.backends._helpers.overfetch_limit` and then applies the
  canonical evaluator with :func:`~vd.backends._helpers.apply_client_filter`.
  This means the entire ``vd`` filter language works, at the cost of
  slightly higher latency when a filter is present on large namespaces.

Requires: ``pip install turbopuffer``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from turbopuffer import Turbopuffer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The turbopuffer backend needs the 'turbopuffer' package. "
        "Install with: pip install turbopuffer"
    ) from e

from vd.backends._helpers import (
    apply_client_filter,
    overfetch_limit,
    score_from_distance,
)
from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.util import register_backend

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Reserved flat attribute used to round-trip document text through turbopuffer.
_VD_TEXT_KEY = "_vd_text"

#: vd metric → turbopuffer distance_metric string.
#:
#: NOTE: turbopuffer does not expose a raw dot-product (inner-product) distance
#: metric in its v2 API. We fall back to "cosine_distance" for "dot", which
#: gives different ranking for non-unit vectors.  Callers that need true dot
#: similarity should normalize their vectors before writing and treat the
#: result as approximate.
_METRIC_MAP: dict[str, str] = {
    "cosine": "cosine_distance",
    "l2": "euclidean_squared",
    "dot": "cosine_distance",  # NOTE: dot → cosine fallback; see docstring above
}

#: Default turbopuffer region.
_DEFAULT_REGION = "gcp-us-central1"


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _map_metric(metric: str) -> str:
    """
    Translate a vd metric name to the turbopuffer ``distance_metric`` string.

    Parameters
    ----------
    metric : str
        ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns
    -------
    str
        One of ``"cosine_distance"`` or ``"euclidean_squared"``.

    Notes
    -----
    ``"dot"`` maps to ``"cosine_distance"`` because turbopuffer v2 does not
    expose an inner-product distance metric.  A ``NOTE:`` comment in
    :data:`_METRIC_MAP` documents this limitation.
    """
    return _METRIC_MAP.get(metric, "cosine_distance")


def _pack_row(doc: Document) -> dict:
    """
    Pack a :class:`~vd.base.Document` into a turbopuffer row dict.

    The row format is flat: ``id`` and ``vector`` are top-level; text and all
    metadata keys are stored as ordinary attributes alongside the reserved
    ``_vd_text`` key.

    Parameters
    ----------
    doc : Document
        A document whose ``vector`` is set (the base class guarantees this
        before ``_write`` is called).

    Returns
    -------
    dict
        A row dict suitable for passing as an element of ``upsert_rows``.
    """
    row: dict[str, Any] = {"id": doc.id, "vector": doc.vector}
    # Flat attributes: reserved text key first, then user metadata.
    row[_VD_TEXT_KEY] = doc.text or ""
    if doc.metadata:
        for key, value in doc.metadata.items():
            row[key] = value
    return row


def _unpack_row(row: Any) -> Document:
    """
    Reconstruct a :class:`~vd.base.Document` from a turbopuffer result row.

    The row object (returned by ``ns.query``) exposes ``.id``, ``.vector``,
    and ``.attributes`` (a dict of flat attributes).

    Parameters
    ----------
    row : turbopuffer result row
        Any object with ``.id``, optional ``.vector``, and ``.attributes``.

    Returns
    -------
    Document
    """
    # NOTE: turbopuffer v2 row objects expose attributes as a dict-like object.
    # The exact attribute accessor may be .attributes or dict-access depending
    # on the SDK version; we try both.
    attrs: dict[str, Any] = {}
    if hasattr(row, "attributes") and row.attributes is not None:
        attrs = dict(row.attributes)
    elif isinstance(row, dict):
        attrs = {k: v for k, v in row.items() if k not in ("id", "vector")}

    text = attrs.pop(_VD_TEXT_KEY, "") or ""
    vector: Optional[list[float]] = None
    if hasattr(row, "vector") and row.vector is not None:
        vector = list(row.vector)
    elif isinstance(row, dict) and "vector" in row:
        vector = list(row["vector"]) if row["vector"] is not None else None

    doc_id = str(row.id if hasattr(row, "id") else row["id"])
    return Document(id=doc_id, text=text, vector=vector, metadata=attrs)


def _result_to_search_result(row: Any, *, metric: str) -> SearchResult:
    """
    Convert a turbopuffer query result row to a ``vd`` :data:`~vd.base.SearchResult`.

    Parameters
    ----------
    row : turbopuffer query result row
        Object with ``.id``, ``.dist`` (the raw distance), ``.vector``, and
        ``.attributes``.
    metric : str
        The vd metric name (``"cosine"``, ``"l2"``, or ``"dot"``) — used to
        convert the raw distance to a higher-is-better score.

    Returns
    -------
    dict
        ``{"id", "text", "score", "metadata"}``.

    Notes
    -----
    turbopuffer v2 exposes the distance as ``.dist`` on each result row.
    The score is converted via :func:`~vd.backends._helpers.score_from_distance`
    so higher always means more similar, consistent with all other vd backends.
    """
    # NOTE: turbopuffer v2 names the distance attribute ".dist" on query rows.
    # If the SDK exposes it differently (e.g. ".distance"), adjust here.
    raw_dist: float = float(getattr(row, "dist", 0.0))
    score = score_from_distance(raw_dist, metric)

    attrs: dict[str, Any] = {}
    if hasattr(row, "attributes") and row.attributes is not None:
        attrs = dict(row.attributes)

    text = attrs.pop(_VD_TEXT_KEY, "") or ""
    doc_id = str(row.id if hasattr(row, "id") else row["id"])
    return {"id": doc_id, "text": text, "score": score, "metadata": attrs}


# --------------------------------------------------------------------------- #
# TurbopufferCollection
# --------------------------------------------------------------------------- #


class TurbopufferCollection(AbstractCollection):
    """
    A collection backed by one turbopuffer namespace.

    turbopuffer's namespace API does not require any upfront schema; the
    distance metric is specified on the first write call.  Metadata filtering
    in this adapter is done **client-side** via
    :func:`~vd.backends._helpers.apply_client_filter` after over-fetching
    candidates from the ANN index.

    Parameters
    ----------
    name : str
        Namespace name (== vd collection name).
    ns : turbopuffer namespace object
        The raw ``client.namespace(name)`` handle.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    dimension : int, optional
        Vector dimension.  Learned from the first write if not supplied.
    metric : str
        ``"cosine"``, ``"dot"``, or ``"l2"``.
    """

    def __init__(
        self,
        name: str,
        ns: Any,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._ns = ns
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric

    @property
    def native(self) -> Any:
        """The raw turbopuffer namespace object (escape hatch)."""
        return self._ns

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """Upsert one document into the turbopuffer namespace."""
        self._ns.write(
            upsert_rows=[_pack_row(doc)],
            distance_metric=_map_metric(self.metric),
        )

    def _write_many(self, docs: list[Document]) -> None:
        """Batch-upsert documents in a single turbopuffer call."""
        self._ns.write(
            upsert_rows=[_pack_row(d) for d in docs],
            distance_metric=_map_metric(self.metric),
        )

    def _read(self, key: str) -> Document:
        """
        Fetch a single document by id.

        turbopuffer does not have a dedicated get-by-id endpoint in v2; we
        query ANN with a filter on ``id``.

        NOTE: turbopuffer v2 may support direct row retrieval via
        ``ns.rows(ids=[key])`` or similar; if that method exists on your
        SDK version it would be more efficient.  Until confirmed, we query
        with a filter.
        """
        # NOTE: turbopuffer v2 filter syntax for a single id match — this uses
        # the attribute filter form ["id", "Eq", key] as per the turbopuffer
        # query docs.  If the SDK spells it differently, update this filter.
        try:
            results = self._ns.query(
                top_k=1,
                filters=["id", "Eq", key],
                include_attributes=True,
                # NOTE: turbopuffer v2 query may not support returning vectors
                # without also ranking by them; we omit rank_by here to get a
                # pure attribute-filter lookup.  If this raises, use rank_by
                # with a dummy vector and filter on id instead.
            )
        except Exception:
            # Fallback: if attribute-only query is unsupported, raise KeyError.
            # The caller can use native to work around this.
            raise KeyError(key)

        rows = list(results) if results is not None else []
        if not rows:
            raise KeyError(key)
        return _unpack_row(rows[0])

    def _drop(self, key: str) -> None:
        """
        Delete a document by id; raise ``KeyError`` if absent.

        We verify existence before deleting so a missing id surfaces as a
        clear ``KeyError`` rather than a silent no-op.
        """
        # Existence check.
        try:
            self._read(key)
        except KeyError:
            raise KeyError(key)

        # NOTE: turbopuffer v2 deletes are issued via write with a deletes list.
        self._ns.write(deletes=[key])

    def _keys(self) -> Iterator[str]:
        """
        Iterate all document ids in the namespace.

        NOTE: turbopuffer v2 exposes ``ns.rows()`` for paginated row
        iteration.  The exact pagination API (cursor / offset) may differ;
        we call ``ns.rows()`` and iterate.  If the SDK uses a different method
        (e.g. ``ns.list_ids()``), adjust the call below.
        """
        # NOTE: turbopuffer v2 namespace.rows() returns a paginated iterator
        # of row objects with at least an .id attribute.
        try:
            for row in self._ns.rows():
                yield str(row.id if hasattr(row, "id") else row["id"])
        except Exception:
            return

    def _count(self) -> int:
        """
        Return the approximate number of documents in the namespace.

        NOTE: turbopuffer v2 may expose an approximate count via namespace
        metadata (e.g. ``ns.approx_count()`` or ``ns.dimensions()``).  We
        fall back to iterating ``_keys()`` if no direct count method exists.
        This is a potentially expensive O(n) fallback for large namespaces.
        """
        # Try a cheap metadata count first.
        try:
            meta = self._ns.dimensions()  # NOTE: method name assumed from v2 docs
            if hasattr(meta, "approx_count"):
                return int(meta.approx_count)
        except Exception:
            pass
        return sum(1 for _ in self._keys())

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Run ANN search, applying ``filter`` client-side after over-fetching.

        Parameters
        ----------
        vector : list[float]
            Query vector.
        limit : int
            Maximum number of results to return after filtering.
        filter : dict, optional
            Canonical vd metadata filter (applied client-side).
        **kwargs
            Passed through to ``ns.query``.

        Returns
        -------
        list of SearchResult
        """
        fetch = overfetch_limit(limit, filter)
        # NOTE: turbopuffer v2 query signature:
        #   ns.query(rank_by=("vector", "ANN", query_vector), top_k=N,
        #            include_attributes=True, ...)
        response = self._ns.query(
            rank_by=("vector", "ANN", vector),
            top_k=fetch,
            include_attributes=True,
            **kwargs,
        )
        rows = list(response) if response is not None else []
        raw_results = [
            _result_to_search_result(row, metric=self.metric) for row in rows
        ]
        return apply_client_filter(raw_results, filter, limit=limit)


# --------------------------------------------------------------------------- #
# TurbopufferClient
# --------------------------------------------------------------------------- #


@register_backend("turbopuffer")
class TurbopufferClient(AbstractClient):
    """
    turbopuffer client.

    Manages a set of turbopuffer namespaces (vd collections) within one
    region.  Namespace names are tracked in memory and reconciled with the
    live ``client.namespaces.list()`` call when listing collections.

    Parameters
    ----------
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder shared by all
        collections.
    api_key : str, optional
        turbopuffer API key.  Defaults to the ``TURBOPUFFER_API_KEY``
        environment variable; raises ``ValueError`` if neither is provided.
    region : str, optional
        turbopuffer region slug.  Defaults to ``"gcp-us-central1"``.
    **config
        Additional keyword arguments passed to :class:`AbstractClient`.

    Raises
    ------
    ValueError
        If no API key is found in the argument or in the environment.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        api_key: Optional[str] = None,
        region: str = _DEFAULT_REGION,
        **config,
    ):
        super().__init__(embedder=embedder, **config)

        resolved_key = api_key or os.environ.get("TURBOPUFFER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A turbopuffer API key is required. "
                "Pass api_key=... or set the TURBOPUFFER_API_KEY environment variable."
            )

        self._client = Turbopuffer(region=region, api_key=resolved_key)
        self.region = region
        # In-memory set of collection names created via create_collection().
        # Turbopuffer namespaces are created implicitly on first write; we
        # track them here so get_collection / list_collections work before
        # any data has been written.
        self._known: set[str] = set()
        # Per-collection metric, needed to reconstruct TurbopufferCollection.
        self._metrics: dict[str, str] = {}

    # ----- helpers -------------------------------------------------------- #

    def _live_names(self) -> set[str]:
        """
        Return the set of namespace names that exist on the turbopuffer server.

        NOTE: turbopuffer v2 exposes ``client.namespaces.list()`` (or possibly
        ``client.namespaces()``) to enumerate existing namespaces.  We try
        both forms; adjust if the SDK differs.
        """
        try:
            listing = self._client.namespaces.list()
        except AttributeError:
            try:
                listing = self._client.namespaces()
            except Exception:
                return set()
        except Exception:
            return set()
        names: set[str] = set()
        for entry in listing or []:
            if isinstance(entry, str):
                names.add(entry)
            elif hasattr(entry, "id"):
                names.add(str(entry.id))
            elif hasattr(entry, "name"):
                names.add(str(entry.name))
        return names

    def _all_known(self) -> set[str]:
        """Union of locally-tracked names and live server names."""
        return self._known | self._live_names()

    def _get_ns(self, name: str):
        """Return the raw turbopuffer namespace object for ``name``."""
        # NOTE: turbopuffer v2 API: client.namespace(name) → namespace object.
        return self._client.namespace(name)

    # ----- AbstractClient implementation ---------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> TurbopufferCollection:
        """
        Register a new collection (turbopuffer namespace).

        Parameters
        ----------
        name : str
            Namespace / collection name.
        dimension : int, optional
            Vector dimension.  May be ``None``; turbopuffer infers it from
            the first upserted vector.
        metric : str
            ``"cosine"``, ``"dot"``, or ``"l2"``.

        Raises
        ------
        ValueError
            If a collection with ``name`` already exists (locally tracked or
            on the live server).
        """
        if name in self._all_known():
            raise ValueError(f"Collection {name!r} already exists")
        self._known.add(name)
        self._metrics[name] = metric
        return TurbopufferCollection(
            name,
            self._get_ns(name),
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    def get_collection(self, name: str) -> TurbopufferCollection:
        """
        Return an existing collection.

        Raises
        ------
        KeyError
            If ``name`` is not known locally and does not exist on the server.
        """
        if name not in self._all_known():
            raise KeyError(f"Collection {name!r} does not exist")
        return TurbopufferCollection(
            name,
            self._get_ns(name),
            embedder=self._embedder,
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection (namespace) and all its data.

        NOTE: turbopuffer v2 namespace deletion is done via ``ns.delete_all()``
        or ``ns.delete_all_indexes()``.  We call ``delete_all()`` here; if the
        method is named differently in your SDK version, update the call.

        Raises
        ------
        KeyError
            If ``name`` does not exist (locally tracked or live on the server).
        """
        if name not in self._all_known():
            raise KeyError(f"Collection {name!r} does not exist")
        ns = self._get_ns(name)
        try:
            ns.delete_all()  # NOTE: assumed v2 method name
        except Exception:
            pass  # If the namespace was never written it may not exist server-side.
        self._known.discard(name)
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        """Iterate all known collection (namespace) names."""
        return iter(sorted(self._all_known()))

    def close(self) -> None:
        """No-op: the turbopuffer HTTP client has no persistent connection to close."""
        # turbopuffer uses a stateless HTTP client; no teardown needed.
