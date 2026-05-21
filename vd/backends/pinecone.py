"""
Pinecone backend.

Pinecone is a fully managed, serverless-first vector database. New indexes are
serverless (no pod provisioning) and are billed per read unit / write unit /
storage. It is the "zero-ops canonical" choice: polished UX, namespace
multi-tenancy, sparse+dense hybrid search (2025), and a consistently maintained
Python SDK (renamed ``pinecone-client`` → ``pinecone`` at v5.1.0; current v9.x
requires Python ≥3.10).

**When to use:** you want managed infrastructure with no servers to operate,
you are on AWS us-east-1 (Starter tier), or you need namespace-level isolation
at scale. Reach for Qdrant Cloud instead if you need a portable exit path or
a permanent free 1 GB cluster outside AWS us-east-1.

**How this adapter maps onto Pinecone:**

- A ``vd`` *collection* maps to a Pinecone **serverless index**. Index creation
  requires a vector dimension up front; if ``dimension`` is ``None`` at
  ``create_collection`` time, the index is created lazily on the first write
  (like the Qdrant adapter). Metric mapping: ``cosine``→``"cosine"``,
  ``dot``→``"dotproduct"``, ``l2``→``"euclidean"``.

- A ``vd`` *client* wraps ``pinecone.Pinecone``. The raw client is accessible
  via ``client.client``; the raw index via ``collection.native``.

- **Metadata flattening (important limitation):** Pinecone metadata must be a
  *flat* dict of ``str | int | float | bool | list[str]`` values. Nested dicts
  are not allowed. This adapter stores document text under the reserved key
  ``"_vd_text"`` and merges the user's (already-flat) metadata dict into the
  same flat Pinecone metadata dict. On read, ``"_vd_text"`` is popped back out
  and returned as ``Document.text``; the remainder becomes ``Document.metadata``.
  Consequence: user metadata keys must not collide with ``"_vd_text"``, and
  nested metadata values are not supported — they will be rejected by Pinecone
  at upsert time with a Pinecone API error.

- **Filter dialect:** Pinecone's metadata filter is MongoDB-ish and supports
  ``$eq $ne $gt $gte $lt $lte $in $nin $and $or`` but **not** ``$not`` or
  ``$exists``. Filters using those two operators raise
  :class:`~vd.base.UnsupportedFilterError` before the query is sent.
  A bare ``{"field": value}`` is translated to ``{"field": {"$eq": value}}`` by
  :func:`_to_pinecone_filter`; all explicit operators are passed through.

Requires: ``pip install pinecone``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The pinecone backend needs the 'pinecone' package (NOT 'pinecone-client'). "
        "Install with: pip install pinecone"
    ) from e

from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.filters import SUPPORTED_FILTER_OPERATORS
from vd.util import register_backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Reserved metadata key used to store the document text inside Pinecone's
#: flat metadata dict. Must not appear in user metadata.
_VD_TEXT_KEY = "_vd_text"

#: Operators not supported by Pinecone's metadata filter.
_UNSUPPORTED = frozenset({"$not", "$exists"})

#: Filter operators this adapter can honor.
PINECONE_FILTER_OPERATORS = SUPPORTED_FILTER_OPERATORS - _UNSUPPORTED

#: vd metric name → Pinecone metric string.
_METRIC_MAP: dict[str, str] = {
    "cosine": "cosine",
    "dot": "dotproduct",
    "l2": "euclidean",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _pinecone_metric(vd_metric: str) -> str:
    """
    Map a canonical ``vd`` metric name to the Pinecone metric string.

    Parameters
    ----------
    vd_metric : str
        One of ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns
    -------
    str
        The corresponding Pinecone metric string.

    Raises
    ------
    ValueError
        If ``vd_metric`` is not one of the three known metrics.

    Examples
    --------
    >>> _pinecone_metric("cosine")
    'cosine'
    >>> _pinecone_metric("dot")
    'dotproduct'
    >>> _pinecone_metric("l2")
    'euclidean'
    """
    try:
        return _METRIC_MAP[vd_metric]
    except KeyError:
        known = ", ".join(sorted(_METRIC_MAP))
        raise ValueError(f"Unknown metric {vd_metric!r}. Pinecone supports: {known}.")


def _pack_metadata(text: str, metadata: dict) -> dict:
    """
    Build the flat Pinecone metadata dict for one document.

    The document text is stored under :data:`_VD_TEXT_KEY`; the caller's
    metadata dict is merged in alongside it. The caller's dict is assumed to
    be already flat (``str | int | float | bool | list[str]`` values only).

    Parameters
    ----------
    text : str
        Document text to store.
    metadata : dict
        User-supplied metadata. Must not contain the key ``"_vd_text"``.

    Returns
    -------
    dict
        Flat dict ready for Pinecone upsert.

    Examples
    --------
    >>> _pack_metadata("hello", {"year": 2024})
    {'year': 2024, '_vd_text': 'hello'}
    """
    return {**metadata, _VD_TEXT_KEY: text}


def _unpack_metadata(raw_metadata: dict) -> tuple[str, dict]:
    """
    Split a raw Pinecone metadata dict into ``(text, user_metadata)``.

    Pops :data:`_VD_TEXT_KEY` out and returns the remainder as user metadata.

    Parameters
    ----------
    raw_metadata : dict
        The flat metadata dict returned by Pinecone.

    Returns
    -------
    tuple[str, dict]
        ``(text, user_metadata)`` where ``user_metadata`` does not contain
        ``_VD_TEXT_KEY``.

    Examples
    --------
    >>> _unpack_metadata({'year': 2024, '_vd_text': 'hello'})
    ('hello', {'year': 2024})
    >>> _unpack_metadata({})
    ('', {})
    """
    raw = dict(raw_metadata) if raw_metadata else {}
    text = raw.pop(_VD_TEXT_KEY, "")
    return text, raw


def _to_pinecone_filter(ast: Optional[Filter]) -> Optional[dict]:
    """
    Translate a canonical ``vd`` filter AST to Pinecone's MongoDB-ish dialect.

    Pinecone's filter is nearly identical to the canonical ``vd`` AST, with
    one difference: a bare ``{"field": value}`` (shorthand equality) must be
    expanded to ``{"field": {"$eq": value}}`` because Pinecone requires an
    explicit operator for every field condition. Logical ``$and`` / ``$or``
    subclauses are recursed into; explicit operator dicts (``{"$gt": 5}``, etc.)
    are passed through unchanged.

    ``$not`` and ``$exists`` are already rejected by :meth:`search` validation
    before this function is called; they are included here only for defensive
    completeness.

    Parameters
    ----------
    ast : dict or None
        A ``vd`` filter in the canonical dialect, or ``None`` (no filter).

    Returns
    -------
    dict or None
        A Pinecone-compatible filter dict, or ``None`` if ``ast`` is falsy.

    Examples
    --------
    >>> _to_pinecone_filter(None) is None
    True
    >>> _to_pinecone_filter({'year': 2024})
    {'year': {'$eq': 2024}}
    >>> _to_pinecone_filter({'year': {'$gte': 2020}})
    {'year': {'$gte': 2020}}
    >>> _to_pinecone_filter({'$and': [{'a': 1}, {'b': {'$gt': 0}}]})
    {'$and': [{'a': {'$eq': 1}}, {'b': {'$gt': 0}}]}
    """
    if not ast:
        return None
    out: dict = {}
    for key, condition in ast.items():
        if key in ("$and", "$or"):
            out[key] = [_to_pinecone_filter(sub) for sub in condition]
        elif isinstance(condition, dict):
            # Explicit operator dict — pass through as-is.
            out[key] = condition
        else:
            # Bare value equality — wrap in an explicit $eq.
            out[key] = {"$eq": condition}
    return out


def _index_is_ready(index_description) -> bool:
    """
    Return ``True`` when a Pinecone index description's status is ``"Ready"``.

    Pinecone index creation is asynchronous; the SDK's ``describe_index`` (or
    ``list_indexes`` entry) returns a status dict. This helper abstracts the
    status check so callers are not coupled to the SDK's object layout.

    Parameters
    ----------
    index_description : object
        An index description object returned by ``Pinecone.list_indexes()``
        or ``Pinecone.describe_index()``.
    """
    # Pinecone v6+ index objects expose `.status.ready` (bool) or
    # `.status` as a dict ``{"ready": bool, "state": "Ready"|"Initializing"}``.
    status = getattr(index_description, "status", None)
    if status is None:
        return True  # unknown status — assume ready to avoid hanging
    # Object form (v9): status.ready is a bool.
    if hasattr(status, "ready"):
        return bool(status.ready)
    # Dict form (some v6/v7 builds): {"ready": True, "state": "Ready"}.
    if isinstance(status, dict):
        return bool(status.get("ready", True))
    # String form fall-through.
    return str(status).lower() in ("ready", "")


# ---------------------------------------------------------------------------
# PineconeCollection
# ---------------------------------------------------------------------------


class PineconeCollection(AbstractCollection):
    """
    A ``vd`` :class:`~vd.base.Collection` backed by one Pinecone serverless index.

    Do not instantiate directly — obtain instances via
    :meth:`PineconeClient.create_collection` or
    :meth:`PineconeClient.get_collection`.

    Metadata flattening
    -------------------
    Pinecone only accepts flat metadata (no nested dicts). This adapter stores
    the document text under the reserved key ``"_vd_text"`` inside Pinecone's
    metadata payload and merges user metadata alongside it. On read, that key
    is popped back out. User metadata keys must not collide with ``"_vd_text"``,
    and metadata values must be ``str | int | float | bool | list[str]``.

    Filter support
    --------------
    Pinecone's filter is Mongo-ish but does not support ``$not`` or ``$exists``.
    Filters using those operators are rejected before the query with a clear
    :class:`~vd.base.UnsupportedFilterError`.
    """

    supported_filter_operators = PINECONE_FILTER_OPERATORS

    def __init__(
        self,
        name: str,
        pc: Pinecone,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.name = name
        self._pc = pc
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._cloud = cloud
        self._region = region
        # Lazily-resolved Pinecone Index handle.
        self._index = None

    @property
    def native(self):
        """The raw Pinecone ``Index`` object (escape hatch)."""
        return self._get_index()

    # ----- private helpers -------------------------------------------------- #

    def _get_index(self):
        """Return the Pinecone ``Index`` handle, resolving it once."""
        if self._index is None:
            self._index = self._pc.Index(self.name)
        return self._index

    def _ensure_index(self) -> None:
        """
        Create the Pinecone serverless index if it does not yet exist.

        Called lazily on the first write when ``dimension`` was unknown at
        collection-creation time. Once the dimension is known (set by
        :meth:`~vd.base.AbstractCollection._vet_vector`) the index is created
        with ``ServerlessSpec`` targeting :attr:`_cloud` / :attr:`_region`.

        Raises
        ------
        RuntimeError
            If the dimension is still ``None`` when this is called (should not
            happen because :meth:`~vd.base.AbstractCollection._ensure_vector`
            sets it before ``_write`` is invoked).
        """
        if self.dimension is None:
            raise RuntimeError(
                f"Cannot create Pinecone index {self.name!r}: vector dimension "
                f"is unknown. Write a Document with a vector first, or pass "
                f"dimension= to create_collection."
            )
        existing_names = {idx.name for idx in self._pc.list_indexes()}
        if self.name not in existing_names:
            self._pc.create_index(
                name=self.name,
                dimension=self.dimension,
                metric=_pinecone_metric(self.metric),
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            # Reset the cached index handle so it picks up the new index.
            self._index = None

    # ----- raw primitives --------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """
        Upsert one document into the Pinecone index.

        Creates the index lazily if needed. The document vector must be set
        and dimension-checked by the base class before this is called.
        """
        self._ensure_index()
        flat_meta = _pack_metadata(doc.text, doc.metadata or {})
        self._get_index().upsert(
            vectors=[{"id": doc.id, "values": doc.vector, "metadata": flat_meta}]
        )

    def _write_many(self, docs: list[Document]) -> None:
        """
        Batch-upsert many documents in a single Pinecone ``upsert`` call.

        More efficient than repeated single-document writes for large ingests.
        All documents in ``docs`` must have vectors already set and
        dimension-checked by the base class.
        """
        self._ensure_index()
        vectors = [
            {
                "id": doc.id,
                "values": doc.vector,
                "metadata": _pack_metadata(doc.text, doc.metadata or {}),
            }
            for doc in docs
        ]
        self._get_index().upsert(vectors=vectors)

    def _read(self, key: str) -> Document:
        """
        Fetch one document by id; raise ``KeyError`` if absent.

        Parameters
        ----------
        key : str
            The document id.

        Raises
        ------
        KeyError
            If no document with that id exists in the index.
        """
        try:
            result = self._get_index().fetch(ids=[key])
        except Exception as exc:
            # If the index itself doesn't exist yet, treat as absent.
            raise KeyError(key) from exc
        vectors = getattr(result, "vectors", None) or {}
        if key not in vectors:
            raise KeyError(key)
        vec_obj = vectors[key]
        raw_meta = getattr(vec_obj, "metadata", None) or {}
        text, user_meta = _unpack_metadata(raw_meta)
        raw_values = getattr(vec_obj, "values", None)
        return Document(
            id=key,
            text=text,
            vector=list(raw_values) if raw_values is not None else None,
            metadata=user_meta,
        )

    def _drop(self, key: str) -> None:
        """
        Delete one document; raise ``KeyError`` if absent.

        Parameters
        ----------
        key : str
            The document id to delete.

        Raises
        ------
        KeyError
            If no document with that id exists in the index.
        """
        # Verify existence before deleting (Pinecone's delete is a no-op if
        # the id is absent; we need to raise KeyError instead).
        try:
            result = self._get_index().fetch(ids=[key])
        except Exception as exc:
            raise KeyError(key) from exc
        vectors = getattr(result, "vectors", None) or {}
        if key not in vectors:
            raise KeyError(key)
        self._get_index().delete(ids=[key])

    def _keys(self) -> Iterator[str]:
        """
        Iterate all document ids in the index.

        Pinecone's ``Index.list()`` is paginated; this helper flattens the
        pages transparently. Returns an empty iterator if the index does not
        exist yet (no writes have occurred and dimension was deferred).
        """
        try:
            index = self._get_index()
        except Exception:
            return iter(())
        ids: list[str] = []
        for page in index.list():
            # ``page`` may be a list of ids or a list-like page object with an
            # ``ids`` attribute, depending on the SDK version.
            if isinstance(page, list):
                ids.extend(page)
            else:
                page_ids = getattr(page, "ids", None)
                if page_ids:
                    ids.extend(page_ids)
                else:
                    # Fall back: treat the object itself as iterable.
                    try:
                        ids.extend(list(page))
                    except TypeError:
                        pass
        return iter(ids)

    def _count(self) -> int:
        """
        Return the number of vectors in the index.

        Uses ``describe_index_stats().total_vector_count``. Returns ``0`` if
        the index does not exist yet.
        """
        try:
            stats = self._get_index().describe_index_stats()
        except Exception:
            return 0
        return int(getattr(stats, "total_vector_count", 0) or 0)

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Nearest-neighbour search. Returns ``limit`` results ordered by score
        (higher is better for cosine/dot; Pinecone always returns similarity
        scores, not distances).

        Parameters
        ----------
        vector : list[float]
            The query vector.
        limit : int
            Maximum number of results.
        filter : dict or None
            Canonical ``vd`` filter (validated by the base class before this
            is called). Translated to Pinecone's dialect by
            :func:`_to_pinecone_filter`.
        **kwargs
            Passed through to ``Index.query`` (e.g. ``namespace``).

        Returns
        -------
        list[dict]
            Each element has keys ``id``, ``text``, ``score``, ``metadata``.
        """
        try:
            index = self._get_index()
        except Exception:
            return []

        pinecone_filter = _to_pinecone_filter(filter)
        response = index.query(
            vector=vector,
            top_k=limit,
            filter=pinecone_filter,
            include_metadata=True,
            **kwargs,
        )
        results: list[SearchResult] = []
        for match in getattr(response, "matches", []):
            raw_meta = getattr(match, "metadata", None) or {}
            text, user_meta = _unpack_metadata(raw_meta)
            results.append(
                {
                    "id": match.id,
                    "text": text,
                    "score": float(match.score),
                    "metadata": user_meta,
                }
            )
        return results


# ---------------------------------------------------------------------------
# PineconeClient
# ---------------------------------------------------------------------------


@register_backend("pinecone")
class PineconeClient(AbstractClient):
    """
    Pinecone client.

    Wraps the ``pinecone.Pinecone`` object and maps ``vd`` collection operations
    to Pinecone serverless index operations.

    Parameters
    ----------
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder. Passed to every
        collection so ``collection["k"] = "some text"`` and
        ``collection.search("query text")`` work without pre-computing vectors.
    api_key : str, optional
        Pinecone API key. Defaults to the ``PINECONE_API_KEY`` environment
        variable. A clear error is raised if neither is set.
    cloud : str
        Serverless cloud provider: ``"aws"`` (default), ``"gcp"``, or ``"azure"``.
        Applied to all indexes created through this client.
    region : str
        Cloud region for serverless indexes, e.g. ``"us-east-1"`` (the default
        and the only region available on the Starter free tier).
    **config
        Additional keyword arguments passed through to :class:`AbstractClient`.

    Raises
    ------
    ValueError
        If neither ``api_key`` nor the ``PINECONE_API_KEY`` environment variable
        is set.

    Examples
    --------
    >>> import vd                                                       # doctest: +SKIP
    >>> client = vd.connect('pinecone')                                  # doctest: +SKIP
    >>> col = client.create_collection('my-index', dimension=1536)       # doctest: +SKIP
    >>> col['doc1'] = 'Some text'   # requires embedder to be set        # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        api_key: Optional[str] = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        resolved_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Pinecone API key is required. Pass api_key= or set the "
                "PINECONE_API_KEY environment variable. Get a key at "
                "https://app.pinecone.io."
            )
        self._client = Pinecone(api_key=resolved_key)
        self._cloud = cloud
        self._region = region

    # ----- helpers ---------------------------------------------------------- #

    def _existing_index_names(self) -> set[str]:
        """Return the set of index names currently registered in this account."""
        return {idx.name for idx in self._client.list_indexes()}

    # ----- AbstractClient interface ----------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> PineconeCollection:
        """
        Create a new Pinecone serverless index and return its collection wrapper.

        If ``dimension`` is provided, the Pinecone index is created immediately.
        If ``dimension`` is ``None``, index creation is deferred until the first
        write (lazy creation, like the Qdrant adapter).

        **Metadata flattening note:** Pinecone metadata must be a flat dict. User
        metadata values must be ``str | int | float | bool | list[str]``. Nested
        dicts will cause a Pinecone API error at upsert time. Document text is
        stored under the reserved key ``"_vd_text"`` — do not use that key in
        your metadata.

        Parameters
        ----------
        name : str
            Index / collection name.
        dimension : int, optional
            Vector dimension. Required by Pinecone at index-creation time; if
            omitted here, the index is created on the first write.
        metric : str
            Distance metric: ``"cosine"`` (default), ``"dot"``, or ``"l2"``.
            Mapped to Pinecone's ``"cosine"`` / ``"dotproduct"`` / ``"euclidean"``.
        **index_config
            Reserved for future index-tuning parameters; currently unused.

        Raises
        ------
        ValueError
            If a collection (index) with that name already exists in the account.
        """
        if name in self._existing_index_names():
            raise ValueError(
                f"Collection {name!r} already exists in this Pinecone account."
            )
        collection = PineconeCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
            cloud=self._cloud,
            region=self._region,
        )
        if dimension is not None:
            # Eager index creation when the dimension is known up front.
            collection._ensure_index()
        return collection

    def get_collection(self, name: str) -> PineconeCollection:
        """
        Return a collection wrapper for an existing Pinecone index.

        Parameters
        ----------
        name : str
            Index name to look up.

        Raises
        ------
        KeyError
            If no index with that name exists in the account.
        """
        indexes = {idx.name: idx for idx in self._client.list_indexes()}
        if name not in indexes:
            raise KeyError(
                f"Collection {name!r} does not exist. "
                f"Existing collections: {sorted(indexes)}"
            )
        idx_info = indexes[name]
        # Recover the metric from the index spec so the score interpretation
        # in _query is correct.
        pinecone_metric = getattr(idx_info, "metric", "cosine") or "cosine"
        reverse_metric = {v: k for k, v in _METRIC_MAP.items()}
        vd_metric = reverse_metric.get(pinecone_metric, "cosine")

        # Recover dimension from the index spec (may be None for sparse indexes).
        pinecone_dim = getattr(idx_info, "dimension", None)

        return PineconeCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=pinecone_dim,
            metric=vd_metric,
            cloud=self._cloud,
            region=self._region,
        )

    def delete_collection(self, name: str) -> None:
        """
        Delete the Pinecone index for collection ``name``.

        Parameters
        ----------
        name : str
            Index name to delete.

        Raises
        ------
        KeyError
            If no index with that name exists in the account.
        """
        if name not in self._existing_index_names():
            raise KeyError(f"Collection {name!r} does not exist and cannot be deleted.")
        self._client.delete_index(name)

    def list_collections(self) -> Iterator[str]:
        """
        Iterate the names of all Pinecone indexes in the account.

        Returns
        -------
        Iterator[str]
            Index names in the order returned by ``Pinecone.list_indexes()``.
        """
        return (idx.name for idx in self._client.list_indexes())
