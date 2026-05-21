"""
Milvus backend.

Milvus is a distributed, Apache-2.0 vector database written in Go/C++, built
for billion-vector scale with DiskANN on SSD so RAM does not need to hold every
vector.  It has three deployment modes that all share the same Python SDK:

- **Milvus Lite** (embedded, Linux/macOS only — not native Windows; use WSL2):
  a single `.db` file, no server, up to ~1 M vectors.  Pass ``path=`` to the
  client or leave all connection args ``None`` and a temp file is created.
- **Milvus Standalone**: a Docker container on port 19530.  Pass ``uri=``.
- **Zilliz Cloud** (managed Milvus): pass ``uri=`` (the cloud endpoint) and
  ``token=``.

How this adapter maps onto Milvus:

- Each ``vd`` collection → one Milvus collection with an explicit schema.
- Document ids are arbitrary strings.  The schema uses a ``VARCHAR`` primary
  key (``id``), a ``FLOAT_VECTOR`` field (``vector``), and
  ``enable_dynamic_field=True`` so ``text`` and ``metadata`` are stored as
  dynamic fields in each row — no separate payload store is needed.
- Milvus requires the vector dimension at collection-creation time.  When
  ``dimension`` is ``None`` in :meth:`MilvusBackendClient.create_collection`
  the actual Milvus collection is created lazily on the first write
  (``_ensure_collection``), mirroring the pattern in the Qdrant adapter.
- Metadata filtering is deferred entirely to client-side post-processing
  (``apply_client_filter``).  Translating the canonical ``vd`` MongoDB-style
  filter AST to Milvus's boolean-expression language (e.g.
  ``'metadata["key"] == "val"'``) is fragile across JSON-path semantics and
  Milvus server versions.  Over-fetching then filtering in Python is the
  reliable, backend-version-agnostic choice.

Supported metrics:

==========  ==================
vd name     Milvus metric_type
==========  ==================
``cosine``  ``COSINE``
``dot``     ``IP``   (inner product)
``l2``      ``L2``
==========  ==================

Requires: ``pip install pymilvus``
"""

from __future__ import annotations

import tempfile
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from pymilvus import DataType, MilvusClient
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The milvus backend needs the 'pymilvus' package. "
        "Install with: pip install -U pymilvus"
    ) from e

from vd.backends._helpers import apply_client_filter, overfetch_limit, score_from_distance
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
# Module-level constants
# --------------------------------------------------------------------------- #

#: vd metric name → Milvus metric_type string.
_METRIC_MAP: dict[str, str] = {
    "cosine": "COSINE",
    "dot": "IP",
    "l2": "L2",
}

#: Maximum VARCHAR length for document ids.
_ID_MAX_LENGTH: int = 512

#: Default batch size for Milvus Lite queries that simulate full-collection scan.
_KEYS_BATCH_SIZE: int = 1000


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _milvus_metric(metric: str) -> str:
    """
    Translate a ``vd`` metric name to a Milvus ``metric_type`` string.

    Parameters
    ----------
    metric : str
        One of ``"cosine"``, ``"dot"``, ``"l2"``.

    Returns
    -------
    str
        The Milvus ``metric_type`` value.

    Raises
    ------
    ValueError
        If ``metric`` is not a recognised ``vd`` metric.
    """
    try:
        return _METRIC_MAP[metric]
    except KeyError:
        valid = ", ".join(sorted(_METRIC_MAP))
        raise ValueError(
            f"Unknown metric {metric!r}. Valid values are: {valid}."
        )


def _build_schema(dimension: int) -> Any:
    """
    Build a Milvus ``CollectionSchema`` for a ``vd`` collection.

    The schema has:

    - ``id`` — ``VARCHAR`` primary key (auto_id=False, max_length=512).
    - ``vector`` — ``FLOAT_VECTOR`` with the given ``dimension``.
    - Dynamic fields enabled so ``text`` and ``metadata`` fields in each row
      dict are accepted without being declared in the schema.

    Parameters
    ----------
    dimension : int
        Vector dimension; must be positive.

    Returns
    -------
    CollectionSchema
        Ready to pass to ``MilvusClient.create_collection``.
    """
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX_LENGTH)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
    return schema


def _build_index_params(metric_type: str) -> Any:
    """
    Build Milvus ``IndexParams`` for the ``vector`` field.

    Uses the default ``AUTOINDEX`` so Milvus selects the best physical index
    (HNSW for Lite/standalone, DiskANN on Zilliz).  Pass ``metric_type``
    to tie the index to the correct distance function.

    Parameters
    ----------
    metric_type : str
        One of the Milvus metric_type strings (``"COSINE"``, ``"IP"``,
        ``"L2"``).

    Returns
    -------
    IndexParams
        Ready to pass to ``MilvusClient.create_collection``.
    """
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type=metric_type,
    )
    return index_params


def _row_to_document(row: dict) -> Document:
    """
    Convert a Milvus query/get row dict to a :class:`~vd.base.Document`.

    Parameters
    ----------
    row : dict
        A row dict as returned by ``MilvusClient.get`` or
        ``MilvusClient.query``.  Expected keys: ``id``, optionally ``vector``,
        ``text``, ``metadata``.

    Returns
    -------
    Document
    """
    return Document(
        id=str(row["id"]),
        text=row.get("text") or "",
        vector=row.get("vector"),
        metadata=row.get("metadata") or {},
    )


def _hit_to_result(hit: dict, metric: str) -> SearchResult:
    """
    Convert one Milvus search hit dict to a ``vd`` :data:`~vd.base.SearchResult`.

    The hit is a plain ``dict`` from the pymilvus 2.4+ ``MilvusClient.search``
    response.  The ``distance`` field holds the raw Milvus distance (semantics
    depend on the metric_type); this is converted to a higher-is-better score.

    Parameters
    ----------
    hit : dict
        One element from the inner hit list returned by
        ``MilvusClient.search``.
    metric : str
        The ``vd`` metric name (``"cosine"``, ``"dot"``, ``"l2"``).

    Returns
    -------
    dict
        ``{"id", "text", "score", "metadata"}``
    """
    entity = hit.get("entity") or {}
    distance = hit.get("distance", 0.0)
    # Milvus COSINE distance is in [0, 1] where 0 = identical, 1 = orthogonal.
    # However, MilvusClient.search with COSINE returns *similarity* (1 - dist),
    # already in higher-is-better form.  For IP it returns the inner product
    # (higher-is-better).  For L2 it returns a non-negative L2 distance
    # (lower-is-better).  We normalise via score_from_distance:
    #   cosine  → 1 - distance  (distance already comes as 1 - sim, so
    #             score_from_distance correctly re-inverts to similarity)
    #   dot/IP  → -distance (score_from_distance negates for dot)
    #   l2      → 1 / (1 + distance)
    # Note: pymilvus search actually returns *similarity* for COSINE
    # (i.e., higher-is-better already).  To keep the conversion uniform we
    # map COSINE distance=0 → score=1 via score_from_distance("cosine"):
    #   score_from_distance(d, "cosine") = 1 - d
    # So passing the raw .distance value (which pymilvus already gives as
    # 1 - cos_sim for COSINE) is slightly off.  In practice pymilvus ≥2.4
    # returns the actual cosine *similarity* (not distance) for COSINE metric,
    # so distance ≈ cos_sim (already higher-is-better).  We pass it directly.
    if metric == "cosine":
        score = float(distance)  # already similarity (higher-is-better)
    elif metric == "dot":
        score = float(distance)  # already inner-product (higher-is-better)
    else:
        score = score_from_distance(float(distance), metric)

    return {
        "id": str(hit.get("id", "")),
        "text": entity.get("text") or "",
        "score": score,
        "metadata": entity.get("metadata") or {},
    }


# --------------------------------------------------------------------------- #
# MilvusCollection
# --------------------------------------------------------------------------- #


class MilvusCollection(AbstractCollection):
    """
    A collection backed by one Milvus collection.

    Metadata filtering is performed client-side (over-fetching candidates
    first) because translating the canonical ``vd`` filter AST to Milvus's
    boolean-expression language is fragile across server versions.

    Parameters
    ----------
    name : str
        Milvus collection name.
    client : MilvusClient
        The shared raw Milvus client.
    embedder : callable, optional
        ``text -> vector`` convenience embedder.
    dimension : int, optional
        Vector dimension.  If ``None``, the actual Milvus collection is
        created lazily on the first :meth:`_write`.
    metric : str
        ``vd`` distance metric (``"cosine"``, ``"dot"``, or ``"l2"``).
    """

    def __init__(
        self,
        name: str,
        client: "MilvusClient",
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._client = client
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric

    # ----- escape hatch --------------------------------------------------- #

    @property
    def native(self) -> "MilvusClient":
        """The raw ``MilvusClient`` (escape hatch for Milvus-specific APIs)."""
        return self._client

    # ----- lazy collection creation --------------------------------------- #

    def _collection_exists(self) -> bool:
        """Return ``True`` if the Milvus collection has been materialised."""
        return self._client.has_collection(self.name)

    def _ensure_collection(self) -> None:
        """
        Create the Milvus collection lazily, once the dimension is known.

        Called automatically before the first write.  A no-op if the
        collection already exists in Milvus.
        """
        if self._collection_exists():
            return
        if self.dimension is None:
            raise RuntimeError(
                f"Cannot create Milvus collection {self.name!r}: dimension is "
                f"not yet known. Write a Document with a vector first so the "
                f"dimension can be inferred, or pass dimension= at "
                f"create_collection time."
            )
        metric_type = _milvus_metric(self.metric)
        self._client.create_collection(
            collection_name=self.name,
            schema=_build_schema(self.dimension),
            index_params=_build_index_params(metric_type),
        )

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """Upsert one document (``doc.vector`` is set and dimension-checked)."""
        self._ensure_collection()
        row = {
            "id": doc.id,
            "vector": doc.vector,
            "text": doc.text,
            "metadata": doc.metadata or {},
        }
        self._client.upsert(collection_name=self.name, data=[row])

    def _write_many(self, docs: list[Document]) -> None:
        """Bulk upsert — more efficient than individual :meth:`_write` calls."""
        self._ensure_collection()
        rows = [
            {
                "id": d.id,
                "vector": d.vector,
                "text": d.text,
                "metadata": d.metadata or {},
            }
            for d in docs
        ]
        self._client.upsert(collection_name=self.name, data=rows)

    def _read(self, key: str) -> Document:
        """Fetch one document; raise ``KeyError`` if absent."""
        if not self._collection_exists():
            raise KeyError(key)
        rows = self._client.get(
            collection_name=self.name,
            ids=[key],
            output_fields=["id", "text", "metadata", "vector"],
        )
        if not rows:
            raise KeyError(key)
        return _row_to_document(rows[0])

    def _drop(self, key: str) -> None:
        """Delete one document; raise ``KeyError`` if absent."""
        if not self._collection_exists():
            raise KeyError(key)
        existing = self._client.get(
            collection_name=self.name,
            ids=[key],
            output_fields=["id"],
        )
        if not existing:
            raise KeyError(key)
        self._client.delete(collection_name=self.name, ids=[key])

    def _keys(self) -> Iterator[str]:
        """Iterate all document ids in the collection."""
        if not self._collection_exists():
            return iter(())
        ids: list[str] = []
        offset = 0
        while True:
            rows = self._client.query(
                collection_name=self.name,
                filter="id != ''",
                output_fields=["id"],
                limit=_KEYS_BATCH_SIZE,
                offset=offset,
            )
            if not rows:
                break
            ids.extend(str(row["id"]) for row in rows)
            if len(rows) < _KEYS_BATCH_SIZE:
                break
            offset += len(rows)
        return iter(ids)

    def _count(self) -> int:
        """Return the number of documents in the collection."""
        if not self._collection_exists():
            return 0
        stats = self._client.get_collection_stats(collection_name=self.name)
        return int(stats.get("row_count", 0))

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Raw nearest-neighbour search returning result dicts.

        Metadata filtering is applied client-side after over-fetching
        ``overfetch_limit(limit, filter)`` candidates from Milvus.

        Parameters
        ----------
        vector : list[float]
            Query vector (already vetted and dimension-checked).
        limit : int
            Maximum results to return *after* client-side filtering.
        filter : dict, optional
            Canonical ``vd`` filter AST.  Evaluated client-side.
        **kwargs
            Passed through to ``MilvusClient.search``.

        Returns
        -------
        list[dict]
            Each dict has ``id``, ``text``, ``score``, ``metadata``.
        """
        if not self._collection_exists():
            return []
        fetch = overfetch_limit(limit, filter)
        raw = self._client.search(
            collection_name=self.name,
            data=[vector],
            limit=fetch,
            output_fields=["text", "metadata"],
            **kwargs,
        )
        # raw is a list of one hit-list (one per query vector).
        hits = raw[0] if raw else []
        results: list[SearchResult] = [
            _hit_to_result(hit, self.metric) for hit in hits
        ]
        return apply_client_filter(results, filter, limit=limit)


# --------------------------------------------------------------------------- #
# MilvusBackendClient
# --------------------------------------------------------------------------- #


@register_backend("milvus")
class MilvusBackendClient(AbstractClient):
    """
    Milvus client.

    Supports Milvus Lite (embedded file), Milvus Standalone (Docker), and
    Zilliz Cloud.

    Parameters
    ----------
    embedder : callable, optional
        A ``text -> vector`` convenience embedder.  Passed to every collection
        so text inputs are accepted.  ``None`` (the default) makes the client
        vector-only.
    uri : str, optional
        URI of a running Milvus server or Zilliz Cloud endpoint (e.g.
        ``"http://localhost:19530"`` or
        ``"https://xxxx.api.gcp-us-west1.zillizcloud.com"``).
    token : str, optional
        API token for Zilliz Cloud (``"user:password"`` or an API key string).
        Ignored when connecting to a local server.
    path : str, optional
        Local file path for Milvus Lite embedded mode (e.g.
        ``"./milvus_demo.db"``).  Takes precedence over ``uri``.  If
        neither ``path`` nor ``uri`` is given, a temporary ``.db`` file is
        created automatically.

    Notes
    -----
    Milvus Lite is **not available on native Windows** — use WSL2 there.
    The ``.db`` format is not forward-compatible between major pymilvus
    versions; back up before upgrading.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        path: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        if path is not None:
            self._client = MilvusClient(path)
        elif uri is not None:
            self._client = MilvusClient(uri=uri, token=token)
        else:
            # Milvus Lite: create a temporary .db file.
            _fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="vd_milvus_")
            self._client = MilvusClient(tmp_path)

        #: Track metrics for collections created via this client so they survive
        #: a round-trip through ``get_collection`` (Milvus does not store the
        #: vd metric name; only the low-level metric_type is kept).
        self._metrics: dict[str, str] = {}

    # ----- AbstractClient interface --------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> MilvusCollection:
        """
        Create a new Milvus collection.

        Parameters
        ----------
        name : str
            Collection name.
        dimension : int, optional
            Vector dimension.  If ``None``, the actual Milvus collection is
            created lazily on the first write.
        metric : str
            Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
        **index_config
            Reserved for future backend-specific index tuning; currently
            unused.

        Returns
        -------
        MilvusCollection

        Raises
        ------
        ValueError
            If a collection with ``name`` already exists.
        """
        if self._client.has_collection(name) or name in self._metrics:
            raise ValueError(f"Collection {name!r} already exists")
        self._metrics[name] = metric
        collection = MilvusCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )
        if dimension is not None:
            # Eager creation when the dimension is already known.
            collection._ensure_collection()
        return collection

    def get_collection(self, name: str) -> MilvusCollection:
        """
        Return an existing collection.

        Parameters
        ----------
        name : str
            Collection name.

        Returns
        -------
        MilvusCollection

        Raises
        ------
        KeyError
            If the collection does not exist.
        """
        if not self._client.has_collection(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        return MilvusCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=None,  # dimension is inferred at write / query time
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection.

        Parameters
        ----------
        name : str
            Collection name.

        Raises
        ------
        KeyError
            If the collection does not exist.
        """
        if not self._client.has_collection(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        if self._client.has_collection(name):
            self._client.drop_collection(name)
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        """Iterate collection names (materialised in Milvus + pending-lazy ones)."""
        names = set(self._client.list_collections())
        names |= set(self._metrics)
        return iter(sorted(names))

    def close(self) -> None:
        """Close the underlying Milvus client connection."""
        self._client.close()
