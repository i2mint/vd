"""
Elasticsearch backend.

Elasticsearch is a distributed, REST-based search and analytics engine that
gained native vector search via the ``dense_vector`` field type. Since version
8.x the default index structure is HNSW (``int8_hnsw``), switching to
binary-quantized HNSW (``bbq_hnsw``) for vectors of 384 dims or more in 8.18+.
Hybrid search (BM25 + kNN) is supported natively via the RRF retriever (GA in
8.8+).

This adapter maps a ``vd`` collection onto one Elasticsearch index of the same
name. The index mapping is created lazily on the first write (deferred until the
vector dimension is known, mirroring the Qdrant and FAISS adapters). Metadata
filtering is applied **client-side** — the full canonical ``vd`` filter
language is supported without any translation to ES query DSL.

Mapping layout per index::

    {
        "properties": {
            "embedding": {"type": "dense_vector", "dims": D, "index": True, "similarity": ...},
            "text":      {"type": "text"},
            "metadata":  {"type": "object", "enabled": False},
        }
    }

ES index-name constraints: index names **must be lowercase**. This adapter
does **not** silently mangle names — pass lowercase names or Elasticsearch will
raise a ``RequestError``.

Requires: ``pip install elasticsearch``
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from elasticsearch import Elasticsearch, NotFoundError
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The elasticsearch backend needs the 'elasticsearch' package. "
        "Install with: pip install elasticsearch"
    ) from e

from vd.backends._helpers import apply_client_filter, overfetch_limit
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
# Module-level constants (no magic numbers)
# --------------------------------------------------------------------------- #

#: vd metric -> ES dense_vector similarity value.
_ES_SIMILARITY = {
    "cosine": "cosine",
    "l2": "l2_norm",
    "dot": "dot_product",
}

#: Minimum num_candidates for kNN search — ES requires a sensible floor.
_MIN_NUM_CANDIDATES = 100

#: Page size used when scrolling all document ids from an index.
_KEYS_PAGE_SIZE = 1_000

#: Default ES URL used when no url/hosts argument is given.
_DEFAULT_URL = "http://localhost:9200"


# --------------------------------------------------------------------------- #
# Helper — index mapping
# --------------------------------------------------------------------------- #


def _build_mapping(dimension: int, metric: str) -> dict:
    """
    Return the Elasticsearch index mapping for a ``vd`` collection.

    Parameters
    ----------
    dimension : int
        Vector dimension (must be known before the index is created).
    metric : str
        Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns
    -------
    dict
        A ``mappings`` dict suitable for ``es.indices.create(mappings=...)``.
    """
    similarity = _ES_SIMILARITY.get(metric, "cosine")
    return {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": dimension,
                "index": True,
                "similarity": similarity,
            },
            "text": {"type": "text"},
            "metadata": {"type": "object", "enabled": False},
        }
    }


def _hit_to_result(hit: dict) -> SearchResult:
    """
    Convert one Elasticsearch kNN hit to a ``vd`` :data:`SearchResult` dict.

    ES kNN ``_score`` is higher-is-better for all three similarity functions
    (cosine, dot_product, l2_norm), so it is used directly as ``score``.

    Parameters
    ----------
    hit : dict
        One element from ``es.search(...)[\"hits\"][\"hits\"]``.

    Returns
    -------
    dict
        ``{"id", "text", "score", "metadata"}``.
    """
    source = hit.get("_source") or {}
    return {
        "id": hit["_id"],
        "text": source.get("text", ""),
        "score": hit["_score"],
        "metadata": source.get("metadata") or {},
    }


def _index_exists(es: Elasticsearch, index: str) -> bool:
    """Return ``True`` if *index* exists in Elasticsearch."""
    return bool(es.indices.exists(index=index))


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


class ElasticsearchCollection(AbstractCollection):
    """
    A ``vd`` collection backed by one Elasticsearch index.

    Index creation is deferred until the first write so that ``dimension`` can
    be inferred from the first vector when none is supplied up front. Metadata
    filtering is performed client-side: ``vd`` over-fetches kNN candidates and
    then applies the canonical filter evaluator locally.

    Parameters
    ----------
    name : str
        ES index name. **Must be lowercase** — ES rejects uppercase names.
    es : Elasticsearch
        A live ``elasticsearch.Elasticsearch`` client shared with the owning
        :class:`ElasticsearchClient`.
    embedder : callable, optional
        ``text -> vector`` convenience embedder.
    dimension : int, optional
        Vector dimension. May be deferred until the first write.
    metric : str
        Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
    """

    def __init__(
        self,
        name: str,
        es: Elasticsearch,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._es = es
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric

    @property
    def native(self) -> Elasticsearch:
        """The raw ``Elasticsearch`` client (escape hatch)."""
        return self._es

    # ----- internal helpers ------------------------------------------------- #

    def _ensure_index(self) -> None:
        """
        Create the Elasticsearch index on the first write if it does not exist.

        The dimension must be known (set) before this is called, which is
        guaranteed because :meth:`~vd.base.AbstractCollection._vet_vector`
        sets ``self.dimension`` from the first vector before ``_write`` runs.
        """
        if not _index_exists(self._es, self.name):
            self._es.indices.create(
                index=self.name,
                mappings=_build_mapping(self.dimension, self.metric),
            )

    # ----- raw primitives --------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """Upsert one document and refresh the index so it is immediately searchable."""
        self._ensure_index()
        self._es.index(
            index=self.name,
            id=doc.id,
            document={
                "embedding": doc.vector,
                "text": doc.text,
                "metadata": doc.metadata or {},
            },
        )
        self._es.indices.refresh(index=self.name)

    def _write_many(self, docs: list[Document]) -> None:
        """
        Upsert a batch of documents with a single refresh at the end.

        More efficient than calling ``_write`` in a loop when inserting many
        documents, because the expensive ``refresh`` is issued only once.
        """
        self._ensure_index()
        for doc in docs:
            self._es.index(
                index=self.name,
                id=doc.id,
                document={
                    "embedding": doc.vector,
                    "text": doc.text,
                    "metadata": doc.metadata or {},
                },
            )
        # One refresh after the batch — documents become searchable together.
        self._es.indices.refresh(index=self.name)

    def _read(self, key: str) -> Document:
        """
        Fetch one document by id; raise ``KeyError`` if absent or index missing.
        """
        if not _index_exists(self._es, self.name):
            raise KeyError(key)
        try:
            hit = self._es.get(index=self.name, id=key)
        except NotFoundError:
            raise KeyError(key)
        source = hit["_source"]
        return Document(
            id=key,
            text=source.get("text", ""),
            vector=source.get("embedding"),
            metadata=source.get("metadata") or {},
        )

    def _drop(self, key: str) -> None:
        """Delete one document; raise ``KeyError`` if absent or index missing."""
        if not _index_exists(self._es, self.name):
            raise KeyError(key)
        try:
            self._es.delete(index=self.name, id=key)
        except NotFoundError:
            raise KeyError(key)
        self._es.indices.refresh(index=self.name)

    def _keys(self) -> Iterator[str]:
        """
        Iterate all document ids in the index.

        Uses a ``match_all`` query with pagination (``search_after``) to
        retrieve ids page by page. For very large indices (tens of millions of
        documents) prefer the Scroll API or the ``helpers.scan`` utility; the
        ``search_after`` approach used here avoids the deprecated Scroll deep
        pagination but still loads all ids into memory.
        """
        if not _index_exists(self._es, self.name):
            return iter(())

        ids: list[str] = []
        search_after: Any = None
        while True:
            body: dict = {
                "query": {"match_all": {}},
                "size": _KEYS_PAGE_SIZE,
                "_source": False,
                "sort": [{"_id": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = search_after

            response = self._es.search(index=self.name, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                ids.append(hit["_id"])
            search_after = hits[-1]["sort"]
            if len(hits) < _KEYS_PAGE_SIZE:
                break

        return iter(ids)

    def _count(self) -> int:
        """Return the number of documents; 0 if the index does not exist yet."""
        if not _index_exists(self._es, self.name):
            return 0
        return int(self._es.count(index=self.name)["count"])

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Run kNN vector search and apply client-side metadata filtering.

        ``num_candidates`` is set to ``max(2 * k, _MIN_NUM_CANDIDATES)`` as
        recommended by the ES docs to balance recall against latency. When a
        ``filter`` is present, ``k`` is over-fetched by
        :func:`~vd.backends._helpers.overfetch_limit` and then trimmed
        client-side by :func:`~vd.backends._helpers.apply_client_filter`.

        Parameters
        ----------
        vector : list[float]
            Query vector (already vetted by the base class).
        limit : int
            Maximum results to return after filtering.
        filter : dict, optional
            Canonical ``vd`` metadata filter.
        **kwargs
            Passed through to ``es.search`` for advanced ES options.
        """
        if not _index_exists(self._es, self.name):
            return []

        k = overfetch_limit(limit, filter)
        num_candidates = max(k * 2, _MIN_NUM_CANDIDATES)

        response = self._es.search(
            index=self.name,
            knn={
                "field": "embedding",
                "query_vector": vector,
                "k": k,
                "num_candidates": num_candidates,
            },
            **kwargs,
        )

        hits = response["hits"]["hits"]
        raw_results = [_hit_to_result(h) for h in hits]
        return apply_client_filter(raw_results, filter, limit=limit)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


@register_backend("elasticsearch")
class ElasticsearchClient(AbstractClient):
    """
    Elasticsearch client.

    Connects to a running Elasticsearch 8.x instance and exposes its indices
    as ``vd`` collections.

    Parameters
    ----------
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    url : str
        URL of the Elasticsearch node. Defaults to ``"http://localhost:9200"``.
        Ignored if ``hosts`` is given explicitly.
    hosts : list, optional
        Raw ``hosts`` argument forwarded to the ``Elasticsearch`` constructor
        (useful for multi-node clusters or advanced URL formats). When
        provided, ``url`` is ignored.
    api_key : str or tuple, optional
        API key for Elastic Cloud or a secured self-hosted cluster. Accepts
        the string form (``"id:api_key"``) or a ``(id, api_key)`` tuple, as
        per the ``elasticsearch-py`` 8.x docs. Omit for unauthenticated local
        instances.
    **config
        Additional keyword arguments forwarded to
        ``AbstractClient.__init__`` and stored on ``self.config``.

    Examples
    --------
    Local dev (no auth)::

        from vd.backends.elasticsearch import ElasticsearchClient
        client = ElasticsearchClient()

    Elastic Cloud with an API key::

        client = ElasticsearchClient(
            url="https://my-deployment.es.us-east-1.aws.elastic-cloud.com",
            api_key="my_api_key_string",
        )
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        url: str = _DEFAULT_URL,
        hosts: Optional[list] = None,
        api_key: Optional[Any] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        es_kwargs: dict[str, Any] = {}
        if api_key is not None:
            es_kwargs["api_key"] = api_key
        self._client = Elasticsearch(hosts if hosts is not None else url, **es_kwargs)

    # ----- AbstractClient interface ----------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> ElasticsearchCollection:
        """
        Create a new collection (Elasticsearch index).

        The index is created **lazily** on the first write when ``dimension``
        is ``None``, or **eagerly** here when ``dimension`` is supplied.

        Parameters
        ----------
        name : str
            Index name. **Must be lowercase** — Elasticsearch rejects names
            with uppercase letters.
        dimension : int, optional
            Vector dimension. Required for eager index creation; may be
            deferred until the first write.
        metric : str
            Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
        **index_config
            Additional ES index settings (e.g. ``number_of_shards``) forwarded
            to ``es.indices.create(settings=...)``.

        Raises
        ------
        ValueError
            If a collection (index) named ``name`` already exists.
        """
        if _index_exists(self._client, name):
            raise ValueError(f"Collection {name!r} already exists")
        col = ElasticsearchCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )
        if dimension is not None:
            # Eager index creation: build the mapping now.
            settings = index_config if index_config else None
            create_kwargs: dict[str, Any] = {
                "index": name,
                "mappings": _build_mapping(dimension, metric),
            }
            if settings:
                create_kwargs["settings"] = settings
            self._client.indices.create(**create_kwargs)
        return col

    def get_collection(self, name: str) -> ElasticsearchCollection:
        """
        Return an existing collection.

        Raises
        ------
        KeyError
            If no index named ``name`` exists.
        """
        if not _index_exists(self._client, name):
            raise KeyError(f"Collection {name!r} does not exist")
        return ElasticsearchCollection(
            name,
            self._client,
            embedder=self._embedder,
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection (delete the Elasticsearch index and all its data).

        Raises
        ------
        KeyError
            If no index named ``name`` exists.
        """
        if not _index_exists(self._client, name):
            raise KeyError(f"Collection {name!r} does not exist")
        try:
            self._client.indices.delete(index=name)
        except NotFoundError:
            raise KeyError(f"Collection {name!r} does not exist")

    def list_collections(self) -> Iterator[str]:
        """
        Iterate the names of all user-managed collections.

        System indices (names starting with ``.``) are excluded because they
        are internal to Elasticsearch and not managed by ``vd``.
        """
        all_indices: dict = self._client.indices.get(index="*")
        return iter(
            sorted(name for name in all_indices if not name.startswith("."))
        )

    @property
    def native(self) -> Elasticsearch:
        """The raw ``Elasticsearch`` client (escape hatch)."""
        return self._client

    def close(self) -> None:
        """Close the underlying Elasticsearch transport."""
        self._client.close()
