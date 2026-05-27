"""
Redis 8 backend (Redis Query Engine / RediSearch).

Redis 8 (released May 2025) ships RediSearch, RedisJSON, and RedisBloom as
first-class core modules, licensed AGPLv3 / SSPLv1 / RSALv2. The Redis Query
Engine (formerly RediSearch) provides a fully-featured HNSW vector index over
Redis HASH documents, with metadata filtering via a dedicated tag/numeric
field DSL and — since 8.4 — a native HybridQuery that fuses BM25 and vector
results. It is the right backend when you already operate Redis and want
sub-millisecond KNN on a hot in-RAM dataset.

This adapter maps the ``vd`` facade contract onto the following Redis primitives:

- **Index / collection** — a RediSearch ``FT.CREATE`` index with an ``HNSW``
  ``VectorField``, a ``TextField`` for ``text``, and a ``TextField`` for the
  JSON-serialized ``metadata`` blob. The index is created lazily on the first
  write (so the vector dimension, which is not known until first insert, can be
  inferred).
- **Document** — one Redis ``HASH`` stored at key ``vd:{collection_name}:{doc_id}``,
  with fields ``text`` (plain string), ``metadata`` (JSON), ``embedding`` (float32
  bytes), and ``vd_id`` (the original document id).
- **Key prefix** — ``vd:{name}:`` (all HASH keys for a collection share this
  prefix; the RediSearch index is registered over it so ``FT.SEARCH`` sees only
  that collection's documents).
- **Index name** — ``vd_idx:{name}``.
- **Collection registry** — a Redis SET at key ``"vd:_collections"`` tracks which
  collection names are managed by this adapter, enabling ``list_collections`` and
  existence checks even before a collection's first write.
- **Client-side metadata filtering** — RediSearch's native filter DSL does not map
  cleanly onto the canonical ``vd`` filter language, so this adapter over-fetches
  KNN candidates and filters the result list client-side with
  :func:`~vd.backends._helpers.apply_client_filter`.

*Choose when:* you already run Redis 8 (or redis/redis-stack) and want the KNN
index to live inside your existing Redis instance without a separate process.
*Avoid when:* the HNSW index growing to full RAM is a concern, or you need
billion-scale — HNSW in Redis lives entirely in memory.

Requires: ``pip install redis numpy``
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    import numpy as np
    import redis
    from redis.commands.search.field import TextField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The redis backend needs 'redis' and 'numpy'. "
        "Install with: pip install redis numpy"
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

#: Redis key that holds the set of all vd-managed collection names.
_COLLECTIONS_SET_KEY = "vd:_collections"

#: vd metric name -> RediSearch DISTANCE_METRIC string.
_DISTANCE_METRIC = {
    "cosine": "COSINE",
    "l2": "L2",
    "dot": "IP",
}


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _index_name(collection_name: str) -> str:
    """Return the RediSearch index name for a vd collection."""
    return f"vd_idx:{collection_name}"


def _key_prefix(collection_name: str) -> str:
    """Return the Redis HASH key prefix for all documents in a collection."""
    return f"vd:{collection_name}:"


def _doc_key(collection_name: str, doc_id: str) -> str:
    """Return the Redis HASH key for a single document."""
    return f"vd:{collection_name}:{doc_id}"


def _strip_prefix(raw_key: bytes, prefix: str) -> str:
    """Decode a raw Redis key and strip the collection prefix to get the doc id."""
    decoded = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
    return decoded[len(prefix) :]


def _to_float32_bytes(vector: Any) -> bytes:
    """Convert a vector to a little-endian float32 byte string for Redis storage."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_float32_bytes(blob: bytes) -> list:
    """Reconstruct a list[float] from a float32 byte string retrieved from Redis."""
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _index_exists(r: "redis.Redis", index_name: str) -> bool:
    """Return True if the given RediSearch index already exists on the server."""
    try:
        r.ft(index_name).info()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


class RedisCollection(AbstractCollection):
    """
    A vd collection backed by a RediSearch HNSW vector index.

    Each document is stored as a Redis HASH at ``vd:{name}:{doc_id}``.  The
    RediSearch index is created lazily on the first :meth:`_write` call, once
    the vector dimension is known.  Metadata filtering is applied client-side
    (over-fetch then filter) because RediSearch's native filter DSL does not
    map one-to-one onto the canonical ``vd`` filter language.

    Parameters
    ----------
    name : str
        Collection name. Must be unique per Redis instance.
    client : redis.Redis
        A live ``redis.Redis`` connection shared with the :class:`RedisClient`.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    dimension : int, optional
        Vector dimension.  May be ``None`` when the collection is empty; it
        will be set on the first write.
    metric : str
        Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
    """

    def __init__(
        self,
        name: str,
        client: "redis.Redis",
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._redis = client
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._index_name = _index_name(name)
        self._key_prefix = _key_prefix(name)

    @property
    def native(self) -> "redis.Redis":
        """The raw ``redis.Redis`` connection (escape hatch)."""
        return self._redis

    # ----- lazy index creation -------------------------------------------- #

    def _ensure_index(self) -> None:
        """
        Create the RediSearch index the first time data is written.

        The index is registered over the key prefix ``vd:{name}:`` and defines
        three fields: ``text`` (TextField), ``metadata`` (TextField, not indexed
        for search — stored as a JSON blob), and ``embedding`` (VectorField,
        HNSW, FLOAT32, using the collection's metric).

        Idempotent: does nothing if the index already exists.
        """
        if _index_exists(self._redis, self._index_name):
            return

        distance_metric = _DISTANCE_METRIC.get(self.metric, "COSINE")
        schema = [
            TextField("text"),
            TextField("metadata"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dimension,
                    "DISTANCE_METRIC": distance_metric,
                },
            ),
        ]
        definition = IndexDefinition(
            prefix=[self._key_prefix],
            index_type=IndexType.HASH,
        )
        self._redis.ft(self._index_name).create_index(schema, definition=definition)

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """Upsert one document as a Redis HASH, creating the index if needed."""
        self._ensure_index()
        key = _doc_key(self.name, doc.id)
        mapping = {
            "text": doc.text,
            "metadata": json.dumps(doc.metadata or {}),
            "embedding": _to_float32_bytes(doc.vector),
            "vd_id": doc.id,
        }
        self._redis.hset(key, mapping=mapping)

    def _read(self, key: str) -> Document:
        """
        Fetch one document by id.

        Parameters
        ----------
        key : str
            Document id.

        Raises
        ------
        KeyError
            If no document with that id exists in this collection.
        """
        raw = self._redis.hgetall(_doc_key(self.name, key))
        if not raw:
            raise KeyError(key)
        return _raw_to_document(raw)

    def _drop(self, key: str) -> None:
        """
        Delete one document by id.

        Raises
        ------
        KeyError
            If no document with that id exists in this collection.
        """
        full_key = _doc_key(self.name, key)
        if not self._redis.exists(full_key):
            raise KeyError(key)
        self._redis.delete(full_key)

    def _keys(self) -> Iterator[str]:
        """Iterate over all document ids in this collection."""
        prefix = self._key_prefix
        return (
            _strip_prefix(k, prefix) for k in self._redis.scan_iter(match=f"{prefix}*")
        )

    def _count(self) -> int:
        """
        Return the number of documents in this collection.

        Uses the RediSearch index info when available (cheap O(1)); falls back
        to a key scan if the index has not been created yet (empty collection).
        """
        if _index_exists(self._redis, self._index_name):
            try:
                info = self._redis.ft(self._index_name).info()
                # Redis 4.x+ returns a dict; older clients return a flat list.
                if isinstance(info, dict):
                    return int(info.get("num_docs", 0))
                # Flat list: field names alternate with values.
                pairs = dict(zip(info[::2], info[1::2]))
                return int(pairs.get("num_docs", 0))
            except Exception:
                pass
        return sum(1 for _ in self._redis.scan_iter(match=f"{self._key_prefix}*"))

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Run a KNN nearest-neighbour search via RediSearch.

        Over-fetches candidates when ``filter`` is non-empty so that
        client-side filtering still returns a full ``limit``-length page.
        Returns an empty list if the index does not yet exist (the collection
        is empty).

        Parameters
        ----------
        vector : list[float]
            The query vector (already vetted and dimension-checked by the base).
        limit : int
            Maximum number of results to return after filtering.
        filter : dict, optional
            Canonical ``vd`` metadata filter (applied client-side).
        **kwargs
            Passed through; unused by this adapter.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``text``, ``score``, ``metadata``.
        """
        if not _index_exists(self._redis, self._index_name):
            return []

        fetch = overfetch_limit(limit, filter)
        query_bytes = _to_float32_bytes(vector)

        q = (
            Query(f"*=>[KNN {fetch} @embedding $vec AS vd_score]")
            .sort_by("vd_score")
            .return_fields("text", "metadata", "vd_score", "vd_id")
            .dialect(2)
        )

        try:
            response = self._redis.ft(self._index_name).search(
                q, query_params={"vec": query_bytes}
            )
        except Exception:
            # Index may exist structurally but contain no vectors yet.
            return []

        results: list[SearchResult] = []
        for doc in response.docs:
            raw_score = float(getattr(doc, "vd_score", 1.0))
            score = score_from_distance(raw_score, self.metric)
            try:
                metadata = json.loads(getattr(doc, "metadata", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            results.append(
                {
                    "id": getattr(doc, "vd_id", ""),
                    "text": getattr(doc, "text", ""),
                    "score": score,
                    "metadata": metadata,
                }
            )

        return apply_client_filter(results, filter, limit=limit)

    # ----- native hybrid via RediSearch BM25 + dense, fused client-side --- #

    def _lexical_query(
        self,
        text: str,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> list[SearchResult]:
        """
        BM25 lexical search via ``FT.SEARCH`` on the ``text`` field.

        Used as the lexical side of :meth:`hybrid_search`. Metadata filtering
        is applied client-side (vd stores metadata as a JSON blob, not as
        individually indexed RediSearch fields).
        """
        del kwargs
        if not _index_exists(self._redis, self._index_name):
            return []
        fetch = overfetch_limit(limit, filter)
        # Escape RediSearch query-language special characters; keep tokens.
        escaped = _escape_redisearch_query(text)
        if not escaped:
            return []
        q = (
            Query(f"@text:({escaped})")
            .return_fields("text", "metadata", "vd_id")
            .paging(0, fetch)
            .dialect(2)
        )
        try:
            response = self._redis.ft(self._index_name).search(q)
        except Exception:
            return []
        results: list[SearchResult] = []
        for doc in response.docs:
            try:
                metadata = json.loads(getattr(doc, "metadata", "{}") or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            results.append(
                {
                    "id": getattr(doc, "vd_id", ""),
                    "text": getattr(doc, "text", ""),
                    "score": 0.0,  # RRF fuser uses ranks, not raw scores.
                    "metadata": metadata,
                }
            )
        return list(apply_client_filter(results, filter, limit=limit))

    def hybrid_search(
        self,
        query,
        *,
        query_text=None,
        limit: int = 10,
        filter: Optional[Filter] = None,
        k_dense: Optional[int] = None,
        k_lexical: Optional[int] = None,
        rrf_k: int = 60,
        egress=None,
        **kwargs,
    ):
        """
        Hybrid (KNN + BM25) search via RediSearch, fused client-side with RRF.

        See :class:`vd.SupportsHybrid` for the canonical contract. Backend
        notes: Redis 8.4+ has a native ``HybridQuery`` that fuses BM25 and
        KNN server-side; we deliberately run them separately and fuse
        client-side so the fused score is uniform across vd backends and so
        the adapter works on Redis < 8.4 as well. Pass ``query_text=...``
        explicitly when ``query`` is a vector.
        """
        return self._hybrid_via_rrf(
            query,
            self._lexical_query,
            query_text=query_text,
            limit=limit,
            filter=filter,
            k_dense=k_dense,
            k_lexical=k_lexical,
            rrf_k=rrf_k,
            egress=egress,
            **kwargs,
        )


# --------------------------------------------------------------------------- #
# Module-level helper used by _read (defined after the class for readability)
# --------------------------------------------------------------------------- #


_REDISEARCH_SPECIAL_CHARS = set(",.<>{}[]\"':;!@#$%^&*()-+=~|\\/")


def _escape_redisearch_query(text: str) -> str:
    """
    Escape RediSearch query-language special characters in ``text``.

    Returns a space-joined OR-able token string. Returns an empty string when
    no usable tokens remain.
    """
    cleaned: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch in _REDISEARCH_SPECIAL_CHARS:
            if current:
                cleaned.append("".join(current))
                current = []
        elif ch.isspace():
            if current:
                cleaned.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        cleaned.append("".join(current))
    # Filter out one-char tokens that RediSearch treats as stopwords.
    tokens = [t for t in cleaned if len(t) >= 2]
    return " | ".join(tokens)


def _raw_to_document(raw: dict) -> Document:
    """
    Rebuild a :class:`~vd.base.Document` from a Redis HGETALL response.

    All values in ``raw`` may be ``bytes``; this function decodes them and
    reconstructs the typed Document fields.
    """

    def _decode(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else (v or "")

    def _key(name: str) -> bytes | str:
        """Look up a field regardless of whether the key is bytes or str."""
        return raw.get(name.encode()) or raw.get(name) or b""

    doc_id = _decode(_key("vd_id"))
    text = _decode(_key("text"))

    raw_meta = _key("metadata")
    try:
        metadata = json.loads(_decode(raw_meta)) if raw_meta else {}
    except (json.JSONDecodeError, TypeError):
        metadata = {}

    raw_embedding = _key("embedding")
    vector: Optional[list] = None
    if raw_embedding:
        blob = (
            raw_embedding
            if isinstance(raw_embedding, bytes)
            else raw_embedding.encode("latin-1")
        )
        vector = _from_float32_bytes(blob)

    return Document(id=doc_id, text=text, vector=vector, metadata=metadata)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


@register_backend("redis")
class RedisClient(AbstractClient):
    """
    Redis 8 client for the ``vd`` facade.

    Connects to a Redis server (standalone or Redis Cloud) and manages
    collections backed by RediSearch HNSW vector indexes.

    Parameters
    ----------
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder, forwarded to every
        collection so that ``collection["k"] = "some text"`` and
        ``collection.search("query text")`` work without a pre-computed vector.
    url : str, optional
        A Redis URL (e.g. ``"redis://localhost:6379"`` or
        ``"rediss://user:pass@host:6380"``). When provided, ``host`` and
        ``port`` are ignored.
    host : str
        Hostname of the Redis server.  Default: ``"localhost"``.
    port : int
        Port of the Redis server.  Default: ``6379``.
    **config
        Additional keyword arguments forwarded to :class:`redis.Redis` (e.g.
        ``password``, ``db``, ``ssl``, ``decode_responses=False``).

    Examples
    --------
    >>> client = RedisClient()                           # localhost:6379
    >>> client = RedisClient(url="redis://my-host:6379")
    >>> client = RedisClient(host="my-host", port=6380, password="s3cr3t")
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        **config,
    ):
        super().__init__(embedder=embedder)
        if url is not None:
            self._client = redis.Redis.from_url(url, **config)
        else:
            self._client = redis.Redis(host=host, port=port, **config)

    # ----- adapter contract ----------------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> RedisCollection:
        """
        Create a new collection.

        Parameters
        ----------
        name : str
            Collection name.  Must be unique within this Redis instance.
        dimension : int, optional
            Vector dimension.  May be ``None``; the dimension will be learned
            from the first inserted vector.
        metric : str
            Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
        **index_config
            Currently unused; reserved for future HNSW tuning parameters
            (``M``, ``EF_CONSTRUCTION``).

        Raises
        ------
        ValueError
            If a collection with ``name`` already exists.
        """
        if self._collection_exists(name):
            raise ValueError(f"Collection {name!r} already exists")
        self._client.sadd(_COLLECTIONS_SET_KEY, name)
        return RedisCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    def get_collection(self, name: str) -> RedisCollection:
        """
        Return an existing collection.

        The returned collection does not store any in-process state about
        dimension or metric — those are re-derived from the RediSearch index
        the first time they are needed.

        Raises
        ------
        KeyError
            If no collection with ``name`` exists.
        """
        if not self._collection_exists(name):
            raise KeyError(f"Collection {name!r} does not exist")
        dimension, metric = _read_index_params(self._client, _index_name(name))
        return RedisCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection and its RediSearch index.

        All documents stored under the ``vd:{name}:*`` key prefix are removed
        together with the index (``FT.DROPINDEX … DD``).

        Raises
        ------
        KeyError
            If no collection with ``name`` exists.
        """
        if not self._collection_exists(name):
            raise KeyError(f"Collection {name!r} does not exist")
        idx = _index_name(name)
        if _index_exists(self._client, idx):
            # delete_documents=True removes all documents under the index prefix.
            self._client.ft(idx).dropindex(delete_documents=True)
        self._client.srem(_COLLECTIONS_SET_KEY, name)

    def list_collections(self) -> Iterator[str]:
        """Iterate over the names of all vd-managed collections."""
        members = self._client.smembers(_COLLECTIONS_SET_KEY)
        return iter(sorted(m.decode() if isinstance(m, bytes) else m for m in members))

    def close(self) -> None:
        """Release the underlying Redis connection."""
        self._client.close()

    # ----- internal helpers ----------------------------------------------- #

    def _collection_exists(self, name: str) -> bool:
        """Return True if ``name`` is registered in the vd collections set."""
        return bool(self._client.sismember(_COLLECTIONS_SET_KEY, name))


# --------------------------------------------------------------------------- #
# Helper to reconstruct index parameters from an existing RediSearch index
# --------------------------------------------------------------------------- #


def _read_index_params(
    r: "redis.Redis",
    index_name: str,
) -> tuple[Optional[int], str]:
    """
    Read the vector dimension and metric from an existing RediSearch index.

    Returns ``(None, "cosine")`` when the index does not exist yet (the
    collection is registered but not yet written to).

    Parameters
    ----------
    r : redis.Redis
        A live Redis connection.
    index_name : str
        The RediSearch index name (e.g. ``"vd_idx:my_collection"``).

    Returns
    -------
    tuple[int | None, str]
        ``(dimension, metric)`` where ``metric`` is a vd canonical metric name.
    """
    if not _index_exists(r, index_name):
        return None, "cosine"

    _METRIC_REVERSE = {v: k for k, v in _DISTANCE_METRIC.items()}

    try:
        info = r.ft(index_name).info()
        # Navigate the nested attribute structure to find the VectorField params.
        fields = info.get("attributes") if isinstance(info, dict) else None
        if fields is None:
            # Flat list response: convert to dict first.
            flat = dict(zip(info[::2], info[1::2]))
            fields = flat.get(b"attributes") or flat.get("attributes") or []
        for field_info in fields:
            # field_info is a list of alternating name/value pairs or a dict.
            if isinstance(field_info, dict):
                pairs = field_info
            else:
                pairs = {}
                it = iter(field_info)
                for k in it:
                    pairs[k.decode() if isinstance(k, bytes) else k] = next(it, None)
            identifier = pairs.get("identifier") or pairs.get(b"identifier") or b""
            if isinstance(identifier, bytes):
                identifier = identifier.decode()
            if identifier == "embedding":
                dim_val = (
                    pairs.get("DIM") or pairs.get(b"DIM") or pairs.get("dim") or None
                )
                dist_val = (
                    pairs.get("DISTANCE_METRIC")
                    or pairs.get(b"DISTANCE_METRIC")
                    or pairs.get("distance_metric")
                    or b"COSINE"
                )
                dimension = int(dim_val) if dim_val is not None else None
                dist_str = (
                    dist_val.decode() if isinstance(dist_val, bytes) else dist_val
                )
                metric = _METRIC_REVERSE.get(dist_str.upper(), "cosine")
                return dimension, metric
    except Exception:
        pass

    return None, "cosine"
