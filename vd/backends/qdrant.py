"""
Qdrant backend.

Qdrant is the cleanest mapping of a real vector database onto the ``vd``
facade: one Rust binary that runs embedded (in-memory or a local path), as a
Docker server, or as a managed cloud cluster — all behind the same Python SDK.
It has rich native payload filtering, so this adapter is the one that performs
a *real* filter translation: the canonical ``vd`` filter AST is compiled to a
``qdrant_client.models.Filter`` (no client-side post-filtering).

Document ids are arbitrary strings; Qdrant point ids must be UUIDs or
unsigned ints, so each id is mapped to a deterministic UUID5 and the original
is kept in the point payload.

The backend is also **natively async** against a server:
``vd.connect_async("qdrant", url=...)`` returns a :class:`NativeAsyncQdrantClient`
built on ``qdrant_client.AsyncQdrantClient``. Embedded mode (no ``url``) gets
the thread-pool wrapper, because qdrant-client's embedded async client would
block the event loop.

Requires: ``pip install qdrant-client``
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Optional

try:
    from qdrant_client import AsyncQdrantClient, QdrantClient, models
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The qdrant backend needs the 'qdrant-client' package. "
        "Install it with: pip install qdrant-client"
    ) from e

from vd.asynchronous import (
    AsyncAbstractClient,
    AsyncAbstractCollection,
    register_async_backend,
)
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

#: vd metric -> Qdrant distance.
_DISTANCE = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "l2": models.Distance.EUCLID,
}

#: Payload keys vd reserves; user metadata lives nested under ``metadata``.
_ID_KEY = "_vd_id"
_TEXT_KEY = "_vd_text"


def _point_id(doc_id: str) -> str:
    """Map an arbitrary document id to a deterministic Qdrant UUID point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def _to_qdrant_filter(ast: Optional[Filter]) -> Optional["models.Filter"]:
    """
    Compile a canonical ``vd`` filter AST to a ``qdrant_client.models.Filter``.

    User metadata is stored nested under the ``metadata`` payload key, so every
    field reference is translated to a ``metadata.<field>`` path.
    """
    if not ast:
        return None
    must: list = []
    should: list = []
    must_not: list = []

    for key, cond in ast.items():
        if key == "$and":
            must += [_to_qdrant_filter(sub) for sub in cond]
        elif key == "$or":
            should += [_to_qdrant_filter(sub) for sub in cond]
        elif key == "$not":
            must_not.append(_to_qdrant_filter(cond))
        else:
            qkey = f"metadata.{key}"
            if not isinstance(cond, dict):
                must.append(
                    models.FieldCondition(key=qkey, match=models.MatchValue(value=cond))
                )
            else:
                _compile_field(qkey, cond, must, must_not)

    return models.Filter(
        must=must or None, should=should or None, must_not=must_not or None
    )


def _compile_field(qkey: str, cond: dict, must: list, must_not: list) -> None:
    """Translate one ``{field: {op: operand}}`` clause into Qdrant conditions."""
    range_kw: dict[str, Any] = {}
    for op, operand in cond.items():
        if op == "$eq":
            must.append(
                models.FieldCondition(key=qkey, match=models.MatchValue(value=operand))
            )
        elif op == "$ne":
            must_not.append(
                models.FieldCondition(key=qkey, match=models.MatchValue(value=operand))
            )
        elif op == "$in":
            must.append(
                models.FieldCondition(
                    key=qkey, match=models.MatchAny(any=list(operand))
                )
            )
        elif op == "$nin":
            must_not.append(
                models.FieldCondition(
                    key=qkey, match=models.MatchAny(any=list(operand))
                )
            )
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            range_kw[op[1:]] = operand
        elif op == "$exists":
            empty = models.IsEmptyCondition(is_empty=models.PayloadField(key=qkey))
            (must_not if operand else must).append(empty)
    if range_kw:
        must.append(models.FieldCondition(key=qkey, range=models.Range(**range_kw)))


def _to_point(doc: Document) -> "models.PointStruct":
    """Build the Qdrant point for ``doc`` (id mapped to a UUID, payload nested)."""
    return models.PointStruct(
        id=_point_id(doc.id),
        vector=doc.vector,
        payload={_ID_KEY: doc.id, _TEXT_KEY: doc.text, "metadata": doc.metadata or {}},
    )


def _point_to_result(point, metric: str) -> SearchResult:
    """Convert a scored Qdrant point to a ``vd`` result dict."""
    payload = point.payload or {}
    score = point.score
    # Qdrant `point.score` per metric (see vd.base "Score semantics"):
    #   - cosine: cosine similarity in [-1, 1]  → matches vd canonical
    #   - dot:    raw inner product              → matches vd canonical
    #   - euclid: a *distance* value (lower-is-better); Qdrant's
    #     own sort orders ascending in that case. The existing
    #     transform 1/(1+d) matches vd's canonical l2 score directly
    #     (no un-negation), so leave it as-is. If a future Qdrant
    #     client version switches Euclid to higher-is-better, this
    #     branch must be revisited.
    return {
        "id": payload.get(_ID_KEY, str(point.id)),
        "text": payload.get(_TEXT_KEY, ""),
        "score": 1.0 / (1.0 + score) if metric == "l2" else score,
        "metadata": payload.get("metadata", {}),
    }


def _to_document(point) -> Document:
    """Convert a retrieved Qdrant point (with payload and vector) to a Document."""
    payload = point.payload or {}
    vector = point.vector
    return Document(
        id=payload.get(_ID_KEY, str(point.id)),
        text=payload.get(_TEXT_KEY, ""),
        vector=list(vector) if vector is not None else None,
        metadata=payload.get("metadata", {}),
    )


def _vectors_config(dimension: int, metric: str) -> "models.VectorParams":
    """The Qdrant vector config for a collection of ``dimension`` and ``metric``."""
    return models.VectorParams(
        size=dimension, distance=_DISTANCE.get(metric, models.Distance.COSINE)
    )


class QdrantCollection(AbstractCollection):
    """A collection backed by one Qdrant collection. Native payload filtering."""

    # Qdrant covers the entire canonical filter language natively.
    supported_filter_operators = SUPPORTED_FILTER_OPERATORS

    def __init__(
        self,
        name: str,
        client: QdrantClient,
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

    @property
    def native(self) -> QdrantClient:
        """The raw ``QdrantClient`` (escape hatch)."""
        return self._client

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection lazily, once the dimension is known."""
        if not self._client.collection_exists(self.name):
            self._client.create_collection(
                collection_name=self.name,
                vectors_config=_vectors_config(self.dimension, self.metric),
            )

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._ensure_collection()
        self._client.upsert(self.name, points=[_to_point(doc)])

    def _write_many(self, docs: list[Document]) -> None:
        self._ensure_collection()
        self._client.upsert(self.name, points=[_to_point(d) for d in docs])

    def _read(self, key: str) -> Document:
        if not self._client.collection_exists(self.name):
            raise KeyError(key)
        points = self._client.retrieve(
            self.name, ids=[_point_id(key)], with_payload=True, with_vectors=True
        )
        if not points:
            raise KeyError(key)
        return _to_document(points[0])

    def _drop(self, key: str) -> None:
        if not self._client.collection_exists(self.name) or not self._client.retrieve(
            self.name, ids=[_point_id(key)]
        ):
            raise KeyError(key)
        self._client.delete(
            self.name, points_selector=models.PointIdsList(points=[_point_id(key)])
        )

    def _keys(self) -> Iterator[str]:
        if not self._client.collection_exists(self.name):
            return iter(())
        ids: list[str] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.name, limit=256, offset=offset, with_payload=[_ID_KEY]
            )
            ids += [p.payload[_ID_KEY] for p in points]
            if offset is None:
                break
        return iter(ids)

    def _count(self) -> int:
        if not self._client.collection_exists(self.name):
            return 0
        return self._client.count(self.name).count

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        if not self._client.collection_exists(self.name):
            return []
        response = self._client.query_points(
            self.name,
            query=vector,
            limit=limit,
            query_filter=_to_qdrant_filter(filter),
            with_payload=True,
            **kwargs,
        )
        return [_point_to_result(point, self.metric) for point in response.points]


@register_backend("qdrant")
class QdrantClientAdapter(AbstractClient):
    """
    Qdrant client.

    Parameters
    ----------
    path : str, optional
        A local directory for embedded persistent mode.
    url : str, optional
        URL of a Qdrant server / cloud cluster. With ``api_key`` for cloud.
    api_key : str, optional
        API key for Qdrant Cloud.
    location : str, optional
        Passed straight to ``QdrantClient`` (e.g. ``":memory:"``). The default,
        when neither ``path`` nor ``url`` is given, is ``":memory:"``.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        self._client = QdrantClient(
            **_qdrant_client_kwargs(
                path=path, url=url, api_key=api_key, location=location, config=config
            )
        )
        self._metrics: dict[str, str] = {}

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> QdrantCollection:
        if self._client.collection_exists(name) or name in self._metrics:
            raise ValueError(f"Collection {name!r} already exists")
        self._metrics[name] = metric
        collection = QdrantCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )
        if dimension is not None:  # eager create when the dimension is known
            collection._ensure_collection()
        return collection

    def get_collection(self, name: str) -> QdrantCollection:
        if not self._client.collection_exists(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        return QdrantCollection(
            name,
            self._client,
            embedder=self._embedder,
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        if not self._client.collection_exists(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        names = {c.name for c in self._client.get_collections().collections}
        names |= set(self._metrics)
        return iter(sorted(names))

    def close(self) -> None:
        """Close the underlying Qdrant client."""
        self._client.close()


# --------------------------------------------------------------------------- #
# Native async — qdrant_client.AsyncQdrantClient
# --------------------------------------------------------------------------- #


def _qdrant_client_kwargs(
    *,
    path: Optional[str],
    url: Optional[str],
    api_key: Optional[str],
    location: Optional[str],
    config: dict,
) -> dict:
    """Constructor kwargs shared by the sync and async Qdrant clients."""
    if url is not None:
        return {"url": url, "api_key": api_key, **config}
    if path is not None:
        return {"path": path}
    return {"location": location or ":memory:"}


class NativeAsyncQdrantCollection(AsyncAbstractCollection):
    """
    A Qdrant collection driven by ``AsyncQdrantClient`` — non-blocking I/O.

    Same storage layout, filter translation and scores as
    :class:`QdrantCollection`, so data written by either is readable by both.
    """

    supported_filter_operators = SUPPORTED_FILTER_OPERATORS

    def __init__(
        self,
        name: str,
        client: AsyncQdrantClient,
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

    @property
    def native(self) -> AsyncQdrantClient:
        """The raw ``AsyncQdrantClient`` (escape hatch)."""
        return self._client

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection lazily, once the dimension is known."""
        if not await self._client.collection_exists(self.name):
            await self._client.create_collection(
                collection_name=self.name,
                vectors_config=_vectors_config(self.dimension, self.metric),
            )

    # ----- async raw primitives ------------------------------------------ #

    async def _write_many(self, docs: list[Document]) -> None:
        await self._ensure_collection()
        await self._client.upsert(self.name, points=[_to_point(d) for d in docs])

    async def _read(self, key: str) -> Document:
        if not await self._client.collection_exists(self.name):
            raise KeyError(key)
        points = await self._client.retrieve(
            self.name, ids=[_point_id(key)], with_payload=True, with_vectors=True
        )
        if not points:
            raise KeyError(key)
        return _to_document(points[0])

    async def _drop(self, key: str) -> None:
        if not await self._client.collection_exists(
            self.name
        ) or not await self._client.retrieve(self.name, ids=[_point_id(key)]):
            raise KeyError(key)
        await self._client.delete(
            self.name, points_selector=models.PointIdsList(points=[_point_id(key)])
        )

    async def _keys(self) -> AsyncIterator[str]:
        if not await self._client.collection_exists(self.name):
            return
        offset = None
        while True:
            points, offset = await self._client.scroll(
                self.name, limit=256, offset=offset, with_payload=[_ID_KEY]
            )
            for point in points:
                yield point.payload[_ID_KEY]
            if offset is None:
                break

    async def _count(self) -> int:
        if not await self._client.collection_exists(self.name):
            return 0
        return (await self._client.count(self.name)).count

    async def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> list[SearchResult]:
        if not await self._client.collection_exists(self.name):
            return []
        response = await self._client.query_points(
            self.name,
            query=vector,
            limit=limit,
            query_filter=_to_qdrant_filter(filter),
            with_payload=True,
            **kwargs,
        )
        return [_point_to_result(point, self.metric) for point in response.points]


class NativeAsyncQdrantClient(AsyncAbstractClient):
    """
    Native async Qdrant client — what ``await vd.connect_async("qdrant", url=...)``
    returns for a Qdrant server or cloud cluster.

    Takes the same arguments as :class:`QdrantClientAdapter` (``path``,
    ``url``, ``api_key``, ``location``, ``embedder``). It also works embedded
    (no ``url``), but there qdrant-client's async client runs synchronous code
    inside its coroutines and blocks the event loop, so
    :func:`vd.connect_async` returns the thread-pool wrapper for embedded mode
    instead (see :func:`_connect_async_qdrant`).

    Examples
    --------
    >>> import asyncio, vd
    >>> async def go():
    ...     async with NativeAsyncQdrantClient() as client:  # embedded, for the demo
    ...         col = await client.create_collection("docs", dimension=2)
    ...         await col.set("a", vd.Document(id="a", text="x", vector=[1.0, 0.0]))
    ...         return client.native_async, await col.count()
    >>> asyncio.run(go())
    (True, 1)
    """

    backend_name = "qdrant"

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        self._client = AsyncQdrantClient(
            **_qdrant_client_kwargs(
                path=path, url=url, api_key=api_key, location=location, config=config
            )
        )
        self._metrics: dict[str, str] = {}

    def _collection(
        self, name: str, *, dimension: Optional[int], metric: str
    ) -> NativeAsyncQdrantCollection:
        return NativeAsyncQdrantCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    async def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> NativeAsyncQdrantCollection:
        if await self._client.collection_exists(name) or name in self._metrics:
            raise ValueError(f"Collection {name!r} already exists")
        self._metrics[name] = metric
        collection = self._collection(name, dimension=dimension, metric=metric)
        if dimension is not None:  # eager create when the dimension is known
            await collection._ensure_collection()
        return collection

    async def get_collection(self, name: str) -> NativeAsyncQdrantCollection:
        if not await self._client.collection_exists(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        return self._collection(
            name, dimension=None, metric=self._metrics.get(name, "cosine")
        )

    async def delete_collection(self, name: str) -> None:
        exists = await self._client.collection_exists(name)
        if not exists and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        if exists:
            await self._client.delete_collection(name)
        self._metrics.pop(name, None)

    async def list_collections(self) -> AsyncIterator[str]:
        response = await self._client.get_collections()
        names = {c.name for c in response.collections} | set(self._metrics)
        for name in sorted(names):
            yield name


async def _connect_async_qdrant(**kwargs) -> Any:
    """
    The :func:`vd.connect_async` factory for ``qdrant``.

    With ``url=`` (a Qdrant server or cloud cluster) it returns
    :class:`NativeAsyncQdrantClient`, which does real non-blocking network I/O.
    Without it (embedded ``:memory:`` / ``path=`` mode), qdrant-client's local
    async client would run blocking code on the event loop, so it returns the
    ``asyncio.to_thread`` wrapper around the sync adapter instead.
    """
    if kwargs.get("url") is not None:
        return NativeAsyncQdrantClient(**kwargs)
    import asyncio

    from vd.asynchronous import AsyncClientWrapper
    from vd.util import connect

    return AsyncClientWrapper(await asyncio.to_thread(connect, "qdrant", **kwargs))


register_async_backend("qdrant", _connect_async_qdrant)
