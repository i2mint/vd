"""
ChromaDB backend.

Chroma is the fastest path to a working RAG demo: ``pip install chromadb``,
no server, optional on-disk persistence. This adapter supports the three
Chroma deployment modes — ephemeral in-process, persistent on disk, and a
remote Chroma server — selected by the arguments to ``vd.connect('chroma', ...)``.

Requires: ``pip install chromadb``
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    import chromadb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The chroma backend needs the 'chromadb' package. "
        "Install it with: pip install chromadb"
    ) from e

from vd.backends._helpers import score_from_distance
from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.filters import SUPPORTED_FILTER_OPERATORS, STRING_OPERATORS
from vd.util import register_backend

#: Operators Chroma's ``where`` clause supports natively. Chroma has no
#: ``$not`` and no ``$exists`` — those raise UnsupportedFilterError so the
#: caller can simplify the filter or drop to ``collection.native``.
CHROMA_FILTER_OPERATORS = (
    SUPPORTED_FILTER_OPERATORS - {"$not", "$exists"} - STRING_OPERATORS
)

#: vd metric name -> Chroma HNSW space name.
_SPACE = {"cosine": "cosine", "dot": "ip", "l2": "l2"}
_METRIC_OF_SPACE = {space: metric for metric, space in _SPACE.items()}
#: Chroma's own default space: a collection created without one uses it.
_CHROMA_DEFAULT_SPACE = "l2"


def _configuration_of(chroma_collection) -> dict:
    """The collection's configuration as a plain dict, or ``{}``.

    ``configuration_json`` first: in chromadb 1.x the ``configuration``
    property rebuilds the collection's embedding function, and raises when it
    cannot (an OpenAI function in a process without its key) -- which would
    make a collection another client created impossible to open.
    """
    raw = getattr(chroma_collection, "configuration_json", None)
    if isinstance(raw, dict):
        return raw
    try:
        configuration = getattr(chroma_collection, "configuration", None)
    except Exception:  # noqa: BLE001 - an embedding function that cannot be built
        return {}
    return configuration if isinstance(configuration, dict) else {}


def _recorded_metric(chroma_collection) -> str:
    """The ``vd`` metric of an existing Chroma collection, read from Chroma.

    Chroma keeps the space in the collection's configuration (``hnsw`` or
    ``spann``) and, on older releases or collections created with
    ``metadata={"hnsw:space": ...}``, in its metadata. A collection that names
    none uses Chroma's default, ``l2``. This is what lets a second client, or
    another process, score an ``l2`` collection as ``l2`` (i2mint/vd#31).
    """
    configuration = _configuration_of(chroma_collection)
    spaces = [
        (configuration.get(index) or {}).get("space")
        for index in ("hnsw", "spann")
        if isinstance(configuration, dict)
    ]
    metadata = getattr(chroma_collection, "metadata", None) or {}
    spaces.append(metadata.get("hnsw:space"))
    space = next((s for s in spaces if s), _CHROMA_DEFAULT_SPACE)
    return _METRIC_OF_SPACE.get(space, space)


def _to_chroma_where(filter: Optional[Filter]) -> Optional[dict]:
    """
    Translate a canonical ``vd`` filter to a Chroma ``where`` clause.

    Chroma's ``where`` is itself MongoDB-ish, so the translation is mostly
    structural: bare values become explicit ``$eq``, and any dict with more
    than one condition is wrapped in an explicit ``$and`` (Chroma requires it).

    Examples
    --------
    >>> _to_chroma_where({'year': 2024})
    {'year': {'$eq': 2024}}
    >>> _to_chroma_where({'a': 1, 'b': 2})
    {'$and': [{'a': {'$eq': 1}}, {'b': {'$eq': 2}}]}
    """
    if not filter:
        return None
    clauses: list[dict] = []
    for key, cond in filter.items():
        if key in ("$and", "$or"):
            clauses.append({key: [_to_chroma_where(f) for f in cond]})
        elif isinstance(cond, dict):
            # One {field: {op: operand}} clause per operator.
            for op, operand in cond.items():
                clauses.append({key: {op: operand}})
        else:
            clauses.append({key: {"$eq": cond}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaCollection(AbstractCollection):
    """A :class:`~vd.base.Collection` wrapping a ``chromadb`` collection."""

    supported_filter_operators = CHROMA_FILTER_OPERATORS

    def __init__(
        self,
        chroma_collection,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        metric: str = "cosine",
    ):
        self.name = chroma_collection.name
        self._collection = chroma_collection
        self._embedder = embedder
        self.dimension = None  # Chroma infers it from the first vector
        self.metric = metric

    @property
    def native(self):
        """The raw ``chromadb`` collection (escape hatch)."""
        return self._collection

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._collection.upsert(
            ids=[doc.id],
            documents=[doc.text],
            embeddings=[doc.vector],
            metadatas=[doc.metadata or None],
        )

    def _write_many(self, docs: list[Document]) -> None:
        self._collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=[d.vector for d in docs],
            metadatas=[d.metadata or None for d in docs],
        )

    def _read(self, key: str) -> Document:
        got = self._collection.get(
            ids=[key], include=["documents", "embeddings", "metadatas"]
        )
        if not got["ids"]:
            raise KeyError(key)
        embeddings = got.get("embeddings")
        return Document(
            id=got["ids"][0],
            text=(got["documents"] or [""])[0] or "",
            vector=list(embeddings[0])
            if embeddings is not None and len(embeddings)
            else None,
            metadata=(got["metadatas"] or [{}])[0] or {},
        )

    def _drop(self, key: str) -> None:
        if not self._collection.get(ids=[key])["ids"]:
            raise KeyError(key)
        self._collection.delete(ids=[key])

    def _keys(self) -> Iterator[str]:
        return iter(self._collection.get(include=[])["ids"])

    def _count(self) -> int:
        return self._collection.count()

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=_to_chroma_where(filter),
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        ids, docs = result["ids"][0], result["documents"][0]
        metas, dists = result["metadatas"][0], result["distances"][0]
        return [
            {
                "id": ids[i],
                "text": docs[i] if docs else "",
                "score": score_from_distance(dists[i], self.metric),
                "metadata": metas[i] or {} if metas else {},
            }
            for i in range(len(ids))
        ]


@register_backend("chroma")
class ChromaClient(AbstractClient):
    """
    ChromaDB client.

    Parameters
    ----------
    persist_directory : str, optional
        Persist on disk at this path (``chromadb.PersistentClient``). If
        omitted and no ``host`` is given, an ephemeral in-process client is
        used (data lost on exit).
    host, port : str, int, optional
        Connect to a running Chroma server (``chromadb.HttpClient``).
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.

    Examples
    --------
    >>> import vd                                                  # doctest: +SKIP
    >>> client = vd.connect('chroma', persist_directory='./db')    # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        if host is not None:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> ChromaCollection:
        space = _SPACE.get(metric, "cosine")
        try:
            chroma_collection = self._client.create_collection(
                name=name, configuration={"hnsw": {"space": space}}, **index_config
            )
        except TypeError:
            # Older chromadb: space goes in metadata, not configuration.
            chroma_collection = self._client.create_collection(
                name=name, metadata={"hnsw:space": space}, **index_config
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                raise ValueError(f"Collection {name!r} already exists") from e
            raise
        return ChromaCollection(
            chroma_collection, embedder=self._embedder, metric=metric
        )

    def get_collection(self, name: str) -> ChromaCollection:
        try:
            chroma_collection = self._client.get_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e
        return ChromaCollection(
            chroma_collection,
            embedder=self._embedder,
            metric=_recorded_metric(chroma_collection),
        )

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e

    def list_collections(self) -> Iterator[str]:
        return (c.name for c in self._client.list_collections())


# --------------------------------------------------------------------------- #
# Native async — chromadb.AsyncHttpClient (a Chroma server)
# --------------------------------------------------------------------------- #

from vd.asynchronous import (  # noqa: E402
    AsyncAbstractClient,
    AsyncAbstractCollection,
    register_async_backend,
)


class NativeAsyncChromaCollection(AsyncAbstractCollection):
    """
    A Chroma collection driven by ``chromadb.AsyncHttpClient`` — non-blocking I/O.

    Same storage layout, filter translation and scores as :class:`ChromaCollection`,
    so data written by either is readable by both.
    """

    supported_filter_operators = CHROMA_FILTER_OPERATORS

    def __init__(
        self,
        chroma_collection,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        metric: str = "cosine",
    ):
        self.name = chroma_collection.name
        self._collection = chroma_collection
        self._embedder = embedder
        self.dimension = None  # Chroma infers it from the first vector
        self.metric = metric

    @property
    def native(self):
        """The raw async ``chromadb`` collection (escape hatch)."""
        return self._collection

    async def _write_many(self, docs: list[Document]) -> None:
        await self._collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=[d.vector for d in docs],
            metadatas=[d.metadata or None for d in docs],
        )

    async def _read(self, key: str) -> Document:
        got = await self._collection.get(
            ids=[key], include=["documents", "embeddings", "metadatas"]
        )
        if not got["ids"]:
            raise KeyError(key)
        embeddings = got.get("embeddings")
        return Document(
            id=got["ids"][0],
            text=(got["documents"] or [""])[0] or "",
            vector=list(embeddings[0])
            if embeddings is not None and len(embeddings)
            else None,
            metadata=(got["metadatas"] or [{}])[0] or {},
        )

    async def _drop(self, key: str) -> None:
        if not (await self._collection.get(ids=[key]))["ids"]:
            raise KeyError(key)
        await self._collection.delete(ids=[key])

    async def _keys(self):
        for key in (await self._collection.get(include=[]))["ids"]:
            yield key

    async def _count(self) -> int:
        return await self._collection.count()

    async def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> list[SearchResult]:
        result = await self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=_to_chroma_where(filter),
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        ids, docs = result["ids"][0], result["documents"][0]
        metas, dists = result["metadatas"][0], result["distances"][0]
        return [
            {
                "id": ids[i],
                "text": docs[i] if docs else "",
                "score": score_from_distance(dists[i], self.metric),
                "metadata": metas[i] or {} if metas else {},
            }
            for i in range(len(ids))
        ]


class NativeAsyncChromaClient(AsyncAbstractClient):
    """
    Native async Chroma client — what ``await vd.connect_async("chroma", host=...)``
    returns for a Chroma server.

    It wraps an already-connected ``chromadb`` async client (``AsyncHttpClient`` is a
    coroutine, so :func:`_connect_async_chroma` awaits it and hands it in). Ephemeral
    and on-disk Chroma run in-process, where there is no I/O to overlap, so
    :func:`vd.connect_async` gives those the thread-pool wrapper instead.
    """

    backend_name = "chroma"

    def __init__(
        self,
        client,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        self._client = client

    def _wrap(self, chroma_collection, metric: str) -> NativeAsyncChromaCollection:
        return NativeAsyncChromaCollection(
            chroma_collection, embedder=self._embedder, metric=metric
        )

    async def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> NativeAsyncChromaCollection:
        space = _SPACE.get(metric, "cosine")
        try:
            chroma_collection = await self._client.create_collection(
                name=name, configuration={"hnsw": {"space": space}}, **index_config
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                raise ValueError(f"Collection {name!r} already exists") from e
            raise
        return self._wrap(chroma_collection, metric)

    async def get_collection(self, name: str) -> NativeAsyncChromaCollection:
        try:
            chroma_collection = await self._client.get_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e
        return self._wrap(chroma_collection, _recorded_metric(chroma_collection))

    async def delete_collection(self, name: str) -> None:
        try:
            await self._client.delete_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e

    async def list_collections(self):
        for c in await self._client.list_collections():
            yield getattr(c, "name", c)

    async def close(self) -> None:
        """
        Release the HTTP connection pool chromadb keeps for the current event loop.

        chromadb's async HTTP client pools one ``httpx.AsyncClient`` per event loop
        and only closes it when the chromadb client itself is used as an async
        context manager. Without this, every ``asyncio.run`` that connects leaves a
        client and its open socket behind. A later request on the same loop simply
        opens a fresh pool.
        """
        await _release_async_http_pool(self._client)


async def _release_async_http_pool(client) -> None:
    """
    Close the ``httpx.AsyncClient`` chromadb pooled for the running event loop.

    chromadb (``AsyncFastAPI``) keeps these in a class-level dict keyed by the
    loop's hash and offers no public per-loop cleanup: its ``__aexit__`` closes the
    clients of *every* loop, including ones bound to other, possibly closed, loops.
    So only the current loop's entry is popped and closed here, and the lookup is
    defensive: if a chromadb release changes these internals, ``close()`` becomes a
    no-op rather than an error.
    """
    server = getattr(client, "_server", None)
    pools = getattr(server, "_clients", None)
    if not isinstance(pools, dict):
        return
    http_client = pools.pop(hash(asyncio.get_running_loop()), None)
    aclose = getattr(http_client, "aclose", None)
    if aclose is not None:
        await aclose()


async def _connect_async_chroma(**kwargs) -> Any:
    """
    The :func:`vd.connect_async` factory for ``chroma``.

    With ``host=`` (a Chroma server) it returns :class:`NativeAsyncChromaClient`
    over ``chromadb.AsyncHttpClient``. Ephemeral and ``persist_directory=`` modes run
    in-process, so they get the ``asyncio.to_thread`` wrapper around the sync
    adapter.
    """
    if kwargs.get("host") is not None:
        host, port = kwargs.pop("host"), kwargs.pop("port", 8000)
        embedder = kwargs.pop("embedder", None)
        client = await chromadb.AsyncHttpClient(host=host, port=port)
        return NativeAsyncChromaClient(client, embedder=embedder, **kwargs)
    from vd.asynchronous import AsyncClientWrapper
    from vd.util import connect

    return AsyncClientWrapper(await asyncio.to_thread(connect, "chroma", **kwargs))


register_async_backend("chroma", _connect_async_chroma)
