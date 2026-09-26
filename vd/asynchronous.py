"""
Async support for ``vd``: universal wrapper + opt-in native implementations.

This module gives every ``vd`` backend an ``async``/``await`` surface day one,
without forking the adapter hierarchy. Three pieces:

- :class:`AsyncCollectionWrapper` / :class:`AsyncClientWrapper` —
  thin adapters that take any sync :class:`vd.Collection` / :class:`vd.Client`
  and dispatch every method to :func:`asyncio.to_thread`. This is the
  **universal fallback**: every backend works through it.
- :class:`AsyncAbstractCollection` / :class:`AsyncAbstractClient` — bases for
  **native** async adapters, which do real non-blocking I/O through a
  backend's own async SDK. A backend implements a few ``async`` raw
  primitives and registers its client with :func:`register_async_backend`.
  Native today: ``qdrant`` against a server (``url=``), on
  ``qdrant_client.AsyncQdrantClient``.
- :func:`connect_async` — the entry point. Mirrors :func:`vd.connect`. It
  returns the backend's registered native client when there is one, and the
  wrapper otherwise (or when called with ``native=False``).

The asyncio.to_thread wrapper does **not** make I/O non-blocking — it moves
blocking calls off the event loop onto a worker thread. For real
non-blocking I/O against a network backend, use a client whose
``native_async`` attribute is ``True`` (see :class:`vd.SupportsNativeAsync`).

The module name is ``vd.asynchronous`` (not ``vd.async``) because ``async``
is a Python keyword.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Union

from vd.base import (
    AsyncClient,
    AsyncCollection,
    Document,
    DocumentInput,
    Filter,
    SearchResult,
    StaticIndexError,
    SupportsHybrid,
    Vector,
    _coerce_document,
    _CollectionPolicy,
)

# --------------------------------------------------------------------------- #
# Universal wrappers
# --------------------------------------------------------------------------- #


class AsyncCollectionWrapper:
    """
    Adapt a sync :class:`~vd.Collection` to the :class:`~vd.AsyncCollection`
    contract by dispatching every method to :func:`asyncio.to_thread`.

    Use :func:`connect_async` rather than instantiating this directly — it
    will pick this wrapper or a native async adapter as appropriate.

    Parameters
    ----------
    sync_collection :
        A live :class:`~vd.Collection` (typically obtained from a
        :class:`~vd.Client`).

    Attributes
    ----------
    native_async : bool
        Always ``False`` for this wrapper. The wrapper still satisfies
        :class:`~vd.SupportsNativeAsync` structurally (the attribute is
        present), but the boolean tells callers that I/O is happening in a
        thread pool rather than on the event loop. Prefer a native
        implementation for high-concurrency workloads.
    """

    #: This wrapper offloads to a thread pool; it doesn't do non-blocking I/O.
    native_async: bool = False

    def __init__(self, sync_collection: Any):
        self._sync = sync_collection

    # ----- escape hatch — the wrapped sync collection ---------------------- #

    @property
    def sync(self) -> Any:
        """The underlying sync :class:`~vd.Collection` — a documented escape hatch."""
        return self._sync

    @property
    def native(self) -> Any:
        """Pass through to the wrapped collection's :attr:`~vd.Collection.native`."""
        return getattr(self._sync, "native", None)

    # ----- AsyncCollection contract ---------------------------------------- #

    async def get(self, key: str) -> Document:
        """Fetch one document; raises ``KeyError`` if absent."""
        return await asyncio.to_thread(self._sync.__getitem__, key)

    async def set(self, key: str, value: Union[str, tuple, Document]) -> None:
        """Insert or replace a document (idempotent upsert)."""
        await asyncio.to_thread(self._sync.__setitem__, key, value)

    async def delete(self, key: str) -> None:
        """Delete a document; raises ``KeyError`` if absent."""
        await asyncio.to_thread(self._sync.__delitem__, key)

    async def keys(self) -> AsyncIterator[str]:
        """Yield document ids."""
        # We materialize once in a worker thread, then yield from memory.
        # Streaming through asyncio.to_thread per-item would be much slower
        # for the common case where _keys() is already O(N) iteration.
        ids = await asyncio.to_thread(lambda: list(self._sync))
        for doc_id in ids:
            yield doc_id

    async def count(self) -> int:
        """Return the number of documents."""
        return await asyncio.to_thread(self._sync.__len__)

    async def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> AsyncIterator[SearchResult]:
        """
        Yield the ``limit`` documents most similar to ``query``.

        The underlying search runs once on a worker thread; results stream
        from memory. (Most backends' sync ``search`` already returns a list
        or a fully-realized iterator under the hood.)
        """

        def _run() -> list[SearchResult]:
            return list(
                self._sync.search(
                    query, limit=limit, filter=filter, egress=egress, **kwargs
                )
            )

        results = await asyncio.to_thread(_run)
        for hit in results:
            yield hit

    # ----- batch convenience (also satisfies an async SupportsBatch) ------ #

    async def add_documents(
        self,
        documents: Iterable[Any],
        *,
        batch_size: int = 100,
    ) -> None:
        """Batch upsert — mirrors :meth:`~vd.AbstractCollection.add_documents`."""
        await asyncio.to_thread(
            self._sync.add_documents, list(documents), batch_size=batch_size
        )

    async def upsert(self, document: Document) -> None:
        """Insert or replace ``document``."""
        await asyncio.to_thread(self._sync.upsert, document)


class AsyncClientWrapper:
    """
    Adapt a sync :class:`~vd.Client` to the :class:`~vd.AsyncClient`
    contract by dispatching every method to :func:`asyncio.to_thread`.

    Use :func:`connect_async` rather than instantiating this directly.

    Parameters
    ----------
    sync_client :
        A live :class:`~vd.Client` (typically obtained from :func:`vd.connect`).

    Attributes
    ----------
    native_async : bool
        Always ``False`` for this wrapper.
    """

    native_async: bool = False

    def __init__(self, sync_client: Any):
        self._sync = sync_client

    # ----- escape hatches -------------------------------------------------- #

    @property
    def sync(self) -> Any:
        """The underlying sync :class:`~vd.Client` — a documented escape hatch."""
        return self._sync

    @property
    def client(self) -> Any:
        """Pass through to the wrapped client's :attr:`~vd.Client.client`."""
        return getattr(self._sync, "client", None)

    # ----- AsyncClient contract -------------------------------------------- #

    async def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> AsyncCollection:
        """Create a new collection; raise ``ValueError`` if it exists."""
        col = await asyncio.to_thread(
            self._sync.create_collection,
            name,
            dimension=dimension,
            metric=metric,
            **index_config,
        )
        return AsyncCollectionWrapper(col)

    async def get_collection(self, name: str) -> AsyncCollection:
        """Return an existing collection; raise ``KeyError`` if absent."""
        col = await asyncio.to_thread(self._sync.get_collection, name)
        return AsyncCollectionWrapper(col)

    async def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> AsyncCollection:
        """Return collection ``name``, creating it if missing."""
        col = await asyncio.to_thread(
            self._sync.get_or_create_collection,
            name,
            dimension=dimension,
            metric=metric,
            **index_config,
        )
        return AsyncCollectionWrapper(col)

    async def delete_collection(self, name: str) -> None:
        """Drop a collection; raise ``KeyError`` if absent."""
        await asyncio.to_thread(self._sync.delete_collection, name)

    async def list_collections(self) -> AsyncIterator[str]:
        """Yield collection names."""
        names = await asyncio.to_thread(lambda: list(self._sync.list_collections()))
        for name in names:
            yield name

    # ----- lifecycle / context manager ------------------------------------ #

    async def close(self) -> None:
        """Release backend resources. Calls ``close()`` on the sync client if present."""
        close = getattr(self._sync, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    async def __aenter__(self) -> "AsyncClientWrapper":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


# --------------------------------------------------------------------------- #
# Native async bases — for backends whose SDK ships an async client
# --------------------------------------------------------------------------- #


class AsyncAbstractCollection(_CollectionPolicy, metaclass=ABCMeta):
    """
    Base class for **native** async collections (the async sibling of
    :class:`vd.AbstractCollection`).

    A backend subclasses this and implements ``async`` raw primitives; the
    user-facing :class:`~vd.AsyncCollection` surface is provided here, with
    the same input coercion, embedding, dimension checks, filter validation
    and ``egress`` handling as the sync base (both share one policy mixin).

    Subclass responsibilities (async raw primitives)
    ------------------------------------------------
    ``async _write_many(docs)``
        Upsert documents; each ``vector`` is set and dimension-checked.
    ``async _read(key) -> Document``
        Fetch one document; raise ``KeyError`` if absent.
    ``async _drop(key)``
        Delete one document; raise ``KeyError`` if absent.
    ``_keys() -> AsyncIterator[str]``
        An async generator of document ids.
    ``async _count() -> int``
        Number of documents.
    ``async _query(vector, *, limit, filter, **kwargs) -> list[SearchResult]``
        Raw nearest-neighbor search; ``filter`` is the canonical AST.
    """

    #: Real non-blocking I/O through the backend's async SDK.
    native_async: bool = True

    # ----- escape hatch --------------------------------------------------- #

    @property
    def native(self) -> Any:
        """The raw backend handle (escape hatch), or ``None``."""
        return getattr(self, "_native", None)

    # ----- AsyncCollection contract --------------------------------------- #

    async def get(self, key: str) -> Document:
        """Fetch one document; raises ``KeyError`` if absent."""
        return await self._read(key)

    async def set(self, key: str, value: Union[str, tuple, Document]) -> None:
        """Insert or replace a document (idempotent upsert)."""
        self._check_writable()
        doc = self._ensure_vector(_coerce_document(key, value))
        await self._write_many([doc])

    async def delete(self, key: str) -> None:
        """Delete a document; raises ``KeyError`` if absent."""
        self._check_writable()
        await self._drop(key)

    async def keys(self) -> AsyncIterator[str]:
        """Yield document ids."""
        async for key in self._keys():
            yield key

    async def count(self) -> int:
        """Return the number of documents."""
        return await self._count()

    async def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> AsyncIterator[SearchResult]:
        """Yield the ``limit`` documents most similar to ``query``.

        Same contract as :meth:`vd.AbstractCollection.search`.
        """
        from vd.filters import validate_filter

        validate_filter(filter, supported=self.supported_filter_operators)
        vector = self._resolve_query(query)
        for result in await self._query(vector, limit=limit, filter=filter, **kwargs):
            yield egress(result) if egress is not None else result

    # ----- batch convenience ---------------------------------------------- #

    async def add_documents(
        self,
        documents: Iterable[DocumentInput],
        *,
        batch_size: int = 100,
    ) -> None:
        """Batch upsert — mirrors :meth:`vd.AbstractCollection.add_documents`."""
        from vd.util import normalize_document_input

        self._check_writable()
        batch: list[Document] = []
        for item in documents:
            doc = normalize_document_input(item, auto_id=True)
            self._ensure_vector(doc)
            batch.append(doc)
            if len(batch) >= batch_size:
                await self._write_many(batch)
                batch = []
        if batch:
            await self._write_many(batch)

    async def upsert(self, document: Document) -> None:
        """Insert or replace ``document``."""
        await self.set(document.id, document)

    def _check_writable(self) -> None:
        if not self.supports_incremental_writes:
            raise StaticIndexError(
                f"Collection {self.name!r} uses a static index and cannot "
                f"accept writes after creation. Rebuild it instead."
            )

    # ----- raw primitives — adapters MUST implement ----------------------- #

    @abstractmethod
    async def _write_many(self, docs: list[Document]) -> None:
        """Upsert documents (vectors set and dimension-checked)."""

    @abstractmethod
    async def _read(self, key: str) -> Document:
        """Fetch one document; raise ``KeyError`` if absent."""

    @abstractmethod
    async def _drop(self, key: str) -> None:
        """Delete one document; raise ``KeyError`` if absent."""

    @abstractmethod
    def _keys(self) -> AsyncIterator[str]:
        """Async-iterate document ids."""

    @abstractmethod
    async def _count(self) -> int:
        """Number of documents."""

    @abstractmethod
    async def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> list[SearchResult]:
        """Raw nearest-neighbor search."""


class AsyncAbstractClient(metaclass=ABCMeta):
    """
    Base class for **native** async clients (the async sibling of
    :class:`vd.AbstractClient`).

    A backend implements :meth:`create_collection`, :meth:`get_collection`,
    :meth:`delete_collection` and :meth:`list_collections` as coroutines /
    async generators; :meth:`get_or_create_collection`, the ``client`` escape
    hatch, :meth:`close` and ``async with`` support come for free. Register
    the class with :func:`register_async_backend` so :func:`connect_async`
    returns it.

    Parameters
    ----------
    embedder : callable, optional
        A ``text -> vector`` function handed to every collection.
    **config
        Backend-specific connection configuration.
    """

    #: Real non-blocking I/O through the backend's async SDK.
    native_async: bool = True

    #: The registry name of this backend (set by :func:`register_async_backend`).
    backend_name: str = ""

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        **config,
    ):
        self._embedder = embedder
        self.config = config

    @property
    def client(self) -> Any:
        """The raw async backend client — a supported, documented escape hatch."""
        return getattr(self, "_client", None)

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> AsyncAbstractCollection:
        """Create a new collection; raise ``ValueError`` if it exists."""

    @abstractmethod
    async def get_collection(self, name: str) -> AsyncAbstractCollection:
        """Return an existing collection; raise ``KeyError`` if absent."""

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Drop a collection; raise ``KeyError`` if absent."""

    @abstractmethod
    def list_collections(self) -> AsyncIterator[str]:
        """Async-iterate collection names."""

    async def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> AsyncAbstractCollection:
        """Return collection ``name``, creating it if missing."""
        try:
            return await self.get_collection(name)
        except KeyError:
            return await self.create_collection(
                name, dimension=dimension, metric=metric, **index_config
            )

    async def close(self) -> None:
        """Release backend resources (closes the raw client if it can)."""
        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> "AsyncAbstractClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


# --------------------------------------------------------------------------- #
# Native async backend registry
# --------------------------------------------------------------------------- #

#: name -> factory returning a native async client (or an awaitable of one).
_async_backends: dict[str, Callable[..., Any]] = {}


def register_async_backend(name: str, factory: Optional[Callable[..., Any]] = None):
    """
    Register a native async client factory for backend ``name``.

    Use as a class decorator on an :class:`AsyncAbstractClient` subclass, or
    call it with a ``factory`` (a class or a function, sync or ``async``,
    taking the :func:`connect_async` keyword arguments). Once registered,
    :func:`connect_async` returns the factory's client instead of the
    ``to_thread`` wrapper.

    Examples
    --------
    >>> @register_async_backend('example')           # doctest: +SKIP
    ... class ExampleAsyncClient(AsyncAbstractClient):
    ...     ...
    """

    def decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        if isinstance(factory, type):
            factory.backend_name = name
        _async_backends[name] = factory
        return factory

    return decorator(factory) if factory is not None else decorator


def list_async_backends() -> list[str]:
    """
    Return the names of backends with a registered native async client.

    Every other backend still works with :func:`connect_async`, through the
    universal ``to_thread`` wrapper.
    """
    return sorted(_async_backends)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def connect_async(backend: str, *, native: bool = True, **kwargs) -> AsyncClient:
    """
    Async sibling of :func:`vd.connect`.

    Returns an :class:`~vd.AsyncClient`. When the backend has a native async
    client (see :func:`list_async_backends`; today ``qdrant``), its registered
    factory decides: it may return a native client doing real non-blocking
    I/O (qdrant with ``url=``) or the wrapper (embedded qdrant, whose async
    client would block the loop). Every other backend goes
    through the universal :class:`AsyncClientWrapper`, built on
    :func:`asyncio.to_thread`. Check ``client.native_async`` to tell them
    apart.

    Parameters
    ----------
    backend : str
        Backend name — same vocabulary as :func:`vd.connect`.
    native : bool
        Use the backend's native async client when one is registered
        (default). ``False`` forces the ``to_thread`` wrapper around the
        sync adapter.
    **kwargs
        Forwarded to the native client's constructor, or to
        :func:`vd.connect` for the wrapper. Both take the same arguments
        (``embedder``, ``url``, ``path``, ...).

    Returns
    -------
    AsyncClient
        A live async client. ``await`` once at session start::

            client = await vd.connect_async("memory")

    Examples
    --------
    >>> import asyncio, vd
    >>> async def go():
    ...     client = await vd.connect_async("memory")
    ...     col = await client.create_collection("docs", dimension=2)
    ...     await col.set("a", vd.Document(id="a", text="x", vector=[1.0, 0.0]))
    ...     return await col.count()
    >>> asyncio.run(go())
    1
    """
    # Late import to avoid a top-level cycle (vd.util imports nothing in here,
    # but keep it lazy so this module is safe to import standalone).
    from vd.util import connect

    factory = _async_backends.get(backend) if native else None
    if factory is not None:
        client = factory(**kwargs)
        if inspect.isawaitable(client):
            client = await client
        return client
    sync_client = await asyncio.to_thread(connect, backend, **kwargs)
    return AsyncClientWrapper(sync_client)


# --------------------------------------------------------------------------- #
# hybrid_search_async — async sibling of vd.hybrid_search
# --------------------------------------------------------------------------- #


async def hybrid_search_async(
    collection: AsyncCollection,
    query: Union[str, Vector],
    *,
    query_text: Optional[str] = None,
    limit: int = 10,
    filter: Optional[Filter] = None,
    k_dense: Optional[int] = None,
    k_lexical: Optional[int] = None,
    rrf_k: int = 60,
    lexical_search: Optional[Callable[..., list[SearchResult]]] = None,
    egress: Optional[Callable[[SearchResult], Any]] = None,
    **kwargs,
) -> AsyncIterator[SearchResult]:
    """
    Async sibling of :func:`vd.hybrid_search`.

    For a wrapped sync collection, runs :func:`vd.hybrid_search` (native
    hybrid if the backend has it, else the client-side BM25 + RRF fallback)
    on a worker thread. For a native async collection it awaits the
    collection's own ``hybrid_search`` if it has one, and otherwise fuses the
    collection's async dense search with a client-side BM25 scan (O(N): it
    reads every document) via RRF. On a native collection, an ``async def``
    ``lexical_search`` receives the async collection; a sync one receives a
    materialized ``{id: Document}`` dict (every document is read per call)
    and runs on a worker thread. Either way the awaitable + async iterator
    interface stays uniform.

    Parameters mirror :func:`vd.hybrid_search` exactly; see that function for
    the full docs.

    Yields
    ------
    dict
        Fused result dicts.

    Examples
    --------
    >>> import asyncio, vd
    >>> async def go():
    ...     client = await vd.connect_async("memory")
    ...     col = await client.create_collection("docs", dimension=2)
    ...     await col.set("a", vd.Document(id="a", text="cats",
    ...                                    vector=[1.0, 0.0]))
    ...     await col.set("b", vd.Document(id="b", text="dogs",
    ...                                    vector=[0.0, 1.0]))
    ...     hits = []
    ...     async for h in vd.hybrid_search_async(col, [0.9, 0.1],
    ...                                           query_text="cats", limit=1):
    ...         hits.append(h["id"])
    ...     return hits
    >>> asyncio.run(go())
    ['a']
    """
    from vd.search import hybrid_search as sync_hybrid_search

    if getattr(collection, "native_async", False):
        async for hit in _native_hybrid_search_async(
            collection,
            query,
            query_text=query_text,
            limit=limit,
            filter=filter,
            k_dense=k_dense,
            k_lexical=k_lexical,
            rrf_k=rrf_k,
            lexical_search=lexical_search,
            egress=egress,
            **kwargs,
        ):
            yield hit
        return

    sync_collection = getattr(collection, "sync", collection)

    def _run() -> list[SearchResult]:
        return list(
            sync_hybrid_search(
                sync_collection,
                query,
                query_text=query_text,
                limit=limit,
                filter=filter,
                k_dense=k_dense,
                k_lexical=k_lexical,
                rrf_k=rrf_k,
                lexical_search=lexical_search,
                egress=egress,
                **kwargs,
            )
        )

    results = await asyncio.to_thread(_run)
    for hit in results:
        yield hit


async def _native_hybrid_search_async(
    collection: Any,
    query: Union[str, Vector],
    *,
    query_text: Optional[str],
    limit: int,
    filter: Optional[Filter],
    k_dense: Optional[int],
    k_lexical: Optional[int],
    rrf_k: int,
    lexical_search: Optional[Callable[..., Any]],
    egress: Optional[Callable[[SearchResult], Any]],
    **kwargs,
) -> AsyncIterator[SearchResult]:
    """Hybrid search over a native async collection (see :func:`hybrid_search_async`)."""
    from vd.search import _HYBRID_OVERFETCH_FLOOR, BM25Index, _rrf_fuse

    k_dense_eff = (
        k_dense if k_dense is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    )
    k_lexical_eff = (
        k_lexical if k_lexical is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    )
    native_hybrid = getattr(collection, "hybrid_search", None)
    if native_hybrid is not None:
        if lexical_search is not None:
            from vd.search import _warn_lexical_search_ignored

            _warn_lexical_search_ignored(collection)
        async for hit in native_hybrid(
            query,
            query_text=query_text,
            limit=limit,
            filter=filter,
            k_dense=k_dense_eff,
            k_lexical=k_lexical_eff,
            rrf_k=rrf_k,
            egress=egress,
            **kwargs,
        ):
            yield hit
        return

    if isinstance(query, str):
        text = query_text if query_text is not None else query
    else:
        if query_text is None:
            raise ValueError(
                "hybrid_search needs a `query_text` for the lexical side when "
                "`query` is a vector. Either pass query_text=..., or pass "
                "`query` as a string and let the embedder handle both."
            )
        text = query_text
    if not text:
        raise ValueError("hybrid_search needs a non-empty lexical query string.")

    dense = [
        hit async for hit in collection.search(query, limit=k_dense_eff, filter=filter)
    ]
    if lexical_search is not None and inspect.iscoroutinefunction(lexical_search):
        # An ``async def`` lexical search gets the async collection itself.
        lexical = await lexical_search(
            collection, text, limit=k_lexical_eff, filter=filter
        )
    else:
        # The default BM25 scan and sync callables (e.g. vd.bm25_lexical_search)
        # expect a sync ``id -> Document`` mapping: materialize one.
        docs = {key: await collection.get(key) async for key in collection.keys()}
        if lexical_search is None:
            lexical = BM25Index(docs, filter=filter).search(text, limit=k_lexical_eff)
        else:
            lexical = await asyncio.to_thread(
                lexical_search, docs, text, limit=k_lexical_eff, filter=filter
            )
            if inspect.isawaitable(lexical):
                lexical = await lexical
    for hit in _rrf_fuse([dense, list(lexical)], rrf_k=rrf_k, limit=limit):
        yield egress(hit) if egress is not None else hit


# Re-export SupportsHybrid so users importing from vd.asynchronous have the
# whole hybrid surface in one place — even if their native-async adapter
# decides to also satisfy SupportsHybrid directly.
__all__ = [
    "AsyncCollectionWrapper",
    "AsyncClientWrapper",
    "AsyncAbstractCollection",
    "AsyncAbstractClient",
    "register_async_backend",
    "list_async_backends",
    "connect_async",
    "hybrid_search_async",
    "SupportsHybrid",
]
