"""
Async support for ``vd``: universal wrapper + opt-in native implementations.

This module gives every ``vd`` backend an ``async``/``await`` surface day one,
without forking the adapter hierarchy. Two pieces:

- :class:`AsyncCollectionWrapper` / :class:`AsyncClientWrapper` —
  thin adapters that take any sync :class:`vd.Collection` / :class:`vd.Client`
  and dispatch every method to :func:`asyncio.to_thread`. This is the
  **universal fallback**: every backend works through it.
- :func:`connect_async` — the entry point. Mirrors :func:`vd.connect`. If a
  backend ships a native async client (Phase 2 follow-ups: chroma, qdrant,
  weaviate, elasticsearch, redis, mongodb, lancedb, milvus, pinecone,
  turbopuffer), :func:`connect_async` returns *that*; otherwise it returns
  the wrapper.

The asyncio.to_thread wrapper does **not** unblock the event loop — it just
moves blocking calls off the main thread, freeing the loop. For real
non-blocking I/O against a network backend, use a client that satisfies
:class:`vd.SupportsNativeAsync`.

The module name is ``vd.asynchronous`` (not ``vd.async``) because ``async``
is a Python keyword.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Union

from vd.base import (
    AsyncClient,
    AsyncCollection,
    Document,
    Filter,
    SearchResult,
    SupportsHybrid,
    Vector,
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
# Entry point
# --------------------------------------------------------------------------- #


async def connect_async(backend: str, **kwargs) -> AsyncClient:
    """
    Async sibling of :func:`vd.connect`.

    Returns an :class:`~vd.AsyncClient`. Today every backend goes through
    the universal :class:`AsyncClientWrapper` (built on
    :func:`asyncio.to_thread`); Phase 2 follow-ups will plug in native async
    clients per backend, which :func:`connect_async` will return instead.

    Parameters
    ----------
    backend : str
        Backend name — same vocabulary as :func:`vd.connect`.
    **kwargs
        Forwarded to :func:`vd.connect`.

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

    # Per-backend native async adapters can be wired here in Phase 2 by
    # checking a registry for an async constructor before falling back. For
    # Phase 1 every backend uses the universal wrapper.
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

    If the wrapped sync collection's class supports native hybrid (i.e.
    satisfies :class:`~vd.SupportsHybrid`), dispatches the whole fused call
    to a worker thread. Otherwise runs the universal client-side BM25 + RRF
    fallback in a worker thread too. In both cases the awaitable + async
    iterator interface stays uniform.

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


# Re-export SupportsHybrid so users importing from vd.asynchronous have the
# whole hybrid surface in one place — even if their native-async adapter
# decides to also satisfy SupportsHybrid directly.
__all__ = [
    "AsyncCollectionWrapper",
    "AsyncClientWrapper",
    "connect_async",
    "hybrid_search_async",
    "SupportsHybrid",
]
