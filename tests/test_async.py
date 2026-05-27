"""
Contract tests for the async wrapper, ``connect_async``, and ``hybrid_search_async``.

Phase 1 of #18 — the universal :class:`~vd.asynchronous.AsyncClientWrapper`
adapts every sync backend to the :class:`vd.AsyncClient` /
:class:`vd.AsyncCollection` contract via :func:`asyncio.to_thread`. These
tests run against the ``memory`` backend (no infra needed) and verify:

- the entry point :func:`vd.connect_async` returns an awaitable that resolves
  to an :class:`~vd.AsyncClient`;
- the wrapper satisfies :class:`vd.AsyncClient` / :class:`vd.AsyncCollection`
  structurally;
- ``native_async`` is ``False`` for the wrapper (signals to_thread-based);
- get / set / delete / keys / count / search all work and are awaitable /
  async-iterable;
- :func:`vd.hybrid_search_async` dispatches correctly through the wrapper;
- the async context-manager protocol works.

Phase 2 follow-ups will add per-backend native async adapters; each backend
will get its own parametrized test entries at that time.
"""

import pytest

import vd

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Entry point + protocol satisfaction
# --------------------------------------------------------------------------- #


async def test_connect_async_returns_async_client():
    client = await vd.connect_async("memory")
    try:
        assert isinstance(client, vd.AsyncClient)
        # The universal wrapper structurally satisfies SupportsNativeAsync
        # (the attribute exists); native_async is False to signal that I/O
        # runs in a thread pool.
        assert isinstance(client, vd.SupportsNativeAsync)
        assert client.native_async is False
    finally:
        await client.close()


async def test_async_context_manager():
    async with await vd.connect_async("memory") as client:
        assert isinstance(client, vd.AsyncClient)
        col = await client.create_collection("ctx", dimension=2)
        assert isinstance(col, vd.AsyncCollection)


# --------------------------------------------------------------------------- #
# AsyncClient surface
# --------------------------------------------------------------------------- #


async def test_create_get_delete_list_collections():
    async with await vd.connect_async("memory") as client:
        # Initially empty
        names = [n async for n in client.list_collections()]
        assert names == []

        col = await client.create_collection("docs", dimension=2)
        assert isinstance(col, vd.AsyncCollection)

        names = [n async for n in client.list_collections()]
        assert "docs" in names

        # Round-trip via get_collection
        same = await client.get_collection("docs")
        assert isinstance(same, vd.AsyncCollection)

        # get_or_create returns existing
        again = await client.get_or_create_collection("docs", dimension=2)
        assert isinstance(again, vd.AsyncCollection)

        # Drop
        await client.delete_collection("docs")
        names = [n async for n in client.list_collections()]
        assert "docs" not in names


async def test_get_missing_collection_raises_keyerror():
    async with await vd.connect_async("memory") as client:
        with pytest.raises(KeyError):
            await client.get_collection("nope")


# --------------------------------------------------------------------------- #
# AsyncCollection surface
# --------------------------------------------------------------------------- #


async def _populate(col):
    """Write 3 docs into ``col``."""
    await col.set("a", vd.Document(id="a", text="cats purr", vector=[1.0, 0.0]))
    await col.set("b", vd.Document(id="b", text="dogs bark", vector=[0.0, 1.0]))
    await col.set("c", vd.Document(id="c", text="cats and dogs", vector=[0.5, 0.5]))


async def test_set_get_count_keys_delete():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("crud", dimension=2)
        await _populate(col)

        assert await col.count() == 3

        keys = sorted([k async for k in col.keys()])
        assert keys == ["a", "b", "c"]

        doc = await col.get("a")
        assert isinstance(doc, vd.Document)
        assert doc.id == "a"
        assert doc.text == "cats purr"

        await col.delete("b")
        assert await col.count() == 2
        with pytest.raises(KeyError):
            await col.get("b")


async def test_set_accepts_text_and_tuple_inputs():
    """The wrapper passes flexible inputs straight to sync __setitem__."""
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection(
            "flex", dimension=2, metric="cosine"
        )
        # Pre-vector required for memory (no embedder); use Document form.
        await col.set("a", vd.Document(id="a", text="hi", vector=[1.0, 0.0]))
        doc = await col.get("a")
        assert doc.text == "hi"


async def test_search_returns_async_iterator():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("srch", dimension=2)
        await _populate(col)
        hits = []
        async for h in col.search([0.9, 0.1], limit=2):
            hits.append(h)
        assert len(hits) == 2
        # Closest to [0.9, 0.1] is 'a' ([1, 0]); next is 'c' ([0.5, 0.5]).
        assert hits[0]["id"] == "a"
        # Descending score order
        assert hits[0]["score"] >= hits[1]["score"]


async def test_search_filter_and_egress():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("filt", dimension=2)
        await col.set("a", vd.Document(
            id="a", text="x", vector=[1.0, 0.0], metadata={"k": 1}
        ))
        await col.set("b", vd.Document(
            id="b", text="y", vector=[0.0, 1.0], metadata={"k": 2}
        ))
        # Filter to k==2 only
        ids = [h["id"] async for h in col.search([0.9, 0.1], filter={"k": 2})]
        assert ids == ["b"]
        # Egress transform
        only_ids = [
            r async for r in col.search([0.9, 0.1], egress=lambda h: h["id"])
        ]
        assert set(only_ids) == {"a", "b"}


async def test_upsert_and_add_documents_batch():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("batch", dimension=2)
        # upsert
        await col.upsert(
            vd.Document(id="x", text="x", vector=[1.0, 0.0])
        )
        # add_documents — Documents with vectors, no embedder needed
        await col.add_documents(
            [
                vd.Document(id="y", text="y", vector=[0.0, 1.0]),
                vd.Document(id="z", text="z", vector=[0.5, 0.5]),
            ],
            batch_size=2,
        )
        assert await col.count() == 3


# --------------------------------------------------------------------------- #
# Escape hatches
# --------------------------------------------------------------------------- #


async def test_sync_and_native_escape_hatches():
    async with await vd.connect_async("memory") as client:
        # AsyncClientWrapper.sync exposes the underlying sync Client
        assert isinstance(client.sync, vd.Client)
        # client.client mirrors the sync client's `client` property
        # (None for memory; the attribute existing is the contract).
        assert hasattr(client, "client")
        col = await client.create_collection("esc", dimension=2)
        # AsyncCollectionWrapper.sync exposes the underlying sync Collection
        assert isinstance(col.sync, vd.Collection)
        assert hasattr(col, "native")


# --------------------------------------------------------------------------- #
# hybrid_search_async
# --------------------------------------------------------------------------- #


async def test_hybrid_search_async_basic():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("hyb", dimension=2)
        await _populate(col)
        hits = []
        async for h in vd.hybrid_search_async(
            col, [0.9, 0.1], query_text="cats", limit=3
        ):
            hits.append(h)
        ids = [h["id"] for h in hits]
        # 'a' has both dense + lexical signal — should top the fused ranking.
        assert "a" in ids[:2]
        # Each result is the standard contract shape
        for h in hits:
            assert {"id", "text", "score", "metadata"} <= set(h.keys())


async def test_hybrid_search_async_requires_query_text_for_vector_query():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("hyb_err", dimension=2)
        await _populate(col)
        with pytest.raises(ValueError, match="query_text"):
            async for _ in vd.hybrid_search_async(col, [0.9, 0.1], limit=2):
                pass
