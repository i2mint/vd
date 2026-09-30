"""
Tests for native async backends (Phase 2 of the async work, #20).

Covers the dispatch layer and the first native adapter:

- :func:`vd.register_async_backend` + :func:`vd.connect_async` dispatch: a
  registered native factory is returned instead of the ``to_thread`` wrapper,
  and ``native=False`` forces the wrapper;
- the native qdrant adapter (``qdrant_client.AsyncQdrantClient``), exercised
  in Qdrant's embedded ``:memory:`` mode so no server is needed. Its results
  must match the wrapped sync adapter's exactly;
- :func:`vd.hybrid_search_async` over a native collection (no ``.sync`` to
  fall back on).
"""

import pytest

import vd
from vd.asynchronous import AsyncClientWrapper

from tests.conftest import make_embedder

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Registry / dispatch
# --------------------------------------------------------------------------- #


async def test_registered_native_factory_is_used_and_can_be_bypassed():
    made = {}

    class FakeNativeClient:
        native_async = True

        def __init__(self, **kwargs):
            made.update(kwargs)

    vd.register_async_backend("memory_fake_native")(FakeNativeClient)
    try:
        client = await vd.connect_async("memory_fake_native", flavor="x")
        assert isinstance(client, FakeNativeClient)
        assert made == {"flavor": "x"}
        assert "memory_fake_native" in vd.list_async_backends()
    finally:
        from vd.asynchronous import _async_backends

        _async_backends.pop("memory_fake_native", None)


async def test_async_factory_coroutine_is_awaited():
    class FakeNativeClient:
        native_async = True

    async def factory(**kwargs):
        return FakeNativeClient()

    vd.register_async_backend("coro_fake_native", factory)
    try:
        assert isinstance(
            await vd.connect_async("coro_fake_native"), FakeNativeClient
        )
    finally:
        from vd.asynchronous import _async_backends

        _async_backends.pop("coro_fake_native", None)


async def test_backend_without_native_gets_wrapper():
    client = await vd.connect_async("memory")
    assert isinstance(client, AsyncClientWrapper)
    assert client.native_async is False


# --------------------------------------------------------------------------- #
# Native qdrant
# --------------------------------------------------------------------------- #


@pytest.fixture
def qdrant_available():
    pytest.importorskip("qdrant_client")
    if "qdrant" not in vd.list_backends():
        pytest.skip("qdrant backend not installed")


def _native(**kwargs):
    """The native async qdrant client, in embedded mode (no server needed).

    ``connect_async`` only picks it for a server (``url=``): qdrant-client's
    embedded async client runs synchronous code and would block the event
    loop. Constructing it directly is how these tests exercise its code
    without a server.
    """
    from vd.backends.qdrant import NativeAsyncQdrantClient

    return NativeAsyncQdrantClient(**kwargs)


async def test_qdrant_connect_async_dispatch(qdrant_available):
    from qdrant_client import AsyncQdrantClient

    from vd.backends.qdrant import NativeAsyncQdrantClient

    # Embedded mode: the thread-pool wrapper (keeps the event loop free).
    embedded = await vd.connect_async("qdrant")
    assert isinstance(embedded, AsyncClientWrapper)
    await embedded.close()
    # Server mode: the native client (construction does not connect).
    remote = await vd.connect_async("qdrant", url="http://localhost:6399",
                                    check_compatibility=False)
    assert isinstance(remote, NativeAsyncQdrantClient)
    assert remote.native_async is True
    assert isinstance(remote.client, AsyncQdrantClient)
    await remote.close()
    remote2 = await vd.connect_async("qdrant", location="http://localhost:6399",
                                     check_compatibility=False)
    assert isinstance(remote2, NativeAsyncQdrantClient)
    await remote2.close()
    # native=False always gives the wrapper
    wrapped = await vd.connect_async("qdrant", url="http://localhost:6399", check_compatibility=False,
                                     native=False)
    assert isinstance(wrapped, AsyncClientWrapper)
    await wrapped.close()


async def test_qdrant_native_client_surface_types(qdrant_available):
    async with _native() as client:
        assert isinstance(client, vd.AsyncClient)
        assert isinstance(client, vd.SupportsNativeAsync)
        col = await client.create_collection("docs", dimension=2)
        assert isinstance(col, vd.AsyncCollection)
        assert col.native_async is True
        assert col.native is client.client


async def _populate(col):
    await col.set(
        "a", vd.Document(id="a", text="cats purr", vector=[1.0, 0.0],
                         metadata={"k": 1, "tag": "pet"})
    )
    await col.set(
        "b", vd.Document(id="b", text="dogs bark", vector=[0.0, 1.0],
                         metadata={"k": 2, "tag": "pet"})
    )
    await col.set(
        "c", vd.Document(id="c", text="cats and dogs", vector=[0.6, 0.8],
                         metadata={"k": 3})
    )


async def test_qdrant_native_crud(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("crud", dimension=2)
        assert await col.count() == 0
        assert [k async for k in col.keys()] == []
        with pytest.raises(KeyError):
            await col.get("a")

        await _populate(col)
        assert await col.count() == 3
        assert sorted([k async for k in col.keys()]) == ["a", "b", "c"]

        doc = await col.get("a")
        assert (doc.id, doc.text, doc.metadata) == ("a", "cats purr",
                                                    {"k": 1, "tag": "pet"})
        assert doc.vector == pytest.approx([1.0, 0.0])

        # set is an idempotent replace
        await col.set("a", vd.Document(id="a", text="new", vector=[1.0, 0.0]))
        assert (await col.get("a")).text == "new"
        assert await col.count() == 3

        await col.delete("b")
        assert await col.count() == 2
        with pytest.raises(KeyError):
            await col.get("b")
        with pytest.raises(KeyError):
            await col.delete("b")


async def test_qdrant_native_search_filter_egress(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("srch", dimension=2)
        await _populate(col)
        hits = [h async for h in col.search([0.9, 0.1], limit=2)]
        assert [h["id"] for h in hits] == ["a", "c"]
        assert hits[0]["score"] >= hits[1]["score"]
        assert {"id", "text", "score", "metadata"} <= set(hits[0])

        ids = [h["id"] async for h in col.search(
            [0.9, 0.1], filter={"k": {"$gte": 2}, "tag": "pet"})]
        assert ids == ["b"]

        only_ids = [r async for r in col.search([0.9, 0.1], limit=3,
                                                egress=lambda h: h["id"])]
        assert only_ids == ["a", "c", "b"]


async def test_qdrant_native_matches_wrapped_sync(qdrant_available):
    """Native and wrapped-sync adapters return identical results."""
    results = []
    for native in (True, False):
        client = _native() if native else await vd.connect_async("qdrant")
        async with client:
            col = await client.create_collection("parity", dimension=2,
                                                 metric="l2")
            await _populate(col)
            hits = [
                (h["id"], round(h["score"], 6), h["text"], h["metadata"])
                async for h in col.search([0.7, 0.3], limit=3,
                                          filter={"k": {"$in": [1, 3]}})
            ]
            results.append(hits)
    assert results[0] == results[1]
    assert [r[0] for r in results[0]] == ["a", "c"]


async def test_qdrant_native_embedder_and_dimension_checks(qdrant_available):
    embed = make_embedder()
    async with _native(embedder=embed) as client:
        col = await client.create_collection("emb")
        await col.set("a", "cats and kittens")
        await col.set("b", ("dogs and puppies", {"kind": "dog"}))
        assert (await col.get("b")).metadata == {"kind": "dog"}
        assert col.dimension == len(embed("x"))
        hits = [h["id"] async for h in col.search("cats and kittens", limit=1)]
        assert hits == ["a"]
        with pytest.raises(ValueError, match="dimension mismatch"):
            await col.set("c", vd.Document(id="c", text="x", vector=[1.0, 2.0]))
        with pytest.raises(ValueError, match="dimension mismatch"):
            async for _ in col.search([1.0, 2.0]):
                pass

    async with _native() as client:
        col = await client.create_collection("noemb", dimension=2)
        with pytest.raises(vd.EmbeddingRequiredError):
            await col.set("a", "raw text needs an embedder")
        with pytest.raises(vd.UnsupportedFilterError):
            async for _ in col.search([1.0, 0.0], filter={"x": {"$regex": "a"}}):
                pass


async def test_qdrant_native_batch_ops(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("batch", dimension=2)
        await col.upsert(vd.Document(id="x", text="x", vector=[1.0, 0.0]))
        await col.add_documents(
            [vd.Document(id=str(i), text=f"t{i}", vector=[1.0, float(i)])
             for i in range(5)],
            batch_size=2,
        )
        assert await col.count() == 6
        assert sorted([k async for k in col.keys()]) == [
            "0", "1", "2", "3", "4", "x"
        ]


async def test_qdrant_native_keys_paginate(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("many", dimension=2)
        await col.add_documents(
            [vd.Document(id=f"d{i}", text="t", vector=[1.0, float(i)])
             for i in range(600)],
            batch_size=250,
        )
        keys = [k async for k in col.keys()]
        assert len(keys) == len(set(keys)) == 600


async def test_qdrant_native_client_surface(qdrant_available):
    async with _native() as client:
        assert [n async for n in client.list_collections()] == []
        await client.create_collection("one", dimension=2)
        lazy = await client.create_collection("lazy")  # no dimension yet
        assert sorted([n async for n in client.list_collections()]) == [
            "lazy", "one"
        ]
        with pytest.raises(ValueError):
            await client.create_collection("one", dimension=2)
        with pytest.raises(KeyError):
            await client.get_collection("nope")
        with pytest.raises(KeyError):
            await client.delete_collection("nope")

        # the lazy collection learns its dimension on first write
        await lazy.set("a", vd.Document(id="a", text="t", vector=[0.0, 1.0]))
        again = await client.get_collection("lazy")
        assert (await again.get("a")).text == "t"

        same = await client.get_or_create_collection("one", dimension=2)
        assert await same.count() == 0
        fresh = await client.get_or_create_collection("two", dimension=2)
        assert isinstance(fresh, vd.AsyncCollection)

        await client.delete_collection("one")
        assert "one" not in [n async for n in client.list_collections()]


async def test_hybrid_search_async_over_native_collection(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("hyb", dimension=2)
        await _populate(col)
        hits = [
            h async for h in vd.hybrid_search_async(
                col, [0.9, 0.1], query_text="cats", limit=3
            )
        ]
        ids = [h["id"] for h in hits]
        assert ids[0] == "a"  # top on both the dense and the lexical side
        assert set(ids) == {"a", "b", "c"}
        filtered = [
            h["id"] async for h in vd.hybrid_search_async(
                col, [0.9, 0.1], query_text="cats", limit=3,
                filter={"tag": "pet"},
            )
        ]
        assert set(filtered) == {"a", "b"}
        with pytest.raises(ValueError, match="query_text"):
            async for _ in vd.hybrid_search_async(col, [0.9, 0.1], limit=2):
                pass


async def test_hybrid_search_async_native_custom_lexical_and_errors(
    qdrant_available,
):
    async with _native() as client:
        col = await client.create_collection("hyb2", dimension=2)
        await _populate(col)

        async def my_lex(collection, text, *, limit, filter):
            return [{"id": "b", "text": "dogs bark", "score": 9.0, "metadata": {}}]

        ids = [
            h["id"] async for h in vd.hybrid_search_async(
                col, [1.0, 0.0], query_text="zzz", limit=3, lexical_search=my_lex,
                egress=lambda h: {**h, "seen": True},
            )
        ]
        assert "b" in ids[:2]
        with pytest.raises(ValueError, match="non-empty"):
            async for _ in vd.hybrid_search_async(col, [1.0, 0.0], query_text=""):
                pass


async def test_hybrid_search_async_delegates_to_native_hybrid():
    calls = {}

    class FakeNativeHybrid:
        native_async = True

        async def hybrid_search(self, query, **kwargs):
            calls.update(kwargs, query=query)
            yield {"id": "x", "text": "", "score": 1.0, "metadata": {}}

    hits = [
        h async for h in vd.hybrid_search_async(
            FakeNativeHybrid(), "q", limit=2, alpha=0.5
        )
    ]
    assert [h["id"] for h in hits] == ["x"]
    assert calls["query"] == "q" and calls["alpha"] == 0.5
    assert calls["k_dense"] == calls["k_lexical"] == 50


async def test_qdrant_native_lazy_collection_before_first_write(qdrant_available):
    async with _native() as client:
        col = await client.create_collection("lazy")  # no dimension → not created
        assert await col.count() == 0
        assert [k async for k in col.keys()] == []
        assert [h async for h in col.search([1.0, 0.0])] == []
        with pytest.raises(KeyError):
            await col.get("a")
        with pytest.raises(KeyError):
            await col.delete("a")
        await client.delete_collection("lazy")  # registered but never created
        assert [n async for n in client.list_collections()] == []


async def test_hybrid_search_async_native_accepts_sync_lexical_callable(
    qdrant_available,
):
    """A sync lexical_search (e.g. vd.bm25_lexical_search) gets a mapping of docs."""
    async with _native() as client:
        col = await client.create_collection("hyb3", dimension=2)
        await _populate(col)
        ids = [
            h["id"] async for h in vd.hybrid_search_async(
                col, [0.0, 1.0], query_text="cats", limit=3,
                lexical_search=vd.bm25_lexical_search,
            )
        ]
        assert set(ids) == {"a", "b", "c"}
        assert ids.index("a") < ids.index("b") or ids[0] == "b"


async def test_qdrant_native_against_live_server(qdrant_available):
    """With a Qdrant server up, connect_async(url=) is native and matches sync."""
    from tests.conftest import _connect_kwargs, _unavailable_reason

    reason = _unavailable_reason("qdrant_server")
    if reason:
        pytest.skip(reason)
    kwargs = _connect_kwargs("qdrant_server")
    sync = vd.connect("qdrant", **kwargs)
    name = "vd_async_live"
    if name in sync:
        sync.delete_collection(name)
    try:
        async with await vd.connect_async("qdrant", **kwargs) as client:
            assert client.native_async is True
            col = await client.create_collection(name, dimension=2, metric="l2")
            await _populate(col)
            await col.add_documents(
                [vd.Document(id=f"d{i}", text="t", vector=[0.1, 0.01 * i])
                 for i in range(300)], batch_size=128)
            assert await col.count() == 303
            assert len({k async for k in col.keys()}) == 303
            native_hits = [(h["id"], round(h["score"], 6)) async for h in
                           col.search([0.7, 0.3], limit=3,
                                      filter={"k": {"$in": [1, 3]}})]
        sync_hits = [(h["id"], round(h["score"], 6)) for h in
                     sync.get_collection(name).search(
                         [0.7, 0.3], limit=3, filter={"k": {"$in": [1, 3]}})]
        assert native_hits == sync_hits and [i for i, _ in native_hits] == ["a", "c"]
    finally:
        if name in sync:
            sync.delete_collection(name)
        sync.close()
