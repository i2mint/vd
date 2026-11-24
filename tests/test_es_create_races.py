"""Concurrent index creation in the Elasticsearch adapters (sync and async).

Checking for an index and then creating it are two calls, so a concurrent creator
can win in between. Losing that race must look like the contract (``ValueError`` from
``create_collection``, success from ``get_or_create_collection`` and from a write),
never like Elasticsearch's own ``resource_already_exists_exception``. Runs against a
fake ``elasticsearch`` module, so no server is needed.
"""

import asyncio
import importlib
import sys
import types

import pytest

ALREADY = "resource_already_exists_exception"


class _BadRequestError(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error = error


class _NotFoundError(Exception):
    pass


class _AsyncIndices:
    def __init__(self, store):
        self.store = store

    async def exists(self, index):
        await asyncio.sleep(0)
        return index in self.store

    async def create(self, index, mappings=None, settings=None):
        await asyncio.sleep(0)
        if index in self.store:
            raise _BadRequestError(ALREADY)
        self.store[index] = {"mappings": dict(mappings or {}), "docs": {}}

    async def get_mapping(self, index):
        await asyncio.sleep(0)
        return {index: {"mappings": self.store[index]["mappings"]}}

    async def put_mapping(self, index, properties):
        await asyncio.sleep(0)
        self.store[index]["mappings"].setdefault("properties", {}).update(properties)

    async def refresh(self, index):
        await asyncio.sleep(0)


class _AsyncElasticsearch:
    def __init__(self, hosts=None, **kwargs):
        self.store = {}
        self.indices = _AsyncIndices(self.store)

    async def index(self, index, id, document, **kwargs):
        await asyncio.sleep(0)
        self.store.setdefault(index, {"mappings": {}, "docs": {}})["docs"][id] = document

    async def count(self, index):
        return {"count": len(self.store[index]["docs"])}

    async def close(self):
        pass


class _SyncIndices:
    """Sync indices whose ``exists`` lies once, as if a peer created it meanwhile."""

    def __init__(self, store):
        self.store = store
        self.stale_exists = set()

    def exists(self, index):
        if index in self.stale_exists:
            self.stale_exists.discard(index)
            return False
        return index in self.store

    def create(self, index, mappings=None, settings=None):
        if index in self.store:
            raise _BadRequestError(ALREADY)
        self.store[index] = {"mappings": dict(mappings or {}), "docs": {}}

    def get_mapping(self, index):
        return {index: {"mappings": self.store[index]["mappings"]}}

    def put_mapping(self, index, properties):
        self.store[index]["mappings"].setdefault("properties", {}).update(properties)

    def refresh(self, index):
        pass


class _Elasticsearch:
    def __init__(self, hosts=None, **kwargs):
        self.store = {}
        self.indices = _SyncIndices(self.store)

    def index(self, index, id, document, **kwargs):
        self.store.setdefault(index, {"mappings": {}, "docs": {}})["docs"][id] = document

    def count(self, index):
        return {"count": len(self.store[index]["docs"])}


@pytest.fixture
def es_mod(monkeypatch):
    fake = types.ModuleType("elasticsearch")
    fake.Elasticsearch = _Elasticsearch
    fake.AsyncElasticsearch = _AsyncElasticsearch
    fake.NotFoundError = _NotFoundError
    fake.BadRequestError = _BadRequestError
    helpers = types.ModuleType("elasticsearch.helpers")
    helpers.scan = helpers.async_scan = None
    fake.helpers = helpers
    monkeypatch.setitem(sys.modules, "elasticsearch", fake)
    monkeypatch.setitem(sys.modules, "elasticsearch.helpers", helpers)
    monkeypatch.delitem(sys.modules, "vd.backends.elasticsearch", raising=False)
    mod = importlib.import_module("vd.backends.elasticsearch")
    yield mod
    sys.modules.pop("vd.backends.elasticsearch", None)


def test_already_exists_is_recognised(es_mod):
    assert es_mod._already_exists(_BadRequestError(ALREADY))
    assert not es_mod._already_exists(_BadRequestError("mapper_parsing_exception"))


def test_async_concurrent_create_collection_loses_with_value_error(es_mod):
    client = es_mod.NativeAsyncElasticsearchClient()

    async def run():
        return await asyncio.gather(
            client.create_collection("fresh"),
            client.create_collection("fresh"),
            return_exceptions=True,
        )

    results = asyncio.run(run())
    assert [type(r).__name__ for r in results].count("ValueError") == 1
    assert not any(isinstance(r, _BadRequestError) for r in results)


def test_async_concurrent_get_or_create_both_succeed(es_mod):
    client = es_mod.NativeAsyncElasticsearchClient()

    async def run():
        return await asyncio.gather(
            client.get_or_create_collection("fresh"),
            client.get_or_create_collection("fresh"),
            return_exceptions=True,
        )

    results = asyncio.run(run())
    assert not any(isinstance(r, Exception) for r in results), results


def test_async_concurrent_first_writes_to_a_missing_index_both_land(es_mod):
    import vd

    client = es_mod.NativeAsyncElasticsearchClient()
    col = client._collection("gone", dimension=2, metric="cosine")  # index absent

    async def run():
        results = await asyncio.gather(
            col.set("a", vd.Document(id="a", text="x", vector=[1.0, 0.0])),
            col.set("b", vd.Document(id="b", text="y", vector=[0.0, 1.0])),
            return_exceptions=True,
        )
        return results, await col.count()

    results, count = asyncio.run(run())
    assert results == [None, None]
    assert count == 2


def test_sync_create_collection_race_is_a_value_error(es_mod):
    client = es_mod.ElasticsearchClient()
    client._client.indices.store["fresh"] = {"mappings": {}, "docs": {}}
    client._client.indices.stale_exists.add("fresh")  # the check saw no index
    with pytest.raises(ValueError, match="already exists"):
        client.create_collection("fresh")


def test_sync_ensure_index_tolerates_a_concurrent_create(es_mod):
    client = es_mod.ElasticsearchClient()
    col = es_mod.ElasticsearchCollection(
        "fresh", client._client, dimension=2, metric="cosine"
    )
    client._client.indices.store["fresh"] = {"mappings": {}, "docs": {}}
    client._client.indices.stale_exists.add("fresh")
    col._ensure_index()  # must not raise
    props = client._client.indices.store["fresh"]["mappings"].get("properties", {})
    assert "embedding" in props
