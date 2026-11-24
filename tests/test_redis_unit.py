"""Server-free unit tests for the redis backend (fake ``redis`` clients).

The RediSearch index is created lazily on first write, as check-then-create.
Two first writers to a fresh collection can both see no index; the one that
loses the creation race gets ``Index already exists``, which must not cost it
its document (#69).
"""

import asyncio

import pytest

pytest.importorskip("numpy")
pytest.importorskip("redis")

from redis.exceptions import ResponseError  # noqa: E402

import vd  # noqa: E402
from vd.backends.redis import (  # noqa: E402
    NativeAsyncRedisCollection,
    RedisCollection,
)


class _AsyncFT:
    """``FT.INFO`` and ``FT.CREATE`` each take one network round trip."""

    def __init__(self, server, name):
        self.server, self.name = server, name

    async def info(self):
        await asyncio.sleep(0)
        if self.name not in self.server.indexes:
            raise ResponseError("Unknown index name")
        return {"num_docs": 0}

    async def create_index(self, schema, definition=None):
        await asyncio.sleep(0)
        if self.name in self.server.indexes:
            raise ResponseError("Index already exists")
        self.server.indexes.add(self.name)


class _AsyncPipeline:
    def __init__(self, server):
        self.server, self.ops = server, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def hset(self, key, mapping):
        self.ops.append((key, mapping))

    async def execute(self):
        for key, mapping in self.ops:
            self.server.data[key] = mapping


class _FakeAsyncRedis:
    def __init__(self):
        self.indexes, self.data = set(), {}

    def ft(self, name):
        return _AsyncFT(self, name)

    def pipeline(self, transaction=False):
        return _AsyncPipeline(self)


def _doc(i, vector):
    return vd.Document(id=i, text=i, vector=vector)


@pytest.mark.asyncio
async def test_async_concurrent_first_writes_both_land():
    server = _FakeAsyncRedis()
    col = NativeAsyncRedisCollection("fresh", server, dimension=2)
    results = await asyncio.gather(
        col.set("a", _doc("a", [1.0, 0.0])),
        col.set("b", _doc("b", [0.0, 1.0])),
        return_exceptions=True,
    )
    assert results == [None, None]
    assert len(server.data) == 2


class _RacedSyncFT:
    """Another writer creates the index between our ``FT.INFO`` and ``FT.CREATE``."""

    def __init__(self, server, name):
        self.server, self.name = server, name

    def info(self):
        raise ResponseError("Unknown index name")

    def create_index(self, schema, definition=None):
        raise ResponseError(self.server.create_error)


class _FakeSyncRedis:
    def __init__(self, create_error="Index already exists"):
        self.create_error, self.data = create_error, {}

    def ft(self, name):
        return _RacedSyncFT(self, name)

    def hset(self, key, mapping):
        self.data[key] = mapping


def test_sync_write_survives_losing_the_index_creation_race():
    server = _FakeSyncRedis()
    col = RedisCollection("fresh", server, dimension=2)
    col["a"] = _doc("a", [1.0, 0.0])
    assert len(server.data) == 1


def test_sync_other_index_creation_errors_still_raise():
    server = _FakeSyncRedis(create_error="Invalid field type for field `embedding`")
    col = RedisCollection("fresh", server, dimension=2)
    with pytest.raises(ResponseError, match="Invalid field type"):
        col["a"] = _doc("a", [1.0, 0.0])
    assert server.data == {}
