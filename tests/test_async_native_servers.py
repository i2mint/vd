"""Native async clients of the server backends (#40), against the live servers.

One parametrized suite for every backend in ``conftest.SERVER_BACKENDS`` that
registers a native async client: it must be what :func:`vd.connect_async` returns,
pass CRUD, and give the sync adapter's results on the same server (ids and scores,
with and without a filter, for each metric). A backend whose server is not reachable
(``docker compose -f tests/docker-compose.yml up -d <service>``) is skipped.
"""

import uuid

import pytest

import vd

from tests.conftest import SERVER_BACKENDS, _tcp_open

pytestmark = pytest.mark.asyncio

DOCS = [
    vd.Document(
        id="a", text="cats purr", vector=[1.0, 0.0], metadata={"k": 1, "tag": "pet"}
    ),
    vd.Document(
        id="b", text="dogs bark", vector=[0.0, 1.0], metadata={"k": 2, "tag": "pet"}
    ),
    vd.Document(id="c", text="cats and dogs", vector=[0.6, 0.8], metadata={"k": 3}),
]


def _native_servers():
    import vd.backends  # noqa: F401 - registers what is installed

    out = []
    for key, spec in SERVER_BACKENDS.items():
        backend = spec.get("backend", key)
        if backend in vd.list_async_backends():
            out.append(pytest.param(key, id=key))
    return out


@pytest.fixture(params=_native_servers())
def server(request):
    spec = SERVER_BACKENDS[request.param]
    if not _tcp_open(*spec["probe"]):
        pytest.skip(f"{request.param} server not reachable on {spec['probe']}")
    return spec.get("backend", request.param), spec["connect_kwargs"]()


def _name(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


async def test_connect_async_returns_the_native_client(server):
    backend, kwargs = server
    client = await vd.connect_async(backend, **kwargs)
    try:
        assert client.native_async is True
    finally:
        await client.close()


async def test_crud(server):
    backend, kwargs = server
    name = _name("crud")
    async with await vd.connect_async(backend, **kwargs) as client:
        col = await client.create_collection(name, dimension=2)
        try:
            assert await col.count() == 0
            with pytest.raises(KeyError):
                await col.get("a")
            await col.add_documents(DOCS)
            assert await col.count() == 3
            assert sorted([k async for k in col.keys()]) == ["a", "b", "c"]
            doc = await col.get("a")
            assert (doc.text, doc.metadata) == ("cats purr", {"k": 1, "tag": "pet"})
            assert doc.vector == pytest.approx([1.0, 0.0])
            await col.delete("a")
            with pytest.raises(KeyError):
                await col.delete("a")
            assert await col.count() == 2
            with pytest.raises(ValueError):
                await client.create_collection(name, dimension=2)
            assert name in [n async for n in client.list_collections()]
        finally:
            await client.delete_collection(name)
        with pytest.raises(KeyError):
            await client.get_collection(name)


@pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
async def test_search_matches_the_sync_adapter(server, metric):
    backend, kwargs = server
    name = _name(f"parity_{metric}")
    async with await vd.connect_async(backend, **kwargs) as client:
        col = await client.create_collection(name, dimension=2, metric=metric)
        try:
            await col.add_documents(DOCS)
            got = [r async for r in col.search([0.9, 0.1], limit=3)]
            filtered = [
                r async for r in col.search([0.9, 0.1], limit=3, filter={"tag": "pet"})
            ]
            reopened = await client.get_collection(name)
            assert (reopened.metric, reopened.dimension) == (metric, 2)
            sync = vd.connect(backend, **kwargs).get_collection(name)
            want = list(sync.search([0.9, 0.1], limit=3))
            want_filtered = list(
                sync.search([0.9, 0.1], limit=3, filter={"tag": "pet"})
            )
        finally:
            await client.delete_collection(name)
    assert [r["id"] for r in got] == [r["id"] for r in want]
    assert [r["score"] for r in got] == pytest.approx([r["score"] for r in want])
    assert [r["id"] for r in filtered] == [r["id"] for r in want_filtered]
