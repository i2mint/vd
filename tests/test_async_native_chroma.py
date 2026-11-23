"""Native async Chroma (#40): ``chromadb.AsyncHttpClient`` against a real ``chroma run`` server.

The server is started for the module on a free port with a temporary path, and the
tests skip when ``chromadb`` or its ``chroma`` CLI is missing or the server does not
come up. Results must match the sync adapter's on the same server.
"""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import vd
from vd.asynchronous import AsyncClientWrapper

pytestmark = pytest.mark.asyncio

chromadb = pytest.importorskip("chromadb")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def chroma_server(tmp_path_factory):
    cli = shutil.which("chroma") or str(Path(sys.executable).with_name("chroma"))
    if not Path(cli).exists():
        pytest.skip("the chroma CLI is not installed")
    port = _free_port()
    proc = subprocess.Popen(
        [
            cli,
            "run",
            "--path",
            str(tmp_path_factory.mktemp("chroma")),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 90  # a loaded machine starts it slowly
        while time.time() < deadline:
            try:
                chromadb.HttpClient(host="localhost", port=port).heartbeat()
                break
            except Exception:
                time.sleep(0.3)
        else:
            pytest.skip("chroma run did not come up")
        yield {"host": "localhost", "port": port}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def test_connect_async_dispatch(chroma_server):
    from vd.backends.chroma import NativeAsyncChromaClient

    native = await vd.connect_async("chroma", **chroma_server)
    assert isinstance(native, NativeAsyncChromaClient) and native.native_async
    await native.close()
    wrapped = await vd.connect_async("chroma")  # in-process: the thread wrapper
    assert isinstance(wrapped, AsyncClientWrapper)
    await wrapped.close()
    forced = await vd.connect_async("chroma", native=False, **chroma_server)
    assert isinstance(forced, AsyncClientWrapper)


DOCS = [
    vd.Document(
        id="a", text="cats purr", vector=[1.0, 0.0], metadata={"k": 1, "tag": "pet"}
    ),
    vd.Document(
        id="b", text="dogs bark", vector=[0.0, 1.0], metadata={"k": 2, "tag": "pet"}
    ),
    vd.Document(id="c", text="cats and dogs", vector=[0.6, 0.8], metadata={"k": 3}),
]


async def test_crud_and_collections(chroma_server):
    async with await vd.connect_async("chroma", **chroma_server) as client:
        col = await client.create_collection("crud", metric="cosine")
        assert col.native_async is True
        assert await col.count() == 0
        with pytest.raises(KeyError):
            await col.get("a")
        for d in DOCS:
            await col.set(d.id, d)
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
            await client.create_collection("crud")
        assert "crud" in [n async for n in client.list_collections()]
        await client.delete_collection("crud")
        with pytest.raises(KeyError):
            await client.get_collection("crud")


@pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
async def test_search_matches_the_sync_adapter(chroma_server, metric):
    name = f"parity_{metric}"
    async with await vd.connect_async("chroma", **chroma_server) as client:
        col = await client.create_collection(name, metric=metric)
        await col.add_documents(DOCS)
        got = [r async for r in col.search([0.9, 0.1], limit=3)]
        filtered = [
            r async for r in col.search([0.9, 0.1], limit=3, filter={"tag": "pet"})
        ]
        reopened = await client.get_collection(name)
        assert reopened.metric == metric
    sync = vd.connect("chroma", **chroma_server).get_collection(name)
    want = list(sync.search([0.9, 0.1], limit=3))
    want_filtered = list(sync.search([0.9, 0.1], limit=3, filter={"tag": "pet"}))
    assert [r["id"] for r in got] == [r["id"] for r in want]
    assert [r["score"] for r in got] == pytest.approx([r["score"] for r in want])
    assert [r["id"] for r in filtered] == [r["id"] for r in want_filtered]
