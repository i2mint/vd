"""
Regression tests that need a live server backend.

Each test is skipped unless its server answers (start them with
``docker compose -f tests/docker-compose.yml up -d``). They cover behaviour the
parametrized contract suites only hit by accident of test ordering.
"""

import pytest

import vd
from tests.conftest import SERVER_BACKENDS, _connect_kwargs, _tcp_open


def _live_client(name):
    if name not in vd.list_backends():
        pytest.skip(f"backend {name!r} is not installed")
    host, port = SERVER_BACKENDS[name]["probe"]
    if not _tcp_open(host, port):
        pytest.skip(f"{name!r} server unreachable at {host}:{port}")
    return vd.connect(name, **_connect_kwargs(name))


def test_mongodb_search_after_drop_and_recreate_same_name():
    """A recreated collection gets a fresh vector index (#29).

    Atlas briefly lists the dropped collection's index as DOES_NOT_EXIST; the
    adapter used to take that as "exists", skip creation, and time out.
    """
    client = _live_client("mongodb")
    name = "vd_regress_recreate"
    try:
        for _ in range(2):
            if name in list(client.list_collections()):
                client.delete_collection(name)
            col = client.create_collection(name, dimension=2)
            col["a"] = vd.Document(id="a", text="x", vector=[1.0, 0.0])
            assert [h["id"] for h in col.search([1.0, 0.0], limit=1)] == ["a"]
    finally:
        if name in list(client.list_collections()):
            client.delete_collection(name)
        client.close()
