"""The turbopuffer adapter against a fake client at the SDK seam (i2mint/vd#57).

No account and no network: ``Turbopuffer`` is replaced by an in-memory fake that
keeps the rows, scores them by cosine distance, and records the calls it receives.
This pins the row packing, the metric mapping, the client-side filter and the
collection bookkeeping that the live service cannot be asked about in CI.
"""

import importlib
import math
import sys
import types

import pytest

import vd
from vd.util import _backends


class FakeRow:
    def __init__(self, id, vector, attributes, dist=0.0):
        self.id, self.vector, self.attributes, self.dist = id, vector, attributes, dist


def _cos_dist(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 1 - dot / (na * nb)


class FakeNamespace:
    def __init__(self, on_delete_all=lambda: None):
        self._on_delete_all = on_delete_all
        self.rows_by_id = {}
        self.writes = []
        self.metric = None

    def write(self, *, upsert_rows=None, deletes=None, distance_metric=None):
        self.writes.append({"upsert": upsert_rows, "deletes": deletes, "metric": distance_metric})
        for row in upsert_rows or []:
            self.rows_by_id[row["id"]] = dict(row)
            self.metric = distance_metric
        for key in deletes or []:
            self.rows_by_id.pop(key, None)

    def query(self, *, top_k, include_attributes=True, rank_by=None, filters=None, **_):
        if filters is not None:
            _, _, wanted = filters
            rows = [r for r in self.rows_by_id.values() if r["id"] == wanted]
            return [FakeRow(r["id"], r["vector"], _attrs(r)) for r in rows][:top_k]
        _, _, query_vector = rank_by
        scored = sorted(
            (_cos_dist(r["vector"], query_vector), r) for r in self.rows_by_id.values()
        )
        return [FakeRow(r["id"], r["vector"], _attrs(r), dist) for dist, r in scored[:top_k]]

    def rows(self):
        return [FakeRow(r["id"], r["vector"], _attrs(r)) for r in self.rows_by_id.values()]

    def schema(self):
        return {}

    def delete_all(self):
        self.rows_by_id.clear()
        self._on_delete_all()  # an empty namespace no longer exists server-side


def _attrs(row):
    return {k: v for k, v in row.items() if k not in ("id", "vector")}


class FakeTurbopuffer:
    instances = []

    def __init__(self, *, region, api_key):
        self.region, self.api_key = region, api_key
        self._namespaces = {}
        self.namespaces = types.SimpleNamespace(list=lambda: list(self._namespaces))
        FakeTurbopuffer.instances.append(self)

    def namespace(self, name):
        if name not in self._namespaces:
            self._namespaces[name] = FakeNamespace(lambda: self._namespaces.pop(name, None))
        return self._namespaces[name]


@pytest.fixture
def tpuf(monkeypatch):
    saved_backend = _backends.get("turbopuffer")
    stub = types.ModuleType("turbopuffer")
    stub.Turbopuffer = FakeTurbopuffer
    monkeypatch.setitem(sys.modules, "turbopuffer", stub)
    monkeypatch.delitem(sys.modules, "vd.backends.turbopuffer", raising=False)
    FakeTurbopuffer.instances.clear()
    module = importlib.import_module("vd.backends.turbopuffer")
    yield module
    sys.modules.pop("vd.backends.turbopuffer", None)
    if saved_backend is None:
        _backends.pop("turbopuffer", None)
    else:
        _backends["turbopuffer"] = saved_backend


def embed(text):
    return {"red": [1.0, 0.0], "blue": [0.0, 1.0]}.get(text.split()[0], [0.7, 0.7])


@pytest.fixture
def client(tpuf):
    return tpuf.TurbopufferClient(api_key="k", region="r1", embedder=embed)


def test_api_key_is_required(tpuf, monkeypatch):
    monkeypatch.delenv("TURBOPUFFER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        tpuf.TurbopufferClient()


def test_api_key_comes_from_the_environment(tpuf, monkeypatch):
    monkeypatch.setenv("TURBOPUFFER_API_KEY", "from-env")
    tpuf.TurbopufferClient()
    assert FakeTurbopuffer.instances[-1].api_key == "from-env"


@pytest.mark.parametrize(
    "metric,expected",
    [("cosine", "cosine_distance"), ("l2", "euclidean_squared"), ("dot", "cosine_distance")],
)
def test_metric_mapping(tpuf, metric, expected):
    assert tpuf._map_metric(metric) == expected


def test_row_round_trip_keeps_text_out_of_metadata(tpuf):
    doc = vd.Document(id="a", text="hello", vector=[1.0, 0.0], metadata={"k": 1})
    row = tpuf._pack_row(doc)
    assert row == {"id": "a", "vector": [1.0, 0.0], "_vd_text": "hello", "k": 1}
    back = tpuf._unpack_row(FakeRow("a", [1.0, 0.0], {"_vd_text": "hello", "k": 1}))
    assert (back.text, back.metadata, back.vector) == ("hello", {"k": 1}, [1.0, 0.0])


def test_collection_lifecycle(client):
    col = client.create_collection("docs", metric="l2")
    assert "docs" in list(client.list_collections())
    with pytest.raises(ValueError):
        client.create_collection("docs")
    assert client.get_collection("docs").metric == "l2"
    client.delete_collection("docs")
    assert "docs" not in list(client.list_collections())
    with pytest.raises(KeyError):
        client.get_collection("docs")
    with pytest.raises(KeyError):
        client.delete_collection("docs")


def test_write_read_search_delete(client):
    col = client.create_collection("docs")
    col["r"] = vd.Document(id="r", text="red apple", vector=embed("red"), metadata={"color": "red"})
    col["b"] = vd.Document(id="b", text="blue sky", vector=embed("blue"), metadata={"color": "blue"})
    assert col.native.writes[0]["metric"] == "cosine_distance"
    assert col["r"].text == "red apple" and col["r"].metadata == {"color": "red"}
    assert sorted(col) == ["b", "r"]
    assert len(col) == 2
    hits = list(col.search("red thing", limit=2))
    assert [h["id"] for h in hits][0] == "r"
    assert hits[0]["score"] == pytest.approx(1.0) and "_vd_text" not in hits[0]["metadata"]
    filtered = list(col.search("red thing", limit=2, filter={"color": "blue"}))
    assert [h["id"] for h in filtered] == ["b"]
    del col["r"]
    assert sorted(col) == ["b"]
    with pytest.raises(KeyError):
        col["r"]


def test_missing_key_reads_raise_keyerror_even_when_the_service_errors(client):
    col = client.create_collection("docs")

    def boom(**_):
        raise RuntimeError("service down")

    col.native.query = boom
    with pytest.raises(KeyError):
        col["x"]


def test_recorded_metric_is_read_from_the_schema(tpuf):
    ns = types.SimpleNamespace(schema=lambda: {"vector": {"ann": {"distance_metric": "euclidean_squared"}}})
    assert tpuf._recorded_metric(ns) == "l2"
    assert tpuf._recorded_metric(types.SimpleNamespace(schema=lambda: {"vector": {"ann": True}})) is None

    def fail():
        raise RuntimeError

    assert tpuf._recorded_metric(types.SimpleNamespace(schema=fail)) is None
