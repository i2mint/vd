"""A client that did not create a collection still scores it with its metric (#31).

Chroma, LanceDB, Milvus and turbopuffer adapters remembered each collection's
metric only in the client that created it, so a second client (or a new
process on a persisted store) got ``cosine`` from ``get_collection`` whatever
the collection used. Each adapter now reads the metric back from the backend.
"""

import os
import tempfile

import pytest

import vd

_QUERY = [0.9, 0.1]


def _fill(col):
    col["x"] = vd.Document(id="x", text="", vector=[1.0, 0.0])
    col["y"] = vd.Document(id="y", text="", vector=[0.0, 3.0])


def _scores(col):
    return [(h["id"], round(h["score"], 4)) for h in col.search(_QUERY)]


_EMBEDDED = [
    ("chroma", "chromadb", "persist_directory"),
    ("lancedb", "lancedb", "path"),
]


@pytest.mark.parametrize("backend, module, kw", _EMBEDDED)
@pytest.mark.parametrize("metric", ["l2", "dot", "cosine"])
def test_a_second_client_scores_with_the_collections_metric(backend, module, kw, metric):
    pytest.importorskip(module)
    where = tempfile.mkdtemp(prefix=f"vd_metric_{backend}_")
    col = vd.connect(backend, **{kw: where}).create_collection(
        "metric_probe", dimension=2, metric=metric
    )
    _fill(col)
    other = vd.connect(backend, **{kw: where}).get_collection("metric_probe")
    assert other.metric == metric
    assert _scores(other) == _scores(col)


def test_a_chroma_collection_created_without_vd_uses_chromas_default():
    chromadb = pytest.importorskip("chromadb")
    where = tempfile.mkdtemp(prefix="vd_metric_chroma_raw_")
    chromadb.PersistentClient(path=where).create_collection("made_elsewhere")
    assert vd.connect("chroma", persist_directory=where).get_collection(
        "made_elsewhere"
    ).metric == "l2"


def test_a_lancedb_collection_not_yet_written_keeps_its_metric():
    pytest.importorskip("lancedb")
    client = vd.connect("lancedb", path=tempfile.mkdtemp(prefix="vd_metric_lazy_"))
    client.create_collection("lazy", metric="l2")
    assert client.get_collection("lazy").metric == "l2"


def test_a_lancedb_table_recording_no_metric_reads_as_cosine():
    """Tables written before #31 (vd's default was cosine), or by other tools."""
    lancedb = pytest.importorskip("lancedb")
    where = tempfile.mkdtemp(prefix="vd_metric_lance_raw_")
    lancedb.connect(where).create_table(
        "old", data=[{"id": "x", "text": "", "vector": [1.0, 0.0], "metadata": "{}"}]
    )
    assert vd.connect("lancedb", path=where).get_collection("old").metric == "cosine"


@pytest.mark.parametrize("metric", ["l2", "dot", "cosine"])
def test_a_second_milvus_client_reads_the_metric_from_the_index(metric):
    pytest.importorskip("pymilvus")
    pytest.importorskip("milvus_lite")
    where = os.path.join(tempfile.mkdtemp(prefix="vd_metric_milvus_"), "m.db")
    maker = vd.connect("milvus", path=where)
    col = maker.create_collection("metric_probe", dimension=2, metric=metric)
    _fill(col)
    expected = _scores(col)
    maker.close()
    reader = vd.connect("milvus", path=where)
    other = reader.get_collection("metric_probe")
    assert other.metric == metric
    assert _scores(other) == expected
    reader.close()


class _FakeNamespace:
    """Just enough of a turbopuffer namespace for ``get_collection``."""

    def __init__(self, distance_metric):
        self._schema = {
            "id": {"type": "string"},
            "vector": {"type": "[2]f32", "ann": {"distance_metric": distance_metric}},
        }

    def schema(self):
        return self._schema


@pytest.mark.parametrize(
    "distance, metric", [("euclidean_squared", "l2"), ("cosine_distance", "cosine")]
)
def test_turbopuffer_reads_the_metric_from_the_namespace_schema(distance, metric):
    pytest.importorskip("turbopuffer")
    from vd.backends.turbopuffer import _recorded_metric

    assert _recorded_metric(_FakeNamespace(distance)) == metric


def test_turbopuffer_without_a_readable_schema_records_no_metric():
    pytest.importorskip("turbopuffer")
    from vd.backends.turbopuffer import _recorded_metric

    class Unwritten:
        def schema(self):
            raise RuntimeError("namespace not found")

    assert _recorded_metric(Unwritten()) is None


def test_a_chroma_collection_whose_embedder_cannot_be_built_still_opens():
    """Review finding: reading ``configuration`` builds the embedding function."""
    pytest.importorskip("chromadb")
    from vd.backends.chroma import _recorded_metric

    class Collection:
        configuration_json = {"hnsw": {"space": "ip"}}
        metadata = None

        @property
        def configuration(self):
            raise ValueError("Could not build embedding function openai")

    assert _recorded_metric(Collection()) == "dot"
    del Collection.configuration_json
    assert _recorded_metric(Collection()) == "l2"  # nothing readable: the default


def test_milvus_reads_only_the_vector_fields_index():
    pytest.importorskip("pymilvus")
    from vd.backends.milvus import _recorded_metric

    class Client:
        def list_indexes(self, collection_name):
            return ["other_idx", "vector_idx"]

        def describe_index(self, collection_name, index_name):
            return {
                "other_idx": {"field_name": "image", "metric_type": "IP"},
                "vector_idx": {"field_name": "vector", "metric_type": "L2"},
            }[index_name]

    assert _recorded_metric(Client(), "c") == "l2"
