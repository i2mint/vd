"""Weaviate, Elasticsearch and MongoDB ``get_collection`` read the metric back (#33).

The server adapters remembered a collection's metric only in the client that
created it, so a second client (or a new process) got ``cosine`` from
``get_collection`` whatever the collection used. These tests need no service:
each fakes just the SDK calls the adapter makes, in the shape the real server
answers them.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vd

METRICS = ["cosine", "dot", "l2"]


# --------------------------------------------------------------------------- #
# Weaviate
# --------------------------------------------------------------------------- #

_WEAVIATE_DISTANCE = {"cosine": "cosine", "dot": "dot", "l2": "l2-squared"}


@pytest.fixture
def weaviate_server(monkeypatch):
    """A fake Weaviate server: ``collections.create`` records the index distance."""
    pytest.importorskip("weaviate")
    from weaviate.classes.config import VectorDistances

    import vd.backends.weaviate as mod

    classes = {}

    def create(*, name, vector_config, **_):
        classes[name] = vector_config.vectorIndexConfig.distance

    def get(name):
        distance = classes[name]
        index = SimpleNamespace(distance_metric=distance)
        config = SimpleNamespace(
            vector_config={"default": SimpleNamespace(vector_index_config=index)}
        )
        return SimpleNamespace(config=SimpleNamespace(get=lambda: config))

    server = MagicMock()
    server.collections.create.side_effect = create
    server.collections.get.side_effect = get
    server.collections.exists.side_effect = lambda name: name in classes
    monkeypatch.setattr(mod.weaviate, "connect_to_local", lambda **kw: server)
    server.classes = classes
    server.distances = VectorDistances
    return server


@pytest.mark.parametrize("metric", METRICS)
def test_weaviate_creates_the_class_with_the_metric_and_reads_it_back(
    weaviate_server, metric
):
    vd.connect("weaviate").create_collection("probe", metric=metric)
    assert weaviate_server.classes["Probe"].value == _WEAVIATE_DISTANCE[metric]
    assert vd.connect("weaviate").get_collection("probe").metric == metric


def test_weaviate_class_whose_distance_vd_does_not_map_reads_as_cosine(weaviate_server):
    weaviate_server.classes["Raw"] = weaviate_server.distances.HAMMING
    assert vd.connect("weaviate").get_collection("raw").metric == "cosine"


def test_weaviate_reads_the_single_index_config_of_older_sdks(weaviate_server):
    config = SimpleNamespace(
        vector_index_config=SimpleNamespace(distance_metric="l2-squared")
    )
    weaviate_server.collections.exists.side_effect = lambda name: True
    weaviate_server.collections.get.side_effect = lambda name: SimpleNamespace(
        config=SimpleNamespace(get=lambda: config)
    )
    assert vd.connect("weaviate").get_collection("old").metric == "l2"


# --------------------------------------------------------------------------- #
# Elasticsearch
# --------------------------------------------------------------------------- #


@pytest.fixture
def es_server(monkeypatch):
    """A fake Elasticsearch: indices keep the mappings they were created with."""
    pytest.importorskip("elasticsearch")
    import vd.backends.elasticsearch as mod

    indices = {}
    server = MagicMock()
    server.indices.exists.side_effect = lambda index: index in indices
    server.indices.create.side_effect = lambda index, mappings, **_: indices.update(
        {index: mappings}
    )
    server.indices.get_mapping.side_effect = lambda index: {
        index: {"mappings": indices[index]}
    }
    monkeypatch.setattr(mod, "Elasticsearch", lambda *a, **kw: server)
    server.mappings = indices
    return server


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("dimension", [2, None])
def test_elasticsearch_second_client_reads_the_metric(es_server, metric, dimension):
    vd.connect("elasticsearch").create_collection(
        "probe", dimension=dimension, metric=metric
    )
    assert vd.connect("elasticsearch").get_collection("probe").metric == metric


def test_elasticsearch_second_client_maps_its_first_write_with_the_metric(es_server):
    """The motivating failure: the handle that writes first maps the vector field."""
    vd.connect("elasticsearch").create_collection("probe", metric="l2")
    other = vd.connect("elasticsearch").get_collection("probe")
    other.dimension = 2
    other["x"] = vd.Document(id="x", text="", vector=[1.0, 0.0])
    put = es_server.indices.put_mapping.call_args.kwargs["properties"]
    assert put["embedding"]["similarity"] == "l2_norm"


def test_elasticsearch_index_made_outside_vd_reads_as_cosine(es_server):
    es_server.mappings["raw"] = {"properties": {"text": {"type": "text"}}}
    assert vd.connect("elasticsearch").get_collection("raw").metric == "cosine"


# --------------------------------------------------------------------------- #
# MongoDB
# --------------------------------------------------------------------------- #

_ATLAS_SIMILARITY = {"cosine": "cosine", "dot": "dotProduct", "l2": "euclidean"}


def _atlas_index(similarity, **extra):
    definition = {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 2,
             "similarity": similarity}
        ]
    }
    return {"name": "vector_index", "type": "vectorSearch",
            "latestDefinition": definition, **extra}


@pytest.fixture
def mongo_server(monkeypatch):
    pytest.importorskip("pymongo")
    import vd.backends.mongodb as mod

    server = MagicMock()
    server.__getitem__.return_value.__getitem__.return_value.indexes = []
    coll = server.__getitem__.return_value.__getitem__.return_value
    coll.list_search_indexes.side_effect = lambda *a, **kw: iter(coll.indexes)
    db = server.__getitem__.return_value
    db.list_collection_names.return_value = ["probe"]
    monkeypatch.setattr(mod, "MongoClient", lambda *a, **kw: server)
    server.coll = coll
    return server


@pytest.mark.parametrize("metric", METRICS)
def test_mongodb_second_client_reads_the_metric_from_the_index(mongo_server, metric):
    mongo_server.coll.indexes = [_atlas_index(_ATLAS_SIMILARITY[metric])]
    client = vd.connect("mongodb", uri="mongodb://fake")
    assert client.get_collection("probe").metric == metric


def test_mongodb_reads_the_older_definition_key(mongo_server):
    index = _atlas_index("euclidean")
    index["definition"] = index.pop("latestDefinition")
    mongo_server.coll.indexes = [index]
    assert vd.connect("mongodb", uri="mongodb://fake").get_collection("probe").metric == "l2"


def test_mongodb_ignores_an_index_that_is_going_away(mongo_server):
    mongo_server.coll.indexes = [_atlas_index("euclidean", status="DELETING")]
    assert vd.connect("mongodb", uri="mongodb://fake").get_collection("probe").metric == "cosine"


def test_mongodb_without_an_index_yet_or_search_support_reads_as_cosine(mongo_server):
    client = vd.connect("mongodb", uri="mongodb://fake")
    assert client.get_collection("probe").metric == "cosine"  # no index yet
    mongo_server.coll.list_search_indexes.side_effect = RuntimeError("not Atlas")
    assert client.get_collection("probe").metric == "cosine"
