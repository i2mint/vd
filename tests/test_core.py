"""
Core contract tests for the vd facade.

Most tests here take the parametrized ``client`` fixture, so they run once
per testable backend — that is how the suite proves the *same* facade
contract holds across memory, chroma, faiss, duckdb, lancedb, and qdrant.
"""

import pytest

import vd
from vd import Document


# --------------------------------------------------------------------------- #
# Package surface & data model (backend-independent)
# --------------------------------------------------------------------------- #


def test_package_surface():
    for name in ("connect", "Document", "Client", "Collection", "list_backends"):
        assert hasattr(vd, name), name


def test_list_backends_includes_memory():
    assert "memory" in vd.list_backends()


def test_document_dataclass():
    doc = Document(id="d1", text="hello")
    assert (doc.id, doc.text, doc.vector, doc.metadata) == ("d1", "hello", None, {})
    doc2 = Document(id="d2", text="t", vector=[0.1], metadata={"k": 1})
    assert doc2.vector == [0.1] and doc2.metadata == {"k": 1}


# --------------------------------------------------------------------------- #
# Client behaves as a Mapping[str, Collection]
# --------------------------------------------------------------------------- #


def test_client_is_a_mapping(client):
    assert "absent" not in client
    col = client.create_collection("coll_one")
    assert "coll_one" in client
    assert client["coll_one"] is not None
    assert "coll_one" in list(client)
    assert len(client) >= 1


def test_create_get_delete_collection(client):
    client.create_collection("lifecycle")
    assert "lifecycle" in list(client.list_collections())
    assert client.get_collection("lifecycle") is not None
    client.delete_collection("lifecycle")
    assert "lifecycle" not in list(client.list_collections())


def test_get_missing_collection_raises_keyerror(client):
    with pytest.raises(KeyError):
        client.get_collection("never_created")


def test_create_duplicate_collection_raises_valueerror(client):
    client.create_collection("dup")
    with pytest.raises(ValueError):
        client.create_collection("dup")


def test_delete_missing_collection_raises_keyerror(client):
    with pytest.raises(KeyError):
        client.delete_collection("never_created")


def test_get_or_create_collection(client):
    a = client.get_or_create_collection("goc")
    b = client.get_or_create_collection("goc")  # second call must not raise
    assert a is not None and b is not None
    assert "goc" in list(client.list_collections())


# --------------------------------------------------------------------------- #
# Collection behaves as a MutableMapping[str, Document]
# --------------------------------------------------------------------------- #


def _doc(id_, vec, **meta):
    return Document(id=id_, text=f"text of {id_}", vector=vec, metadata=meta)


def test_collection_crud(client):
    col = client.create_collection("crud")
    col["a"] = _doc("a", [1.0] + [0.0] * 15, kind="x")
    assert len(col) == 1
    got = col["a"]
    assert got.id == "a" and got.text == "text of a"
    assert got.metadata.get("kind") == "x"
    del col["a"]
    assert len(col) == 0
    with pytest.raises(KeyError):
        _ = col["a"]
    with pytest.raises(KeyError):
        del col["a"]


def test_collection_iter_and_contains(client):
    col = client.create_collection("itr")
    for i in range(3):
        v = [0.0] * 16
        v[i] = 1.0
        col[f"k{i}"] = _doc(f"k{i}", v)
    assert set(col) == {"k0", "k1", "k2"}
    assert "k1" in col
    assert "k9" not in col
    assert len(col) == 3


def test_setitem_accepts_text_via_embedder(client):
    # The client was created with an embedder, so raw text is accepted.
    col = client.create_collection("textwrite")
    col["t1"] = "the quick brown fox"
    assert col["t1"].text == "the quick brown fox"
    assert col["t1"].vector is not None


def test_setitem_accepts_tuple_with_metadata(client):
    col = client.create_collection("tuplewrite")
    col["t1"] = ("some text", {"lang": "en"})
    assert col["t1"].metadata.get("lang") == "en"


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


def _seed(col):
    """Three orthogonal-ish docs; returns nothing (mutates col)."""
    col["cat"] = _doc("cat", [1.0] + [0.0] * 15, kind="animal")
    col["dog"] = _doc("dog", [0.0, 1.0] + [0.0] * 14, kind="animal")
    col["pie"] = _doc("pie", [0.0, 0.0, 1.0] + [0.0] * 13, kind="food")


def test_search_ranks_by_similarity(client):
    col = client.create_collection("search_rank")
    _seed(col)
    hits = list(col.search([0.95, 0.05] + [0.0] * 14, limit=3))
    assert hits[0]["id"] == "cat"
    for hit in hits:
        assert set(hit) >= {"id", "text", "score", "metadata"}


def test_search_respects_limit(client):
    col = client.create_collection("search_limit")
    _seed(col)
    assert len(list(col.search([1.0] + [0.0] * 15, limit=2))) == 2


def test_search_with_metadata_filter(client):
    col = client.create_collection("search_filter")
    _seed(col)
    hits = list(col.search([0.3] * 16, limit=10, filter={"kind": "food"}))
    assert [h["id"] for h in hits] == ["pie"]


def test_search_with_comparison_filter(client):
    col = client.create_collection("search_cmp")
    for i in range(6):
        v = [0.0] * 16
        # (i + 1), not i — a zero-magnitude vector has undefined cosine
        # similarity and some backends (Elasticsearch) reject it outright.
        v[0] = (i + 1) / 6
        col[str(i)] = _doc(str(i), v, n=i)
    hits = list(col.search([0.5] * 16, limit=10, filter={"n": {"$gte": 3}}))
    assert sorted(int(h["id"]) for h in hits) == [3, 4, 5]


def test_search_with_in_filter(client):
    col = client.create_collection("search_in")
    for i in range(5):
        v = [0.0] * 16
        # (i + 1), not i — avoid a zero-magnitude vector (see comment above).
        v[0] = (i + 1) / 5
        col[str(i)] = _doc(str(i), v, n=i)
    hits = list(col.search([0.5] * 16, limit=10, filter={"n": {"$in": [1, 3]}}))
    assert sorted(int(h["id"]) for h in hits) == [1, 3]


def test_search_with_egress(client):
    col = client.create_collection("search_egress")
    _seed(col)
    out = list(col.search([1.0] + [0.0] * 15, limit=1, egress=vd.id_only))
    assert out == ["cat"]


def test_search_by_text_query(client):
    # Client has an embedder, so a text query is embedded automatically.
    col = client.create_collection("search_text")
    _seed(col)
    hits = list(col.search("anything", limit=2))
    assert len(hits) == 2


# --------------------------------------------------------------------------- #
# batch & upsert
# --------------------------------------------------------------------------- #


def test_add_documents_batch(client):
    col = client.create_collection("batch")
    docs = []
    for i in range(12):
        v = [0.0] * 16
        v[i % 16] = 1.0
        docs.append(Document(id=f"b{i}", text=f"doc {i}", vector=v))
    col.add_documents(docs, batch_size=5)
    assert len(col) == 12


def test_upsert_is_idempotent(client):
    col = client.create_collection("upsert")
    v = [1.0] + [0.0] * 15
    col.upsert(Document(id="x", text="first", vector=v))
    col.upsert(Document(id="x", text="second", vector=v))
    assert len(col) == 1
    assert col["x"].text == "second"


# --------------------------------------------------------------------------- #
# dimension safety
# --------------------------------------------------------------------------- #


def test_dimension_mismatch_raises(client):
    col = client.create_collection("dimcheck")
    # A non-zero vector — a zero-magnitude one has undefined cosine similarity
    # and some backends (Elasticsearch) reject it outright.
    col["ok"] = Document(id="ok", text="t", vector=[1.0] * 16)
    with pytest.raises(ValueError):
        col["bad"] = Document(id="bad", text="t", vector=[1.0] * 8)
