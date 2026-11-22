"""BM25Index's engines (i2mint/vd#47): the optional bm25s engine matches the built-in."""

import random
from importlib.util import find_spec

import pytest

import vd
from vd.search import BM25_ENGINES, BM25Index


class _Doc:
    def __init__(self, text, metadata=None):
        self.text, self.metadata = text, metadata or {}


def _corpus(n=400, vocab=300, seed=7):
    rnd = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]  # Zipf-ish, so common words exist
    return {
        f"d{i}": _Doc(" ".join(rnd.choices(words, weights, k=rnd.randint(5, 40))))
        for i in range(n)
    }


def test_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown BM25 engine"):
        BM25Index({}, engine="nope")


def test_search_defaults_to_the_index_hyperparameters():
    docs = {"a": _Doc("quick brown fox"), "b": _Doc("lazy dog")}
    custom = BM25Index(docs, k1=1.2, b=0.5)
    assert custom.search("fox") == custom.search("fox", k1=1.2, b=0.5)
    assert custom.search("fox") != custom.search("fox", k1=1.5, b=0.75)


def test_a_missing_bm25s_says_how_to_get_it(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "bm25s", None)
    with pytest.raises(ImportError, match=r"vd\[bm25s\]"):
        BM25Index({"a": _Doc("x y")}, engine="bm25s")


needs_bm25s = pytest.mark.skipif(
    find_spec("bm25s") is None, reason="bm25s not installed (vd[bm25s])"
)


@needs_bm25s
@pytest.mark.parametrize("query", ["w0 w1", "w3 w50 w200", "w0 w0 w7", "w299 w298", "zzz"])
def test_bm25s_gives_the_same_ranking_and_scores(query):
    docs = _corpus()
    py, fast = BM25Index(docs), BM25Index(docs, engine="bm25s")
    a, b = py.search(query, limit=25), fast.search(query, limit=25)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    for x, y in zip(a, b):
        assert y["score"] == pytest.approx(x["score"], rel=1e-5)
        assert (y["text"], y["metadata"]) == (x["text"], x["metadata"])


@needs_bm25s
def test_bm25s_keeps_ties_in_document_order():
    docs = {k: _Doc("same words here") for k in ("c", "a", "b")}
    assert [r["id"] for r in BM25Index(docs, engine="bm25s").search("words")] == [
        "c", "a", "b"
    ]


@needs_bm25s
def test_bm25s_takes_k1_and_b_per_query_like_the_python_engine():
    docs = _corpus(n=120)
    py, fast = BM25Index(docs), BM25Index(docs, engine="bm25s")
    a = py.search("w0 w9", limit=20, k1=1.2, b=0.5)
    c = fast.search("w0 w9", limit=20, k1=1.2, b=0.5)
    assert [r["id"] for r in a] == [r["id"] for r in c]
    assert [r["score"] for r in c] == pytest.approx([r["score"] for r in a], rel=1e-5)


@needs_bm25s
@pytest.mark.parametrize("query", ["fox  dog", " "])
def test_bm25s_survives_a_tokenizer_that_emits_empty_tokens(query):
    docs = {"a": _Doc("fox dog"), "b": _Doc("cat")}
    split = lambda s: s.split(" ")  # noqa: E731
    py = BM25Index(docs, tokenize=split)
    fast = BM25Index(docs, tokenize=split, engine="bm25s")
    assert [r["id"] for r in fast.search(query)] == [r["id"] for r in py.search(query)]


@needs_bm25s
def test_a_bm25s_index_pickles():
    import pickle

    index = BM25Index(_corpus(n=50), engine="bm25s")
    again = pickle.loads(pickle.dumps(index))
    assert again.search("w0 w1") == index.search("w0 w1")


def test_a_negative_limit_is_refused():
    with pytest.raises(ValueError, match="limit"):
        BM25Index({"a": _Doc("x")}).search("x", limit=-1)


@needs_bm25s
def test_bm25s_respects_the_build_filter():
    c = vd.connect("memory").create_collection("t", dimension=2)
    c["a"] = vd.Document(id="a", text="red fox", vector=[1.0, 0.0], metadata={"k": 1})
    c["b"] = vd.Document(id="b", text="red dog", vector=[0.0, 1.0], metadata={"k": 2})
    index = BM25Index(c, filter={"k": 2}, engine="bm25s")
    assert [r["id"] for r in index.search("red")] == ["b"]
    assert set(BM25_ENGINES) == {"python", "bm25s"}
