"""
Tests for :mod:`vd.analytics` against the in-memory backend.

The analytics functions only need iteration, ``__getitem__`` and ``len``, so
edge cases a real backend refuses to store (vector-less documents, mixed
dimensions) are exercised with a plain ``dict[str, Document]``.
"""

import pytest

import vd
from vd import Document
from vd.analytics import (
    collection_stats,
    find_duplicates,
    find_outliers,
    metadata_distribution,
    sample_collection,
    validate_collection,
)
from vd.util import cosine_similarity


@pytest.fixture
def col():
    """A small memory collection: two near-identical vectors and one opposite."""
    c = vd.connect("memory").create_collection("t", dimension=2)
    c["a"] = Document(id="a", text="hello", vector=[1.0, 0.0], metadata={"cat": "x"})
    c["b"] = Document(
        id="b", text="hello", vector=[1.0, 0.01], metadata={"cat": "x", "tags": ["p", "q"]}
    )
    c["c"] = Document(id="c", text="far away", vector=[-1.0, 0.0], metadata={"cat": "y"})
    return c


# ----- collection_stats ---------------------------------------------------- #


def test_collection_stats_empty():
    stats = collection_stats(vd.connect("memory").create_collection("e"))
    assert stats["total_documents"] == 0
    assert stats["metadata_fields"] == set()
    assert stats["embedding_dimension"] is None
    assert stats["has_vectors"] == 0


def test_collection_stats_counts_text_metadata_and_vectors(col):
    stats = collection_stats(col)
    assert stats["total_documents"] == 3
    assert stats["total_chars"] == 5 + 5 + 8
    assert stats["min_text_length"] == 5
    assert stats["max_text_length"] == 8
    assert stats["avg_text_length"] == pytest.approx(18 / 3)
    assert stats["metadata_fields"] == {"cat", "tags"}
    assert stats["metadata_field_counts"] == {"cat": 3, "tags": 1}
    assert stats["embedding_dimension"] == 2
    assert stats["has_vectors"] == 3


def test_collection_stats_counts_only_documents_with_vectors():
    docs = {
        "a": Document(id="a", text="t", vector=[1.0, 2.0, 3.0]),
        "b": Document(id="b", text="tt"),
    }
    stats = collection_stats(docs)
    assert stats["has_vectors"] == 1
    assert stats["embedding_dimension"] == 3


# ----- metadata_distribution ----------------------------------------------- #


def test_metadata_distribution_counts_values(col):
    assert metadata_distribution(col, "cat") == {"x": 2, "y": 1}


def test_metadata_distribution_expands_list_values_and_skips_missing(col):
    assert metadata_distribution(col, "tags") == {"p": 1, "q": 1}
    assert metadata_distribution(col, "absent") == {}


def test_metadata_distribution_top_n(col):
    assert metadata_distribution(col, "cat", top_n=1) == {"x": 2}


# ----- find_duplicates ----------------------------------------------------- #


def test_find_duplicates_cosine(col):
    dups = find_duplicates(col, threshold=0.99)
    assert [(a, b) for a, b, _ in dups] == [("a", "b")]
    assert dups[0][2] == pytest.approx(cosine_similarity([1.0, 0.0], [1.0, 0.01]))


def test_find_duplicates_threshold_excludes_dissimilar(col):
    assert find_duplicates(col, threshold=0.99999999) == []


def test_find_duplicates_exact_text(col):
    assert find_duplicates(col, method="exact") == [("a", "b", 1.0)]


def test_find_duplicates_cosine_ignores_documents_without_vectors():
    docs = {
        "a": Document(id="a", text="same", vector=[1.0, 0.0]),
        "b": Document(id="b", text="same"),
    }
    assert find_duplicates(docs) == []
    assert find_duplicates(docs, method="exact") == [("a", "b", 1.0)]


# ----- find_outliers ------------------------------------------------------- #


def test_find_outliers_flags_the_document_far_from_its_neighbours(col):
    outliers = find_outliers(col, n_neighbors=1, threshold=0.0)
    assert [doc_id for doc_id, _ in outliers] == ["c"]
    assert outliers[0][1] < 0


def test_find_outliers_sorted_lowest_similarity_first():
    docs = {
        "a": Document(id="a", vector=[1.0, 0.0]),
        "b": Document(id="b", vector=[0.0, 1.0]),
        "c": Document(id="c", vector=[-1.0, 0.0]),
    }
    outliers = find_outliers(docs, n_neighbors=2, threshold=1.0)
    sims = [s for _, s in outliers]
    assert sims == sorted(sims)
    assert {doc_id for doc_id, _ in outliers} == {"a", "b", "c"}


def test_find_outliers_skips_documents_without_vectors_or_neighbours():
    docs = {
        "a": Document(id="a", vector=[1.0, 0.0]),
        "b": Document(id="b", text="no vector"),
    }
    # "b" has no vector; "a" has no neighbour with a vector: nothing to judge.
    assert find_outliers(docs, threshold=1.0) == []


# ----- sample_collection --------------------------------------------------- #


def test_sample_collection_returns_everything_when_n_is_large(col):
    assert sorted(sample_collection(col, 10)) == ["a", "b", "c"]


def test_sample_collection_first(col):
    assert sample_collection(col, 2, method="first") == ["a", "b"]


def test_sample_collection_random_is_reproducible_with_seed(col):
    s1 = sample_collection(col, 2, method="random", seed=7)
    s2 = sample_collection(col, 2, method="random", seed=7)
    assert s1 == s2
    assert len(set(s1)) == 2 and set(s1) <= {"a", "b", "c"}


@pytest.mark.xfail(
    strict=True,
    reason="'diverse' keeps the candidate with the HIGHEST min-similarity, i.e. the most similar one",
)
def test_sample_collection_diverse_picks_the_least_similar_next(col):
    picked = sample_collection(col, 2, method="diverse", seed=3)
    first, second = picked
    others = [i for i in ("a", "b", "c") if i != first]
    expected = min(others, key=lambda i: cosine_similarity(col[first].vector, col[i].vector))
    assert second == expected


def test_sample_collection_diverse_returns_n_distinct_ids(col):
    picked = sample_collection(col, 2, method="diverse", seed=3)
    assert len(picked) == 2 and len(set(picked)) == 2
    assert set(picked) <= {"a", "b", "c"}


def test_sample_collection_diverse_falls_back_when_no_vectors():
    docs = {k: Document(id=k, text=k) for k in ("a", "b", "c")}
    picked = sample_collection(docs, 2, method="diverse", seed=1)
    assert len(picked) == 2 and len(set(picked)) == 2


def test_sample_collection_unknown_method_raises(col):
    with pytest.raises(ValueError, match="Unknown sampling method"):
        sample_collection(col, 1, method="nope")


# ----- validate_collection ------------------------------------------------- #


def test_validate_collection_healthy(col):
    report = validate_collection(col)
    assert report["valid"] is True
    assert report["issues"] == []
    assert report["warnings"] == []
    assert report["stats"]["total_documents"] == 3


def test_validate_collection_empty_warns():
    report = validate_collection(vd.connect("memory").create_collection("e"))
    assert report["valid"] is True
    assert report["warnings"] == ["Collection is empty"]


def test_validate_collection_reports_empty_text_missing_vectors_and_dimensions():
    docs = {
        "a": Document(id="a", text="ok", vector=[1.0, 0.0]),
        "b": Document(id="b", text="   ", vector=[1.0, 0.0, 0.0]),
        "c": Document(id="c", text="no vector"),
    }
    report = validate_collection(docs)
    assert report["valid"] is False
    assert report["stats"] == {
        "total_documents": 3,
        "missing_vectors": 1,
        "empty_text": 1,
        "inconsistent_dimensions": 1,
    }
    assert "Document b has empty text" in report["warnings"]
    assert "1 documents missing vectors" in report["warnings"]
    assert any("Inconsistent vector dimensions" in i for i in report["issues"])


def test_validate_collection_records_unreadable_documents():
    class Broken(dict):
        def __getitem__(self, key):
            raise RuntimeError("disk on fire")

    report = validate_collection(Broken(a=None))
    assert report["valid"] is False
    assert report["issues"] == ["Error reading document a: disk on fire"]
