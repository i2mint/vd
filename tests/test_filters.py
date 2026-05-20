"""
Tests for the canonical metadata-filter language (``vd.filters``).

Covers the in-Python evaluator (:func:`vd.filters.matches_filter`), the
fail-loud validator (:func:`vd.filters.validate_filter`), every operator, and
end-to-end filtering through the memory backend's ``search``.
"""

import hashlib

import pytest

import vd
from vd import connect
from vd.base import UnsupportedFilterError
from vd.filters import (
    SUPPORTED_FILTER_OPERATORS,
    matches_filter,
    validate_filter,
)


def mock_embedding_function(text: str) -> list[float]:
    """Deterministic 16-dim mock embedding for testing."""
    text_hash = hashlib.md5(text.encode()).digest()
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]


# --------------------------------------------------------------------------
# matches_filter — operator coverage
# --------------------------------------------------------------------------

META = {"year": 2022, "views": 500, "cat": "tech", "tags": ["python", "ai"]}


def test_empty_filter_matches_everything():
    assert matches_filter(META, None) is True
    assert matches_filter(META, {}) is True


def test_bare_equality_and_implicit_and():
    assert matches_filter(META, {"cat": "tech"}) is True
    assert matches_filter(META, {"cat": "news"}) is False
    # Multiple top-level fields combine with implicit AND.
    assert matches_filter(META, {"cat": "tech", "year": 2022}) is True
    assert matches_filter(META, {"cat": "tech", "year": 1999}) is False


def test_eq_ne():
    assert matches_filter(META, {"year": {"$eq": 2022}}) is True
    assert matches_filter(META, {"year": {"$ne": 2022}}) is False
    assert matches_filter(META, {"year": {"$ne": 1999}}) is True


def test_ordered_comparisons():
    assert matches_filter(META, {"year": {"$gte": 2022}}) is True
    assert matches_filter(META, {"year": {"$gt": 2022}}) is False
    assert matches_filter(META, {"views": {"$lt": 1000}}) is True
    assert matches_filter(META, {"views": {"$lte": 500}}) is True
    # Two operators on one field => implicit AND.
    assert matches_filter(META, {"views": {"$gte": 100, "$lte": 1000}}) is True
    assert matches_filter(META, {"views": {"$gte": 100, "$lte": 200}}) is False


def test_in_nin_scalar_and_list_valued():
    # Scalar field.
    assert matches_filter(META, {"cat": {"$in": ["tech", "news"]}}) is True
    assert matches_filter(META, {"cat": {"$nin": ["news"]}}) is True
    assert matches_filter(META, {"cat": {"$nin": ["tech"]}}) is False
    # List-valued field matches on overlap.
    assert matches_filter(META, {"tags": {"$in": ["ai"]}}) is True
    assert matches_filter(META, {"tags": {"$in": ["rust"]}}) is False
    assert matches_filter(META, {"tags": {"$nin": ["rust"]}}) is True


def test_exists():
    assert matches_filter(META, {"year": {"$exists": True}}) is True
    assert matches_filter(META, {"missing": {"$exists": False}}) is True
    assert matches_filter(META, {"missing": {"$exists": True}}) is False
    assert matches_filter(META, {"year": {"$exists": False}}) is False


def test_logical_and_or_not():
    assert matches_filter(META, {"$and": [{"cat": "tech"}, {"year": 2022}]}) is True
    assert matches_filter(META, {"$and": [{"cat": "tech"}, {"year": 1999}]}) is False
    assert matches_filter(META, {"$or": [{"year": 1999}, {"year": 2022}]}) is True
    assert matches_filter(META, {"$or": [{"year": 1999}, {"year": 2000}]}) is False
    assert matches_filter(META, {"$not": {"cat": "news"}}) is True
    assert matches_filter(META, {"$not": {"cat": "tech"}}) is False


def test_nested_logical():
    flt = {
        "$and": [
            {"$or": [{"cat": "tech"}, {"cat": "science"}]},
            {"year": {"$gte": 2020}},
            {"$not": {"views": {"$lt": 100}}},
        ]
    }
    assert matches_filter(META, flt) is True


def test_missing_field_semantics():
    # A missing field: $eq fails, $ne / $nin succeed, comparisons fail.
    assert matches_filter(META, {"missing": {"$eq": 1}}) is False
    assert matches_filter(META, {"missing": {"$ne": 1}}) is True
    assert matches_filter(META, {"missing": {"$gte": 1}}) is False
    assert matches_filter(META, {"missing": {"$nin": [1, 2]}}) is True
    # Bare-value equality against a missing field never matches.
    assert matches_filter(META, {"missing": 1}) is False


def test_incomparable_types_do_not_match():
    # Comparing a str field with $gte against an int must not raise.
    assert matches_filter(META, {"cat": {"$gte": 5}}) is False


# --------------------------------------------------------------------------
# Fail-loud on unknown operators (the core bug fix)
# --------------------------------------------------------------------------


def test_unknown_field_operator_raises():
    with pytest.raises(UnsupportedFilterError, match=r"\$bogus"):
        matches_filter(META, {"year": {"$bogus": 1}})


def test_unknown_logical_operator_raises():
    with pytest.raises(UnsupportedFilterError, match=r"\$nor"):
        matches_filter(META, {"$nor": [{"year": 2022}]})


def test_supported_operators_constant_is_complete():
    expected = {
        "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists",
        "$and", "$or", "$not",
    }
    assert SUPPORTED_FILTER_OPERATORS == expected


# --------------------------------------------------------------------------
# validate_filter
# --------------------------------------------------------------------------


def test_validate_filter_accepts_valid():
    validate_filter({"year": {"$gte": 2020}, "$or": [{"cat": "tech"}]})
    validate_filter(None)
    validate_filter({})


def test_validate_filter_rejects_unknown_operator():
    with pytest.raises(UnsupportedFilterError, match=r"\$regex"):
        validate_filter({"a": {"$regex": ".*"}})


def test_validate_filter_rejects_operator_outside_supported_subset():
    # $exists is a valid operator, but not in this backend's subset.
    with pytest.raises(UnsupportedFilterError, match="not supported"):
        validate_filter({"a": {"$exists": True}}, supported={"$eq", "$ne"})


def test_validate_filter_recurses_into_logical():
    with pytest.raises(UnsupportedFilterError, match=r"\$regex"):
        validate_filter({"$and": [{"year": 2022}, {"a": {"$regex": "x"}}]})


# --------------------------------------------------------------------------
# End-to-end through the memory backend's search
# --------------------------------------------------------------------------


@pytest.fixture
def collection():
    client = connect("memory", embedding_model=mock_embedding_function)
    coll = client.create_collection("filter_test")
    coll["doc1"] = ("Article 1", {"year": 2020, "views": 100, "tags": ["python"]})
    coll["doc2"] = ("Article 2", {"year": 2021, "views": 500, "tags": ["ai"]})
    coll["doc3"] = ("Article 3", {"year": 2022, "views": 1000})  # no 'tags'
    return coll


def test_search_with_nin(collection):
    results = list(collection.search("article", filter={"year": {"$nin": [2020]}}))
    assert {r["id"] for r in results} == {"doc2", "doc3"}


def test_search_with_not(collection):
    results = list(collection.search("article", filter={"$not": {"year": 2020}}))
    assert {r["id"] for r in results} == {"doc2", "doc3"}


def test_search_with_exists(collection):
    has_tags = list(collection.search("article", filter={"tags": {"$exists": True}}))
    assert {r["id"] for r in has_tags} == {"doc1", "doc2"}
    no_tags = list(collection.search("article", filter={"tags": {"$exists": False}}))
    assert {r["id"] for r in no_tags} == {"doc3"}


def test_search_unknown_operator_raises(collection):
    with pytest.raises(UnsupportedFilterError):
        list(collection.search("article", filter={"year": {"$typo": 2020}}))
