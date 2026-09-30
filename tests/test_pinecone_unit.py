"""Server-free tests for the pinecone adapter's collection-name mapping."""

import pytest

pytest.importorskip("pinecone")

from vd.backends.pinecone import _collection_name, _index_name  # noqa: E402


@pytest.mark.parametrize("name", ["docs", "my_docs", "my-docs", "a1_b2-c3", "a_-b"])
def test_valid_names_round_trip(name):
    assert _collection_name(_index_name(name)) == name


@pytest.mark.parametrize("name", ["a--b", "a-_b", "Docs", "a.b", "", "x" * 46])
def test_ambiguous_or_invalid_names_are_rejected(name):
    with pytest.raises(ValueError):
        _index_name(name)
