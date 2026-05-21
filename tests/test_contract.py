"""
Tests for the facade's contract details: typed errors, capability protocols,
escape hatches, the static-index flag, and vector-only (no-embedder) mode.
"""

import pytest

import vd
from vd import (
    BackendNotInstalledError,
    Document,
    EmbeddingRequiredError,
    StaticIndexError,
    SupportsBatch,
    SupportsHybrid,
    UnsupportedCapabilityError,
    UnsupportedFilterError,
    VdError,
)


# --------------------------------------------------------------------------- #
# Typed errors
# --------------------------------------------------------------------------- #


def test_errors_exported_and_well_formed():
    assert vd.UnsupportedFilterError is UnsupportedFilterError
    assert issubclass(UnsupportedFilterError, ValueError)
    assert issubclass(UnsupportedCapabilityError, NotImplementedError)
    assert issubclass(EmbeddingRequiredError, RuntimeError)
    assert issubclass(BackendNotInstalledError, ImportError)
    # Every vd error shares the VdError base.
    for err in (
        StaticIndexError,
        UnsupportedFilterError,
        UnsupportedCapabilityError,
        EmbeddingRequiredError,
        BackendNotInstalledError,
    ):
        assert issubclass(err, VdError)


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError):
        vd.connect("no_such_backend")


def test_known_uninstalled_backend_raises_backend_not_installed():
    # 'vespa' is in the provider registry but ships no adapter / client here.
    if "vespa" in vd.list_backends():
        pytest.skip("vespa adapter unexpectedly available")
    with pytest.raises(BackendNotInstalledError):
        vd.connect("vespa")


# --------------------------------------------------------------------------- #
# Embedding is external: text without an embedder fails loudly
# --------------------------------------------------------------------------- #


def test_text_without_embedder_raises():
    client = vd.connect("memory")  # no embedder
    col = client.create_collection("noembed")
    with pytest.raises(EmbeddingRequiredError):
        col["x"] = "raw text, but nothing can embed it"
    with pytest.raises(EmbeddingRequiredError):
        list(col.search("a text query"))


def test_vector_only_mode_works_without_embedder():
    client = vd.connect("memory")  # no embedder
    col = client.create_collection("vectoronly")
    col["a"] = Document(id="a", text="cat", vector=[1.0, 0.0])
    col["b"] = Document(id="b", text="dog", vector=[0.0, 1.0])
    hits = list(col.search([0.9, 0.1], limit=1))  # pre-computed query vector
    assert hits[0]["id"] == "a"


# --------------------------------------------------------------------------- #
# Capability protocols
# --------------------------------------------------------------------------- #


def test_capability_protocols_exported():
    assert vd.SupportsBatch is SupportsBatch
    assert vd.SupportsHybrid is SupportsHybrid


def test_memory_collection_supports_batch():
    col = vd.connect("memory").create_collection("c")
    assert isinstance(col, SupportsBatch)


def test_memory_collection_is_not_hybrid():
    col = vd.connect("memory").create_collection("c")
    assert not isinstance(col, SupportsHybrid)


# --------------------------------------------------------------------------- #
# Escape hatches & flags
# --------------------------------------------------------------------------- #


def test_memory_escape_hatches():
    client = vd.connect("memory")
    assert client.client is None  # memory has no external client
    col = client.create_collection("c")
    col["d"] = Document(id="d", text="t", vector=[1.0])
    assert isinstance(col.native, dict) and "d" in col.native


def test_supports_incremental_writes_flag():
    col = vd.connect("memory").create_collection("c")
    assert col.supports_incremental_writes is True


# --------------------------------------------------------------------------- #
# Filter validation against a backend's documented subset
# --------------------------------------------------------------------------- #


def test_chroma_rejects_unsupported_operator():
    pytest.importorskip("chromadb")
    col = vd.connect("chroma").create_collection("contract_chroma")
    col["d1"] = Document(id="d1", text="hi", vector=[0.1, 0.2, 0.3], metadata={"y": 1})
    # Chroma's filter subset has no $exists -> a clear vd error, pre-query.
    with pytest.raises(UnsupportedFilterError):
        list(col.search([0.1, 0.2, 0.3], filter={"missing": {"$exists": True}}))
