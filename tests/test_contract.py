"""
Tests for the hardened facade contract: capability protocols, typed errors,
the ``supports_incremental_writes`` flag, and the backend escape hatch.
"""

import hashlib
import uuid

import pytest

import vd
from vd import connect
from vd.base import (
    StaticIndexError,
    SupportsBatch,
    SupportsHybrid,
    UnsupportedCapabilityError,
    UnsupportedFilterError,
)


def mock_embedding_function(text: str) -> list[float]:
    """Deterministic 16-dim mock embedding for testing."""
    text_hash = hashlib.md5(text.encode()).digest()
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]


@pytest.fixture
def client():
    return connect("memory", embedding_model=mock_embedding_function)


@pytest.fixture
def collection(client):
    return client.create_collection("contract_test")


# --------------------------------------------------------------------------
# Typed errors are exported and well-formed
# --------------------------------------------------------------------------


def test_errors_exported_from_package():
    assert vd.UnsupportedFilterError is UnsupportedFilterError
    assert vd.UnsupportedCapabilityError is UnsupportedCapabilityError


def test_error_hierarchy():
    # UnsupportedFilterError is a bad-input error.
    assert issubclass(UnsupportedFilterError, ValueError)
    # UnsupportedCapabilityError signals a not-implemented capability.
    assert issubclass(UnsupportedCapabilityError, NotImplementedError)


# --------------------------------------------------------------------------
# Capability protocols
# --------------------------------------------------------------------------


def test_protocols_exported_from_package():
    assert vd.SupportsBatch is SupportsBatch
    assert vd.SupportsHybrid is SupportsHybrid


def test_memory_collection_supports_batch(collection):
    # MemoryCollection has add_documents + upsert, so it satisfies SupportsBatch.
    assert isinstance(collection, SupportsBatch)


def test_memory_collection_does_not_support_hybrid(collection):
    # No hybrid_search method => not SupportsHybrid.
    assert not isinstance(collection, SupportsHybrid)


# --------------------------------------------------------------------------
# supports_incremental_writes flag
# --------------------------------------------------------------------------


def test_supports_incremental_writes_flag(client, collection):
    # The memory backend accepts writes at any time.
    assert client.supports_incremental_writes is True
    assert collection.supports_incremental_writes is True


# --------------------------------------------------------------------------
# Escape hatch: .client / .native
# --------------------------------------------------------------------------


def test_backend_client_escape_hatch(client):
    # The memory backend has no external client.
    assert client.client is None


def test_collection_native_escape_hatch(collection):
    # The memory collection's native handle is the backing dict.
    collection["doc1"] = "hello world"
    native = collection.native
    assert isinstance(native, dict)
    assert "doc1" in native


# --------------------------------------------------------------------------
# ChromaDB backend (skipped if chromadb is not installed)
# --------------------------------------------------------------------------


@pytest.fixture
def chroma_collection():
    # ChromaDB's in-memory client shares state across Client() instances within
    # a process, so use a unique collection name per test to avoid collisions.
    pytest.importorskip("chromadb")
    client = connect("chroma", embedding_model=mock_embedding_function)
    name = f"contract_chroma_{uuid.uuid4().hex[:8]}"
    return client, client.create_collection(name)


def test_chroma_escape_hatch(chroma_collection):
    client, coll = chroma_collection
    assert client.client is not None  # the raw chromadb client
    assert coll.native is not None  # the raw chromadb collection


def test_chroma_supports_batch(chroma_collection):
    _, coll = chroma_collection
    assert isinstance(coll, SupportsBatch)


def test_chroma_rejects_unsupported_filter_operator(chroma_collection):
    """ChromaDB has no $exists; vd must raise a clear UnsupportedFilterError."""
    _, coll = chroma_collection
    coll["doc1"] = ("hello", {"cat": "tech"})
    with pytest.raises(UnsupportedFilterError):
        list(coll.search("hello", filter={"missing": {"$exists": True}}))


def test_chroma_accepts_supported_filter(chroma_collection):
    """A filter within ChromaDB's subset must pass validation and run."""
    _, coll = chroma_collection
    coll["doc1"] = ("hello tech", {"year": 2024})
    results = list(coll.search("hello", filter={"year": {"$gte": 2020}}))
    assert all(r["metadata"]["year"] >= 2020 for r in results)
