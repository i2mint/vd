"""
Tests for :mod:`vd.health`: backend and collection health, and the benchmarks.

Benchmarks run with a handful of documents and queries; their timings are
asserted for shape and internal consistency, never for speed.
"""

from types import SimpleNamespace

import pytest

import vd
from vd import Document
from vd.health import (
    benchmark_insert,
    benchmark_search,
    health_check_backend,
    health_check_collection,
)


@pytest.fixture
def col(embedder):
    c = vd.connect("memory", embedder=embedder).create_collection("t")
    for i, text in enumerate(["alpha", "beta", "gamma"]):
        c[f"d{i}"] = text
    return c


# ----- health_check_backend ------------------------------------------------ #


def test_health_check_memory_is_healthy():
    status = health_check_backend("memory")
    assert status["backend"] == "memory"
    assert status["status"] == "healthy"
    assert status["available"] is True
    assert status["registered"] is True
    assert status["message"] == "Backend is operational"
    assert status["details"] == {"collections_count": 0}


def test_health_check_unknown_backend_is_unavailable():
    status = health_check_backend("no_such_backend")
    assert status["status"] == "unavailable"
    assert status["registered"] is False
    assert status["available"] is False
    assert status["message"] == "Backend 'no_such_backend' is not installed"


def test_health_check_known_but_missing_backend_says_how_to_install(monkeypatch):
    monkeypatch.setattr(vd.health, "is_installed", lambda name: False)
    status = health_check_backend("qdrant")
    assert status["status"] == "unavailable"
    assert status["registered"] is True
    assert "Install with: " in status["message"]
    assert vd.health.install_command("qdrant") in status["message"]


def test_health_check_connection_failure_is_unhealthy(monkeypatch):
    def refuse(*args, **kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(vd, "connect", refuse)
    status = health_check_backend("memory")
    assert status["status"] == "unhealthy"
    assert status["message"] == "Failed to connect: refused"


def test_health_check_operation_failure_is_unhealthy(monkeypatch):
    def broken_listing():
        raise RuntimeError("no listing")

    monkeypatch.setattr(
        vd, "connect", lambda *a, **k: SimpleNamespace(list_collections=broken_listing)
    )
    status = health_check_backend("memory")
    assert status["status"] == "unhealthy"
    assert status["message"] == "Backend connected but operations failed: no listing"


def test_health_check_passes_config_and_an_embedder_to_connect(monkeypatch):
    seen = {}

    def fake_connect(backend, **kwargs):
        seen.update(kwargs, backend=backend)
        return SimpleNamespace(list_collections=lambda: iter(["x", "y"]))

    monkeypatch.setattr(vd, "connect", fake_connect)
    status = health_check_backend("memory", path="/somewhere")
    assert status["details"] == {"collections_count": 2}
    assert seen["backend"] == "memory" and seen["path"] == "/somewhere"
    assert len(seen["embedder"]("any text")) == 16


# ----- health_check_collection --------------------------------------------- #


def test_health_check_collection_healthy(col):
    report = health_check_collection(col)
    assert report["status"] == "healthy"
    assert report["valid"] is True
    assert report["total_documents"] == 3
    assert report["has_vectors"] == 3
    assert report["issues"] == []
    assert report["stats"]["embedding_dimension"] == 16


def test_health_check_collection_with_issues():
    docs = {
        "a": Document(id="a", text="x", vector=[1.0, 0.0]),
        "b": Document(id="b", text="", vector=[1.0]),
    }
    report = health_check_collection(docs)
    assert report["status"] == "issues_found"
    assert report["valid"] is False
    assert report["issues"]
    assert "Document b has empty text" in report["warnings"]


# ----- benchmarks ---------------------------------------------------------- #


def test_benchmark_search(col):
    results = benchmark_search(col, "alpha", n_queries=5, limit=2)
    assert results["n_queries"] == 5
    assert results["min_latency"] <= results["p50"] <= results["p95"] <= results["p99"]
    assert results["p99"] <= results["max_latency"]
    assert results["min_latency"] <= results["avg_latency"] <= results["max_latency"]
    assert results["queries_per_second"] == pytest.approx(5 / results["total_time"])


def test_benchmark_search_accepts_a_query_vector(col, embedder):
    assert benchmark_search(col, embedder("beta"), n_queries=2)["n_queries"] == 2


def test_benchmark_insert_writes_the_documents(embedder):
    col = vd.connect("memory", embedder=embedder).create_collection("b")
    calls = []
    original = col.add_documents
    col.add_documents = lambda docs, **kw: (calls.append(len(docs)), original(docs, **kw))
    results = benchmark_insert(col, 5, text_length=30, batch_size=2)
    assert calls == [2, 2, 1]
    assert len(col) == 5
    assert col["bench_doc_3"].metadata == {"benchmark": True, "index": 3}
    assert len(col["bench_doc_3"].text) <= 30
    assert results["n_documents"] == 5
    assert results["batch_size"] == 2
    assert results["avg_time_per_doc"] == pytest.approx(results["total_time"] / 5)


@pytest.mark.parametrize("which", ["search", "insert"])
def test_benchmarks_survive_a_zero_elapsed_time(col, monkeypatch, which):
    """A run inside the clock's resolution (common on Windows) used to divide by zero."""
    monkeypatch.setattr(vd.health, "time", SimpleNamespace(perf_counter=lambda: 0.0))
    if which == "search":
        results = benchmark_search(col, "alpha", n_queries=1)
    else:
        results = benchmark_insert(col, 1)
    assert results["total_time"] == 0.0
