"""
Tests for the provider registry, the backend recommender, and setup checks
(:mod:`vd.providers` and :mod:`vd.requirements`).
"""

import vd
from vd.providers import (
    install_command,
    is_installed,
    provider,
    provider_names,
    recommend_backend,
)


# --------------------------------------------------------------------------- #
# Provider registry
# --------------------------------------------------------------------------- #


def test_registry_loads_and_covers_the_landscape():
    names = provider_names()
    # The report's recommended six plus the embedded set must all be present.
    for expected in ("chroma", "qdrant", "pgvector", "lancedb", "pinecone",
                     "faiss", "sqlite_vec", "duckdb", "weaviate", "milvus",
                     "redis", "elasticsearch", "mongodb", "turbopuffer"):
        assert expected in names, expected


def test_provider_entries_are_well_formed():
    for name in provider_names():
        meta = provider(name)
        assert meta["name"] == name
        assert meta.get("display_name")
        assert meta.get("archetype") in ("embedded", "server", "managed", "library")
        assert isinstance(meta.get("pip_packages", []), list)
        assert "docs" in meta


def test_unknown_provider_is_none():
    assert provider("not_a_real_db") is None


def test_install_command():
    assert install_command("qdrant") == "pip install qdrant-client"
    assert "no installation" in install_command("memory")


def test_memory_always_installed():
    assert is_installed("memory") is True


def test_list_all_backends_has_flags():
    table = vd.list_all_backends()
    assert "chroma" in table
    assert set(table["chroma"]) >= {"installed", "has_adapter", "archetype"}


# --------------------------------------------------------------------------- #
# recommend_backend — the decision framework
# --------------------------------------------------------------------------- #


def test_recommend_tiny_no_persistence_is_memory():
    rec = recommend_backend(corpus_size="tiny", persistence=False)
    assert rec["primary"] == "memory"
    assert rec["reasoning"]


def test_recommend_existing_postgres_is_pgvector():
    rec = recommend_backend(existing_db="postgres")
    assert rec["primary"] == "pgvector"


def test_recommend_no_docker_is_embedded():
    rec = recommend_backend(can_run_docker=False)
    assert rec["primary"] in ("chroma", "lancedb")


def test_recommend_hybrid_wanted():
    rec = recommend_backend(needs_hybrid=True)
    assert rec["primary"] == "weaviate"


def test_recommend_air_gapped_avoids_managed():
    rec = recommend_backend(air_gapped=True)
    managed = {"pinecone", "turbopuffer", "mongodb"}
    assert rec["primary"] not in managed


def test_recommend_always_returns_full_shape():
    rec = recommend_backend()
    assert set(rec) >= {"primary", "runner_up", "reasoning", "alternatives"}


# --------------------------------------------------------------------------- #
# check_requirements — the setup assistant
# --------------------------------------------------------------------------- #


def test_check_requirements_memory_is_ready():
    report = vd.check_requirements("memory", verbose=False)
    assert report["ok"] is True
    assert "checks" in report


def test_check_requirements_unknown_backend():
    report = vd.check_requirements("not_a_backend", verbose=False)
    assert report["ok"] is False


def test_check_requirements_reports_next_step():
    report = vd.check_requirements("pinecone", verbose=False)
    # Either ready, or the next step is actionable text.
    assert isinstance(report["next_step"], str) and report["next_step"]


def test_setup_guide_is_a_string():
    guide = vd.setup_guide("qdrant")
    assert "qdrant" in guide.lower()
    assert "pip install" in guide
