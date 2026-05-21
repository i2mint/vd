"""
The provider registry: choosing and learning about vector databases.

Where :mod:`vd.util` holds the *runtime* registry (name -> adapter class),
this module holds the *descriptive* registry: for every vector database ``vd``
knows about — whether or not an adapter is installed — its deployment
archetype, license, pip package, docs URLs, setup notes, and the structural
facts a developer needs to *choose* one.

The data lives in ``vd/data/providers.yaml`` (lifted from the report
"VectorDB Selection & Setup Guide"). This module loads it and exposes:

- :func:`providers`, :func:`provider` — raw lookup;
- :func:`recommend_backend` — a decision-framework recommender (report §4);
- :func:`compare_backends`, :func:`print_comparison` — side-by-side tables;
- :func:`list_all_backends`, :func:`print_backends_table` — inventory;
- :func:`get_install_instructions`, :func:`install_command` — setup pointers.

The companion :mod:`vd.requirements` turns this metadata into actionable
``check_requirements(backend)`` diagnostics.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_DATA_FILE = Path(__file__).parent / "data" / "providers.yaml"

#: pip-package name -> import name, for the cases where they differ.
_IMPORT_NAMES = {
    "faiss-cpu": "faiss",
    "qdrant-client": "qdrant_client",
    "sqlite-vec": "sqlite_vec",
    "weaviate-client": "weaviate",
    "opensearch-py": "opensearchpy",
    "google-cloud-aiplatform": "google.cloud.aiplatform",
    "azure-search-documents": "azure.search.documents",
    "vald-client-python": "vald",
}


@lru_cache(maxsize=1)
def _registry() -> dict[str, dict[str, Any]]:
    """Load and cache the provider registry as ``{name: entry}``."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Reading the vd provider registry needs PyYAML. "
            "Install it with: pip install pyyaml"
        ) from e
    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {entry["name"]: entry for entry in raw.get("providers", [])}


def registry_as_of() -> str:
    """Return the ``as_of`` date stamp of the provider registry snapshot."""
    try:
        import yaml

        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            return str(yaml.safe_load(f).get("as_of", "unknown"))
    except Exception:  # pragma: no cover
        return "unknown"


# --------------------------------------------------------------------------- #
# Raw lookup
# --------------------------------------------------------------------------- #


def providers() -> dict[str, dict[str, Any]]:
    """Return the full provider registry as ``{name: metadata}``."""
    return dict(_registry())


def provider(name: str) -> Optional[dict[str, Any]]:
    """Return one provider's metadata, or ``None`` if ``name`` is unknown."""
    return _registry().get(name)


def provider_names() -> list[str]:
    """Return every provider name ``vd`` knows about, sorted."""
    return sorted(_registry())


def _import_name(pip_package: str) -> str:
    """Map a pip-package name to the module name used to import it."""
    base = pip_package.split("[", 1)[0]  # drop extras: "psycopg[binary]" -> "psycopg"
    return _IMPORT_NAMES.get(base, base.replace("-", "_"))


def is_installed(name: str) -> bool:
    """
    Return whether a provider's client library is importable right now.

    A provider with no pip packages (``memory``) is always installed.
    """
    meta = provider(name)
    if meta is None:
        return False
    packages = meta.get("pip_packages") or []
    if not packages:
        return True
    # The first package is the primary client library. find_spec can raise
    # ModuleNotFoundError for a dotted name whose parent package is absent.
    try:
        return importlib.util.find_spec(_import_name(packages[0])) is not None
    except (ImportError, ValueError):
        return False


def has_adapter(name: str) -> bool:
    """Return whether ``vd`` ships a facade adapter for this provider."""
    meta = provider(name)
    return bool(meta and meta.get("adapter"))


def install_command(name: str) -> str:
    """
    Return the ``pip install`` command that makes ``name`` usable.

    Examples
    --------
    >>> install_command('qdrant')
    'pip install qdrant-client'
    >>> install_command('memory')
    'memory needs no installation (built into vd)'
    """
    meta = provider(name)
    if meta is None:
        raise ValueError(f"Unknown provider {name!r}")
    packages = meta.get("pip_packages") or []
    if not packages:
        return f"{name} needs no installation (built into vd)"
    return "pip install " + " ".join(packages)


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #


def list_all_backends() -> dict[str, dict[str, Any]]:
    """
    Return every provider with live ``installed`` / ``has_adapter`` flags added.
    """
    out = {}
    for name, meta in _registry().items():
        info = dict(meta)
        info["installed"] = is_installed(name)
        info["has_adapter"] = has_adapter(name)
        out[name] = info
    return out


def list_available_backends() -> list[str]:
    """
    Return providers ``vd`` can :func:`~vd.connect` *right now*.

    A backend is available iff its adapter module imported successfully — which
    happens only when its client library is installed. This is exactly the set
    of registered backends.
    """
    from vd.util import list_backends

    return list_backends()


def get_backend_info(name: str) -> dict[str, Any]:
    """Return one provider's metadata with ``installed``/``has_adapter`` flags."""
    meta = provider(name)
    if meta is None:
        raise ValueError(
            f"Unknown provider {name!r}. Known: {', '.join(provider_names())}"
        )
    info = dict(meta)
    info["installed"] = is_installed(name)
    info["has_adapter"] = has_adapter(name)
    return info


def get_backend_characteristics() -> dict[str, dict[str, Any]]:
    """Return a compact ``{name: characteristics}`` map for comparison tooling."""
    keys = (
        "display_name",
        "archetype",
        "license",
        "embedded_mode",
        "requires_server",
        "managed_free_tier",
        "hybrid_search",
        "filter_dialect",
        "scale",
        "sweet_spot",
    )
    return {name: {k: meta.get(k) for k in keys} for name, meta in _registry().items()}


def get_install_instructions(name: str) -> str:
    """Return a human-readable setup blurb for one provider."""
    info = get_backend_info(name)
    lines = [f"{info['display_name']}  ({name})", "=" * 60]
    if info["installed"]:
        lines.append("Status: client library is installed.")
    else:
        lines.append("Status: NOT installed.")
        lines.append(f"  {install_command(name)}")
    if not info["has_adapter"]:
        lines.append("Note: vd has no facade adapter for this provider yet —")
        lines.append("      it appears here for backend-selection guidance only.")
    lines += [
        "",
        f"Archetype     : {info.get('archetype')}",
        f"License       : {info.get('license')}",
        f"Hybrid search : {info.get('hybrid_search')}",
        f"Sweet spot    : {info.get('sweet_spot')}",
    ]
    docs = info.get("docs") or {}
    if docs.get("install"):
        lines.append(f"Install docs  : {docs['install']}")
    if docs.get("pricing"):
        lines.append(f"Pricing       : {docs['pricing']}  (verify — prices drift)")
    env_vars = info.get("env_vars") or []
    if env_vars:
        lines.append(f"Env vars      : {', '.join(env_vars)}")
    if info.get("notes"):
        lines += ["", f"Notes: {info['notes']}"]
    return "\n".join(lines)


def print_backends_table() -> None:
    """Print every known vector database, grouped by deployment archetype."""
    all_backends = list_all_backends()
    print(f"\nVector databases vd knows about  (registry as of {registry_as_of()})")
    print("=" * 78)
    for archetype in ("embedded", "library", "server", "managed"):
        group = {
            n: m for n, m in all_backends.items() if m.get("archetype") == archetype
        }
        if not group:
            continue
        print(f"\n{archetype.upper()}")
        print("-" * 78)
        for name, m in sorted(group.items()):
            flags = []
            if m["has_adapter"]:
                flags.append("adapter" if m["installed"] else "adapter(not installed)")
            else:
                flags.append("registry-only")
            print(f"  {name:16} {m['display_name']:30} [{', '.join(flags)}]")
            print(f"  {'':16} {m.get('sweet_spot', '')}")
    print("\n" + "=" * 78)
    print("Legend: 'adapter' = vd.connect() works · 'registry-only' = listed for")
    print("        backend-selection guidance, no facade adapter yet.\n")


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare_backends(
    names: list[str],
    *,
    characteristics: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """
    Return a ``{name: {characteristic: value}}`` table for the given providers.
    """
    chars = characteristics or [
        "display_name",
        "archetype",
        "license",
        "embedded_mode",
        "managed_free_tier",
        "hybrid_search",
        "filter_dialect",
        "scale",
    ]
    out = {}
    for name in names:
        meta = provider(name)
        if meta is None:
            raise ValueError(f"Unknown provider {name!r}")
        out[name] = {c: meta.get(c) for c in chars}
    return out


def print_comparison(names: list[str]) -> None:
    """Print a side-by-side comparison table of the given providers."""
    table = compare_backends(names)
    chars = next(iter(table.values())).keys()
    col_w = 22
    header = f"{'characteristic':<18}" + "".join(f"{n:<{col_w}}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for char in chars:
        row = f"{char:<18}"
        for name in names:
            row += f"{str(table[name][char])[: col_w - 1]:<{col_w}}"
        print(row)
    print()


# --------------------------------------------------------------------------- #
# Recommendation — the decision framework (report §4)
# --------------------------------------------------------------------------- #

# Sizes, smallest to largest. Index used for ordered comparisons.
_SIZES = ("tiny", "small", "medium", "large", "huge")


def recommend_backend(
    *,
    corpus_size: str = "medium",
    persistence: bool = True,
    can_run_docker: bool = True,
    cloud_ok: bool = True,
    budget: str = "free",
    existing_db: Optional[str] = None,
    needs_hybrid: bool = False,
    air_gapped: bool = False,
) -> dict[str, Any]:
    """
    Recommend a vector database from a few yes/no facts about the situation.

    A direct encoding of the decision framework in the report's §4. Returns a
    primary pick, a runner-up, and the reasoning trail.

    Parameters
    ----------
    corpus_size : {'tiny', 'small', 'medium', 'large', 'huge'}
        Rough vector count: tiny <100k, small <10M, medium ~10M, large <100M,
        huge >100M.
    persistence : bool
        Must data survive a process restart?
    can_run_docker : bool
        Can the user run Docker / operate a server process?
    cloud_ok : bool
        Is a managed cloud service acceptable (vs. on-prem only)?
    budget : {'free', 'paid'}
        Free-tier-only, or is paid acceptable?
    existing_db : {'postgres', 'redis', 'elastic', 'mongo', 'sqlite', 'duckdb', None}
        A database the user already operates — strongly biases the pick.
    needs_hybrid : bool
        Need keyword + vector ranking fused in one query?
    air_gapped : bool
        Must run with zero network / zero telemetry?

    Returns
    -------
    dict
        ``{"primary", "runner_up", "reasoning", "alternatives"}``.

    Examples
    --------
    >>> rec = recommend_backend(corpus_size='tiny', persistence=False)
    >>> rec['primary']
    'memory'
    >>> rec = recommend_backend(existing_db='postgres')
    >>> rec['primary']
    'pgvector'
    """
    reasoning: list[str] = []

    # Q1/Q2 — do you even need a vector DB?
    if corpus_size == "tiny" and not persistence:
        reasoning.append(
            "Tiny corpus (<100k) with no persistence: a real vector DB is "
            "overkill — brute force in memory is the right answer."
        )
        return _result("memory", "faiss", reasoning, ["chroma"])

    # Q6 — already running a database? Don't add a second one.
    if existing_db:
        mapping = {
            "postgres": (
                "pgvector",
                "vectorchord",
                "Postgres already runs — "
                "pgvector adds vectors with no second database.",
            ),
            "redis": (
                "redis",
                "qdrant",
                "Redis already runs — use its native "
                "vector index (mind the AGPL/RSAL license change).",
            ),
            "elastic": (
                "elasticsearch",
                "opensearch",
                "Elastic already runs — use kNN + RRF in place.",
            ),
            "mongo": (
                "mongodb",
                "qdrant",
                "MongoDB already runs — use Atlas $vectorSearch.",
            ),
            "sqlite": (
                "sqlite_vec",
                "duckdb",
                "SQLite already in the app — "
                "sqlite-vec embeds vector search in the same file.",
            ),
            "duckdb": (
                "duckdb",
                "lancedb",
                "DuckDB already in use — the VSS "
                "extension does ANN in the same SQL engine.",
            ),
        }
        if existing_db in mapping:
            primary, runner, why = mapping[existing_db]
            reasoning.append(why)
            return _result(primary, runner, reasoning, [])

    # Q3 — can't run a server: embedded only.
    if not can_run_docker:
        reasoning.append("No Docker / server: restricted to embedded backends.")
        if needs_hybrid:
            reasoning.append("LanceDB is the embedded backend with native hybrid.")
            return _result("lancedb", "chroma", reasoning, ["sqlite_vec", "duckdb"])
        if corpus_size in ("large", "huge"):
            reasoning.append("Large corpus, embedded: LanceDB scales on local/S3.")
            return _result("lancedb", "faiss", reasoning, ["chroma"])
        reasoning.append("Chroma is the fastest embedded path to a working app.")
        return _result("chroma", "lancedb", reasoning, ["sqlite_vec", "duckdb"])

    # Q4 — air-gapped: self-hostable, no telemetry, permissive license.
    if air_gapped or not cloud_ok:
        reasoning.append(
            "On-prem / air-gapped: pick a self-hostable, permissively licensed "
            "backend with no mandatory telemetry."
        )
        if corpus_size in ("large", "huge"):
            return _result("milvus", "qdrant", reasoning, ["weaviate", "pgvector"])
        return _result("qdrant", "chroma", reasoning, ["milvus", "weaviate", "lancedb"])

    # Q7 — native hybrid wanted.
    if needs_hybrid:
        reasoning.append("Native hybrid (keyword + vector) wanted in one query.")
        return _result(
            "weaviate", "elasticsearch", reasoning, ["qdrant", "redis", "pinecone"]
        )

    # Q1 — huge scale.
    if corpus_size == "huge":
        reasoning.append("Billion-scale: DiskANN-class engines keep RAM bounded.")
        return _result("milvus", "pinecone", reasoning, ["qdrant", "turbopuffer"])

    # Q5 — budget.
    if budget == "free":
        reasoning.append(
            "Free tier only: Qdrant Cloud has a permanent free 1 GB cluster "
            "and the cleanest exit path."
        )
        return _result("qdrant", "pinecone", reasoning, ["chroma", "milvus"])

    reasoning.append(
        "Paid OK, managed, no special constraints: Pinecone serverless is the "
        "zero-ops canonical; Qdrant Cloud is the open runner-up."
    )
    return _result("pinecone", "qdrant", reasoning, ["turbopuffer", "weaviate"])


def _result(
    primary: str,
    runner_up: str,
    reasoning: list[str],
    alternatives: list[str],
) -> dict[str, Any]:
    """Assemble a recommendation dict, attaching a one-line note per backend."""

    def note(name: str) -> str:
        meta = provider(name) or {}
        return meta.get("sweet_spot", "")

    return {
        "primary": primary,
        "primary_note": note(primary),
        "runner_up": runner_up,
        "runner_up_note": note(runner_up),
        "alternatives": alternatives,
        "reasoning": reasoning,
    }


def print_recommendation(**kwargs) -> None:
    """Run :func:`recommend_backend` and print the recommendation readably."""
    rec = recommend_backend(**kwargs)
    print("\nVector database recommendation")
    print("=" * 60)
    for step in rec["reasoning"]:
        print(f"  - {step}")
    print()
    print(f"  PRIMARY    : {rec['primary']}  — {rec['primary_note']}")
    print(f"  RUNNER-UP  : {rec['runner_up']}  — {rec['runner_up_note']}")
    if rec["alternatives"]:
        print(f"  ALSO OK    : {', '.join(rec['alternatives'])}")
    print()
    print(f"  Next: vd.check_requirements({rec['primary']!r}) for setup steps.")
    print()
