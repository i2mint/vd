---
name: vd-backend-choose
description: >-
  Backend-selection and setup tooling for the vd package. Use this skill when
  the user is picking a vector database with vd, asks "which backend should I
  use", weighs persistence / cloud / cost / hybrid / scale trade-offs, hits a
  "backend not installed" error, or needs help installing and starting a
  vectorDB (pip packages, Docker, API keys, env vars).
audience: users
---

# vd — choosing and setting up a backend

`vd` knows ~21 vector databases and ships facade adapters for 15. This skill
has two jobs: **choose** the right one, then **set it up**.

## 1. Choose

### Let vd recommend one

`recommend_backend` encodes the decision framework from the bundled report.
Answer a few facts; it returns a primary pick, a runner-up, and the reasoning.

```python
import vd

vd.print_recommendation(
    corpus_size="medium",     # tiny<100k | small<10M | medium | large<100M | huge>100M
    persistence=True,
    can_run_docker=True,
    cloud_ok=True,
    budget="free",            # "free" | "paid"
    existing_db=None,         # "postgres"|"redis"|"elastic"|"mongo"|"sqlite"|"duckdb"
    needs_hybrid=False,       # keyword + vector fused in one query?
    air_gapped=False,
)
# vd.recommend_backend(...) returns the same as a dict.
```

Key heuristics it applies: tiny + no persistence → `memory`; already running
Postgres → `pgvector`; no Docker → embedded (`chroma`/`lancedb`); air-gapped →
self-hostable Apache/BSD backends; hybrid wanted → `weaviate`; huge scale →
`milvus`; free managed → `qdrant`.

### Browse the landscape

```python
vd.print_backends_table()              # every backend, grouped by archetype
vd.list_backends()                     # adapters installed & ready right now
vd.providers()                         # full registry: {name: metadata}
vd.provider("qdrant")                  # one backend's metadata
vd.compare_backends(["chroma", "qdrant", "pgvector"])
vd.print_comparison(["chroma", "qdrant", "pgvector"])
```

The deep reference is the bundled report **`misc/docs/11 -- VectorDB Selection
& Setup Guide ...md`** — provider profiles, a decision tree, free-tier notes,
license changes, and install playbooks. Point users there for detail; the
provider registry (`vd/data/providers.yaml`) is its machine-usable distillate
and stores **URLs** to live pricing/docs (never cached prices — they drift).

### Deployment archetypes (this dominates setup effort)

- **embedded** — `pip install`, no server: `chroma`, `lancedb`, `sqlite_vec`,
  `duckdb`, `faiss`, plus the always-on `memory`.
- **server** — a process/container you run: `qdrant`, `weaviate`, `milvus`,
  `redis`, `elasticsearch`, `pgvector`. (Qdrant/Weaviate/Milvus also run
  embedded.)
- **managed** — someone else runs it, needs an account + API key: `pinecone`,
  `mongodb` (Atlas), `turbopuffer`.

## 2. Set up

```python
vd.check_requirements("qdrant")   # diagnoses readiness, prints the NEXT STEP
vd.setup_guide("qdrant")          # full copy-pasteable playbook (pip/docker/env)
vd.install_backend("qdrant")      # returns the pip command; run=True to install
```

`check_requirements` is archetype-aware: for embedded backends it checks the
pip package (and quirks like sqlite-vec needing SQLite ≥3.41); for server
backends it checks whether something answers on the default port (non-fatal if
the backend also runs embedded); for managed backends it checks the required
env vars (`PINECONE_API_KEY`, `QDRANT_URL`, `MONGODB_URI`, …). It always ends
with one concrete **next step** — a pip command, a `docker run` one-liner, or
an `export VAR=...`.

## 3. Connect

```python
vd.connect("memory")                                    # embedded, no persistence
vd.connect("chroma", persist_directory="./db")          # embedded, on disk
vd.connect("qdrant", url="http://localhost:6333")       # server
vd.connect("qdrant")                                    # qdrant embedded (:memory:)
vd.connect("pinecone", api_key=...)                     # managed
```

Pass `embedder=` only if you want text-input convenience (see **vd-quickstart**).

A config file can hold the connection (backend + kwargs) under named profiles:

```python
client = vd.connect_from_config("vd.yaml", profile="prod")
```

`vd.create_example_config()` prints a starter. A vd config file describes the
**backend connection only** — embedding is never configured there.

## Gotchas

- `vd.connect("pinecone")` when `pinecone` is not installed raises
  `BackendNotInstalledError` with the exact `pip install` command.
- Free-tier limits and pricing change monthly — the registry links to live
  pricing pages rather than quoting numbers. Re-verify before relying on them.
- Some backends need a running server or a cloud account; `check_requirements`
  tells the user exactly what is missing.
