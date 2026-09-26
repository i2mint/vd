---
name: vd-backend-choose
description: >-
  Backend-selection tooling for the vd package. Use this skill when the user
  is picking a vector database with vd, asks "which backend should I use",
  compares backends, or weighs persistence / cloud / cost / hybrid search /
  async / scale / license trade-offs. For installing, starting and verifying
  the chosen backend (pip, Docker, API keys), use vd-setup-backend.
metadata:
  audience: users
---

# vd — choosing a backend

`vd` knows ~21 vector databases and ships facade adapters for 15. This skill
**chooses** the right one; **vd-setup-backend** then installs, starts and
verifies it.

## 1. Choose

### Let vd recommend one

`recommend_backend` encodes the decision framework from the bundled report.
Answer a few facts; it returns a primary pick, a runner-up, and the reasoning.

```python
import vd

vd.print_recommendation(
    corpus_size="medium",  # tiny<100k | small<10M | medium | large<100M | huge>100M
    persistence=True,
    can_run_docker=True,
    cloud_ok=True,
    budget="free",  # "free" | "paid"
    existing_db=None,  # "postgres"|"redis"|"elastic"|"mongo"|"sqlite"|"duckdb"
    needs_hybrid=False,  # keyword + vector fused in one query?
    air_gapped=False,
)
# vd.recommend_backend(...) returns the same as a dict.
```

Key heuristics it applies: tiny + no persistence → `memory`; already running
Postgres → `pgvector`; no Docker → embedded (`chroma`, or `lancedb` when hybrid
search is wanted); air-gapped → self-hostable Apache/BSD backends; hybrid
wanted → `weaviate`; huge scale → `milvus`; free managed → `qdrant`.

### Capabilities that differ by backend in vd

`vd.hybrid_search` works on every backend (a client-side BM25 + RRF fallback),
but these backends run the lexical side natively, which scales far better:
`weaviate`, `elasticsearch`, `redis`, and `lancedb` (the only embedded one).
Check with `isinstance(collection, vd.SupportsHybrid)`.

`vd.connect_async` also works on every backend (a thread-pool wrapper), but
only these do real non-blocking I/O through a native async SDK:
`vd.list_async_backends()` → currently `qdrant`, when connected to a server
(`url=`); embedded Qdrant uses the wrapper. `client.native_async` tells you
which one you got. Prefer a native client for
high-concurrency async apps (FastAPI, Starlette).

### Browse the landscape

```python
vd.print_backends_table()  # every backend, grouped by archetype
vd.list_backends()  # adapters installed & ready right now
vd.providers()  # full registry: {name: metadata}
vd.provider("qdrant")  # one backend's metadata
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

Hand over to **vd-setup-backend**. In short:

```python
vd.check_requirements("qdrant")  # diagnoses readiness, prints the NEXT STEP
print(vd.setup_guide("qdrant"))  # full copy-pasteable playbook (pip/docker/env)
```

## 3. Connect

```python
vd.connect("memory")  # embedded, no persistence
vd.connect("chroma", persist_directory="./db")  # embedded, on disk
vd.connect("qdrant", url="http://localhost:6333")  # server
vd.connect("qdrant")  # qdrant embedded (:memory:)
vd.connect("pinecone", api_key=...)  # managed
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
