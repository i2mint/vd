# vd

**A facade over vector databases — one Pythonic interface, ~15 backends.**

`vd` lets you operate on any vector database and switch between them with a
one-word change, while keeping each backend's particular power one escape hatch
away. It also helps you *choose* the right backend and *set it up*.

```python
import vd

client = vd.connect("memory")          # switch DB = change this one word
col = client.create_collection("docs")
col["a"] = vd.Document(id="a", text="cats", vector=[0.1, 0.9, 0.0])
col["b"] = vd.Document(id="b", text="pizza", vector=[0.9, 0.0, 0.1])

for hit in col.search([0.1, 0.8, 0.0], limit=2):
    print(hit["id"], hit["score"])
```

## Install

```bash
pip install vd                 # core (zero heavy deps) + the memory backend
pip install vd[chroma]         # + a specific backend's client
pip install vd[embedded]       # + all embedded backends (chroma, qdrant, faiss, …)
pip install vd[all-backends]   # + every backend client
```

The core is near-zero-dependency. Each backend's client library is an optional
extra named after the backend.

## The mental model

`vd` stores and searches **vectors**. Turning text into vectors — *embedding* —
is deliberately **external**: `vd` never embeds on its own. This keeps the
facade honest (most vector DBs do not embed for you) and lightweight.

- **Vector-first.** You hold the embedding model. Hand `vd` `Document`s that
  already carry a `vector`; search with a pre-computed query vector.
- **Text convenience.** Pass an `embedder` (`text -> vector`) to `connect`, and
  then raw text works: `col["k"] = "some text"`, `col.search("a query")`.

With no embedder, passing text raises `EmbeddingRequiredError` — loud, never a
silent wrong-model embedding.

```python
client = vd.connect("chroma", persist_directory="./db", embedder=my_embed_fn)
col = client.create_collection("docs")
col["a"] = "cats and kittens"                  # embedded for you
hits = list(col.search("pets", limit=5))       # query embedded for you
```

## Choosing a backend

`vd` ships a provider registry distilled from a practitioner report
(`misc/docs/11 -- VectorDB Selection & Setup Guide ...md`) and a recommender:

```python
vd.print_recommendation(
    corpus_size="medium", persistence=True, can_run_docker=True,
    cloud_ok=True, budget="free", needs_hybrid=False,
)
vd.print_backends_table()                       # the whole landscape
vd.compare_backends(["chroma", "qdrant", "pgvector"])
```

## Setting a backend up

```python
vd.check_requirements("qdrant")    # diagnoses readiness, prints the next step
vd.setup_guide("qdrant")           # full pip / docker / env-var playbook
vd.install_backend("qdrant")       # the pip command (run=True to install)
```

`check_requirements` is deployment-aware: it checks the pip package for
embedded backends, whether a server answers for self-hosted ones, and the
required environment variables for managed ones — always ending with one
concrete next action.

## The API

| Object | Is a | Plus |
|--------|------|------|
| `Client` (from `connect`) | `Mapping[str, Collection]` | `create_collection`, `get_collection`, `delete_collection`, `get_or_create_collection` |
| `Collection` | `MutableMapping[str, Document]` | `search(...)` |
| `Document` | dataclass | `id`, `text`, `vector`, `metadata` |

```python
col["k"] = vd.Document(id="k", text="…", vector=[...], metadata={"y": 2024})
doc      = col["k"]            # get
del col["k"]                  # delete
"k" in col, len(col), list(col)

col.search(query, *, limit=10, filter=None, egress=None, **backend_kwargs)
```

`search` yields dicts `{"id", "text", "score", "metadata"}` (`score` is
higher-is-better). Transform results with an `egress`: `vd.id_only`,
`vd.id_and_score`, `vd.text_only`, `vd.id_text_score`, or your own.

### Metadata filtering

One backend-agnostic, MongoDB-style filter language — `$eq $ne $gt $gte $lt
$lte $in $nin $exists $and $or $not`:

```python
col.search(qvec, filter={"year": {"$gte": 2020}, "kind": {"$in": ["news", "blog"]}})
```

Each backend declares which operators it honors natively; an unsupported one
raises `UnsupportedFilterError` rather than silently mis-filtering. Backends
with rich native filtering (Qdrant, Pinecone, MongoDB) translate the filter;
the rest apply it client-side with the same semantics.

### Escape hatches

The facade never traps you. `client.client` is the raw backend client;
`collection.native` is the raw backend collection — both supported, documented
API for reaching backend-specific features.

## Backends

| Archetype | Backends |
|-----------|----------|
| **Embedded** (pip-only) | `memory`, `chroma`, `lancedb`, `sqlite_vec`, `duckdb`, `faiss` |
| **Server** (also embedded) | `qdrant`, `weaviate`, `milvus` |
| **Server** | `redis`, `elasticsearch`, `pgvector` |
| **Managed** | `pinecone`, `mongodb` (Atlas), `turbopuffer` |

`vd.list_backends()` shows what is installed and ready now.

## The toolkit

Beyond the facade, `vd` bundles the composite operations people actually do:

- **`vd.search`** — `multi_query_search`, `reciprocal_rank_fusion`,
  `search_similar_to_document`, `deduplicate_results`.
- **`vd.io`** — `export_collection` / `import_collection` (JSONL, JSON,
  directory).
- **`vd.migration`** — `migrate_collection`, `migrate_client`,
  `copy_collection` — move data between *any* two backends.
- **`vd.analytics`** — `collection_stats`, `find_duplicates`, `find_outliers`,
  `validate_collection`.
- **`vd.health`** — `health_check_backend`, `benchmark_search`.
- **`vd.text`** — convenience text cleaning / chunking.
- **`vd.TimeIndexedCollection`** — a time-windowed wrapper over any collection.
- **CLI** — `vd backends`, `vd install`, `vd export/import`, `vd migrate`, …

## Skills

This package ships agent skills you can install into any agent host with
[`gh skill`](https://cli.github.com/manual/gh_skill) (don't have it?
[install gh](https://cli.github.com/)):

```bash
gh skill install i2mint/vd vd-quickstart --agent claude-code
gh skill install i2mint/vd vd-backend-choose
gh skill install i2mint/vd vd-ingest
gh skill install i2mint/vd vd-search
gh skill install i2mint/vd vd-ops
gh skill install i2mint/vd vd-add-backend
```

| Skill | Use it when… |
|-------|--------------|
| `vd-quickstart` | doing basic semantic/vector search — connect, create a collection, add docs, query |
| `vd-backend-choose` | picking a vector DB, weighing trade-offs, or installing/starting a backend |
| `vd-ingest` | loading documents/files into a collection — cleaning, chunking, metadata, bulk insert |
| `vd-search` | going beyond a basic `.search()` — filters, multi-query, RRF, similar-to, dedup, by-vector |
| `vd-ops` | managing a collection — export/import, migrate, stats, integrity, health checks, benchmarks |
| `vd-add-backend` | (developer) implementing or reviewing a new vd backend adapter |

## Design

- **Embedding is external.** The core operates on vectors; an `embedder` is an
  injected, optional convenience — never a hard dependency.
- **Two mappings.** A `Client` is a `Mapping` of collections; a `Collection` is
  a `MutableMapping` of documents plus `search`. Idiomatic, minimal, familiar.
- **Thin adapters.** `AbstractClient` / `AbstractCollection` implement
  everything users see; a backend supplies a handful of raw primitives. Adding
  a backend is ~150 lines — see the `vd-add-backend` skill.
- **Capabilities, not a fat base.** Optional features (`SupportsBatch`,
  `SupportsHybrid`) are `@runtime_checkable` protocols you feature-discover.

## License

MIT
