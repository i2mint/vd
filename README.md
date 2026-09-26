# vd

**One Pythonic interface to ~15 vector databases.** Store vectors, search them, filter by metadata, and switch between databases (memory, Chroma, Qdrant, pgvector, LanceDB, Pinecone, …) by changing one word.

*Still typing code with your own ten fingers? [Your artisanal corner is at the bottom](#for-artisanal-hand-typed-contributions).*

## For AI agents

**Skills.** `vd` ships agent skills in [`vd/data/skills/`](vd/data/skills/). They install into any agent host with [`gh skill`](https://cli.github.com/manual/gh_skill):

```bash
gh skill install i2mint/vd vd-quickstart --agent claude-code
gh skill install i2mint/vd vd-backend-choose
gh skill install i2mint/vd vd-setup-backend
gh skill install i2mint/vd vd-ingest
gh skill install i2mint/vd vd-search
gh skill install i2mint/vd vd-ops
gh skill install i2mint/vd vd-add-backend
```

They also ship inside the wheel, so after `pip install vd` you can link them without GitHub:

```bash
mkdir -p ~/.claude/skills
ln -s "$(python -c 'import vd; print(vd.skills_dir())')"/vd-* ~/.claude/skills/
```

| Skill | Use it when… |
|-------|--------------|
| `vd-quickstart` | doing basic vector search: connect, create a collection, add docs, query |
| `vd-backend-choose` | picking a vector DB and weighing trade-offs |
| `vd-setup-backend` | installing, starting and verifying a backend (pip, Docker, API keys) |
| `vd-ingest` | loading documents: cleaning, chunking, metadata, bulk insert |
| `vd-search` | filters, hybrid (keyword + vector) search, multi-query, RRF, similar-to, dedup |
| `vd-ops` | export/import, migration, stats, integrity, health checks, benchmarks |
| `vd-add-backend` | (developer) implementing or reviewing a backend adapter |

**Project instructions** for coding agents working *on* `vd` live in [`.claude/CLAUDE.md`](.claude/CLAUDE.md): architecture, test commands, conventions and the refactor roadmap.

**What an agent can do with `vd`:** recommend and set up a vector database for a user, index their documents, run filtered or hybrid searches, and move a collection between databases. The whole surface is two `Mapping`s plus `search`:

```python
import vd

client = vd.connect("memory")  # switch DB = change this one word
col = client.create_collection("docs")
col["a"] = vd.Document(
    id="a", text="cats", vector=[0.1, 0.9, 0.0], metadata={"kind": "pet"}
)
col["b"] = vd.Document(
    id="b", text="pizza", vector=[0.9, 0.0, 0.1], metadata={"kind": "food"}
)

for hit in col.search([0.1, 0.8, 0.0], limit=2):
    print(hit["id"], round(hit["score"], 3))
# a 1.0
# b 0.123
```

## Install

```bash
pip install vd                    # core (stdlib + pyyaml) and the memory backend
pip install "vd[chroma]"          # plus one backend's client
pip install "vd[embedded]"        # plus chroma, qdrant, faiss, lancedb, sqlite_vec, duckdb
pip install "vd[all-backends]"    # plus every backend client
```

Each backend's client is an optional extra named after the backend. Keep the quotes: zsh treats `[...]` as a glob.

## The mental model

`vd` stores and searches **vectors**. Turning text into vectors, *embedding*, is deliberately **external**: `vd` never embeds on its own. Hand it `Document`s that carry a `vector`, and search with a query vector.

For convenience, pass any `text -> vector` function as `embedder`, and raw text then works too:

```python
import vd


def embed(text):  # stand-in; use a real embedding model
    return [text.count(c) / (len(text) or 1) for c in "aeiou"]


col = vd.connect("memory", embedder=embed).create_collection("notes")
col["k1"] = "cats and kittens"  # embedded for you
col["k2"] = ("dogs and puppies", {"kind": "dog"})  # text + metadata
print([h["id"] for h in col.search("a kitten", limit=1)])
# ['k1']
```

Without an embedder, passing text raises `EmbeddingRequiredError`: loud, never a silent wrong-model embedding.

## The API

| Object | Is a | Plus |
|--------|------|------|
| `Client` (from `connect`) | `Mapping[str, Collection]` | `create_collection`, `get_collection`, `delete_collection`, `get_or_create_collection` |
| `Collection` | `MutableMapping[str, Document]` | `search(...)` |
| `Document` | dataclass | `id`, `text`, `vector`, `metadata` |

```python
import vd

col = vd.connect("memory").create_collection("api")
col["k"] = vd.Document(id="k", text="hello", vector=[1.0, 0.0], metadata={"year": 2024})
doc = col["k"]  # get
print("k" in col, len(col), list(col))  # membership, count, keys
# True 1 ['k']
del col["k"]  # delete
```

`search(query, *, limit=10, filter=None, egress=None)` yields dicts `{"id", "text", "score", "metadata"}` where `score` is higher-is-better. Transform results with an `egress` such as `vd.id_only`, `vd.id_and_score`, `vd.text_only`, `vd.id_text_score`, or your own function.

### Metadata filtering

One backend-agnostic, MongoDB-style filter language: `$eq $ne $gt $gte $lt $lte $in $nin $exists $and $or $not`.

```python
import vd

col = vd.connect("memory").create_collection("posts")
col["a"] = vd.Document(
    id="a", text="", vector=[1.0, 0.0], metadata={"year": 2023, "kind": "news"}
)
col["b"] = vd.Document(
    id="b", text="", vector=[0.9, 0.1], metadata={"year": 2019, "kind": "blog"}
)
hits = col.search(
    [1.0, 0.0], filter={"year": {"$gte": 2020}, "kind": {"$in": ["news", "blog"]}}
)
print([h["id"] for h in hits])
# ['a']
```

Each backend declares which operators it honors. An unsupported one raises `UnsupportedFilterError` instead of silently mis-filtering. Backends with rich native filtering (Qdrant, Pinecone, MongoDB) translate the filter; the rest apply it client-side with the same semantics.

### Hybrid search

`vd.hybrid_search` fuses a vector search with a keyword (BM25) search using Reciprocal Rank Fusion, on any backend:

```python
import vd

col = vd.connect("memory").create_collection("hybrid")
col["a"] = vd.Document(id="a", text="error code E1234 in the parser", vector=[0.2, 0.8])
col["b"] = vd.Document(id="b", text="parser overview", vector=[0.9, 0.1])
hits = vd.hybrid_search(col, [0.9, 0.1], query_text="E1234", limit=2)
print([h["id"] for h in hits])
# ['a', 'b']
```

Weaviate, Elasticsearch, Redis and LanceDB run the keyword side on their own text index (`isinstance(col, vd.SupportsHybrid)`); other backends use a client-side BM25 scan.

### Async

```python
import asyncio
import vd


async def main():
    async with await vd.connect_async("memory") as client:
        col = await client.create_collection("docs", dimension=2)
        await col.set("a", vd.Document(id="a", text="x", vector=[1.0, 0.0]))
        return [h["id"] async for h in col.search([1.0, 0.0], limit=1)]


print(asyncio.run(main()))
# ['a']
```

Every backend works through a thread-pool wrapper. Backends listed by `vd.list_async_backends()` can return a native async client that does real non-blocking I/O: currently `qdrant` when connected to a server with `url=`. Check `client.native_async`.

### Escape hatches

The facade never traps you. `client.client` is the raw backend client and `collection.native` is the raw backend collection. Both are supported, documented API for reaching backend-specific features.

## Choosing and setting up a backend

```python
import vd

vd.print_recommendation(
    corpus_size="medium",
    persistence=True,
    can_run_docker=True,
    cloud_ok=True,
    budget="free",
    needs_hybrid=False,
)
vd.print_backends_table()  # the whole landscape
vd.compare_backends(["chroma", "qdrant", "pgvector"])

vd.check_requirements("qdrant")  # diagnoses readiness and prints the next step
print(vd.setup_guide("qdrant"))  # pip / docker / env-var playbook
vd.install_command("qdrant")  # 'pip install "vd[qdrant]"'
```

`check_requirements` is deployment-aware. It checks the client library for embedded backends, whether a server answers for self-hosted ones, and the required environment variables for managed ones, and always ends with one concrete next action.

| Archetype | Backends |
|-----------|----------|
| **Embedded** (pip-only) | `memory`, `chroma`, `lancedb`, `sqlite_vec`, `duckdb`, `faiss` |
| **Server** (also embedded) | `qdrant`, `milvus` |
| **Server** | `weaviate`, `redis`, `elasticsearch`, `pgvector` |
| **Managed** | `pinecone`, `mongodb` (Atlas), `turbopuffer` |

`vd.list_backends()` shows which adapters are installed and ready now.

## The toolkit

Beyond the facade, `vd` bundles the composite operations people actually do:

- **`vd.search`**: `multi_query_search`, `reciprocal_rank_fusion`, `hybrid_search`, `BM25Index`, `search_similar_to_document`, `deduplicate_results`.
- **`vd.io`**: `export_collection` / `import_collection` (JSONL, JSON, directory).
- **`vd.migration`**: `migrate_collection`, `migrate_client`, `copy_collection` to move data between any two backends.
- **`vd.analytics`**: `collection_stats`, `find_duplicates`, `find_outliers`, `validate_collection`.
- **`vd.health`**: `health_check_backend`, `benchmark_search`.
- **`vd.text`**: convenience text cleaning and chunking.
- **`vd.TimeIndexedCollection`**: a time-windowed wrapper over any collection.
- **CLI**: `vd backends`, `vd install`, `vd export`, `vd import`, `vd migrate`, …

## For artisanal, hand-typed contributions

Welcome, fellow keyboard enthusiast. Here is what the sections above don't already cover.

**Dev setup and tests.**

```bash
git clone https://github.com/i2mint/vd && cd vd
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[test,dev]"
python -m pytest --doctest-modules -o doctest_optionflags='ELLIPSIS IGNORE_EXCEPTION_DETAIL'
```

That is exactly what CI runs; package doctests are included. The backend-parametrized suites skip server backends that aren't running. To exercise them, start the containers with `docker compose -f tests/docker-compose.yml up -d` and install their clients with `uv pip install -e ".[pgvector,redis,elasticsearch,weaviate,mongodb,milvus]"`.

**Design rationale.**

- **Embedding is external.** The core operates on vectors; an `embedder` is an injected, optional convenience, never a hard dependency.
- **Two mappings.** A `Client` is a `Mapping` of collections; a `Collection` is a `MutableMapping` of documents plus `search`. Idiomatic, minimal, familiar.
- **Thin adapters.** `AbstractClient` / `AbstractCollection` implement everything users see; a backend supplies a handful of raw primitives. Adding a backend is about 150 lines, and the `vd-add-backend` skill walks through it.
- **Capabilities, not a fat base.** Optional features (`SupportsBatch`, `SupportsHybrid`, `SupportsNativeAsync`) are `@runtime_checkable` protocols you feature-discover.

The longer design notes are in [`misc/docs/vd_design_notes.md`](misc/docs/vd_design_notes.md), and the backend-selection research report is in [`misc/docs/`](misc/docs/).

**Contributing.** Open an issue or a pull request at [github.com/i2mint/vd](https://github.com/i2mint/vd/issues). Releases are automated: merging to `master` bumps the version and publishes to PyPI.

## License

MIT
