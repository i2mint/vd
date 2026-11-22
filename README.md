# vd

**One Pythonic interface to ~15 vector databases.** Store vectors, search them, filter by metadata, and switch between databases (memory, Chroma, Qdrant, pgvector, LanceDB, Pinecone, …) by changing one word.

For code that should not be tied to one framework's store. If you already live inside LangChain or LlamaIndex, see [vd versus framework stores](https://github.com/i2mint/vd#vd-versus-framework-stores).

## Install

```bash
pip install vd                    # core (stdlib + pyyaml) and the memory backend
pip install "vd[chroma]"          # plus one backend's client
pip install "vd[embedded]"        # plus chroma, qdrant, faiss, lancedb, sqlite_vec, duckdb
pip install "vd[all-backends]"    # plus every backend client
```

Each backend's client is an optional extra named after the backend. Keep the quotes: zsh treats `[...]` as a glob.

## Quick start

The whole surface is two `Mapping`s plus `search`:

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

*Still typing code with your own ten fingers? [Your artisanal corner is at the bottom](https://github.com/i2mint/vd#for-artisanal-hand-typed-contributions).*

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

Every backend works through a thread-pool wrapper. Backends listed by `vd.list_async_backends()` can return a native async client that does real non-blocking I/O: currently `lancedb` (embedded, over `lancedb.connect_async`) and `qdrant` when connected to a server with `url=`. Check `client.native_async`.

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

- **`vd.search`**: `multi_query_search`, `reciprocal_rank_fusion`, `hybrid_search`, `BM25Index`, `search_similar_to_document`, `deduplicate_results`. `BM25Index` is pure Python by default; `BM25Index(collection, engine="bm25s")` (the `vd[bm25s]` extra) precomputes the scores with [bm25s](https://github.com/xhluca/bm25s) and answers queries with common words hundreds of times faster past ~10k documents, with the same scores and ranking (`k1`/`b` then fixed at build time).
- **`vd.io`**: `export_collection` / `import_collection` (JSONL, JSON, directory).
- **`vd.migration`**: `migrate_collection`, `migrate_client`, `copy_collection` to move data between any two backends.
- **`vd.analytics`**: `collection_stats`, `find_duplicates`, `find_outliers`, `validate_collection`.
- **`vd.health`**: `health_check_backend`, `benchmark_search`.
- **`vd.text`**: convenience text cleaning and chunking.
- **`vd.TimeIndexedCollection`**: a time-windowed wrapper over any collection.
- **CLI**: `vd backends`, `vd install`, `vd export`, `vd import`, `vd migrate`, …

## vd versus framework stores

LlamaIndex, LangChain and Haystack wrap many more vector stores than vd's 15, and if your application lives inside one of those frameworks, its stores are the natural choice. They are built as plug-in points for the framework's retrieval pipeline rather than as a portable storage interface, and it shows in what their base classes guarantee (as of 2026-11-22):

- LangChain's `VectorStore` (langchain-core 1.8.4) requires only `similarity_search` and `from_texts`; `get_by_ids` is optional, there is no listing or length, and `filter` is passed through in each store's own syntax.
- LlamaIndex's `BasePydanticVectorStore` (llama-index-core 0.14.26) requires `add`, `delete` and `query`; `get_nodes`, which you need to read back or export a store, is optional. Its typed `MetadataFilters` is the most portable filter model of the three.
- Haystack's [metadata filtering docs](https://docs.haystack.deepset.ai/docs/metadata-filtering) define one filter syntax but note that the available operators "may vary depending on the specific Document Store integration".

We ran the same ten filters through LlamaIndex on three stores. Its in-memory store got all ten right; its Chroma adapter raised on four it does not support, which is the right behaviour; on Qdrant local mode one filter (`TEXT_MATCH`) returned no rows instead of two, with only a warning. The same documents also could not be added unchanged to all three (Chroma rejects list-valued metadata, Qdrant non-UUID ids).

vd's design answers those points: every backend is the same `Mapping`, so listing, export and `migrate_collection` work on all of them; each backend declares its filter operators and an unsupported one raises `UnsupportedFilterError`; string ids work everywhere. The test script and the full comparison are in [i2mint/vd#48](https://github.com/i2mint/vd/issues/48).

## For AI agents

**Skills.** `vd` ships agent skills in [`vd/data/skills/`](https://github.com/i2mint/vd/tree/master/vd/data/skills/). They install into any agent host with [`gh skill`](https://cli.github.com/manual/gh_skill):

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

**Project instructions** for coding agents working *on* `vd` live in [`.claude/CLAUDE.md`](https://github.com/i2mint/vd/blob/master/.claude/CLAUDE.md): architecture, test commands, conventions and the refactor roadmap.

**What an agent can do with `vd`:** recommend and set up a vector database for a user, index their documents, run filtered or hybrid searches, and move a collection between databases. The quick start above is the whole surface.

## For artisanal, hand-typed contributions

Welcome, fellow keyboard enthusiast. Here is what the sections above don't already cover.

**Dev setup and tests.**

```bash
git clone https://github.com/i2mint/vd && cd vd
uv venv .venv && . .venv/bin/activate
uv pip install -e ".[test,dev]"
python -m pytest --doctest-modules -o doctest_optionflags='ELLIPSIS IGNORE_EXCEPTION_DETAIL'
```

That is what CI runs (CI also installs the `ci-backends` extra); package doctests are included. The backend-parametrized suites skip server backends that aren't running. Every server backend except turbopuffer runs locally without an account, Pinecone and MongoDB Atlas included, through their official emulators:

```bash
DOCKERHUB_MIRROR=mirror.gcr.io docker compose -f tests/docker-compose.yml up -d
uv pip install -e ".[test,dev,pgvector,redis,elasticsearch,weaviate,mongodb,milvus,pinecone]" "pinecone<10"
python -m pytest
```

See [`tests/README.md`](https://github.com/i2mint/vd/blob/master/tests/README.md) for connection settings and caveats.

**Design rationale.**

- **Embedding is external.** The core operates on vectors; an `embedder` is an injected, optional convenience, never a hard dependency.
- **Two mappings.** A `Client` is a `Mapping` of collections; a `Collection` is a `MutableMapping` of documents plus `search`. Idiomatic, minimal, familiar.
- **Thin adapters.** `AbstractClient` / `AbstractCollection` implement everything users see; a backend supplies a handful of raw primitives. Adding a backend is about 150 lines, and the `vd-add-backend` skill walks through it.
- **Capabilities, not a fat base.** Optional features (`SupportsBatch`, `SupportsHybrid`, `SupportsNativeAsync`) are `@runtime_checkable` protocols you feature-discover.

The longer design notes are in [`misc/docs/vd_design_notes.md`](https://github.com/i2mint/vd/blob/master/misc/docs/vd_design_notes.md), and the backend-selection research report is in [`misc/docs/`](https://github.com/i2mint/vd/tree/master/misc/docs/).

**Contributing.** Open an issue or a pull request at [github.com/i2mint/vd](https://github.com/i2mint/vd/issues). Releases are automated: merging to `master` bumps the version and publishes to PyPI.

## License

MIT
