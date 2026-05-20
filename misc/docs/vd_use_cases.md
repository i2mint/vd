# `vd` — User Journeys, Use Cases & Common Operations

> The **living catalogue** of what people do with vector databases — and
> therefore what `vd` must make easy. Add to this doc whenever a new journey is
> encountered. Each item is a short named operation; items marked **★** are
> composite "user journeys" worth a dedicated facade function/method.
>
> Companion docs: [`vd_design_notes.md`](vd_design_notes.md) (the contracts),
> [`../../.claude/CLAUDE.md`](../../.claude/CLAUDE.md) (design constitution).
> Supersedes/extends the earlier `vd_specification.md` / `vd_key_takeaways.md`.

---

## A. Storage / CRUD (single-item)

| # | Operation | Shape |
|---|---|---|
| A1 | Upsert one document | `store[key] = Document(...)` / `= "text"` / `= ("text", meta)` |
| A2 | Get one by id | `store[key]` |
| A3 | Delete one | `del store[key]` |
| A4 | Membership check | `key in store` |
| A5 | Count | `len(store)` |
| A6 | Iterate all keys | `for key in store` (paginate/scroll internally) |
| A7 | Get-many by ids | fetch a list of keys in one round trip |
| A8 | Search by precomputed vector | `store.search([0.1, ...], k)` — skips embedding |

## B. Storage / CRUD (batch)

| # | Operation | Notes |
|---|---|---|
| B1 | Bulk upsert | `add_documents([...])` / `upsert_many({...})` — batched, amortizes round-trips |
| B2 | Bulk delete | delete by id list |
| B3 | Clear collection | drop all records, keep the collection |
| B4 ★ | **Ingest a corpus** | clean → chunk → batch-add with metadata (see `vd-ingest` skill; the real segmentation facade is `ef`'s) |

## C. Collection lifecycle (Client level)

| # | Operation | Notes |
|---|---|---|
| C1 | Create collection | with `dimension` + `distance` + optional `index_config`/`schema`/`quantization` |
| C2 | List collections | `client.list_collections()` |
| C3 | Get / open existing collection | `client.get_collection(name)` |
| C4 | Delete collection | `client.delete_collection(name)` |
| C5 | Create-if-missing (idempotent open) | |
| C6 | Collection stats | size, dimension, index type, metadata fields |

## D. Retrieval — the core

| # | Operation | Notes |
|---|---|---|
| D1 | Vector similarity search | `search(query, k)` — `query` is text (auto-embed) or a vector |
| D2 | Filtered search | `search(query, k, filter=...)` — MongoDB-style metadata filter |
| D3 ★ | **Find similar to an existing document** | "more like this id" — `search_similar_to_document` |
| D4 ★ | **Multi-query search** | several query phrasings, results combined — `multi_query_search` |
| D5 ★ | **Hybrid search** (dense + lexical/BM25) | RRF-fused; `SupportsHybrid` capability or client-side fallback |
| D6 ★ | **Reciprocal rank fusion** of multiple ranked lists | `reciprocal_rank_fusion(lists, k=60)` |
| D7 ★ | **Deduplicate results** | collapse near-duplicate hits (e.g. by `parent_id`) — `deduplicate_results` |
| D8 | MMR / diversified search | max-marginal-relevance to reduce redundancy (optional capability) |
| D9 ★ | **Two-stage rescore** | coarse ANN over quantized vectors → exact rerank top-k on full precision |
| D10 | Reranking | apply a reranker as a decorator over a base result set (separate stage) |
| D11 | Batch search | many queries in one call — `search_many` (capability) |
| D12 | Multi-vector / ColBERT late-interaction search | `SupportsMultiVector` capability |
| D13 | Time-windowed retrieval | search within a time range (`time_indexed`) |
| D14 | Result transformation (`egress`) | project results to text-only / id-only / tuples |

## E. Cross-backend / facade-specific journeys

| # | Operation | Notes |
|---|---|---|
| E1 ★ | **Switch backend with one parameter** | same call sites, `connect('memory')` ↔ `connect('chroma')` ↔ … |
| E2 ★ | **Migrate a collection between backends** | read-all A → write-all B; `preserve_vectors=True` avoids re-embed cost |
| E3 ★ | **Migrate an entire client** (all collections) | `migrate_client` |
| E4 ★ | **Re-embed a corpus** (model change) | write to new collection, dual-write, switch reads, drop old; use collection aliases where available |
| E5 | Capability discovery | `isinstance(store, SupportsHybrid)` before calling hybrid |
| E6 ★ | **Drop to the native client** | escape hatch — `.client` / `.native` for a backend-specific feature |
| E7 ★ | **Compare / recommend backends** | benchmark or rule-based pick by dataset size / persistence / budget — `compare` module |
| E8 | Health check | ping/readiness of a backend — `health` module |
| E9 ★ | **Export / backup a collection** | JSONL / JSON / directory / parquet — `io` module |
| E10 ★ | **Import / restore a collection** | load from a dump |
| E11 | Schema evolution | add a metadata field / change a column |
| E12 | CLI inspection | list/inspect collections from the shell — `cli` |
| E13 | Connect from a config file | YAML/TOML profiles + env-var overrides — `connect_from_config` |

## F. Collection analytics & quality

| # | Operation | Notes |
|---|---|---|
| F1 | Collection statistics | counts, avg text length, metadata fields — `collection_stats` |
| F2 | Metadata-value distribution | `metadata_distribution(coll, field)` |
| F3 ★ | **Find duplicates / near-duplicates** | `find_duplicates(threshold=0.95)` — surfaces overlap from chunking |
| F4 ★ | **Find outliers** | docs dissimilar to their neighbors — broken parses, wrong language |
| F5 | Sample a collection | `random` or `diverse` (spread across embedding space) |
| F6 | Validate integrity | empty texts, missing vectors, malformed metadata — `validate_collection` |
| F7 | Benchmark search / insert | latency percentiles, throughput — `health` module |

## G. Client-side / browser journeys (future — `vd-js` track)

> These belong to a TypeScript sibling, `vd-js`. Recorded here so the Python
> `vd` protocol stays clean enough to re-express in TS. See group design notes §6.

| # | Operation | Notes |
|---|---|---|
| G1 | Offline search | corpus indexed once, searched with no network |
| G2 | Privacy-preserving search | query text never leaves the device |
| G3 | Zero-server-cost search | static-hosted app, no inference backend, no DB bill |
| G4 | Edge embedding | embed client-side, upload only vectors to a remote `vd` backend |
| G5 | Hybrid local/remote | small hot corpus local + large cold corpus remote, RRF-merged |
| G6 | Instant local autocomplete | sub-ms ANN → semantic suggestions per keystroke |
| G7 | First-load corpus bootstrap | ship a prebuilt index or build on first run, persist to IndexedDB |
| G8 | Quantized large-corpus search | 100K vectors via BQ + FP32 rescore within browser memory budget |
| G9 | Bring-your-own-vectors | ingest vectors from another pipeline; no in-browser model |

## H. The "common operations" universal API (the 80/20)

Every surveyed vector store implements these — the irreducible core:

```
upsert(id, vector, metadata)          # __setitem__
get(id) -> (vector, metadata) | None  # __getitem__
delete(id)                            # __delitem__
__iter__ / list_ids()                 # iterate keys
search(query_vector, k, filter=None) -> [(id, score, metadata)]
count() / __len__
clear()
exists(id) / __contains__
```

Everything else (schema, hybrid, namespaces, multi-vector) is **variant** —
keep it out of the required contract, behind capability protocols.

---

## Notes on what makes a good facade method

A composite operation earns a **★ dedicated facade function** when it is (a)
commonly done, (b) annoying to hand-roll correctly, and (c) backend-agnostic.
`reciprocal_rank_fusion`, `migrate_collection`, `search_similar_to_document`,
`find_duplicates` all qualify. Resist adding methods that just rename a single
backend call — those belong to the escape hatch.
