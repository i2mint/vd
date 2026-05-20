# CLAUDE.md — `vd` (Vector Database Facades)

> Design constitution for refactoring/implementing `vd` to a highly functional,
> robust state. Distilled 2026-05-20 from deep research — full notes in
> [`misc/docs/vd_design_notes.md`](../misc/docs/vd_design_notes.md) and use
> cases in [`misc/docs/vd_use_cases.md`](../misc/docs/vd_use_cases.md). Group
> context: [`embeddings/docs/semantic_search_design_notes.md`](../../../priv/data/groups/embeddings/docs/semantic_search_design_notes.md).

---

## 1. What `vd` is

A **facade over vector databases**. Goals, in priority order:

1. Make it easy to **operate on any vectorDB** and **switch between them** with
   a one-parameter change.
2. Offer **common interfaces for the common operations** of vectorDBs.
3. Still allow use of **each backend's particular functionality** (escape hatch).
4. Offer functions/methods for the common **user journeys** — composite
   operations people actually do (migrate, dedup, RRF, hybrid, similar-to-doc…).

`vd` owns **layer L4 (the ANN index) + retrieval primitives**. It does *not*
own embedding (that is `ef`'s job — see §6). Primary research reference:
`research/semantic_search/03 -- Vector Storage and Retrieval`.

## 2. Current state (2026-05)

Already substantial — `vd` is the more mature of the `ef`/`vd` pair:

- `base.py` — `Document` dataclass; `Client` / `Collection` Protocols;
  `BaseBackend` ABC; `StaticIndexError`; type aliases (`Vector`, `Filter`,
  `SearchResult`).
- `backends/` — `memory`, `chroma`. (`pinecone`/`weaviate`/`qdrant`/`milvus`/
  `faiss` are *planned* — in `_backend_metadata`, not implemented.)
- Modules: `analytics`, `compare`, `config`, `health`, `io`, `migration`,
  `search`, `text`, `time_indexed`, `util`, `cli`.
- 5 **user-facing** skills bundled in `vd/data/skills/`: `vd-quickstart`,
  `vd-backend-choose`, `vd-ingest`, `vd-search`, `vd-ops`. (Dev-facing skills
  for working *on* vd live in `.claude/skills/`.)
- MongoDB-style metadata filter; `egress` result transforms; CLI.

**The research validates vd's existing design** (`Client`→`Collection`,
`Document`, `StaticIndexError`, MongoDB-style filter). The refactor is mostly
**hardening + filling gaps**, not a rewrite. See §4.

## 3. Core contracts (the design the refactor should converge on)

### 3.1 The hierarchy: `Client` → `Collection`
Keep the two-level split. `Client` methods stay minimal: `create_collection`,
`get_collection`, `list_collections`, `delete_collection`. A `Collection` is
bound to its `(collection_name, namespace)` **at construction** — namespace is
never a per-call argument. Where natural, make `Client` `Mapping`-shaped over
collections (`client["my_coll"]`).

### 3.2 The `Collection` contract: `MutableMapping` + `search()`
`MutableMapping` is a clean fit for **storage**, not for retrieval. The
universal contract is genuinely small:

```
store[key] = Document(...)        # upsert  (__setitem__ == idempotent replace-or-create)
store[key]                        # get     (__getitem__)
del store[key]                    # delete  (__delitem__)
for key in store: ...             # iterate keys (paginate/scroll internally)
len(store), key in store          # count, membership
store.search(query, k, filter)    # the ONE retrieval extension
```

**Recommended change:** the current `Collection` Protocol bakes in
`add_documents` + `upsert` as *baseline* methods. Report 03 recommends trimming
the *required* surface to `MutableMapping` + `search()` and demoting batch ops
to an optional `SupportsBatch` capability (§3.4). Discuss before landing — it is
a small interface break and `vd` is free-to-change, but keep `upsert`/
`add_documents` available as convenience (mixin) so existing skills stay valid.

### 3.3 Filter language: MongoDB-style JSON as the canonical `FilterAST`
This is the strongest cross-cutting recommendation. Five backends use it
natively; it is LangChain's `SelfQueryRetriever` translation target.

- Canonical filter = MongoDB-style JSON dict.
- Core operator subset every adapter must support: `$eq $ne $gt $gte $lt $lte
  $in $nin $and $or $not`. Optional: `$exists $contains $regex`.
- Each adapter implements `_compile_filter(ast) -> backend_native` (visitor).
- **Document the supported subset per adapter explicitly.** Anything outside it
  raises a new `UnsupportedFilterError` so the caller can drop to a
  backend-specific filter via the escape hatch.
- Provide a thin Python builder (`F("year") >= 2020`) compiling to the JSON.

### 3.4 Capability protocols, not a fat ABC
Use small `@runtime_checkable` capability protocols; callers feature-discover
with `isinstance`. Recommended: `SupportsHybrid`, `SupportsBatch`,
`SupportsMultiVector`, `SupportsExport`, plus a `supports_incremental_writes`
flag. `BaseBackend` ABC may stay as an *adapter-author convenience*; the
*contract clients code against* is the thin `Protocol` + capabilities.

### 3.5 Escape hatch — mandatory, first-class
Every adapter exposes `.client` (raw backend client) and `.native` / `.raw`
(raw collection handle). **Document it as supported API** — it is how users
reach backend-specific power without circumventing the facade. Not a leak.

### 3.6 Create-time vs runtime parameters
`dimension`, `distance` (`"cosine"|"dot"|"l2"`, adapters map spellings),
`index_config: dict` (the *single escape hatch* for HNSW `M`/`efConstruction`,
IVF `nlist`/`nprobe`, PQ…), and optional `schema` are **create-time** params on
`create_collection`. **Do not abstract index strategy** — index choice is
intimately backend-tied; one `index_config: dict` per adapter, documented keys.

### 3.7 Static vs dynamic indexes
`StaticIndexError` is the right pattern — backends whose index can't accept
incremental writes raise it on `__setitem__`/`__delitem__`, with a documented
`rebuild()` path. Surface a `supports_incremental_writes: bool` capability so
callers branch *before* the error.

### 3.8 Hybrid search — opt-in capability, not baseline
No syntactic convergence across backends; conceptual convergence is RRF
(`k=60`). Make pure-vector `search()` the central primitive; expose hybrid via
`SupportsHybrid.hybrid_search(query, query_text, k, filter, alpha, fusion)`.
For backends without native hybrid, vd provides **client-side RRF** as the
fallback engine (`reciprocal_rank_fusion` already exists in `search.py`).

## 4. Refactor priorities (gaps to fill)

In rough order:

1. **`UnsupportedFilterError`** + per-adapter documented filter subset + a
   `_compile_filter` translator per adapter (memory evaluates in Python;
   chroma translates). Add the `F(...)` builder.
2. **Capability protocols** (`Supports*`) + `supports_incremental_writes` flag;
   wire `StaticIndexError` to the flag.
3. **Real backends** — implement at least one of `qdrant` / `pinecone` /
   `lancedb` to prove the facade against a non-trivial backend. **LanceDB** is a
   strong first choice (embedded, columnar, native IVF/HNSW + FTS, cheap schema
   evolution, fully exportable). `chroma` exists — see whether `chromadol`
   should *be* the chroma backend.
4. **Escape hatch** — ensure `.client` / `.native` on every backend.
5. **`quantization=` on `create_collection`** + a backend-agnostic two-stage
   `rescore`/`oversample` option on `search` (benefits server backends too).
6. **`AsyncClient` / `AsyncCollection`** parallel protocols — do not wrap async
   clients in sync shims.
7. **Hybrid** via `SupportsHybrid`; client-side RRF fallback.
8. **Adapter shims** — `LangChainVectorStoreAdapter` (be a LangChain
   `VectorStore`) and `wrap_langchain_vectorstore` (wrap theirs) — a few hundred
   lines, large audience gain.
9. Embedder-identity metadata on `Collection` — store `model_id` + `dim` +
   `metric`; reject dimension-mismatched inserts/queries **loud and early**.

## 5. Module map (keep — each maps to user journeys)

`base` (contracts) · `util` (factory, registry, egress) · `backends/*` ·
`search` (multi-query, RRF, similar-to-doc, dedup) · `migration` ·
`compare` (recommend/compare backends) · `health` (health + benchmark) ·
`io` (export/import) · `analytics` (stats, duplicates, outliers, validate) ·
`text` (clean/chunk — **note:** chunking conceptually belongs in `ef`; keep
`vd`'s as a convenience but `ef` owns the real segmentation facade) ·
`time_indexed` · `config` · `cli`.

## 6. Boundaries — what `vd` must NOT do

- **No embedding model bound into `Collection`.** That was LangChain's mistake.
  Embedding is `ef`'s job. `vd` accepts an **injectable embedder** (or
  pre-computed vectors / `embedder=None` "bring-your-own-vectors" mode). The
  current auto-embed-on-`__setitem__` is acceptable *as a convenience* only if
  the embedder is injected, never imported as a hard dependency.
- No segmentation/RAG/orchestration. No unified hybrid syntax. No unified index
  enum. No Marqo-OSS adapter (deprecated). No `sqlite-vss` adapter (dead — use
  `sqlite-vec`).

## 7. Conventions

- Errors: `StaticIndexError`, `UnsupportedFilterError`, consider
  `UnsupportedCapabilityError`.
- `search` (not `similarity_search`); result type carries `(id, score, text,
  metadata)`; on-disk vector column named `embedding`, dtype `float32`, Arrow
  `FixedSizeList<Float32, dim>`.
- Backend-specific options behind `**kwargs` on the concrete adapter — never in
  the core protocol signature; never as `**kwargs` tails on public facade
  methods.
- Adding a backend → use the **`vd-add-backend`** dev skill in
  [`.claude/skills/vd-add-backend/`](skills/vd-add-backend/SKILL.md).
- Follow the user's global CLAUDE.md (functional > OOP, keyword-only after 3rd
  arg, module docstrings, doctests, progressive disclosure).

## 8. Change-safety

`vd` is **free to change** — no users yet. Refactor, rename, break interfaces to
get the design right. The one caution: the bundled user-facing skills in
`vd/data/skills/` describe the *current* API — when you change the API, update
those skills in the same change (use the `skill-sync` skill).
