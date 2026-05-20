# `vd` — Design Notes (research distillation)

> Deep, actionable notes for the `vd` refactor. Distilled 2026-05-20 from
> `research/semantic_search/03 -- Vector Storage and Retrieval` (primary),
> `04 -- Frameworks as Facade Case Studies`, `05 -- Schemas/Conventions`,
> `08 -- Client-Side AI Vector Search`. Companion: [`vd_use_cases.md`](vd_use_cases.md),
> [`../../.claude/CLAUDE.md`](../../.claude/CLAUDE.md).

---

## 1. The verdict: `vd`'s existing design is endorsed

Report 03 was written explicitly to inform a vectorDB facade. Its conclusions
**validate `vd`'s current `base.py`**: the `Client`→`Collection` hierarchy, the
`Document` model, `StaticIndexError`, and the "unified MongoDB-style filter"
goal are all what the research recommends. The refactor is **hardening + filling
gaps**, not a rewrite. The two divergences to reconcile:

1. `vd`'s `Collection` Protocol bakes `add_documents` + `upsert` into the
   *baseline* — research says trim required surface to `MutableMapping` +
   `search()`, demote batch ops to a `SupportsBatch` capability.
2. `vd` should add `UnsupportedFilterError`, documented per-adapter operator
   subsets, `Supports*` capability protocols, and a first-class `.client` /
   `.native` escape hatch.

## 2. The minimal contract

The universal contract across *every* surveyed store is genuinely small —
`upsert`, `delete`, `search`. Keep the required Protocol surface tiny; push
variance into capability protocols + escape hatches.

```python
from typing import Protocol, Mapping, Optional, runtime_checkable, TypeAlias, NamedTuple
from collections.abc import MutableMapping

Key: TypeAlias = str
Vector: TypeAlias = "list[float] | numpy.ndarray"
Metadata: TypeAlias = Mapping[str, "str|int|float|bool|list|None"]
FilterAST: TypeAlias = "Mapping[str, object]"   # MongoDB-style JSON

class SearchHit(NamedTuple):
    key: Key
    score: float
    metadata: Metadata
    # vd's richer SearchResult dict also carries `text`; keep both — NamedTuple
    # for the typed path, dict for the egress-friendly path.

@runtime_checkable
class VectorStore(MutableMapping[Key, "Document"], Protocol):
    """MutableMapping-shaped vector store with a single `search` extension.

    Storage (inherited): store[k]=doc (upsert) / store[k] / del store[k] /
                         for k in store / len / k in store
    Retrieval (the only addition): search(query, k, filter=None) -> list[SearchHit]
    """
    dimension: int
    distance: str  # "cosine" | "dot" | "l2"

    def search(self, query: Vector, k: int = 10,
               filter: Optional[FilterAST] = None) -> list[SearchHit]: ...
```

### Capability protocols (opt-in, `@runtime_checkable`)

```python
@runtime_checkable
class SupportsHybrid(Protocol):
    def hybrid_search(self, query: Vector, query_text: str, k: int = 10,
                      filter: Optional[FilterAST] = None,
                      alpha: float = 0.5,                  # 0=sparse, 1=dense
                      fusion: str = "rrf") -> list[SearchHit]: ...

@runtime_checkable
class SupportsBatch(Protocol):
    def upsert_many(self, items: Mapping[Key, "Document"]) -> None: ...
    def search_many(self, queries: list[Vector], k: int = 10) -> list[list[SearchHit]]: ...

@runtime_checkable
class SupportsExport(Protocol):
    def export(self, path: str, *, format: str = "jsonl") -> int: ...
```

Plus a plain `supports_incremental_writes: bool` flag; `StaticIndexError` is the
*enforcement*, the flag lets callers branch ahead of time.

## 3. The filter language — adopt MongoDB-style JSON

The strongest cross-cutting recommendation. Canonical filter = MongoDB-style
JSON dict. Core operator subset every adapter must support:

> `$eq $ne $gt $gte $lt $lte $in $nin $and $or $not`
> Optional per-adapter: `$exists $contains $regex`.

A bare `{'field': value}` is sugar for `{'field': {'$eq': value}}`; multiple
top-level fields combine with implicit AND. Anything outside an adapter's
supported subset raises `UnsupportedFilterError`. Each adapter ships a
`_compile_filter(ast) -> native` translator. Provide a Python builder
(`F("year") >= 2020`).

### In-memory MongoDB-style evaluator (copy as-is for the memory backend)

```python
def _match(meta, flt):  # tiny MongoDB-style filter evaluator
    if flt is None: return True
    for k, v in flt.items():
        if k == "$and": return all(_match(meta, f) for f in v)
        if k == "$or":  return any(_match(meta, f) for f in v)
        actual = meta.get(k)
        if isinstance(v, dict):
            for op, val in v.items():
                if op == "$eq"  and actual != val: return False
                if op == "$ne"  and actual == val: return False
                if op == "$gt"  and not (actual is not None and actual >  val): return False
                if op == "$gte" and not (actual is not None and actual >= val): return False
                if op == "$lt"  and not (actual is not None and actual <  val): return False
                if op == "$lte" and not (actual is not None and actual <= val): return False
                if op == "$in"  and actual not in val: return False
                if op == "$nin" and actual in val: return False
        else:
            if actual != v: return False
    return True
```

### MongoDB-AST → Qdrant translator (template for a real adapter)

```python
def _filter_to_qdrant(flt):
    if flt is None: return None
    must, must_not, should = [], [], []
    for k, v in flt.items():
        if   k == "$and": must   += [_filter_to_qdrant(f) for f in v]
        elif k == "$or":  should += [_filter_to_qdrant(f) for f in v]
        elif isinstance(v, dict):
            for op, val in v.items():
                if   op == "$eq":  must.append(FieldCondition(key=k, match=MatchValue(value=val)))
                elif op == "$ne":  must_not.append(FieldCondition(key=k, match=MatchValue(value=val)))
                elif op == "$gt":  must.append(FieldCondition(key=k, range=Range(gt=val)))
                elif op == "$gte": must.append(FieldCondition(key=k, range=Range(gte=val)))
                elif op == "$lt":  must.append(FieldCondition(key=k, range=Range(lt=val)))
                elif op == "$lte": must.append(FieldCondition(key=k, range=Range(lte=val)))
                else: raise UnsupportedFilterError(f"qdrant: op {op}")
        else:
            must.append(FieldCondition(key=k, match=MatchValue(value=v)))
    return Filter(must=must or None, must_not=must_not or None, should=should or None)
```

## 4. Backend landscape — capability facts

### The three paradigms
- **Vector-first** (Pinecone, Qdrant) — user provides vectors, DB does ANN.
- **Document-first** (Chroma) — user provides text, DB embeds.
- **Schema-first** (Weaviate, Milvus) — explicit declared schema.

`vd` default = **schema-flexible** (arbitrary `metadata: Mapping`) — the lowest
common denominator. For schema-on-write backends, accept an optional `schema=`
at `create_collection`. **Portable contract = flat metadata with primitive/list
values**; nested metadata is a per-adapter extension.

### Per-backend quick facts

| Backend | Filter dialect | Hybrid | Index | Notes |
|---|---|---|---|---|
| **Pinecone** | MongoDB-style JSON | yes (sparse-dense) | hidden | managed serverless; namespaces in millions; flat metadata only; `fetch` for by-id |
| **Weaviate** | Pythonic builder (`Filter.by_property`) | yes (BM25F+dense, RRF) | HNSW/flat/SQ | schema-first; **collection aliases** (clean re-embed tool) |
| **Milvus/Zilliz** | boolean expr strings | yes | HNSW/IVF*/DiskANN/GPU | distributed; ≤ 65,536 collections; int64 ids |
| **Qdrant** | structured filter AST | yes (sparse, RRF) | HNSW in-mem/on-disk | Rust; int or UUID ids; `scroll` to enumerate, `retrieve` by-id |
| **Chroma** | MongoDB-style `where` + `where_document` | Cloud only | HNSW | embedded; **filter dialect closest to vd's canonical AST** |
| **pgvector** | full SQL `WHERE` | via `tsvector` | HNSW/IVFFlat | Postgres ext; `pg_dump` backup; native schema evolution |
| **LanceDB** | SQL `WHERE` (DataFusion) | + Tantivy FTS | IVF-PQ/IVF-HNSW/HNSW | embedded columnar; **cheap ACID schema evolution**; fully exportable — strong first real backend |
| **Elasticsearch/OpenSearch** | Bool query DSL | `rrf` retriever | HNSW + int4/8 quant | `sparse_vector` field (ELSER) |
| **FAISS** | none (caller filters) | none | composable | a library, not a DB; static-ish |
| **Annoy** | none | none | static | build-once immutable — classic `StaticIndexError` case |
| **sqlite-vec** | SQL `WHERE` | FTS5 in caller code | brute-force/KNN `vec0` | maintained successor to dead `sqlite-vss` |
| **Turbopuffer** | — | BM25+vector, RRF | centroid/IVF | object-storage-first; millions of namespaces; ~10× cheaper |

**Gotchas:** Marqo OSS is **deprecated** — do not build an adapter. `sqlite-vss`
is **dead** — target `sqlite-vec`. Filter languages are the #1 incompatibility
source — budget real translator effort. HNSW deletes are tombstones (count
may not shrink). IVF/IVF-PQ/ScaNN need periodic retraining. Vector dimension is
immutable post-creation on most backends — change ⇒ rebuild. DuckDB-VSS
persistence is still experimental.

### Index-method tradeoffs

| Method | Build | Query | Memory | Recall | Mutability | Filterable |
|---|---|---|---|---|---|---|
| Flat (brute force) | none | O(n) | full | perfect | trivial | trivial |
| IVF-Flat | fast | fast | full+centroids | high | periodic retrain | yes |
| IVF-PQ | moderate | very fast | ~1/16 | med-high | periodic retrain | yes |
| HNSW | moderate | very fast | full+graph (high) | high | insert ok; delete tombstone | hard (ACORN) |
| DiskANN/Vamana | slow/parallel | fast (SSD) | low | very high | FreshDiskANN updates | yes (Filtered) |
| ScaNN | moderate | very fast | compressed | very high (MIPS) | periodic retrain | yes |

### Quantization (storage scale; see also the client-side track)

| Method | Reduction | Recall | Use when |
|---|---|---|---|
| FP32 | 1× | 100% | small corpus, recall-critical |
| FP16 | 2× | >99.9% | near-free downscale |
| SQ8 / int8 | 4× | 95–98% | default for 10K–1M |
| Binary (BQ) | 32× | 80–85% raw; >95% with rescore | memory-bound; **always pair with two-stage rescore** |

**BQ + rescore pipeline:** (1) Hamming search over the 1-bit index → oversampled
candidate pool N ≫ k; (2) fetch full-precision vectors for those N, exact
cosine rerank → top-k. Recall back above ~95%.

## 5. Prior-art facade lessons

| Facade | Lesson for `vd` |
|---|---|
| LangChain | **Don't** bind the embedder into the store; **don't** put `**kwargs` tails on public methods |
| LlamaIndex | typed `MetadataFilters` AST is the most portable filter model; global `Settings` singleton is an anti-pattern |
| Haystack v2 | **store-CRUD vs retriever-search split is the cleanest design**; per-`DocumentStore` retrievers (type-level commitment) adapt to store-specific features better than a hiding abstraction |
| txtai | bundling the DB makes it an app, not a facade — `vd` stays a facade |
| dlt | write/read split (`append`/`replace`/`merge`) is instructive for `migration` semantics |

**Final architectural verdict (paste into vd's design-principles section):**
> - MongoDB-style `FilterAST` is the canonical filter language; thin Python
>   builder; per-adapter translators compile it.
> - `Collection` stays `MutableMapping`-shaped with `search` as the only
>   retrieval primitive.
> - `Protocol` + capability protocols (`SupportsHybrid`, `SupportsBatch`,
>   `SupportsMultiVector`, `SupportsExport`), not abstract base classes.
> - Index strategy is a backend concern — don't abstract it; one
>   `index_config: dict` escape hatch.
> - Hybrid is opt-in, not baseline — RRF convergence is real, syntax is not.
> - Every adapter exposes an `underlying_client` accessor.

## 6. Data model alignment with `ef`

`ef` produces **segments** (chunks). Align `vd.Document` field names with the
canonical segment schema so segment→index is a trivial map:

- `id` (required), `text` (required), `vector`/`embedding`, `metadata`.
- Promote `parent_id` (a.k.a. `parent_doc_id`) and `chunk_idx` to **top-level
  fields**, not buried in metadata — "query by parent" is the most common
  access pattern (dedup, context expansion).
- Carry `source_hash` + `config_hash` in metadata when `ef` writes — they make
  `ef`'s four staleness conditions (orphan / missing / stale / misconfigured)
  simple filtered queries against `vd`. Store embedder `model_id` + `dim` as
  **collection-level** metadata; reject dimension-mismatched ops loud and early.

## 7. The "same call sites, any backend" demo (use as an integration test)

```python
def index_and_search(store: VectorStore):
    store["a"] = Document(id="a", text="...", vector=[0.1,0.2,0.3],
                          metadata={"src":"wiki","year":2024})
    store["b"] = Document(id="b", text="...", vector=[0.9,0.1,0.0],
                          metadata={"src":"blog","year":2025})
    return store.search([0.1,0.2,0.25], k=2,
                        filter={"year":{"$gte":2024}, "src":{"$in":["wiki","blog"]}})
# Must produce equivalent results for memory, chroma, qdrant, lancedb, ...
```

## 8. Recommended additions checklist

- [ ] `UnsupportedFilterError`; per-adapter documented operator subset
- [ ] `F(...)` filter builder; `_compile_filter` per adapter
- [ ] `Supports*` capability protocols; `supports_incremental_writes` flag
- [ ] `.client` / `.native` escape hatch on every backend
- [ ] `quantization=` on `create_collection`; `rescore`/`oversample` on `search`
- [ ] `AsyncClient` / `AsyncCollection` parallel protocols
- [ ] `SupportsHybrid` + client-side RRF fallback
- [ ] At least one real backend beyond chroma — **LanceDB** recommended
- [ ] `LangChainVectorStoreAdapter` + `wrap_langchain_vectorstore`
- [ ] Collection-level embedder identity metadata (`model_id`, `dim`, `metric`)
- [ ] Resolve `chromadol` ↔ `vd` chroma backend overlap
