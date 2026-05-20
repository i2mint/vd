---
name: vd-add-backend
description: >-
  Developer skill for implementing a new vector-database backend in the vd
  package. Use when adding, refactoring, or reviewing a vd backend adapter
  (qdrant, pinecone, weaviate, lancedb, pgvector, faiss, etc.) — covers the
  Collection/Client contract, the MongoDB-style filter translator, capability
  protocols, the escape hatch, and the static-index pattern. Trigger on
  "add a backend to vd", "implement the X backend", "vd adapter".
audience: developers
---

# Implementing a `vd` backend

This skill is for working **on** `vd` — adding or refactoring a backend adapter.
For *using* `vd`, see the user-facing skills in `vd/data/skills/`. Read
[`../../misc/docs/vd_design_notes.md`](../../misc/docs/vd_design_notes.md) first.

## The contract a backend must satisfy

A backend = a `Client` (manages collections) + a `Collection` (a
`MutableMapping` + `search`). The **required** surface is small:

```python
# Client
create_collection(name, *, dimension, distance, index_config=None, schema=None, **kw) -> Collection
get_collection(name) -> Collection
list_collections() -> Iterator[str]
delete_collection(name) -> None

# Collection  (MutableMapping[str, Document] + ONE retrieval method)
__setitem__(key, value)   # upsert — idempotent replace-or-create
__getitem__(key) -> Document
__delitem__(key)
__iter__() -> Iterator[str]
__len__() -> int
search(query, k=10, *, filter=None, egress=None, **kw) -> Iterator[SearchResult]
```

Everything else — batch ops, hybrid, multi-vector, export — is an **optional
capability protocol** (`SupportsBatch`, `SupportsHybrid`, …). Only implement
what the backend does natively; do not fake it.

## Steps

1. **Create `vd/backends/<name>.py`** with a module docstring.
2. **Register the backend** with `@register_backend('<name>')` on the backend
   class. Add metadata (description, install command, optional dep) to
   `vd`'s `_backend_metadata` so discovery/`vd-backend-choose` works.
3. **Lazy-import the client library** inside `__init__` (or a module-level
   guarded import) so `import vd` stays light. On `ImportError`, raise
   `ValueError` with install instructions — never fail silently.
4. **Map create-time params**: `dimension`, `distance` (translate canonical
   `"cosine"|"dot"|"l2"` to the backend's spelling), `index_config: dict`
   (pass through — do NOT abstract index strategy). Distance + dimension are
   fixed at collection creation everywhere.
5. **Implement the MutableMapping methods.** `__setitem__` is upsert. Accept the
   flexible `Document` input shapes via `vd.util.normalize_document_input`.
   If the backend's index is build-once-immutable, raise `StaticIndexError` on
   `__setitem__`/`__delitem__` after build, and set
   `supports_incremental_writes = False`.
6. **Implement `search`.** Translate the MongoDB-style `filter` AST with a
   `_compile_filter(ast) -> native` function (see the Qdrant translator
   template in `vd_design_notes.md` §3). Operators outside the backend's
   documented subset raise `UnsupportedFilterError`. Apply `egress` to each
   result before yielding. Return an **iterator**, not a list.
7. **Expose the escape hatch** — `.client` (raw backend client) and
   `.native`/`.raw` (raw collection handle) as documented attributes.
8. **Declare capabilities** — implement `SupportsBatch.upsert_many` etc. only
   if native; expose `supports_incremental_writes`.
9. **Document the supported filter-operator subset and `index_config` keys** in
   the module docstring.
10. **Tests** — the backend must pass the shared interface-compliance suite
    (the `index_and_search(store)` cross-backend demo in `vd_design_notes.md`
    §7), plus backend-specific tests.
11. **Update the user-facing skills** in `vd/data/skills/` if the new backend
    changes the happy path (use the `skill-sync` skill).

## Hard don'ts

- Don't bind an embedding model into the backend as a hard dependency — the
  embedder is **injected** (`embedding_model` arg) or vectors are pre-computed.
- Don't abstract index strategy (HNSW vs IVF-PQ vs …) — one `index_config: dict`.
- Don't invent a unified hybrid syntax — `SupportsHybrid` capability only.
- Don't put backend-specific options in the core protocol signature — they go
  in `**kwargs` on the concrete adapter.
- Don't build a Marqo-OSS adapter (deprecated) or a `sqlite-vss` adapter (dead —
  use `sqlite-vec`).
- Don't return a list from `search` — return a lazy iterator.

## Reference

- `vd/backends/memory.py` — the simplest complete reference (in-Python filter
  evaluator, brute-force search).
- `vd/backends/chroma.py` — a real client-backed adapter.
- `misc/docs/vd_design_notes.md` — capability matrix, filter translators,
  index/quantization tradeoffs.
