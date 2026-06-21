---
name: vd-add-backend
description: >-
  Developer skill for implementing a new vector-database backend in the vd
  package. Use when adding, refactoring, or reviewing a vd backend adapter
  (qdrant, pinecone, weaviate, lancedb, pgvector, faiss, etc.) — covers the
  AbstractClient/AbstractCollection raw-primitive contract, filter handling,
  capability protocols, the escape hatch, and the provider registry. Trigger on
  "add a backend to vd", "implement the X backend", "vd adapter".
audience: developers
---

# Implementing a `vd` backend

This skill is for working **on** `vd` — adding or refactoring a backend adapter.
For *using* `vd`, see the user-facing skills in `vd/data/skills/`.

## The model

A backend adapter is one file, `vd/backends/<name>.py`, with two classes built
on the shared bases in `vd/base.py`:

- `<Name>Collection(AbstractCollection)` — `AbstractCollection` already
  implements the whole public `Collection` surface (flexible `__setitem__`
  inputs, optional text embedding, `egress`, batch helpers, dimension checks,
  filter validation). You implement only the **raw primitives**.
- `<Name>Client(AbstractClient)` — `AbstractClient` gives you the `Mapping`
  behavior and `get_or_create_collection`. You implement four methods.

`memory.py` is the simplest reference; `qdrant.py` is the reference for a real
client-backed adapter with native filter translation.

## Raw primitives — what you actually write

```python
class <Name>Collection(AbstractCollection):
    def _write(self, doc):     ...   # upsert one Document (vector guaranteed set)
    def _read(self, key):      ...   # -> Document; raise KeyError if absent
    def _drop(self, key):      ...   # delete; raise KeyError if absent
    def _keys(self):           ...   # -> Iterator[str]
    def _count(self):          ...   # -> int
    def _query(self, vector, *, limit, filter, **kwargs):  ...  # -> Iterable[dict]
    # optional: _write_many(docs) for an efficient batch path
    # optional: native property -> raw backend collection handle
```

`_query` returns dicts with keys `id`, `text`, `score`, `metadata`; `score` is
higher-is-better. `AbstractCollection.search` has already validated `filter`
against `supported_filter_operators` and embedded a text query before calling
`_query`.

```python
@register_backend("<name>")
class <Name>Client(AbstractClient):
    def __init__(self, *, embedder=None, **config):
        super().__init__(embedder=embedder, **config)
        self._client = ...            # the raw backend client (the escape hatch)
    def create_collection(self, name, *, dimension=None, metric="cosine", **index_config): ...
    def get_collection(self, name):    ...   # raise KeyError if absent
    def delete_collection(self, name): ...   # raise KeyError if absent
    def list_collections(self):        ...   # -> Iterator[str]
    # optional: close()
```

## Steps

1. **Module docstring** ending in `Requires: pip install <pkg>`.
2. **Guarded import** at module top: `try: import <lib> except ImportError as e:
   raise ImportError("The <name> backend needs ...") from e`. A missing client
   library then just means the backend is not registered — `vd` still imports.
3. **`@register_backend("<name>")`** on the client class.
4. **Pass `embedder`, `dimension`, `metric`** from the client into every
   collection it constructs.
5. **Filtering — two honest choices:**
   - The backend has native metadata filtering covering the operators: set
     `supported_filter_operators` to that documented subset and translate the
     canonical AST in `_query` (see `qdrant.py`'s `_to_qdrant_filter`,
     `chroma.py`'s `_to_chroma_where`). Operators outside the subset are
     auto-rejected with `UnsupportedFilterError`.
   - Native filtering is absent or weaker: leave `supported_filter_operators`
     at the default (full language) and filter **client-side** with
     `apply_client_filter` + `overfetch_limit` from `vd/backends/_helpers.py`
     (see `faiss.py`).
6. **Static indexes:** if the index cannot accept incremental writes, set
   `supports_incremental_writes = False` — `AbstractCollection` then raises
   `StaticIndexError` on write for you.
7. **Provider registry:** add an entry to `vd/data/providers.yaml` (deployment
   archetype, pip package, license, docs URLs, `verify_command`, notes) and set
   its `adapter:` field to your backend name. Add the name to
   `_BACKEND_MODULES` in `vd/backends/__init__.py`.
8. **Tests:** add the backend to `TESTABLE_BACKENDS` in `tests/conftest.py` if
   it can run in plain CI — the parametrized `test_core.py` suite then exercises
   it automatically. Otherwise it is "correct-by-construction" (no server here).
9. **Update user skills** in `vd/data/skills/` if the happy path changed.

## Hard don'ts

- Don't embed inside the adapter. Embedding is external — the `embedder` is
  injected and optional; `AbstractCollection` handles text→vector and raises
  `EmbeddingRequiredError` when there is no embedder. Never import `imbed` or
  call an embedding API from a backend.
- Don't abstract index strategy (HNSW vs IVF-PQ vs …). Backend-specific index
  tuning goes through `**index_config` on `create_collection`, documented per
  adapter.
- Don't invent a unified hybrid syntax — expose hybrid via the `SupportsHybrid`
  capability only, and only if the backend does it natively.
- Don't return wrong-shaped results — `_query` dicts need `id/text/score/metadata`.
- Don't fake capabilities. Implement what the backend does; nothing more.
