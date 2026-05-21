# `vd` — Notes for Possible Improvements

> A review pass over `vd` (and its main consumer `ef`), 2026-05-21. Captures
> improvement opportunities spotted while reading the code and docs. Nothing
> here is urgent — `vd` is already substantial and functional. This is a
> punch-list to triage, not a mandate.
>
> Companion documents:
> - [`vd_design_notes.md`](vd_design_notes.md) — the facade-design distillation
>   (2026-05-20), still the authority on the *intended* contract.
> - Report **03 — Vector Storage and Retrieval** (April 2026), in the embeddings
>   group `docs/research/semantic_search/` — the facade-design research `vd` was
>   built from.
> - Report **11 — VectorDB Selection & Setup Guide** *(commissioned, not yet
>   written — see `prompt 11 -- deep_research__vectordb_selection_and_setup.md`
>   in the same research folder)* — will supply the install/selection facts
>   several items below depend on.

---

## How the review was framed

The triggering question was: *did `ef` (vd's main consumer) over-influence
`vd`, leaving it suboptimal as an independent vectorDB facade?*

**Short answer: no, not really.** `vd` was designed independently — report 03
was written to inform a vectorDB facade in the abstract, and `vd`'s contracts
(`Client`→`Collection`, `Document`, MongoDB-style filters, `StaticIndexError`,
capability protocols) match that research, not `ef`'s needs. `ef` consumes `vd`
cleanly through the public surface (`connect`, `Document`, `collection.search`,
`collection[id] = …`). The one place the influence runs the *wrong* way is
embedding (see item 2) — and there it is `vd`'s default that forces `ef` to work
around it, not the reverse. `time_indexed.py` is a generic, backend-agnostic
wrapper, not an `ef`-driven feature.

So the work below is **hardening and finishing**, not a redesign.

---

## A. Contract & API consistency

### 1. The documented contract doesn't match the implemented one — *(issue [#7](https://github.com/i2mint/vd/issues/7))*

`misc/docs/vd_design_notes.md`, `.claude/CLAUDE.md`, and the
`.claude/skills/vd-add-backend/` skill describe an API that `vd/base.py` does
not implement:

| Aspect | Design notes / `vd-add-backend` skill | Actual `base.py` |
|---|---|---|
| `create_collection` | `(name, *, dimension, distance, index_config=None, schema=None)` | `(name, *, schema=None, **kwargs)` |
| `search` limit arg | `k=10` | `limit=10` |
| search result type | `SearchHit` NamedTuple | `SearchResult` dict |
| batch ops | `SupportsBatch` capability only | still baseline `Collection` methods |

A developer following `vd-add-backend` would write a non-conforming backend.
The refactor described in the design notes was **half-applied**: the typed
errors and capability protocols landed (issues #4, #5), but the `Collection`
surface trim, the create-time `dimension`/`distance` params, and the result
type did not. Decide the canonical contract and converge code + notes + skill
in one change. `vd` is free-to-change (no users yet) — lock it before more
backends are written.

*Supporting research: `vd_design_notes.md` §2, §3, §8; report 03 §8.1.*

### 2. `connect()` hard-requires an embedder — *(issue [#8](https://github.com/i2mint/vd/issues/8))*

`vd.connect()` always builds an embedding function; `embedding_model=None`
means "default to `imbed.Embed()`", not "no embedder". A pure vector-store
consumer must inject a fake one — `ef` passes a poison-pill `_unused_embedding_model`
that raises if ever called. `vd`'s own constitution (`.claude/CLAUDE.md` §6)
specifies an `embedder=None` "bring-your-own-vectors" mode; it is unimplemented.
Implementing it makes `vd` a cleaner facade for the *vector-first* paradigm
(Pinecone/Qdrant-style) and removes `ef`'s workaround (ef issue
[#22](https://github.com/thorwhalen/ef/issues/22)).

*Supporting research: report 03 §3.1 (the LangChain "don't bind the embedder"
lesson); `vd_design_notes.md` §5, §8.*

### 3. Escape-hatch naming is inconsistent — *(folded into issue [#7](https://github.com/i2mint/vd/issues/7))*

The raw-client escape hatch goes by four names: `BaseBackend.client` (property),
`ChromaCollection.native` (property), and `.raw` / `underlying_client` in prose.
Pick two canonical names — suggest `.client` (raw backend client) and `.native`
(raw collection handle) — and apply them on every backend. The escape hatch is
the *whole* answer to "use a backend's particularities without leaving the
facade", so it must be uniform and documented as supported API.

*Supporting research: `vd_design_notes.md` §5; report 03 §8.3.*

---

## B. Backend correctness

### 4. Chroma search score is metric-blind — *(issue [#9](https://github.com/i2mint/vd/issues/9))*

`ChromaCollection.search` does `score = 1/(1+distance)` for every metric and
never reads the collection's configured distance. Ranking survives (monotonic)
but scores are not interpretable and not comparable to the `memory` backend
(which returns raw cosine similarity). `vd`'s own RRF/dedup helpers and `ef`'s
`SearchHit.score` consume these. Define one canonical score semantics in the
`Collection.search` contract and make every backend honor it.

### 5. Only one real backend — the facade is unproven

`memory` + `chroma` are the only implemented backends; `pinecone`, `weaviate`,
`qdrant`, `milvus`, `faiss` are metadata-only stubs. A facade whose selling
point is "switch vectorDBs freely" cannot demonstrate that with one real
backend. `vd_design_notes.md` §4 recommends **LanceDB** as the first real
backend to add (embedded, columnar, native IVF/HNSW + FTS, cheap schema
evolution, fully exportable — exercises the contract harder than Chroma). A
second non-trivial backend is also what would surface any remaining
Chroma-shaped assumptions baked into `base.py`.

*Supporting research: `vd_design_notes.md` §4, §8; report 03 §2; report 11
§3, §10 will give the current install/feature facts for the candidates.*

### 6. `chromadol` vs the `chroma` backend overlap is unresolved

The ecosystem has two Chroma wrappers: `vd/backends/chroma.py` and the separate
`chromadol` package (a `dol` `MutableMapping` over a Chroma collection). The
embeddings-group ledger (report 00, Opportunity 1) flags this: they are
parallel, not stacked. Decide whether the `chroma` backend should *be*
`chromadol` (or build on it) so there is one Chroma integration to maintain.

---

## C. Install & selection guidance (the user-facing half)

### 7. No "help me install a vectorDB" capability — *(issue [#10](https://github.com/i2mint/vd/issues/10))*

`vd` should help users get set up with the vectorDBs it supports. Today the
whole surface is one-line `pip_install` strings in `_backend_metadata`. There
is no skill that walks a user through downloading / running a backend, and
server-based backends (qdrant, weaviate, milvus) need Docker/a service — not
just pip. **This is the main gap report 11 is being commissioned to fill.**
Deliverables once the report lands:
- a bundled user skill `vd-setup-backend` (per-backend install/run playbooks);
- a `vd.check_requirements(backend)` helper (diagnose + print next step);
- structured `_backend_metadata` (install modes, server-required, doc links).

### 8. Backend metadata is stale / inconsistent — *(folded into issue [#10](https://github.com/i2mint/vd/issues/10))*

In `vd/util.py`: `pinecone` → `pip install pinecone-client` (the PyPI package
was renamed to `pinecone`; `pinecone-client` is the deprecated shim). Install
strings mix `vd[...]` extras with raw client names. `pyproject.toml` defines
only a `chromadb` extra. Report 11 §6 will emit corrected, structured metadata
per provider — refresh `_backend_metadata` and the `pyproject.toml` extras from
it in one pass.

### 9. `compare.py` / `vd-backend-choose` use hardcoded 2024-era heuristics

The recommender's backend characteristics, pricing assumptions, and decision
logic are hardcoded in `compare.py` and the `vd-backend-choose` skill, rooted in
the 2024 imbed discussion #4. Report 11 §4 will provide a current decision
framework (decision tree + use-case rubric) and §3 current per-provider
pros/cons — refresh both from it. Consider moving the backend-characteristics
data out of Python and into a data file so it can be updated without a code
change (open-closed; the user's config-over-hardcoding principle).

*Supporting research: report 11 §3, §4; the 2024 baseline is
[imbed discussion #4](https://github.com/thorwhalen/imbed/discussions/4).*

---

## D. Smaller notes (low priority)

- **`_generate_id` is non-deterministic.** `util._generate_id` mixes a text
  hash with `uuid4`, so `add_documents(["same text"])` twice creates two
  documents. For a facade where upsert should be idempotent, a content-derived
  ID (`sha256` of normalized text) is the safer default; report 03 §4 and `ef`'s
  data model both use content-addressed IDs. Direct `collection[id] = …` is
  unaffected (caller gives the key); only auto-id batch input is.
- **Top-level `__init__.py` exports ~70 names.** `time_indexed` alone adds 8
  (`TimeIndexedCollection`, `WindowSlice`, `count_docs`, `mean_vector`,
  `parse_window`, `to_datetime`, `to_iso`, `TimestampLike`). Fine, but consider
  whether niche helpers belong behind `vd.time_indexed.…` rather than the flat
  top level — progressive disclosure (simple things at the top, the rest one
  level down).
- **`ef`/`vd` data-model alignment.** `vd_design_notes.md` §6 recommends
  promoting `parent_id` / `chunk_idx` to top-level `Document` fields and storing
  embedder identity (`model_id`, `dim`, `metric`) as collection-level metadata,
  with dimension-mismatch rejected loud and early. `ef` already writes
  `source_id`/`source_hash`/`config_hash` into metadata and relies on `vd` for
  staleness queries — worth doing as a pair so `ef`'s four staleness conditions
  stay simple filtered queries.
- **README roadmap claims "Coming soon" backends** (Pinecone, Weaviate, Qdrant,
  FAISS) that are still stubs a year on — keep the README honest about what is
  implemented vs planned.
- **Async.** `vd_design_notes.md` §8 lists `AsyncClient`/`AsyncCollection` as a
  recommended addition; not started. Real managed backends are async-native —
  worth deciding before the second backend lands so it is not retrofitted.

---

## Triage suggestion

If only a little time is available, the highest-leverage order is:
1. **Item 7** (install skill) + **8/9** (refresh metadata & recommender) — the
   explicit user-facing gap; unblocked as soon as report 11 lands.
2. **Item 1** (lock the contract) — cheap now, expensive after more backends.
3. **Item 5** (a second real backend — LanceDB) — turns the facade from
   claimed to proven.
4. **Items 2, 4** — correctness/cleanliness; small, well-scoped.

Everything in §D can wait.
