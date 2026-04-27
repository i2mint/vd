---
name: vd-search
description: >-
  Advanced-search tooling for the vd package. Use this skill when the user goes
  beyond a single basic .search() call with vd — metadata filters with
  MongoDB-style operators, multi-query searches, reciprocal rank fusion,
  finding documents similar to an existing one, deduplicating result sets, or
  searching by a pre-computed query vector.
audience: users
---

# Advanced search with vd

This skill covers everything past `collection.search("query", limit=N)`. For
the basic call, see `vd-quickstart`.

The advanced helpers all live at the top level (`vd.multi_query_search`,
`vd.reciprocal_rank_fusion`, etc.) and take a `Collection` as the first arg.

## Filter syntax (MongoDB-style)

```python
# Equality (sugar — these are equivalent)
docs.search("query", filter={'category': 'tech'})
docs.search("query", filter={'category': {'$eq': 'tech'}})

# Comparison
docs.search("query", filter={'year': {'$gte': 2020}})
docs.search("query", filter={'views': {'$lt': 1000}})

# Membership
docs.search("query", filter={'tags': {'$in': ['python', 'ai']}})

# Negation
docs.search("query", filter={'category': {'$ne': 'spam'}})

# Logical
docs.search("query", filter={
    '$and': [
        {'year': {'$gte': 2020}},
        {'category': 'tech'},
    ],
})

docs.search("query", filter={
    '$or': [
        {'priority': 'high'},
        {'pinned': True},
    ],
})
```

Supported operators:

| Operator | Meaning |
|---|---|
| `$eq` | equal (default when value is a scalar) |
| `$ne` | not equal |
| `$gt`, `$gte`, `$lt`, `$lte` | comparison |
| `$in` | value is in list |
| `$and`, `$or` | logical combinators (take a list of subfilters) |

A bare `{'field': value}` is sugar for `{'field': {'$eq': value}}`. Multiple
top-level fields combine with implicit AND:

```python
filter={'category': 'tech', 'year': {'$gte': 2020}}
# means: category == 'tech' AND year >= 2020
```

The same filter dict is accepted by `multi_query_search` and
`search_similar_to_document` via their `filter` kwarg.

## Multi-query search

Run several queries and combine the results — useful when the user is unsure
how to phrase a question or when you want to broaden recall.

```python
results = vd.multi_query_search(
    docs,
    queries=["machine learning", "neural networks", "deep learning"],
    limit=10,
    combine='best',   # 'interleave' | 'concatenate' | 'union' | 'best'
    filter=None,
)
```

`combine` modes:

- `'interleave'` (default) — round-robin one from each query's results.
  Good when you want diversity across phrasings.
- `'concatenate'` — first query's results, then second query's, etc.
- `'union'` — deduplicate by `id`, keep the highest score per doc.
- `'best'` — for each doc found, keep the best score across queries; sort by
  that score. Usually the right choice for "give me the top-10 across all
  these phrasings."

`limit` is the **final** result count after combining, not per-query. The
function pulls more than `limit` per query under the hood and trims.

## Reciprocal rank fusion (RRF)

When you already have multiple ranked result lists (from different queries,
backends, or scoring methods) and want a single combined ranking, RRF is the
simplest robust merge:

```python
list_a = list(docs.search("query A", limit=20))
list_b = list(docs.search("query B", limit=20))

merged = vd.reciprocal_rank_fusion([list_a, list_b], k=60)
# returns a list[SearchResult] sorted by RRF score
```

`k=60` is the standard constant — leave it alone unless you have a reason. RRF
is rank-based, so it doesn't care that scores from different lists aren't
comparable. Prefer RRF over a hand-rolled score average.

## Finding documents similar to an existing document

```python
similar = vd.search_similar_to_document(
    docs,
    doc_id='doc1',
    limit=10,
    exclude_self=True,    # default — don't return doc1 itself
    filter={'category': 'tech'},   # optional metadata filter
)
```

Internally this re-runs search using `doc1`'s vector. Useful for "more like
this" recommendations. Set `exclude_self=False` only if you need a sanity
check that the doc is in fact most similar to itself.

## Deduplicating results

```python
results = list(docs.search("query", limit=50))
unique = vd.deduplicate_results(results, key='id', keep='first')
```

`key` defaults to `'id'`. Use a metadata key (e.g., `'parent_id'`) when
results came from chunked documents and you want one result per parent
document. `keep='first'` keeps the highest-ranked occurrence; `keep='last'`
keeps the lowest. `deduplicate_results` returns an iterator.

## Searching by a pre-computed vector

If you already have a query embedding (e.g., from caching or a different
model), pass the vector directly to `.search`:

```python
query_vec = my_embed("how do neural networks learn?")
results = docs.search(query_vec, limit=5)
```

`search`'s first arg accepts `str | list[float]`. The Vector path skips
embedding entirely. Make sure the vector dimension matches the collection's
embedding model.

## Egress functions

Every search-returning helper in `vd` accepts (or composes with) `egress` to
project results before they're yielded. Built-ins:

```python
vd.text_only(r)        # -> str
vd.id_only(r)          # -> str
vd.id_and_score(r)     # -> (id, score)
vd.id_text_score(r)    # -> (id, text, score)
```

Or pass any `Callable[[dict], Any]`. The `multi_query_search`,
`search_similar_to_document`, etc. helpers don't take `egress` directly — wrap
them with `map(egress, ...)` if you need projection there.

## Putting it together: a "best of both worlds" search

```python
import vd

# 1. Run a focused vector search
vector_hits = list(docs.search(
    "transformer attention mechanism",
    limit=20,
    filter={'year': {'$gte': 2017}},
))

# 2. Run a broader paraphrase
paraphrase_hits = list(docs.search(
    "how do transformers attend to inputs",
    limit=20,
    filter={'year': {'$gte': 2017}},
))

# 3. Fuse the rankings
merged = vd.reciprocal_rank_fusion([vector_hits, paraphrase_hits], k=60)

# 4. Dedup by parent document
final = list(vd.deduplicate_results(merged, key='id', keep='first'))[:10]
```

## Common gotchas

- **Filters depend on backend support.** `memory` evaluates filters in
  Python after retrieval. `chroma` translates supported operators into its
  native query and may reject unsupported combinations. If a filter raises
  on `chroma`, simplify (split `$or` into multiple queries + RRF).
- **`multi_query_search` returns an iterator, not a list.** Wrap in `list()`
  if you need to use it twice.
- **RRF needs lists, not iterators.** Materialize each input list with
  `list(docs.search(...))` before passing to `reciprocal_rank_fusion`.
- **`search_similar_to_document` raises `KeyError`** if `doc_id` doesn't
  exist. Check `doc_id in docs` first if input is user-supplied.
- **Vector queries must match dimension.** A vector from a different embedding
  model than the collection's will either error or silently return garbage
  matches.
- **Don't mistake "score" for "probability".** It's a similarity score, scale
  depends on the backend (cosine ∈ [-1, 1] for memory; chroma may return a
  distance instead of a similarity). Don't threshold across backends without
  checking.

## See also

- `vd-quickstart` — basic `.search` and egress functions
- `vd-ingest` — chunking / metadata that powers good filters
- `vd-ops` — `find_duplicates` and `find_outliers` (related, but for whole
  collections rather than a single query result set)
