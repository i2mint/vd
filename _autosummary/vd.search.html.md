# vd.search

Advanced search utilities for vd.

Provides functions for multi-query search, search result merging, and
other advanced search patterns.

### Functions

| [`bm25_lexical_search`](#vd.search.bm25_lexical_search)(collection, query_text, \*)    | Brute-force BM25 lexical search over a vd collection's stored `text`.   |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`deduplicate_results`](#vd.search.deduplicate_results)(results, \*[, key, keep])      | Remove duplicate results.                                               |
| [`hybrid_search`](#vd.search.hybrid_search)(collection, query, \*[, ...])        | Hybrid (dense + lexical) search that works on any vd Collection.        |
| [`multi_query_search`](#vd.search.multi_query_search)(collection, queries, \*[, ...]) | Search with multiple queries and combine results.                       |
| [`reciprocal_rank_fusion`](#vd.search.reciprocal_rank_fusion)(result_lists, \*[, k])      | Combine multiple result lists using Reciprocal Rank Fusion.             |
| [`search_similar_to_document`](#vd.search.search_similar_to_document)(collection, doc_id, \*) | Find documents similar to a specific document.                          |
| [`search_with_feedback`](#vd.search.search_with_feedback)(collection, query, \*[, ...]) | Search with relevance feedback (Rocchio algorithm).                     |

### Classes

| [`BM25Index`](#vd.search.BM25Index)(collection, \*[, filter, tokenize])   | A reusable Okapi BM25 index over a vd collection's stored `text`.   |
|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|

### *class* vd.search.BM25Index(collection, \*, filter=None, tokenize=<function \_tokenize>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reusable Okapi BM25 index over a vd collection’s stored `text`.

Builds the **query-independent** term statistics — per-document token
lists, document frequencies, document lengths, and the mean length — once
in `__init__()`, then answers many queries against them via
[`search()`](#vd.search.BM25Index.search). This is the build-once / query-many companion to
[`bm25_lexical_search()`](#vd.search.bm25_lexical_search) (which builds a throwaway index for a single
query): for batch evaluation or any repeated querying of the same
collection it turns an O(N · Q) workload (re-tokenizing every document on
every query) into O(N + scoring · Q).

Construction is **O(N)** in the collection size; each [`search()`](#vd.search.BM25Index.search) is
O(matching documents). Fine for prototypes and collections up to ~100k
documents; for larger workloads switch to a backend with a native text
index (weaviate, elasticsearch, redis, lancedb, …).

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection (or mapping-like `id -> obj` exposing `.text` and
    `.metadata`). Documents whose `text` is empty contribute zero score
    and are dropped at build time.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter, applied **once** at build time (via
    [`vd.filters.matches_filter()`](vd.filters.html.md#vd.filters.matches_filter)) so the index covers only the
    matching documents and its statistics reflect that subset.
  * **tokenize** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Tokenizer (default: lowercased `\w+` tokens). Pass a custom one for
    stemming, CJK, etc.

### Examples

```pycon
>>> import vd
>>> c = vd.connect('memory').create_collection('t', dimension=2)
>>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
>>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
>>> index = vd.BM25Index(c)
>>> index.search('quick fox', limit=1)[0]['id']
'a'
```

#### search(query_text, , limit=10, k1=1.5, b=0.75)

Okapi BM25 scores for `query_text` over the indexed documents.

Returns result dicts in the same shape as `Collection.search()` —
`{"id", "text", "score", "metadata"}` — sorted by descending score.
`k1` / `b` are the standard Okapi hyperparameters (scoring-time, so
one index can be queried with different settings).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.search.bm25_lexical_search(collection, query_text, , limit=10, filter=None, k1=1.5, b=0.75)

Brute-force BM25 lexical search over a vd collection’s stored `text`.

Builds a throwaway [`BM25Index`](#vd.search.BM25Index) over `collection` and runs a single
query against it. Used as the default lexical side of [`hybrid_search()`](#vd.search.hybrid_search)
when a collection does not implement `SupportsHybrid`.

Cost is **O(N)** in the collection size — fine for prototypes and
collections up to ~100k documents. \*\*For repeated queries over the same
collection, build a\*\* [`BM25Index`](#vd.search.BM25Index) **once and call**
[`BM25Index.search()`](#vd.search.BM25Index.search) **per query** instead of calling this function in a
loop — the term statistics are then computed once rather than on every call.
For larger workloads, switch to a backend with native hybrid search
(weaviate, elasticsearch, redis, lancedb, …) or pass a custom `lexical_search`
callable to [`hybrid_search()`](#vd.search.hybrid_search) that consults a real text index.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection. Documents whose `text` is empty contribute zero
    score and are filtered out of the result.
  * **query_text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The lexical query.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of results.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter. Applied client-side via
    [`vd.filters.matches_filter()`](vd.filters.html.md#vd.filters.matches_filter).
  * **k1** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – BM25 hyperparameters. Defaults match the standard Okapi BM25.
  * **b** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – BM25 hyperparameters. Defaults match the standard Okapi BM25.
* **Returns:**
  Result dicts in the same shape as `Collection.search()` —
  `{"id", "text", "score", "metadata"}` — sorted by descending score.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> c = vd.connect('memory').create_collection('t', dimension=2)
>>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
>>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
>>> hits = bm25_lexical_search(c, 'quick fox', limit=1)
>>> hits[0]['id']
'a'
```

### vd.search.deduplicate_results(results, , key='id', keep='first')

Remove duplicate results.

* **Parameters:**
  * **results** ([`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Search results
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field to check for duplicates
  * **keep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which duplicate to keep: ‘first’ or ‘highest_score’
* **Yields:**
  *dict* – Deduplicated results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> results = [
...     {'id': 'doc1', 'score': 0.9},
...     {'id': 'doc1', 'score': 0.8},
...     {'id': 'doc2', 'score': 0.7}
... ]
>>> unique = list(deduplicate_results(iter(results)))
>>> len(unique)
2
```

### vd.search.hybrid_search(collection, query, , query_text=None, limit=10, filter=None, k_dense=None, k_lexical=None, rrf_k=60, lexical_search=None, egress=None, \*\*kwargs)

Hybrid (dense + lexical) search that works on any vd Collection.

Dispatches to the collection’s native `hybrid_search` when it implements
[`SupportsHybrid`](vd.html.md#vd.SupportsHybrid) (efficient, server-side). Otherwise fuses the
collection’s own dense [`search()`](vd.html.md#vd.Collection.search) with a client-side
lexical scan (default: [`bm25_lexical_search()`](#vd.search.bm25_lexical_search)) via \*\*Reciprocal Rank
Fusion\*\*.

The portable contract is RRF. Backend-specific knobs (weighted blend
`alpha`, fusion-type variants, native ranker choices) are accepted via
`**kwargs` and forwarded to the adapter when it has a native
implementation; they are ignored by the client-side fallback.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection — native-hybrid or not.
  * **query** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Query text (embedded by the collection if it has an embedder) or a
    pre-computed query vector. When `query` is a vector, `query_text`
    is **required**.
  * **query_text** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Explicit text for the lexical side. Defaults to `query` when
    `query` is a string.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of fused results to return.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter, applied to both sub-searches.
  * **k_dense** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – How many results to fetch from each sub-search before fusion. Default
    is `max(4 * limit, 50)` for each side. Widen for higher recall.
  * **k_lexical** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – How many results to fetch from each sub-search before fusion. Default
    is `max(4 * limit, 50)` for each side. Widen for higher recall.
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Reciprocal Rank Fusion constant (typically 60).
  * **lexical_search** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]]) – Custom `lexical_search(collection, query_text, *, limit, filter,
    \*\*kwargs) -> list[SearchResult]`. Defaults to
    [`bm25_lexical_search()`](#vd.search.bm25_lexical_search). Used only on the fallback path; on the
    native path it is ignored with a `UserWarning`.
  * **egress** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Per-result transform applied before yielding.
  * **\*\*kwargs** – Extra options. On the native path they are forwarded to the adapter
    (e.g. `alpha=0.7` on weaviate). On the fallback path they are
    ignored.
* **Yields:**
  *dict* – Fused result dicts. `score` is the RRF score on the fallback path,
  or the adapter’s fused score on the native path.
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> col = client.create_collection('docs', dimension=2)
>>> col['a'] = vd.Document(id='a', text='cats purr',
...                        vector=[1.0, 0.0])
>>> col['b'] = vd.Document(id='b', text='dogs bark',
...                        vector=[0.0, 1.0])
>>> hits = list(vd.hybrid_search(col, [0.9, 0.1], query_text='cats',
...                              limit=1))
>>> hits[0]['id']
'a'
```

### vd.search.multi_query_search(collection, queries, , limit=10, combine='interleave', filter=None, \*\*kwargs)

Search with multiple queries and combine results.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to search
  * **queries** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Multiple query strings
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Total number of results to return
  * **combine** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – 

    How to combine results:
    - ’interleave’: Interleave results from each query
    - ’concatenate’: Concatenate all results
    - ’union’: Remove duplicates across queries
    - ’best’: Take best results across all queries
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Metadata filter
  * **\*\*kwargs** – Additional search options
* **Yields:**
  *dict* – Search results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> results = vd.multi_query_search(
...     docs,
...     ["What is AI?", "How does ML work?"],
...     limit=10
... )
```

### vd.search.reciprocal_rank_fusion(result_lists, , k=60)

Combine multiple result lists using Reciprocal Rank Fusion.

RRF is a simple yet effective way to combine rankings from multiple sources.

* **Parameters:**
  * **result_lists** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]) – Multiple lists of search results
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Constant for RRF formula (typically 60)
* **Returns:**
  Combined and re-ranked results
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> results1 = [{'id': 'doc1', 'score': 0.9}, {'id': 'doc2', 'score': 0.8}]
>>> results2 = [{'id': 'doc2', 'score': 0.95}, {'id': 'doc3', 'score': 0.7}]
>>> combined = reciprocal_rank_fusion([results1, results2])
```

### vd.search.search_similar_to_document(collection, doc_id, , limit=10, exclude_self=True, filter=None, \*\*kwargs)

Find documents similar to a specific document.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to search
  * **doc_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ID of the reference document
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of similar documents to return
  * **exclude_self** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to exclude the reference document from results
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Metadata filter
  * **\*\*kwargs** – Additional search options
* **Yields:**
  *dict* – Search results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> similar = vd.search_similar_to_document(docs, 'doc1', limit=5)
```

### vd.search.search_with_feedback(collection, query, , relevant_ids=None, irrelevant_ids=None, alpha=1.0, beta=0.75, gamma=0.15, limit=10, \*\*kwargs)

Search with relevance feedback (Rocchio algorithm).

Adjusts the query based on relevant and irrelevant documents.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to search
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Original query
  * **relevant_ids** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – IDs of relevant documents
  * **irrelevant_ids** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – IDs of irrelevant documents
  * **alpha** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Weight for original query
  * **beta** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Weight for relevant documents
  * **gamma** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Weight for irrelevant documents (negative)
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of results
  * **\*\*kwargs** – Additional search options
* **Yields:**
  *dict* – Search results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> results = vd.search_with_feedback(
...     docs,
...     "machine learning",
...     relevant_ids=['doc1', 'doc2'],
...     irrelevant_ids=['doc5']
... )
```
