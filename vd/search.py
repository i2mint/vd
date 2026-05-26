"""
Advanced search utilities for vd.

Provides functions for multi-query search, search result merging, and
other advanced search patterns.
"""

import math
import re
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from vd.base import Collection, SearchResult, SupportsHybrid, Vector


def multi_query_search(
    collection: Collection,
    queries: list[str],
    *,
    limit: int = 10,
    combine: str = "interleave",
    filter: Optional[dict] = None,
    **kwargs,
) -> Iterator[SearchResult]:
    """
    Search with multiple queries and combine results.

    Parameters
    ----------
    collection : Collection
        Collection to search
    queries : list of str
        Multiple query strings
    limit : int, default 10
        Total number of results to return
    combine : str, default 'interleave'
        How to combine results:
        - 'interleave': Interleave results from each query
        - 'concatenate': Concatenate all results
        - 'union': Remove duplicates across queries
        - 'best': Take best results across all queries
    filter : dict, optional
        Metadata filter
    **kwargs
        Additional search options

    Yields
    ------
    dict
        Search results

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> results = vd.multi_query_search(  # doctest: +SKIP
    ...     docs,
    ...     ["What is AI?", "How does ML work?"],
    ...     limit=10
    ... )
    """
    # Get results for each query
    all_results = []
    for query in queries:
        results = list(collection.search(query, limit=limit, filter=filter, **kwargs))
        all_results.append(results)

    if combine == "interleave":
        # Interleave results
        combined = []
        max_len = max(len(r) for r in all_results) if all_results else 0

        for i in range(max_len):
            for results in all_results:
                if i < len(results):
                    combined.append(results[i])

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for item in combined:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)

        yield from unique[:limit]

    elif combine == "concatenate":
        # Simply concatenate
        for results in all_results:
            yield from results

    elif combine == "union":
        # Combine and remove duplicates, keeping best score
        by_id = {}
        for results in all_results:
            for item in results:
                doc_id = item["id"]
                if doc_id not in by_id or item["score"] > by_id[doc_id]["score"]:
                    by_id[doc_id] = item

        # Sort by score
        sorted_results = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)
        yield from sorted_results[:limit]

    elif combine == "best":
        # Get best results across all queries
        all_items = []
        for results in all_results:
            all_items.extend(results)

        # Sort by score
        all_items.sort(key=lambda x: x["score"], reverse=True)

        # Remove duplicates
        seen = set()
        unique = []
        for item in all_items:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)

        yield from unique[:limit]

    else:
        raise ValueError(f"Unknown combine method: {combine}")


def search_similar_to_document(
    collection: Collection,
    doc_id: str,
    *,
    limit: int = 10,
    exclude_self: bool = True,
    filter: Optional[dict] = None,
    **kwargs,
) -> Iterator[SearchResult]:
    """
    Find documents similar to a specific document.

    Parameters
    ----------
    collection : Collection
        Collection to search
    doc_id : str
        ID of the reference document
    limit : int, default 10
        Number of similar documents to return
    exclude_self : bool, default True
        Whether to exclude the reference document from results
    filter : dict, optional
        Metadata filter
    **kwargs
        Additional search options

    Yields
    ------
    dict
        Search results

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> similar = vd.search_similar_to_document(docs, 'doc1', limit=5)  # doctest: +SKIP
    """
    # Get the reference document
    ref_doc = collection[doc_id]

    # Search using its vector or text
    if ref_doc.vector:
        query = ref_doc.vector
    else:
        query = ref_doc.text

    # Search
    for result in collection.search(query, limit=limit + 1, filter=filter, **kwargs):
        # Optionally exclude self
        if exclude_self and result["id"] == doc_id:
            continue

        yield result


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    *,
    k: int = 60,
) -> list[SearchResult]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.

    RRF is a simple yet effective way to combine rankings from multiple sources.

    Parameters
    ----------
    result_lists : list of lists
        Multiple lists of search results
    k : int, default 60
        Constant for RRF formula (typically 60)

    Returns
    -------
    list
        Combined and re-ranked results

    Examples
    --------
    >>> results1 = [{'id': 'doc1', 'score': 0.9}, {'id': 'doc2', 'score': 0.8}]
    >>> results2 = [{'id': 'doc2', 'score': 0.95}, {'id': 'doc3', 'score': 0.7}]
    >>> combined = reciprocal_rank_fusion([results1, results2])  # doctest: +SKIP
    """
    # Compute RRF scores
    rrf_scores = {}

    for results in result_lists:
        for rank, item in enumerate(results, 1):
            doc_id = item["id"]
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0.0, "item": item}

            rrf_scores[doc_id]["score"] += rrf_score

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    # Return results with RRF score
    results = []
    for doc_id, data in sorted_items:
        item = data["item"].copy()
        item["rrf_score"] = data["score"]
        results.append(item)

    return results


def search_with_feedback(
    collection: Collection,
    query: str,
    *,
    relevant_ids: Optional[list[str]] = None,
    irrelevant_ids: Optional[list[str]] = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
    limit: int = 10,
    **kwargs,
) -> Iterator[SearchResult]:
    """
    Search with relevance feedback (Rocchio algorithm).

    Adjusts the query based on relevant and irrelevant documents.

    Parameters
    ----------
    collection : Collection
        Collection to search
    query : str
        Original query
    relevant_ids : list of str, optional
        IDs of relevant documents
    irrelevant_ids : list of str, optional
        IDs of irrelevant documents
    alpha : float, default 1.0
        Weight for original query
    beta : float, default 0.75
        Weight for relevant documents
    gamma : float, default 0.15
        Weight for irrelevant documents (negative)
    limit : int, default 10
        Number of results
    **kwargs
        Additional search options

    Yields
    ------
    dict
        Search results

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> results = vd.search_with_feedback(  # doctest: +SKIP
    ...     docs,
    ...     "machine learning",
    ...     relevant_ids=['doc1', 'doc2'],
    ...     irrelevant_ids=['doc5']
    ... )
    """
    # Get query vector
    # Note: This is a simplified version - in practice, we'd need access to the
    # embedding function
    raise NotImplementedError(
        "Relevance feedback requires direct access to embedding function. "
        "This will be implemented in a future version."
    )


def deduplicate_results(
    results: Iterator[SearchResult],
    *,
    key: str = "id",
    keep: str = "first",
) -> Iterator[SearchResult]:
    """
    Remove duplicate results.

    Parameters
    ----------
    results : iterator
        Search results
    key : str, default 'id'
        Field to check for duplicates
    keep : str, default 'first'
        Which duplicate to keep: 'first' or 'highest_score'

    Yields
    ------
    dict
        Deduplicated results

    Examples
    --------
    >>> results = [
    ...     {'id': 'doc1', 'score': 0.9},
    ...     {'id': 'doc1', 'score': 0.8},
    ...     {'id': 'doc2', 'score': 0.7}
    ... ]
    >>> unique = list(deduplicate_results(iter(results)))
    >>> len(unique)
    2
    """
    seen = {}

    for result in results:
        key_value = result[key]

        if key_value not in seen:
            seen[key_value] = result
            yield result
        elif keep == "highest_score":
            if result["score"] > seen[key_value]["score"]:
                seen[key_value] = result
                yield result


# --------------------------------------------------------------------------- #
# Hybrid search — top-level entry + client-side BM25 fallback
# --------------------------------------------------------------------------- #

#: Minimum default for `k_dense`/`k_lexical` when callers don't override.
_HYBRID_OVERFETCH_FLOOR = 50

#: Simple ASCII-word tokenizer for the built-in BM25 fallback. Adapters that
#: want better tokenization (stemming, CJK, etc.) should pass a custom
#: ``lexical_search`` callable.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens — used by the built-in BM25 fallback."""
    return _TOKEN_RE.findall(text.lower())


def bm25_lexical_search(
    collection: Collection,
    query_text: str,
    *,
    limit: int = 10,
    filter: Optional[dict] = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    """
    Brute-force BM25 lexical search over a vd collection's stored ``text``.

    Iterates every document in ``collection``, tokenizes its ``text`` field,
    and computes Okapi BM25 scores against ``query_text``. Used as the default
    lexical side of :func:`hybrid_search` when a collection does not implement
    :class:`SupportsHybrid`.

    Cost is **O(N)** in the collection size — fine for prototypes and
    collections up to ~100k documents. For larger workloads, either switch to
    a backend with native hybrid search (weaviate, elasticsearch, redis, …)
    or pass a custom ``lexical_search`` callable to :func:`hybrid_search` that
    consults a real text index.

    Parameters
    ----------
    collection : Collection
        Any vd Collection. Documents whose ``text`` is empty contribute zero
        score and are filtered out of the result.
    query_text : str
        The lexical query.
    limit : int
        Maximum number of results.
    filter : dict, optional
        Canonical ``vd`` metadata filter. Applied client-side via
        :func:`vd.filters.matches_filter`.
    k1, b : float
        BM25 hyperparameters. Defaults match the standard Okapi BM25.

    Returns
    -------
    list[dict]
        Result dicts in the same shape as :meth:`Collection.search` —
        ``{"id", "text", "score", "metadata"}`` — sorted by descending score.

    Examples
    --------
    >>> import vd
    >>> c = vd.connect('memory').create_collection('t', dimension=2)
    >>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
    >>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
    >>> hits = bm25_lexical_search(c, 'quick fox', limit=1)
    >>> hits[0]['id']
    'a'
    """
    from vd.filters import matches_filter

    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    # Pass 1: tokenize once, compute document frequencies and lengths.
    docs: list[tuple[str, str, dict, list[str]]] = []
    df: dict[str, int] = {}
    for doc_id in collection:
        doc = collection[doc_id]
        if filter is not None and not matches_filter(doc.metadata or {}, filter):
            continue
        tokens = _tokenize(doc.text or "")
        if not tokens:
            continue
        docs.append((doc_id, doc.text, dict(doc.metadata or {}), tokens))
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    if not docs:
        return []

    n_docs = len(docs)
    avg_len = sum(len(toks) for _, _, _, toks in docs) / n_docs

    # Pass 2: BM25 scoring (Okapi).
    query_terms = set(query_tokens)
    idf = {
        term: math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        for term in query_terms
        if term in df
    }

    scored: list[SearchResult] = []
    for doc_id, text, metadata, tokens in docs:
        score = 0.0
        doc_len = len(tokens)
        tf: dict[str, int] = {}
        for tok in tokens:
            if tok in idf:
                tf[tok] = tf.get(tok, 0) + 1
        if not tf:
            continue
        for term, freq in tf.items():
            numer = freq * (k1 + 1)
            denom = freq + k1 * (1 - b + b * doc_len / avg_len)
            score += idf[term] * numer / denom
        scored.append(
            {"id": doc_id, "text": text, "score": score, "metadata": metadata}
        )

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:limit]


def _rrf_fuse(
    result_lists: Iterable[list[SearchResult]],
    *,
    rrf_k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over result lists, returning the top ``limit``."""
    scores: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            doc_id = item["id"]
            contribution = 1.0 / (rrf_k + rank)
            if doc_id not in scores:
                # Keep the first occurrence's payload for text/metadata.
                scores[doc_id] = {"item": dict(item), "score": 0.0}
            scores[doc_id]["score"] += contribution
    fused = []
    for doc_id, entry in scores.items():
        item = entry["item"]
        # Replace the source score with the fused RRF score so downstream
        # consumers can rely on result["score"] = fused score.
        item["score"] = entry["score"]
        fused.append(item)
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:limit]


def hybrid_search(
    collection: Collection,
    query: Union[str, Vector],
    *,
    query_text: Optional[str] = None,
    limit: int = 10,
    filter: Optional[dict] = None,
    k_dense: Optional[int] = None,
    k_lexical: Optional[int] = None,
    rrf_k: int = 60,
    lexical_search: Optional[Callable[..., list[SearchResult]]] = None,
    egress: Optional[Callable[[SearchResult], Any]] = None,
    **kwargs,
) -> Iterator[SearchResult]:
    """
    Hybrid (dense + lexical) search that works on any vd Collection.

    Dispatches to the collection's native ``hybrid_search`` when it implements
    :class:`~vd.SupportsHybrid` (efficient, server-side). Otherwise fuses the
    collection's own dense :meth:`~vd.Collection.search` with a client-side
    lexical scan (default: :func:`bm25_lexical_search`) via **Reciprocal Rank
    Fusion**.

    The portable contract is RRF. Backend-specific knobs (weighted blend
    ``alpha``, fusion-type variants, native ranker choices) are accepted via
    ``**kwargs`` and forwarded to the adapter when it has a native
    implementation; they are ignored by the client-side fallback.

    Parameters
    ----------
    collection : Collection
        Any vd Collection — native-hybrid or not.
    query : str or list[float]
        Query text (embedded by the collection if it has an embedder) or a
        pre-computed query vector. When ``query`` is a vector, ``query_text``
        is **required**.
    query_text : str, optional
        Explicit text for the lexical side. Defaults to ``query`` when
        ``query`` is a string.
    limit : int
        Number of fused results to return.
    filter : dict, optional
        Canonical ``vd`` metadata filter, applied to both sub-searches.
    k_dense, k_lexical : int, optional
        How many results to fetch from each sub-search before fusion. Default
        is ``max(4 * limit, 50)`` for each side. Widen for higher recall.
    rrf_k : int
        Reciprocal Rank Fusion constant (typically 60).
    lexical_search : callable, optional
        Custom ``lexical_search(collection, query_text, *, limit, filter,
        **kwargs) -> list[SearchResult]``. Defaults to
        :func:`bm25_lexical_search`. Used only on the fallback path.
    egress : callable, optional
        Per-result transform applied before yielding.
    **kwargs
        Extra options. On the native path they are forwarded to the adapter
        (e.g. ``alpha=0.7`` on weaviate). On the fallback path they are
        ignored.

    Yields
    ------
    dict
        Fused result dicts. ``score`` is the RRF score on the fallback path,
        or the adapter's fused score on the native path.

    Examples
    --------
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
    """
    k_dense_eff = k_dense if k_dense is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    k_lexical_eff = (
        k_lexical if k_lexical is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    )

    # Native path.
    if isinstance(collection, SupportsHybrid):
        for hit in collection.hybrid_search(
            query,
            query_text=query_text,
            limit=limit,
            filter=filter,
            k_dense=k_dense_eff,
            k_lexical=k_lexical_eff,
            rrf_k=rrf_k,
            egress=egress,
            **kwargs,
        ):
            yield hit
        return

    # Fallback: dense via collection.search() + lexical via callable, fused by RRF.
    # Resolve the text for the lexical side. (The dense side accepts the original
    # `query` directly — the collection's search() does its own embed/vet.)
    if isinstance(query, str):
        text = query_text if query_text is not None else query
    else:
        if query_text is None:
            raise ValueError(
                "hybrid_search needs a `query_text` for the lexical side when "
                "`query` is a vector. Either pass query_text=..., or pass "
                "`query` as a string and let the embedder handle both."
            )
        text = query_text
    if not text:
        raise ValueError("hybrid_search needs a non-empty lexical query string.")

    dense_hits = list(
        collection.search(query, limit=k_dense_eff, filter=filter)
    )

    lex_fn = lexical_search if lexical_search is not None else bm25_lexical_search
    lex_hits = lex_fn(collection, text, limit=k_lexical_eff, filter=filter)

    fused = _rrf_fuse([dense_hits, lex_hits], rrf_k=rrf_k, limit=limit)
    for hit in fused:
        yield egress(hit) if egress is not None else hit
