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


class BM25Index:
    """A reusable Okapi BM25 index over a vd collection's stored ``text``.

    Builds the **query-independent** term statistics — postings lists
    (``term -> [(doc, tf)]``), document lengths, and the mean length — once
    in :meth:`__init__`, then answers many queries against them via
    :meth:`search`. This is the build-once / query-many companion to
    :func:`bm25_lexical_search` (which builds a throwaway index for a single
    query): for batch evaluation or any repeated querying of the same
    collection it turns an O(N · Q) workload (re-tokenizing every document on
    every query) into O(N + scoring · Q).

    Construction is **O(N)** in the collection size; each :meth:`search` is
    O(matching documents). Fine for prototypes and collections up to ~100k
    documents; for larger workloads switch to a backend with a native text
    index (weaviate, elasticsearch, redis, lancedb, …).

    Parameters
    ----------
    collection : Collection
        Any vd Collection (or mapping-like ``id -> obj`` exposing ``.text`` and
        ``.metadata``). Documents whose ``text`` is empty contribute zero score
        and are dropped at build time.
    filter : dict, optional
        Canonical ``vd`` metadata filter, applied **once** at build time (via
        :func:`vd.filters.matches_filter`) so the index covers only the
        matching documents and its statistics reflect that subset.
    tokenize : Callable[[str], list[str]], optional
        Tokenizer (default: lowercased ``\\w+`` tokens). Pass a custom one for
        stemming, CJK, etc.
    k1, b : float
        The Okapi hyperparameters :meth:`search` uses when it is given none.
    engine : str
        ``"python"`` (default): the pure-Python postings index, no dependency,
        ``k1``/``b`` free at query time. ``"bm25s"``: the scores are computed
        once at build time by `bm25s <https://github.com/xhluca/bm25s>`_ (MIT;
        ``pip install vd[bm25s]``), which answers a query with common words
        tens to hundreds of times faster past ~10k documents. Same scores (to
        float32 rounding) and the same ranking; ``k1``/``b`` are then fixed at
        build time. See :data:`BM25_ENGINES`.

    Examples
    --------
    >>> import vd
    >>> c = vd.connect('memory').create_collection('t', dimension=2)
    >>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
    >>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
    >>> index = vd.BM25Index(c)
    >>> index.search('quick fox', limit=1)[0]['id']
    'a'
    """

    def __init__(
        self,
        collection: Collection,
        *,
        filter: Optional[dict] = None,
        tokenize: Callable[[str], list[str]] = _tokenize,
        k1: float = 1.5,
        b: float = 0.75,
        engine: str = "python",
    ):
        from vd.filters import matches_filter

        if engine not in BM25_ENGINES:
            raise ValueError(
                f"unknown BM25 engine {engine!r} (known: {', '.join(BM25_ENGINES)})"
            )
        self._tokenize = tokenize
        self.k1, self.b, self.engine = k1, b, engine
        # Build (query-independent): tokenize once; keep postings
        # ``term -> [(doc_idx, tf, first_pos)]`` plus per-document lengths.
        # ``first_pos`` is the index of the term's first occurrence in the
        # document, so scoring can sum terms in the same order as the original
        # per-document scan (bit-identical scores, unchanged tie order).
        docs: list[tuple[str, str, dict]] = []
        collection_tokens: list[list[str]] = []
        doc_lens: list[int] = []
        postings: dict[str, list[tuple[int, int, int]]] = {}
        for doc_id in collection:
            doc = collection[doc_id]
            if filter is not None and not matches_filter(doc.metadata or {}, filter):
                continue
            tokens = tokenize(doc.text or "")
            if not tokens:
                continue
            doc_idx = len(docs)
            docs.append((doc_id, doc.text, dict(doc.metadata or {})))
            doc_lens.append(len(tokens))
            if engine != "python":  # the engine builds its own statistics
                collection_tokens.append(tokens)
                continue
            seen: dict[str, list[int]] = {}  # term -> [tf, first_pos]
            for pos, tok in enumerate(tokens):
                entry = seen.get(tok)
                if entry is None:
                    seen[tok] = [1, pos]
                else:
                    entry[0] += 1
            for term, (freq, first_pos) in seen.items():
                postings.setdefault(term, []).append((doc_idx, freq, first_pos))
        self._docs = docs
        self._doc_lens = doc_lens
        self._postings = postings
        self._n_docs = len(docs)
        self._avg_len = sum(doc_lens) / self._n_docs if docs else 0.0
        self._engine = (
            BM25_ENGINES[engine](collection_tokens, k1=k1, b=b)
            if engine != "python" and docs
            else None
        )

    def __len__(self) -> int:
        """Number of non-empty, filter-surviving documents in the index."""
        return self._n_docs

    def search(
        self,
        query_text: str,
        *,
        limit: int = 10,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> list[SearchResult]:
        """Okapi BM25 scores for ``query_text`` over the indexed documents.

        Returns result dicts in the same shape as :meth:`Collection.search` —
        ``{"id", "text", "score", "metadata"}`` — sorted by descending score,
        ties in document order. ``k1`` / ``b`` are the standard Okapi
        hyperparameters, defaulting to the index's. The ``"python"`` engine
        scores at query time, so one index can be queried with different
        settings; the ``"bm25s"`` engine scored at build time and raises
        ``ValueError`` for others.
        """
        k1 = self.k1 if k1 is None else k1
        b = self.b if b is None else b
        query_tokens = self._tokenize(query_text)
        if not query_tokens or not self._docs:
            return []
        if self._engine is not None:
            if (k1, b) != (self.k1, self.b):
                raise ValueError(
                    f"the {self.engine!r} engine scored with k1={self.k1}, b={self.b} "
                    f"at build time; build another BM25Index for k1={k1}, b={b}"
                )
            return [
                {"id": d[0], "text": d[1], "score": score, "metadata": d[2]}
                for idx, score in self._engine(query_tokens, limit=limit)
                for d in (self._docs[idx],)
            ]

        # Query-dependent: idf for the query terms, then Okapi scoring of only
        # the documents that appear in those terms' postings.
        n_docs, avg_len, postings = self._n_docs, self._avg_len, self._postings
        doc_lens = self._doc_lens
        idf: dict[str, float] = {}
        for term in set(query_tokens):
            plist = postings.get(term)
            if plist:
                df = len(plist)
                idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        hits: dict[int, list[tuple[int, str, int]]] = {}
        for term in idf:
            for doc_idx, freq, first_pos in postings[term]:
                hits.setdefault(doc_idx, []).append((first_pos, term, freq))

        scored: list[SearchResult] = []
        for doc_idx in sorted(hits):  # document order, so score ties keep it
            doc_len = doc_lens[doc_idx]
            score = 0.0
            for _, term, freq in sorted(hits[doc_idx]):  # first-appearance order
                numer = freq * (k1 + 1)
                denom = freq + k1 * (1 - b + b * doc_len / avg_len)
                score += idf[term] * numer / denom
            doc_id, text, metadata = self._docs[doc_idx]
            scored.append(
                {"id": doc_id, "text": text, "score": score, "metadata": metadata}
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]


def _bm25s_engine(corpus_tokens: list[list[str]], *, k1: float, b: float):
    """A ``query_tokens, limit -> [(doc_idx, score)]`` scorer built by bm25s.

    bm25s's Lucene variant drops the constant ``(k1 + 1)`` from the numerator, so
    its scores are multiplied back by it: the result is vd's own Okapi score (to
    float32 rounding), which matters to callers that calibrate on score values.
    Ties are put back in document order, as the pure-Python engine returns them.
    """
    try:
        import bm25s
        import numpy as np
    except ImportError as error:
        raise ImportError(
            "BM25Index(engine='bm25s') needs bm25s: pip install 'vd[bm25s]'"
        ) from error

    retriever = bm25s.BM25(k1=k1, b=b, method="lucene")
    retriever.index(corpus_tokens, show_progress=False)
    n_docs, scale = len(corpus_tokens), k1 + 1

    def score(query_tokens: list[str], *, limit: int) -> list[tuple[int, float]]:
        # each distinct term once, as the pure-Python engine counts it
        known = [t for t in dict.fromkeys(query_tokens) if t in retriever.vocab_dict]
        if not known or limit <= 0:
            return []
        scores = retriever.get_scores(known)
        hit = np.flatnonzero(scores > 0)
        order = hit[np.lexsort((hit, -scores[hit]))][:limit]  # score desc, then doc order
        return [(int(i), float(scores[i]) * scale) for i in order]

    return score


#: The scoring engines :class:`BM25Index` can build: name -> ``(corpus_tokens, *,
#: k1, b) -> scorer``. ``"python"`` is the built-in postings index itself.
BM25_ENGINES: dict = {"python": None, "bm25s": _bm25s_engine}


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

    Builds a throwaway :class:`BM25Index` over ``collection`` and runs a single
    query against it. Used as the default lexical side of :func:`hybrid_search`
    when a collection does not implement :class:`SupportsHybrid`.

    Cost is **O(N)** in the collection size — fine for prototypes and
    collections up to ~100k documents. **For repeated queries over the same
    collection, build a** :class:`BM25Index` **once and call**
    :meth:`BM25Index.search` **per query** instead of calling this function in a
    loop — the term statistics are then computed once rather than on every call.
    For larger workloads, switch to a backend with native hybrid search
    (weaviate, elasticsearch, redis, lancedb, …) or pass a custom ``lexical_search``
    callable to :func:`hybrid_search` that consults a real text index.

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
    return BM25Index(collection, filter=filter).search(
        query_text, limit=limit, k1=k1, b=b
    )


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


def _warn_lexical_search_ignored(collection: Any) -> None:
    """Warn that a custom ``lexical_search`` is ignored on a native hybrid path."""
    import warnings

    warnings.warn(
        f"lexical_search= is ignored: {type(collection).__name__} runs hybrid "
        f"search natively (it satisfies vd.SupportsHybrid), so the backend's own "
        f"text index is the lexical side. Call vd.bm25_lexical_search / your "
        f"callable and vd.reciprocal_rank_fusion yourself to force a custom "
        f"lexical side.",
        UserWarning,
        stacklevel=3,
    )


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
        :func:`bm25_lexical_search`. Used only on the fallback path; on the
        native path it is ignored with a ``UserWarning``.
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
    k_dense_eff = (
        k_dense if k_dense is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    )
    k_lexical_eff = (
        k_lexical if k_lexical is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
    )

    # Native path.
    if isinstance(collection, SupportsHybrid):
        if lexical_search is not None:
            _warn_lexical_search_ignored(collection)
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

    dense_hits = list(collection.search(query, limit=k_dense_eff, filter=filter))

    lex_fn = lexical_search if lexical_search is not None else bm25_lexical_search
    lex_hits = lex_fn(collection, text, limit=k_lexical_eff, filter=filter)

    fused = _rrf_fuse([dense_hits, lex_hits], rrf_k=rrf_k, limit=limit)
    for hit in fused:
        yield egress(hit) if egress is not None else hit
