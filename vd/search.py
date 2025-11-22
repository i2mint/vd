"""
Advanced search utilities for vd.

Provides functions for multi-query search, search result merging, and
other advanced search patterns.
"""

from typing import Any, Callable, Iterator, Optional, Union

from vd.base import Collection, SearchResult


def multi_query_search(
    collection: Collection,
    queries: list[str],
    *,
    limit: int = 10,
    combine: str = 'interleave',
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

    if combine == 'interleave':
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
            if item['id'] not in seen:
                seen.add(item['id'])
                unique.append(item)

        yield from unique[:limit]

    elif combine == 'concatenate':
        # Simply concatenate
        for results in all_results:
            yield from results

    elif combine == 'union':
        # Combine and remove duplicates, keeping best score
        by_id = {}
        for results in all_results:
            for item in results:
                doc_id = item['id']
                if doc_id not in by_id or item['score'] > by_id[doc_id]['score']:
                    by_id[doc_id] = item

        # Sort by score
        sorted_results = sorted(by_id.values(), key=lambda x: x['score'], reverse=True)
        yield from sorted_results[:limit]

    elif combine == 'best':
        # Get best results across all queries
        all_items = []
        for results in all_results:
            all_items.extend(results)

        # Sort by score
        all_items.sort(key=lambda x: x['score'], reverse=True)

        # Remove duplicates
        seen = set()
        unique = []
        for item in all_items:
            if item['id'] not in seen:
                seen.add(item['id'])
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
        if exclude_self and result['id'] == doc_id:
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
            doc_id = item['id']
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {'score': 0.0, 'item': item}

            rrf_scores[doc_id]['score'] += rrf_score

    # Sort by RRF score
    sorted_items = sorted(
        rrf_scores.items(), key=lambda x: x[1]['score'], reverse=True
    )

    # Return results with RRF score
    results = []
    for doc_id, data in sorted_items:
        item = data['item'].copy()
        item['rrf_score'] = data['score']
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
    key: str = 'id',
    keep: str = 'first',
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
        elif keep == 'highest_score':
            if result['score'] > seen[key_value]['score']:
                seen[key_value] = result
                yield result
