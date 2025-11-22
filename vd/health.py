"""
Health check and validation utilities for vd.

Provides functions to check backend health, validate configurations,
and benchmark performance.
"""

import time
from typing import Any, Optional

from vd.base import Client, Collection
from vd.util import _check_backend_available, _backend_metadata


def health_check_backend(
    backend_name: str,
    **config,
) -> dict[str, Any]:
    """
    Check if a backend is healthy and accessible.

    Parameters
    ----------
    backend_name : str
        Backend name to check
    **config
        Backend-specific configuration

    Returns
    -------
    dict
        Health report with keys:
        - status: 'healthy', 'unhealthy', or 'unavailable'
        - available: Whether backend is installed
        - registered: Whether backend is registered
        - message: Status message
        - details: Additional details (if connected successfully)

    Examples
    --------
    >>> import vd
    >>> status = vd.health_check_backend('memory')  # doctest: +SKIP
    >>> print(status['status'])  # doctest: +SKIP
    'healthy'
    """
    result = {
        'backend': backend_name,
        'available': _check_backend_available(backend_name),
        'registered': backend_name in _backend_metadata,
        'status': 'unknown',
        'message': '',
        'details': {},
    }

    # Check if available
    if not result['available']:
        result['status'] = 'unavailable'
        result['message'] = f"Backend '{backend_name}' is not installed"
        if backend_name in _backend_metadata:
            install_cmd = _backend_metadata[backend_name].get('pip_install')
            if install_cmd:
                result['message'] += f". Install with: {install_cmd}"
        return result

    # Try to connect
    try:
        from vd import connect

        # Use a mock embedding function for testing
        def mock_embed(text):
            return [0.0] * 16

        client = connect(backend_name, embedding_model=mock_embed, **config)

        # Try basic operations
        try:
            collections = list(client.list_collections())
            result['details']['collections_count'] = len(collections)
            result['status'] = 'healthy'
            result['message'] = 'Backend is operational'
        except Exception as e:
            result['status'] = 'unhealthy'
            result['message'] = f'Backend connected but operations failed: {str(e)}'

    except Exception as e:
        result['status'] = 'unhealthy'
        result['message'] = f'Failed to connect: {str(e)}'

    return result


def health_check_collection(collection: Collection) -> dict[str, Any]:
    """
    Check collection health and compute basic stats.

    Parameters
    ----------
    collection : Collection
        Collection to check

    Returns
    -------
    dict
        Health report

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> status = vd.health_check_collection(docs)  # doctest: +SKIP
    """
    from vd.analytics import validate_collection, collection_stats

    # Run validation
    validation = validate_collection(collection)

    # Get stats
    stats = collection_stats(collection)

    return {
        'status': 'healthy' if validation['valid'] else 'issues_found',
        'valid': validation['valid'],
        'total_documents': stats['total_documents'],
        'has_vectors': stats['has_vectors'],
        'issues': validation['issues'],
        'warnings': validation['warnings'],
        'stats': stats,
    }


def benchmark_search(
    collection: Collection,
    query: str,
    *,
    n_queries: int = 100,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Benchmark search performance on a collection.

    Parameters
    ----------
    collection : Collection
        Collection to benchmark
    query : str
        Query text to use
    n_queries : int, default 100
        Number of queries to run
    limit : int, default 10
        Number of results per query

    Returns
    -------
    dict
        Benchmark results with:
        - total_time: Total time for all queries
        - avg_latency: Average query latency
        - min_latency: Minimum latency
        - max_latency: Maximum latency
        - p50, p95, p99: Latency percentiles
        - queries_per_second: Throughput

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> # Add some documents...
    >>> results = vd.benchmark_search(docs, "test query", n_queries=50)  # doctest: +SKIP
    """
    latencies = []

    start_time = time.time()

    for _ in range(n_queries):
        query_start = time.time()
        list(collection.search(query, limit=limit))
        query_end = time.time()
        latencies.append(query_end - query_start)

    end_time = time.time()
    total_time = end_time - start_time

    # Sort for percentiles
    latencies_sorted = sorted(latencies)

    def percentile(data, p):
        index = int(len(data) * p / 100)
        return data[min(index, len(data) - 1)]

    return {
        'n_queries': n_queries,
        'total_time': total_time,
        'avg_latency': sum(latencies) / len(latencies),
        'min_latency': min(latencies),
        'max_latency': max(latencies),
        'p50': percentile(latencies_sorted, 50),
        'p95': percentile(latencies_sorted, 95),
        'p99': percentile(latencies_sorted, 99),
        'queries_per_second': n_queries / total_time,
    }


def benchmark_insert(
    collection: Collection,
    n_documents: int = 100,
    *,
    text_length: int = 100,
    batch_size: int = 10,
) -> dict[str, Any]:
    """
    Benchmark document insertion performance.

    Parameters
    ----------
    collection : Collection
        Collection to benchmark
    n_documents : int, default 100
        Number of documents to insert
    text_length : int, default 100
        Length of test documents
    batch_size : int, default 10
        Batch size for insertion

    Returns
    -------
    dict
        Benchmark results

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> results = vd.benchmark_insert(docs, n_documents=50)  # doctest: +SKIP
    """
    from vd.base import Document

    # Generate test documents
    test_docs = []
    for i in range(n_documents):
        text = f"Test document {i} " * (text_length // 20)
        doc = Document(
            id=f"bench_doc_{i}",
            text=text[:text_length],
            metadata={'benchmark': True, 'index': i},
        )
        test_docs.append(doc)

    # Benchmark
    start_time = time.time()

    batch = []
    for doc in test_docs:
        batch.append(doc)
        if len(batch) >= batch_size:
            collection.add_documents(batch)
            batch = []

    if batch:
        collection.add_documents(batch)

    end_time = time.time()
    total_time = end_time - start_time

    return {
        'n_documents': n_documents,
        'total_time': total_time,
        'avg_time_per_doc': total_time / n_documents,
        'documents_per_second': n_documents / total_time,
        'batch_size': batch_size,
    }
