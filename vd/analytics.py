"""
Analytics and statistics for vd collections.

Provides functions to analyze collections, find duplicates, compute statistics,
and gain insights into your vector database.
"""

from collections import Counter, defaultdict
from typing import Any, Optional

from vd.base import Collection, Document
from vd.util import cosine_similarity


def collection_stats(collection: Collection) -> dict[str, Any]:
    """
    Compute comprehensive statistics for a collection.

    Parameters
    ----------
    collection : Collection
        Collection to analyze

    Returns
    -------
    dict
        Statistics including:
        - total_documents: Number of documents
        - avg_text_length: Average text length in characters
        - min_text_length: Minimum text length
        - max_text_length: Maximum text length
        - total_chars: Total characters across all documents
        - metadata_fields: Set of all metadata fields used
        - metadata_field_counts: Count of documents with each metadata field
        - embedding_dimension: Dimension of embeddings (if available)
        - has_vectors: Number of documents with vectors

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> docs['doc1'] = ("Hello", {'category': 'greeting'})  # doctest: +SKIP
    >>> stats = vd.collection_stats(docs)  # doctest: +SKIP
    >>> print(stats['total_documents'])  # doctest: +SKIP
    1
    """
    if len(collection) == 0:
        return {
            'total_documents': 0,
            'avg_text_length': 0,
            'min_text_length': 0,
            'max_text_length': 0,
            'total_chars': 0,
            'metadata_fields': set(),
            'metadata_field_counts': {},
            'embedding_dimension': None,
            'has_vectors': 0,
        }

    text_lengths = []
    total_chars = 0
    metadata_fields = set()
    metadata_field_counts = defaultdict(int)
    embedding_dim = None
    has_vectors = 0

    for doc_id in collection:
        doc = collection[doc_id]

        # Text stats
        text_len = len(doc.text)
        text_lengths.append(text_len)
        total_chars += text_len

        # Metadata stats
        for field in doc.metadata.keys():
            metadata_fields.add(field)
            metadata_field_counts[field] += 1

        # Vector stats
        if doc.vector:
            has_vectors += 1
            if embedding_dim is None:
                embedding_dim = len(doc.vector)

    return {
        'total_documents': len(collection),
        'avg_text_length': total_chars / len(collection),
        'min_text_length': min(text_lengths),
        'max_text_length': max(text_lengths),
        'total_chars': total_chars,
        'metadata_fields': metadata_fields,
        'metadata_field_counts': dict(metadata_field_counts),
        'embedding_dimension': embedding_dim,
        'has_vectors': has_vectors,
    }


def metadata_distribution(
    collection: Collection,
    field: str,
    *,
    top_n: Optional[int] = None,
) -> dict[Any, int]:
    """
    Get the distribution of values for a metadata field.

    Parameters
    ----------
    collection : Collection
        Collection to analyze
    field : str
        Metadata field name
    top_n : int, optional
        If specified, return only the top N most common values

    Returns
    -------
    dict
        Mapping of field values to their counts

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> docs['doc1'] = ("Hello", {'category': 'A'})  # doctest: +SKIP
    >>> docs['doc2'] = ("World", {'category': 'A'})  # doctest: +SKIP
    >>> docs['doc3'] = ("Test", {'category': 'B'})  # doctest: +SKIP
    >>> dist = vd.metadata_distribution(docs, 'category')  # doctest: +SKIP
    >>> print(dist)  # doctest: +SKIP
    {'A': 2, 'B': 1}
    """
    counter = Counter()

    for doc_id in collection:
        doc = collection[doc_id]
        value = doc.metadata.get(field)
        if value is not None:
            # Handle list values
            if isinstance(value, list):
                counter.update(value)
            else:
                counter[value] += 1

    if top_n is not None:
        return dict(counter.most_common(top_n))

    return dict(counter)


def find_duplicates(
    collection: Collection,
    *,
    threshold: float = 0.95,
    method: str = 'cosine',
) -> list[tuple[str, str, float]]:
    """
    Find near-duplicate documents in a collection.

    Parameters
    ----------
    collection : Collection
        Collection to analyze
    threshold : float, default 0.95
        Similarity threshold above which documents are considered duplicates
    method : str, default 'cosine'
        Similarity method: 'cosine' or 'exact'

    Returns
    -------
    list of tuples
        List of (doc_id1, doc_id2, similarity) tuples for duplicates

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> docs['doc1'] = "Hello world"  # doctest: +SKIP
    >>> docs['doc2'] = "Hello world"  # doctest: +SKIP
    >>> duplicates = vd.find_duplicates(docs)  # doctest: +SKIP
    >>> len(duplicates) > 0  # doctest: +SKIP
    True
    """
    duplicates = []
    doc_ids = list(collection)

    for i, doc_id1 in enumerate(doc_ids):
        doc1 = collection[doc_id1]

        for doc_id2 in doc_ids[i + 1 :]:
            doc2 = collection[doc_id2]

            if method == 'exact':
                if doc1.text == doc2.text:
                    duplicates.append((doc_id1, doc_id2, 1.0))

            elif method == 'cosine':
                if doc1.vector and doc2.vector:
                    similarity = cosine_similarity(doc1.vector, doc2.vector)
                    if similarity >= threshold:
                        duplicates.append((doc_id1, doc_id2, similarity))

    return duplicates


def find_outliers(
    collection: Collection,
    *,
    n_neighbors: int = 5,
    threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """
    Find outlier documents (those dissimilar to their neighbors).

    Parameters
    ----------
    collection : Collection
        Collection to analyze
    n_neighbors : int, default 5
        Number of neighbors to consider
    threshold : float, default 0.3
        Average similarity threshold below which a document is an outlier

    Returns
    -------
    list of tuples
        List of (doc_id, avg_similarity) for outliers

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> # Add some documents...
    >>> outliers = vd.find_outliers(docs)  # doctest: +SKIP
    """
    outliers = []

    for doc_id in collection:
        doc = collection[doc_id]

        if not doc.vector:
            continue

        # Find nearest neighbors
        similarities = []
        for other_id in collection:
            if other_id == doc_id:
                continue

            other_doc = collection[other_id]
            if not other_doc.vector:
                continue

            sim = cosine_similarity(doc.vector, other_doc.vector)
            similarities.append(sim)

        if not similarities:
            continue

        # Get top n_neighbors
        similarities.sort(reverse=True)
        top_sims = similarities[:n_neighbors]
        avg_sim = sum(top_sims) / len(top_sims)

        if avg_sim < threshold:
            outliers.append((doc_id, avg_sim))

    # Sort by similarity (lowest first)
    outliers.sort(key=lambda x: x[1])

    return outliers


def sample_collection(
    collection: Collection,
    n: int,
    *,
    method: str = 'random',
    seed: Optional[int] = None,
) -> list[str]:
    """
    Sample document IDs from a collection.

    Parameters
    ----------
    collection : Collection
        Collection to sample from
    n : int
        Number of documents to sample
    method : str, default 'random'
        Sampling method: 'random', 'first', 'diverse'
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list of str
        Sampled document IDs

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> # Add 100 documents...
    >>> sample = vd.sample_collection(docs, 10, method='random')  # doctest: +SKIP
    >>> len(sample)  # doctest: +SKIP
    10
    """
    doc_ids = list(collection)

    if n >= len(doc_ids):
        return doc_ids

    if method == 'first':
        return doc_ids[:n]

    elif method == 'random':
        import random

        if seed is not None:
            random.seed(seed)
        return random.sample(doc_ids, n)

    elif method == 'diverse':
        # Sample diverse documents using greedy selection
        if len(doc_ids) == 0:
            return []

        selected = []
        remaining = set(doc_ids)

        # Start with a random document
        import random

        if seed is not None:
            random.seed(seed)

        first_id = random.choice(list(remaining))
        selected.append(first_id)
        remaining.remove(first_id)

        # Greedily select most diverse documents
        while len(selected) < n and remaining:
            best_id = None
            best_min_sim = -1

            for candidate_id in remaining:
                candidate_doc = collection[candidate_id]
                if not candidate_doc.vector:
                    continue

                # Find minimum similarity to selected documents
                min_sim = float('inf')
                for selected_id in selected:
                    selected_doc = collection[selected_id]
                    if not selected_doc.vector:
                        continue

                    sim = cosine_similarity(candidate_doc.vector, selected_doc.vector)
                    min_sim = min(min_sim, sim)

                if min_sim > best_min_sim:
                    best_min_sim = min_sim
                    best_id = candidate_id

            if best_id:
                selected.append(best_id)
                remaining.remove(best_id)
            else:
                # Fallback to random if no vectors
                best_id = random.choice(list(remaining))
                selected.append(best_id)
                remaining.remove(best_id)

        return selected

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def validate_collection(collection: Collection) -> dict[str, Any]:
    """
    Validate collection integrity and identify issues.

    Parameters
    ----------
    collection : Collection
        Collection to validate

    Returns
    -------
    dict
        Validation report with:
        - valid: Whether collection is valid
        - issues: List of issue descriptions
        - warnings: List of warning messages
        - stats: Basic stats

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> report = vd.validate_collection(docs)  # doctest: +SKIP
    >>> print(report['valid'])  # doctest: +SKIP
    True
    """
    issues = []
    warnings = []

    # Check for empty collection
    if len(collection) == 0:
        warnings.append("Collection is empty")

    # Check each document
    missing_vectors = 0
    empty_text = 0
    inconsistent_dimensions = []

    expected_dim = None

    for doc_id in collection:
        try:
            doc = collection[doc_id]

            # Check for empty text
            if not doc.text or not doc.text.strip():
                empty_text += 1
                warnings.append(f"Document {doc_id} has empty text")

            # Check vectors
            if doc.vector is None:
                missing_vectors += 1
            else:
                dim = len(doc.vector)
                if expected_dim is None:
                    expected_dim = dim
                elif dim != expected_dim:
                    inconsistent_dimensions.append((doc_id, dim))

        except Exception as e:
            issues.append(f"Error reading document {doc_id}: {str(e)}")

    # Summary warnings
    if missing_vectors > 0:
        warnings.append(f"{missing_vectors} documents missing vectors")

    if inconsistent_dimensions:
        issues.append(
            f"Inconsistent vector dimensions found: {len(inconsistent_dimensions)} documents"
        )

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
        'stats': {
            'total_documents': len(collection),
            'missing_vectors': missing_vectors,
            'empty_text': empty_text,
            'inconsistent_dimensions': len(inconsistent_dimensions),
        },
    }
