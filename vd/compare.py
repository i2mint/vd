"""
Backend comparison and recommendation tools for vd.

Provides utilities to compare different vector database backends and
recommend the best backend for specific use cases.
"""

from typing import Any, Optional

from vd.util import _backend_metadata, get_backend_info, list_available_backends


def compare_backends(
    backends: Optional[list[str]] = None,
    *,
    criteria: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Compare multiple backends across various criteria.

    Parameters
    ----------
    backends : list of str, optional
        List of backend names to compare. If not provided, compares all
        available backends.
    criteria : list of str, optional
        Specific criteria to compare: 'scalability', 'persistence',
        'cloud_support', 'local_support', 'cost', 'performance'.
        If not provided, shows all criteria.

    Returns
    -------
    dict
        Comparison results with backend names as keys

    Examples
    --------
    >>> import vd
    >>> comparison = vd.compare_backends(['memory', 'chroma'])  # doctest: +SKIP
    >>> for backend, info in comparison.items():  # doctest: +SKIP
    ...     print(f"{backend}: {info['scalability']}")
    """
    if backends is None:
        # Compare all registered backends
        backends = list(_backend_metadata.keys())

    comparison = {}
    for backend in backends:
        try:
            info = get_backend_info(backend)
            backend_data = {
                'available': info['available'],
                'description': info['description'],
                'features': info['features'],
            }

            # Add backend-specific characteristics
            if backend == 'memory':
                backend_data.update(
                    {
                        'scalability': 'Small datasets only',
                        'persistence': 'No (in-memory only)',
                        'cloud_support': 'No',
                        'local_support': 'Yes',
                        'cost': 'Free',
                        'performance': 'Fast for small datasets',
                        'best_for': 'Prototyping, testing, small datasets',
                    }
                )
            elif backend == 'chroma':
                backend_data.update(
                    {
                        'scalability': 'Medium (up to millions of vectors)',
                        'persistence': 'Yes (local files)',
                        'cloud_support': 'Limited',
                        'local_support': 'Yes',
                        'cost': 'Free (open source)',
                        'performance': 'Good for medium datasets',
                        'best_for': 'Local development, medium datasets',
                    }
                )
            elif backend == 'pinecone':
                backend_data.update(
                    {
                        'scalability': 'Very high (billions of vectors)',
                        'persistence': 'Yes (cloud)',
                        'cloud_support': 'Yes (managed service)',
                        'local_support': 'No',
                        'cost': 'Paid (with free tier)',
                        'performance': 'Excellent at scale',
                        'best_for': 'Production, large-scale applications',
                    }
                )
            elif backend == 'weaviate':
                backend_data.update(
                    {
                        'scalability': 'High (billions of vectors)',
                        'persistence': 'Yes (cloud or self-hosted)',
                        'cloud_support': 'Yes (managed or self-hosted)',
                        'local_support': 'Yes (Docker)',
                        'cost': 'Free (open source) or paid (cloud)',
                        'performance': 'Excellent',
                        'best_for': 'Production, semantic search, GraphQL',
                    }
                )
            elif backend == 'qdrant':
                backend_data.update(
                    {
                        'scalability': 'High (billions of vectors)',
                        'persistence': 'Yes (local or cloud)',
                        'cloud_support': 'Yes (managed or self-hosted)',
                        'local_support': 'Yes',
                        'cost': 'Free (open source) or paid (cloud)',
                        'performance': 'Excellent',
                        'best_for': 'Production, filtering-heavy workloads',
                    }
                )
            elif backend == 'milvus':
                backend_data.update(
                    {
                        'scalability': 'Very high (billions of vectors)',
                        'persistence': 'Yes (distributed storage)',
                        'cloud_support': 'Yes (managed or self-hosted)',
                        'local_support': 'Yes (Docker)',
                        'cost': 'Free (open source) or paid (cloud)',
                        'performance': 'Excellent at large scale',
                        'best_for': 'Enterprise, very large datasets',
                    }
                )
            elif backend == 'faiss':
                backend_data.update(
                    {
                        'scalability': 'High (in-memory)',
                        'persistence': 'Manual (save/load indexes)',
                        'cloud_support': 'No (library)',
                        'local_support': 'Yes',
                        'cost': 'Free (open source)',
                        'performance': 'Excellent (optimized)',
                        'best_for': 'Research, custom implementations',
                    }
                )

            # Filter by criteria if specified
            if criteria:
                backend_data = {
                    k: v for k, v in backend_data.items() if k in criteria or k in ['available', 'description']
                }

            comparison[backend] = backend_data

        except ValueError:
            # Skip backends that don't exist
            continue

    return comparison


def print_comparison(
    backends: Optional[list[str]] = None,
    *,
    criteria: Optional[list[str]] = None,
) -> None:
    """
    Print a formatted comparison table of backends.

    Parameters
    ----------
    backends : list of str, optional
        List of backend names to compare
    criteria : list of str, optional
        Specific criteria to compare

    Examples
    --------
    >>> import vd
    >>> vd.print_comparison(['memory', 'chroma'])  # doctest: +SKIP
    """
    comparison = compare_backends(backends, criteria=criteria)

    if not comparison:
        print("No backends to compare")
        return

    print("\n" + "=" * 80)
    print("BACKEND COMPARISON")
    print("=" * 80)

    for backend, data in comparison.items():
        status = "✓ AVAILABLE" if data['available'] else "✗ NOT AVAILABLE"
        print(f"\n{backend.upper()}: {status}")
        print("-" * 80)

        for key, value in data.items():
            if key not in ['available']:
                # Format key nicely
                display_key = key.replace('_', ' ').title()
                print(f"  {display_key:20s}: {value}")

    print("\n" + "=" * 80)


def recommend_backend(
    *,
    dataset_size: str = 'small',
    persistence_required: bool = False,
    cloud_required: bool = False,
    budget: str = 'free',
    performance_priority: str = 'balanced',
) -> dict:
    """
    Recommend backends based on requirements.

    Parameters
    ----------
    dataset_size : str, default 'small'
        Dataset size: 'small' (< 10K docs), 'medium' (10K-1M docs),
        'large' (1M-100M docs), 'very_large' (> 100M docs)
    persistence_required : bool, default False
        Whether data persistence is required
    cloud_required : bool, default False
        Whether cloud/managed service is required
    budget : str, default 'free'
        Budget constraint: 'free', 'low', 'medium', 'high'
    performance_priority : str, default 'balanced'
        Performance priority: 'speed', 'scalability', 'balanced'

    Returns
    -------
    dict
        Recommendations with 'primary', 'alternatives', and 'reasoning'

    Examples
    --------
    >>> import vd
    >>> rec = vd.recommend_backend(
    ...     dataset_size='medium',
    ...     persistence_required=True,
    ...     budget='free'
    ... )  # doctest: +SKIP
    >>> print(rec['primary'])  # doctest: +SKIP
    'chroma'
    >>> print(rec['reasoning'])  # doctest: +SKIP
    """
    recommendations = {
        'primary': None,
        'alternatives': [],
        'reasoning': [],
    }

    # Get available backends
    available = set(list_available_backends())

    # Decision logic
    if dataset_size == 'small':
        if not persistence_required:
            recommendations['primary'] = 'memory'
            recommendations['reasoning'].append(
                "Memory backend is perfect for small datasets without persistence needs"
            )
            if 'chroma' in available:
                recommendations['alternatives'].append('chroma')
        else:
            if 'chroma' in available:
                recommendations['primary'] = 'chroma'
                recommendations['reasoning'].append(
                    "ChromaDB provides persistence for small to medium datasets"
                )
            recommendations['alternatives'].append('memory')

    elif dataset_size == 'medium':
        if budget == 'free':
            if 'chroma' in available:
                recommendations['primary'] = 'chroma'
                recommendations['reasoning'].append(
                    "ChromaDB is free and handles medium datasets well"
                )
            if 'qdrant' in available:
                recommendations['alternatives'].append('qdrant')
            if 'weaviate' in available:
                recommendations['alternatives'].append('weaviate')
        else:
            if cloud_required:
                recommendations['primary'] = 'pinecone'
                recommendations['reasoning'].append(
                    "Pinecone offers managed cloud service for medium datasets"
                )
                recommendations['alternatives'].extend(['weaviate', 'qdrant'])
            else:
                if 'qdrant' in available:
                    recommendations['primary'] = 'qdrant'
                elif 'chroma' in available:
                    recommendations['primary'] = 'chroma'
                recommendations['alternatives'].extend(['weaviate', 'faiss'])

    elif dataset_size == 'large':
        if cloud_required:
            recommendations['primary'] = 'pinecone'
            recommendations['reasoning'].append(
                "Pinecone scales well for large datasets with managed service"
            )
            recommendations['alternatives'].extend(['weaviate', 'qdrant', 'milvus'])
        else:
            if performance_priority == 'speed':
                recommendations['primary'] = 'faiss'
                recommendations['reasoning'].append(
                    "FAISS provides excellent performance for large datasets"
                )
                recommendations['alternatives'].extend(['qdrant', 'milvus'])
            else:
                recommendations['primary'] = 'qdrant'
                recommendations['reasoning'].append(
                    "Qdrant balances performance and features for large datasets"
                )
                recommendations['alternatives'].extend(['milvus', 'weaviate', 'faiss'])

    elif dataset_size == 'very_large':
        if cloud_required:
            recommendations['primary'] = 'pinecone'
            recommendations['reasoning'].append(
                "Pinecone handles billions of vectors with managed infrastructure"
            )
            recommendations['alternatives'].extend(['milvus', 'weaviate'])
        else:
            recommendations['primary'] = 'milvus'
            recommendations['reasoning'].append(
                "Milvus is designed for very large scale deployments"
            )
            recommendations['alternatives'].extend(['qdrant', 'pinecone', 'weaviate'])

    # Add budget reasoning
    if budget == 'free':
        recommendations['reasoning'].append(
            "Recommendation prioritizes free/open-source options"
        )

    # Add cloud reasoning
    if cloud_required:
        recommendations['reasoning'].append(
            "Recommendation includes managed cloud services"
        )

    # Filter alternatives to only available backends
    recommendations['alternatives'] = [
        b for b in recommendations['alternatives'] if b in available or b in _backend_metadata
    ]

    # Remove duplicates and primary from alternatives
    if recommendations['primary'] in recommendations['alternatives']:
        recommendations['alternatives'].remove(recommendations['primary'])

    return recommendations


def print_recommendation(
    *,
    dataset_size: str = 'small',
    persistence_required: bool = False,
    cloud_required: bool = False,
    budget: str = 'free',
    performance_priority: str = 'balanced',
) -> None:
    """
    Print backend recommendation based on requirements.

    Parameters
    ----------
    dataset_size : str, default 'small'
        Dataset size: 'small', 'medium', 'large', 'very_large'
    persistence_required : bool, default False
        Whether data persistence is required
    cloud_required : bool, default False
        Whether cloud/managed service is required
    budget : str, default 'free'
        Budget constraint: 'free', 'low', 'medium', 'high'
    performance_priority : str, default 'balanced'
        Performance priority: 'speed', 'scalability', 'balanced'

    Examples
    --------
    >>> import vd
    >>> vd.print_recommendation(
    ...     dataset_size='large',
    ...     persistence_required=True,
    ...     cloud_required=True
    ... )  # doctest: +SKIP
    """
    rec = recommend_backend(
        dataset_size=dataset_size,
        persistence_required=persistence_required,
        cloud_required=cloud_required,
        budget=budget,
        performance_priority=performance_priority,
    )

    print("\n" + "=" * 80)
    print("BACKEND RECOMMENDATION")
    print("=" * 80)
    print(f"\nRequirements:")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Persistence required: {persistence_required}")
    print(f"  Cloud required: {cloud_required}")
    print(f"  Budget: {budget}")
    print(f"  Performance priority: {performance_priority}")

    print(f"\n{'Recommended Backend:':20s} {rec['primary'] or 'No recommendation'}")

    if rec['alternatives']:
        print(f"\nAlternatives:")
        for alt in rec['alternatives'][:5]:  # Show top 5
            print(f"  - {alt}")

    print(f"\nReasoning:")
    for reason in rec['reasoning']:
        print(f"  • {reason}")

    # Show availability info
    if rec['primary']:
        try:
            info = get_backend_info(rec['primary'])
            if not info['available']:
                print(f"\n⚠ Note: {rec['primary']} is not currently installed")
                if 'install_instructions' in info:
                    print(f"  Install with: {info['install_instructions']}")
        except ValueError:
            pass

    print("\n" + "=" * 80)


def get_backend_characteristics() -> dict[str, dict]:
    """
    Get detailed characteristics of all backends.

    Returns
    -------
    dict
        Backend characteristics including strengths, weaknesses, and use cases

    Examples
    --------
    >>> import vd
    >>> chars = vd.get_backend_characteristics()  # doctest: +SKIP
    >>> print(chars['chroma']['strengths'])  # doctest: +SKIP
    """
    return {
        'memory': {
            'strengths': [
                'Zero setup required',
                'Fast for small datasets',
                'Perfect for testing',
                'No dependencies',
            ],
            'weaknesses': [
                'No persistence',
                'Limited scalability',
                'Data lost on restart',
                'Memory constrained',
            ],
            'use_cases': [
                'Prototyping and experimentation',
                'Unit testing',
                'Temporary caching',
                'Small datasets (< 10K documents)',
            ],
        },
        'chroma': {
            'strengths': [
                'Easy to use',
                'Good documentation',
                'Local persistence',
                'Active development',
            ],
            'weaknesses': [
                'Limited scalability',
                'Single-node only',
                'Basic filtering',
                'No distributed mode',
            ],
            'use_cases': [
                'Local development',
                'Medium-sized datasets (10K-1M)',
                'RAG applications',
                'Document search',
            ],
        },
        'pinecone': {
            'strengths': [
                'Fully managed',
                'Excellent scalability',
                'Great performance',
                'Simple API',
            ],
            'weaknesses': [
                'Requires cloud account',
                'Costs money (after free tier)',
                'Vendor lock-in',
                'No local deployment',
            ],
            'use_cases': [
                'Production applications',
                'Large-scale search',
                'Recommendation systems',
                'Enterprise deployments',
            ],
        },
        'weaviate': {
            'strengths': [
                'GraphQL interface',
                'Rich filtering',
                'Hybrid search',
                'Multi-modal support',
            ],
            'weaknesses': [
                'Complex setup',
                'Requires Docker/K8s',
                'Learning curve',
                'Resource intensive',
            ],
            'use_cases': [
                'Semantic search',
                'Knowledge graphs',
                'Multi-modal search',
                'Complex filtering needs',
            ],
        },
        'qdrant': {
            'strengths': [
                'Excellent filtering',
                'Good performance',
                'Rust-based (fast)',
                'Rich query language',
            ],
            'weaknesses': [
                'Relatively new',
                'Smaller community',
                'Requires setup',
            ],
            'use_cases': [
                'Production applications',
                'Filter-heavy workloads',
                'High-performance search',
                'Custom deployments',
            ],
        },
        'milvus': {
            'strengths': [
                'Very scalable',
                'Distributed architecture',
                'Good for billion-scale',
                'Enterprise features',
            ],
            'weaknesses': [
                'Complex deployment',
                'Resource intensive',
                'Steep learning curve',
                'Requires infrastructure',
            ],
            'use_cases': [
                'Enterprise scale',
                'Billion-vector datasets',
                'Distributed deployments',
                'Mission-critical applications',
            ],
        },
        'faiss': {
            'strengths': [
                'Excellent performance',
                'Highly optimized',
                'Research-grade',
                'Flexible',
            ],
            'weaknesses': [
                'Low-level library',
                'No built-in persistence',
                'Manual index management',
                'No server mode',
            ],
            'use_cases': [
                'Research',
                'Custom implementations',
                'Performance-critical apps',
                'Embedding exploration',
            ],
        },
    }
