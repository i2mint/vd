"""
VectorDB facades - A unified interface for vector databases.

The `vd` package provides a consistent, Pythonic API for interacting with
various vector databases. It abstracts away the specifics of each database's
API to offer a database-agnostic interface for semantic search operations.

Key Components
--------------
- connect(): Factory function to create database clients
- Document: Standardized document representation
- Collection: MutableMapping interface for document collections
- Client: Protocol for database connections

Examples
--------
>>> import vd
>>>
>>> # Connect to a backend
>>> client = vd.connect('memory')  # doctest: +SKIP
>>>
>>> # Create a collection
>>> docs = client.create_collection('my_docs')  # doctest: +SKIP
>>>
>>> # Add documents (dict-like syntax)
>>> docs['doc1'] = "This is a test document"  # doctest: +SKIP
>>> docs['doc2'] = ("Another document", {'category': 'test'})  # doctest: +SKIP
>>>
>>> # Search
>>> results = docs.search("test query", limit=5)  # doctest: +SKIP
>>> for result in results:  # doctest: +SKIP
...     print(result['id'], result['score'])

Available Backends
------------------
- 'memory': In-memory storage (always available)
- 'chroma': ChromaDB (requires: pip install chromadb)

To list all available backends:
>>> vd.list_backends()  # doctest: +SKIP
['memory', 'chroma']
"""

# Import backends to trigger registration
import vd.backends  # noqa: F401

# Public API
from vd.base import (
    Client,
    Collection,
    Document,
    Filter,
    SearchResult,
    StaticIndexError,
    Vector,
)
from vd.util import (
    connect,
    get_backend_info,
    get_install_instructions,
    id_and_score,
    id_only,
    id_text_score,
    list_all_backends,
    list_available_backends,
    list_backends,
    print_backends_table,
    text_only,
)

# Import/Export utilities
from vd.io import (
    export_collection,
    export_to_directory,
    export_to_json,
    export_to_jsonl,
    import_collection,
    import_from_directory,
    import_from_json,
    import_from_jsonl,
)

# Migration utilities
from vd.migration import (
    copy_collection,
    migrate_client,
    migrate_collection,
)

# Analytics utilities
from vd.analytics import (
    collection_stats,
    find_duplicates,
    find_outliers,
    metadata_distribution,
    sample_collection,
    validate_collection,
)

# Text preprocessing
from vd.text import (
    chunk_documents,
    chunk_text,
    clean_text,
    extract_metadata,
    normalize_whitespace,
    truncate_text,
)

# Health checks
from vd.health import (
    benchmark_insert,
    benchmark_search,
    health_check_backend,
    health_check_collection,
)

# Advanced search
from vd.search import (
    deduplicate_results,
    multi_query_search,
    reciprocal_rank_fusion,
    search_similar_to_document,
)

# Configuration management
from vd.config import (
    connect_from_config,
    create_example_config,
    load_config,
    save_config,
)

# Backend comparison and recommendation
from vd.compare import (
    compare_backends,
    get_backend_characteristics,
    print_comparison,
    print_recommendation,
    recommend_backend,
)

__version__ = '0.1.0'

__all__ = [
    # Main entry point
    'connect',
    # Core types
    'Document',
    'Client',
    'Collection',
    # Type aliases
    'Vector',
    'Filter',
    'SearchResult',
    # Exceptions
    'StaticIndexError',
    # Backend discovery
    'list_backends',
    'list_available_backends',
    'list_all_backends',
    'print_backends_table',
    'get_backend_info',
    'get_install_instructions',
    # Egress functions
    'text_only',
    'id_only',
    'id_and_score',
    'id_text_score',
    # Import/Export
    'export_collection',
    'import_collection',
    'export_to_jsonl',
    'import_from_jsonl',
    'export_to_json',
    'import_from_json',
    'export_to_directory',
    'import_from_directory',
    # Migration
    'migrate_collection',
    'migrate_client',
    'copy_collection',
    # Analytics
    'collection_stats',
    'metadata_distribution',
    'find_duplicates',
    'find_outliers',
    'sample_collection',
    'validate_collection',
    # Text preprocessing
    'chunk_text',
    'chunk_documents',
    'clean_text',
    'normalize_whitespace',
    'truncate_text',
    'extract_metadata',
    # Health checks
    'health_check_backend',
    'health_check_collection',
    'benchmark_search',
    'benchmark_insert',
    # Advanced search
    'multi_query_search',
    'search_similar_to_document',
    'reciprocal_rank_fusion',
    'deduplicate_results',
    # Configuration
    'load_config',
    'save_config',
    'connect_from_config',
    'create_example_config',
    # Backend comparison
    'compare_backends',
    'print_comparison',
    'recommend_backend',
    'print_recommendation',
    'get_backend_characteristics',
]