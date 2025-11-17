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
    id_and_score,
    id_only,
    id_text_score,
    list_backends,
    text_only,
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
    # Utilities
    'list_backends',
    'text_only',
    'id_only',
    'id_and_score',
    'id_text_score',
]