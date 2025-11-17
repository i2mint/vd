"""
Utility functions and facades for the vd package.

This module provides:
- Backend registration system
- Connection factory function
- Utility functions for common operations
- Egress functions for search result transformation
"""

import hashlib
import uuid
from typing import Any, Callable, Optional

from vd.base import (
    BaseBackend,
    Client,
    Document,
    DocumentInput,
    SearchResult,
    Vector,
)

# Backend registry
_backends: dict[str, type[BaseBackend]] = {}


def register_backend(name: str):
    """
    Decorator to register a backend implementation.

    This allows backends to be dynamically registered and accessed via the
    connect() function.

    Parameters
    ----------
    name : str
        Name identifier for the backend (e.g., 'memory', 'chroma', 'pinecone')

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_backend('custom')  # doctest: +SKIP
    ... class CustomBackend(BaseBackend):
    ...     pass
    """

    def decorator(backend_class: type[BaseBackend]):
        _backends[name] = backend_class
        return backend_class

    return decorator


def get_backend(name: str) -> type[BaseBackend]:
    """
    Get a registered backend class by name.

    Parameters
    ----------
    name : str
        Backend identifier

    Returns
    -------
    type[BaseBackend]
        The backend class

    Raises
    ------
    ValueError
        If backend is not registered
    """
    if name not in _backends:
        available = ', '.join(_backends.keys())
        raise ValueError(
            f"Backend '{name}' not found. Available backends: {available}"
        )
    return _backends[name]


def list_backends() -> list[str]:
    """
    List all registered backend names.

    Returns
    -------
    list of str
        Names of all registered backends

    Examples
    --------
    >>> backends = list_backends()  # doctest: +SKIP
    >>> 'memory' in backends  # doctest: +SKIP
    True
    """
    return list(_backends.keys())


def connect(
    backend: str,
    *,
    embedding_model: Optional[str | Callable] = None,
    **backend_kwargs,
) -> Client:
    """
    Connect to a vector database backend.

    This is the main entry point for creating a client to interact with
    vector databases. It uses a factory pattern to instantiate the appropriate
    backend based on the backend name.

    Parameters
    ----------
    backend : str
        Backend identifier ('memory', 'chroma', 'pinecone', etc.)
    embedding_model : str or callable, optional
        Embedding model specification. Can be:
        - Model name string (e.g., 'text-embedding-3-small')
        - Callable that takes text and returns a vector
        - None to use the default from imbed
    **backend_kwargs
        Backend-specific configuration (API keys, URLs, persist_directory, etc.)

    Returns
    -------
    Client
        A client instance for the specified backend

    Raises
    ------
    ValueError
        If backend is not registered

    Examples
    --------
    >>> # Connect to in-memory backend
    >>> client = connect('memory')  # doctest: +SKIP
    >>>
    >>> # Connect to ChromaDB with persistence
    >>> client = connect('chroma', persist_directory='./data')  # doctest: +SKIP
    >>>
    >>> # Connect with custom embedding model
    >>> client = connect(  # doctest: +SKIP
    ...     'memory',
    ...     embedding_model='text-embedding-3-large'
    ... )
    """
    backend_class = get_backend(backend)

    # Get embedding function
    embed_func = _get_embedding_function(embedding_model)

    # Instantiate and return backend
    return backend_class(embedding_model=embed_func, **backend_kwargs)


def _get_embedding_function(
    embedding_model: Optional[str | Callable] = None,
) -> Callable[[str], Vector]:
    """
    Get the embedding function to use.

    Parameters
    ----------
    embedding_model : str, callable, or None
        Embedding model specification

    Returns
    -------
    callable
        Function that takes text and returns a vector
    """
    if callable(embedding_model):
        # Already a function, use it directly
        return embedding_model

    # Import imbed to get embedding function
    try:
        from imbed import Embed
    except ImportError:
        raise ImportError(
            "The 'imbed' package is required for embedding generation. "
            "Install it with: pip install imbed"
        )

    # Create Embed instance
    embed = Embed(model=embedding_model) if embedding_model else Embed()

    # Return a function that uses the Embed instance
    return embed


# Utility functions for document handling


def normalize_document_input(
    doc_input: DocumentInput,
    *,
    auto_id: bool = True,
) -> Document:
    """
    Normalize various document input formats to a Document object.

    Parameters
    ----------
    doc_input : DocumentInput
        Input in one of the supported formats:
        - str: Just text
        - (text, id): Text with ID
        - (text, metadata): Text with metadata
        - (text, id, metadata): Full specification
        - Document: Already a Document object
    auto_id : bool, default True
        If True, auto-generate ID when not provided

    Returns
    -------
    Document
        Normalized Document object

    Examples
    --------
    >>> doc = normalize_document_input("Hello world")
    >>> doc.text
    'Hello world'
    >>> doc.id  # doctest: +SKIP
    '...'  # Auto-generated ID
    >>>
    >>> doc = normalize_document_input(("Hello", "doc1"))
    >>> doc.id
    'doc1'
    >>> doc.text
    'Hello'
    >>>
    >>> doc = normalize_document_input(("Hello", {'category': 'test'}))
    >>> doc.metadata
    {'category': 'test'}
    """
    if isinstance(doc_input, Document):
        return doc_input

    if isinstance(doc_input, str):
        # Just text
        doc_id = _generate_id(doc_input) if auto_id else None
        return Document(id=doc_id, text=doc_input)

    if isinstance(doc_input, tuple):
        if len(doc_input) == 2:
            text, second = doc_input
            if isinstance(second, str):
                # (text, id)
                return Document(id=second, text=text)
            else:
                # (text, metadata)
                doc_id = _generate_id(text) if auto_id else None
                return Document(id=doc_id, text=text, metadata=second)
        elif len(doc_input) == 3:
            # (text, id, metadata)
            text, doc_id, metadata = doc_input
            return Document(id=doc_id, text=text, metadata=metadata)

    raise ValueError(f"Invalid document input format: {type(doc_input)}")


def _generate_id(text: str, *, prefix: str = 'doc') -> str:
    """
    Generate a unique ID for a document based on its text.

    Uses a hash of the text combined with a UUID to ensure uniqueness.

    Parameters
    ----------
    text : str
        Text to generate ID from
    prefix : str, default 'doc'
        Prefix for the generated ID

    Returns
    -------
    str
        Generated document ID

    Examples
    --------
    >>> doc_id = _generate_id("Hello world")
    >>> doc_id.startswith('doc_')
    True
    """
    # Use hash of text + short uuid for reasonable uniqueness
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    unique_suffix = str(uuid.uuid4())[:8]
    return f"{prefix}_{text_hash}_{unique_suffix}"


# Egress functions for search result transformation


def text_only(result: SearchResult) -> str:
    """
    Extract only the text from a search result.

    Parameters
    ----------
    result : dict
        Search result dictionary

    Returns
    -------
    str
        The text content

    Examples
    --------
    >>> result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
    >>> text_only(result)
    'Hello'
    """
    return result['text']


def id_only(result: SearchResult) -> str:
    """
    Extract only the ID from a search result.

    Parameters
    ----------
    result : dict
        Search result dictionary

    Returns
    -------
    str
        The document ID

    Examples
    --------
    >>> result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
    >>> id_only(result)
    'doc1'
    """
    return result['id']


def id_and_score(result: SearchResult) -> tuple[str, float]:
    """
    Extract ID and score from a search result.

    Parameters
    ----------
    result : dict
        Search result dictionary

    Returns
    -------
    tuple of (str, float)
        Document ID and similarity score

    Examples
    --------
    >>> result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
    >>> id_and_score(result)
    ('doc1', 0.9)
    """
    return result['id'], result['score']


def id_text_score(result: SearchResult) -> tuple[str, str, float]:
    """
    Extract ID, text, and score from a search result.

    Parameters
    ----------
    result : dict
        Search result dictionary

    Returns
    -------
    tuple of (str, str, float)
        Document ID, text, and similarity score

    Examples
    --------
    >>> result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
    >>> id_text_score(result)
    ('doc1', 'Hello', 0.9)
    """
    return result['id'], result['text'], result['score']


# Similarity/distance utilities


def cosine_similarity(vec1: Vector, vec2: Vector) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : list of float
        First vector
    vec2 : list of float
        Second vector

    Returns
    -------
    float
        Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)

    Examples
    --------
    >>> v1 = [1.0, 0.0, 0.0]
    >>> v2 = [1.0, 0.0, 0.0]
    >>> cosine_similarity(v1, v2)
    1.0
    >>> v3 = [0.0, 1.0, 0.0]
    >>> cosine_similarity(v1, v3)
    0.0
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = sum(a * a for a in vec1) ** 0.5
    mag2 = sum(b * b for b in vec2) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)


def euclidean_distance(vec1: Vector, vec2: Vector) -> float:
    """
    Compute Euclidean distance between two vectors.

    Parameters
    ----------
    vec1 : list of float
        First vector
    vec2 : list of float
        Second vector

    Returns
    -------
    float
        Euclidean distance (0.0 = identical, larger = more different)

    Examples
    --------
    >>> v1 = [1.0, 0.0, 0.0]
    >>> v2 = [1.0, 0.0, 0.0]
    >>> euclidean_distance(v1, v2)
    0.0
    >>> v3 = [0.0, 1.0, 0.0]
    >>> euclidean_distance(v1, v3)  # doctest: +ELLIPSIS
    1.414...
    """
    return sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
