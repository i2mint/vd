"""
Core protocols, base classes, and data models for the vd package.

This module defines the fundamental abstractions used throughout the vd package:
- Document: Standardized representation of searchable documents
- Client: Protocol for vector database connections
- Collection: Protocol for document collections (MutableMapping-based)
- BaseBackend: Abstract base class for backend implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Union,
)

# Type aliases
Text = str
TextKey = str  # Document ID / URI
Metadata = dict[str, Any]
Vector = list[float]
VectorMapping = Mapping[TextKey, Vector]
SearchResult = dict[str, Any]  # {id, text, score, metadata, ...}
Filter = dict[str, Any]  # MongoDB-style filter syntax

# Document input can be specified in multiple flexible ways
DocumentInput = Union[
    str,  # Just text (ID auto-generated)
    tuple[str, str],  # (text, id)
    tuple[str, Metadata],  # (text, metadata)
    tuple[str, str, Metadata],  # (text, id, metadata)
    'Document',  # Full document object
]


@dataclass
class Document:
    """
    Standardized document representation.

    A Document represents a searchable text segment with its embedding vector
    and associated metadata.

    Parameters
    ----------
    id : str
        Unique identifier (URI to source). Acts as the key in collections.
    text : str
        The text content to be embedded and searched.
    vector : list[float], optional
        Embedding vector. If None, will be auto-generated using the collection's
        embedding model.
    metadata : dict[str, Any]
        Associated metadata for filtering and retrieval.

    Examples
    --------
    >>> doc = Document(id="doc1", text="Hello world")
    >>> doc.id
    'doc1'
    >>> doc.text
    'Hello world'
    >>> doc.metadata
    {}

    >>> doc_with_meta = Document(
    ...     id="doc2",
    ...     text="AI article",
    ...     metadata={'category': 'tech', 'year': 2024}
    ... )
    >>> doc_with_meta.metadata['category']
    'tech'
    """

    id: str
    text: str
    vector: Optional[Vector] = None
    metadata: Metadata = field(default_factory=dict)


class Collection(Protocol):
    """
    Protocol for a collection of searchable documents.

    A Collection follows the MutableMapping interface, storing Document objects
    and enabling intuitive CRUD operations via dict-like syntax while providing
    semantic search capabilities.

    The collection automatically handles embedding generation when documents
    are added without pre-computed vectors.

    Examples
    --------
    >>> # Add documents (dict-like syntax)
    >>> collection['doc1'] = "This is a test document"
    >>> collection['doc2'] = Document(id='doc2', text='Another doc')
    >>>
    >>> # Retrieve documents
    >>> doc = collection['doc1']
    >>> doc.text
    'This is a test document'
    >>>
    >>> # Delete documents
    >>> del collection['doc1']
    >>>
    >>> # Iterate over document IDs
    >>> list(collection)  # doctest: +SKIP
    ['doc2']
    >>>
    >>> # Search the collection
    >>> results = collection.search('test query', limit=5)  # doctest: +SKIP
    >>> for result in results:  # doctest: +SKIP
    ...     print(result['id'], result['score'])
    """

    # MutableMapping interface methods
    def __setitem__(self, key: str, value: Union[str, Document]) -> None:
        """
        Add or update a document in the collection.

        Parameters
        ----------
        key : str
            Document ID
        value : str or Document
            Document content. If str, creates a Document with this text.
        """
        ...

    def __getitem__(self, key: str) -> Document:
        """
        Retrieve a document by ID.

        Parameters
        ----------
        key : str
            Document ID

        Returns
        -------
        Document
            The document with the specified ID

        Raises
        ------
        KeyError
            If document ID not found
        """
        ...

    def __delitem__(self, key: str) -> None:
        """
        Delete a document from the collection.

        Parameters
        ----------
        key : str
            Document ID to delete

        Raises
        ------
        KeyError
            If document ID not found
        """
        ...

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over document IDs.

        Yields
        ------
        str
            Document IDs in the collection
        """
        ...

    def __len__(self) -> int:
        """
        Get the number of documents in the collection.

        Returns
        -------
        int
            Number of documents
        """
        ...

    # Search methods
    def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> Iterator[SearchResult]:
        """
        Search the collection for similar documents.

        Parameters
        ----------
        query : str or list of float
            Query text (will be embedded) or pre-computed query vector
        limit : int, default 10
            Maximum number of results to return
        filter : dict, optional
            Metadata filter using unified filter syntax (MongoDB-style)
        egress : callable, optional
            Function to transform results. If None, returns full result dict.
        **kwargs
            Backend-specific options (e.g., alpha=0.5 for hybrid search)

        Yields
        ------
        dict or transformed result
            Search results with keys: 'id', 'text', 'score', 'metadata'
            (or transformed by egress function)

        Examples
        --------
        >>> # Basic search
        >>> docs = collection.search("machine learning", limit=5)  # doctest: +SKIP
        >>> for doc in docs:  # doctest: +SKIP
        ...     print(doc['id'], doc['score'])
        >>>
        >>> # With metadata filter
        >>> docs = collection.search(  # doctest: +SKIP
        ...     "AI research",
        ...     filter={'year': {'$gte': 2020}}
        ... )
        >>>
        >>> # With custom egress to extract just text
        >>> texts = collection.search(  # doctest: +SKIP
        ...     "neural networks",
        ...     egress=lambda r: r['text']
        ... )
        """
        ...

    # Batch operations
    def add_documents(
        self,
        documents: Iterator[DocumentInput],
        *,
        batch_size: int = 100,
    ) -> None:
        """
        Batch add documents to the collection.

        This method efficiently adds multiple documents in batches, which is
        typically faster than adding documents one at a time.

        Parameters
        ----------
        documents : iterable of DocumentInput
            Documents to add. Each can be:
            - str: Just text (ID auto-generated)
            - (text, id): Text with specific ID
            - (text, metadata): Text with metadata (ID auto-generated)
            - (text, id, metadata): Full specification
            - Document: Full document object
        batch_size : int, default 100
            Number of documents to process in each batch

        Examples
        --------
        >>> collection.add_documents([  # doctest: +SKIP
        ...     "First article about AI",
        ...     ("Second article", "doc2"),
        ...     ("Third article", {'category': 'tech'})
        ... ])
        """
        ...

    def upsert(self, document: Document) -> None:
        """
        Insert or update a document (idempotent operation).

        This is equivalent to __setitem__ but takes a Document object.

        Parameters
        ----------
        document : Document
            Document to insert or update
        """
        self[document.id] = document


class Client(Protocol):
    """
    Protocol for vector database clients.

    A Client manages connections to vector database backends and provides
    methods to create, retrieve, and manage collections.

    Examples
    --------
    >>> import vd  # doctest: +SKIP
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> collection = client.create_collection('my_docs')  # doctest: +SKIP
    >>> collection['doc1'] = "Hello world"  # doctest: +SKIP
    """

    def create_collection(
        self,
        name: str,
        *,
        schema: Optional[dict] = None,
        **kwargs,
    ) -> Collection:
        """
        Create a new collection.

        Parameters
        ----------
        name : str
            Name of the collection
        schema : dict, optional
            Schema definition (backend-specific)
        **kwargs
            Additional backend-specific options

        Returns
        -------
        Collection
            The newly created collection

        Raises
        ------
        ValueError
            If collection already exists
        """
        ...

    def get_collection(self, name: str) -> Collection:
        """
        Get an existing collection.

        Parameters
        ----------
        name : str
            Name of the collection

        Returns
        -------
        Collection
            The requested collection

        Raises
        ------
        KeyError
            If collection does not exist
        """
        ...

    def list_collections(self) -> Iterator[str]:
        """
        List all collection names.

        Yields
        ------
        str
            Collection names
        """
        ...

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection.

        Parameters
        ----------
        name : str
            Name of the collection to delete

        Raises
        ------
        KeyError
            If collection does not exist
        """
        ...


class BaseBackend(ABC):
    """
    Abstract base class for vector database backends.

    Backend implementations should inherit from this class and implement
    all abstract methods.

    Parameters
    ----------
    embedding_model : callable
        Function to generate embeddings from text
    **config
        Backend-specific configuration
    """

    def __init__(
        self,
        *,
        embedding_model: Callable[[str], Vector],
        **config,
    ):
        """
        Initialize the backend.

        Parameters
        ----------
        embedding_model : callable
            Function that takes text and returns an embedding vector
        **config
            Backend-specific configuration options
        """
        self.embedding_model = embedding_model
        self.config = config

    @abstractmethod
    def create_collection(
        self,
        name: str,
        *,
        schema: Optional[dict] = None,
        **kwargs,
    ) -> Collection:
        """Create a new collection."""
        ...

    @abstractmethod
    def get_collection(self, name: str) -> Collection:
        """Get an existing collection."""
        ...

    @abstractmethod
    def list_collections(self) -> Iterator[str]:
        """List all collection names."""
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        ...


class StaticIndexError(Exception):
    """
    Raised when attempting write operations on a static index.

    Some backends (like FAISS, Annoy) use static indexes that cannot be
    modified after they are built. This exception is raised when users
    attempt write operations on such backends.
    """

    pass
