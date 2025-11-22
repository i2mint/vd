"""
ChromaDB vector database backend.

This backend provides integration with ChromaDB, a popular open-source
vector database. It supports both in-memory and persistent storage.

Requires: pip install chromadb
"""

from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, Optional, Union

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "ChromaDB backend requires the 'chromadb' package. "
        "Install it with: pip install chromadb"
    )

from vd.base import (
    BaseBackend,
    Collection,
    Document,
    DocumentInput,
    Filter,
    SearchResult,
    Vector,
)
from vd.util import normalize_document_input, register_backend


class ChromaCollection(MutableMapping):
    """
    ChromaDB collection implementation.

    Wraps a ChromaDB collection to provide the vd Collection interface.

    Parameters
    ----------
    chroma_collection : chromadb.Collection
        The underlying ChromaDB collection
    embedding_model : callable
        Function to generate embeddings from text
    """

    def __init__(
        self,
        chroma_collection,
        *,
        embedding_model: Callable[[str], Vector],
    ):
        """Initialize the ChromaDB collection wrapper."""
        self._collection = chroma_collection
        self.embedding_model = embedding_model
        self.name = chroma_collection.name

    def __setitem__(self, key: str, value: Union[str, Document]) -> None:
        """Add or update a document."""
        if isinstance(value, str):
            doc = Document(id=key, text=value)
        elif isinstance(value, tuple):
            doc = normalize_document_input(value, auto_id=False)
            doc.id = key
        else:
            doc = value

        # Generate embedding if not provided
        if doc.vector is None:
            doc.vector = self.embedding_model(doc.text)

        # Add to ChromaDB
        self._collection.upsert(
            ids=[doc.id],
            documents=[doc.text],
            embeddings=[doc.vector],
            metadatas=[doc.metadata] if doc.metadata else None,
        )

    def __getitem__(self, key: str) -> Document:
        """Retrieve a document by ID."""
        result = self._collection.get(ids=[key], include=['documents', 'embeddings', 'metadatas'])

        if not result['ids']:
            raise KeyError(f"Document '{key}' not found")

        return Document(
            id=result['ids'][0],
            text=result['documents'][0],
            vector=result['embeddings'][0] if result['embeddings'] else None,
            metadata=result['metadatas'][0] if result['metadatas'] else {},
        )

    def __delitem__(self, key: str) -> None:
        """Delete a document."""
        # Check if exists first
        result = self._collection.get(ids=[key])
        if not result['ids']:
            raise KeyError(f"Document '{key}' not found")

        self._collection.delete(ids=[key])

    def __iter__(self) -> Iterator[str]:
        """Iterate over document IDs."""
        # Get all IDs from the collection
        result = self._collection.get()
        return iter(result['ids'])

    def __len__(self) -> int:
        """Get number of documents."""
        return self._collection.count()

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
        Search the collection using ChromaDB.

        Parameters
        ----------
        query : str or list of float
            Query text or pre-computed vector
        limit : int, default 10
            Maximum number of results
        filter : dict, optional
            Metadata filter (ChromaDB where clause format)
        egress : callable, optional
            Transform function for results
        **kwargs
            Additional ChromaDB query options

        Yields
        ------
        dict or transformed result
            Search results sorted by similarity
        """
        # Get query vector
        if isinstance(query, str):
            query_vector = self.embedding_model(query)
        else:
            query_vector = query

        # Prepare query arguments
        query_args = {
            'query_embeddings': [query_vector],
            'n_results': limit,
            'include': ['documents', 'metadatas', 'distances'],
        }

        # Add filter if provided
        if filter:
            query_args['where'] = filter

        # Query ChromaDB
        results = self._collection.query(**query_args)

        # Convert to standard format
        if results['ids']:
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances, convert to similarity scores
                # Lower distance = higher similarity
                distance = results['distances'][0][i]
                # Convert L2 distance to similarity (inverse)
                # For cosine similarity, ChromaDB returns 1 - cosine
                score = 1.0 / (1.0 + distance)

                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': score,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                }

                if egress is not None:
                    yield egress(result)
                else:
                    yield result

    def add_documents(
        self,
        documents: Iterator[DocumentInput],
        *,
        batch_size: int = 100,
    ) -> None:
        """
        Batch add documents.

        Parameters
        ----------
        documents : iterable of DocumentInput
            Documents to add
        batch_size : int, default 100
            Number of documents to process in each batch
        """
        batch_ids = []
        batch_texts = []
        batch_embeddings = []
        batch_metadatas = []

        for doc_input in documents:
            doc = normalize_document_input(doc_input, auto_id=True)

            # Generate embedding if needed
            if doc.vector is None:
                doc.vector = self.embedding_model(doc.text)

            batch_ids.append(doc.id)
            batch_texts.append(doc.text)
            batch_embeddings.append(doc.vector)
            batch_metadatas.append(doc.metadata)

            # Flush batch if full
            if len(batch_ids) >= batch_size:
                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                batch_ids = []
                batch_texts = []
                batch_embeddings = []
                batch_metadatas = []

        # Flush remaining
        if batch_ids:
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

    def upsert(self, document: Document) -> None:
        """
        Insert or update a document (idempotent operation).

        Parameters
        ----------
        document : Document
            Document to insert or update
        """
        self[document.id] = document


@register_backend('chroma')
class ChromaBackend(BaseBackend):
    """
    ChromaDB vector database backend.

    Supports both in-memory and persistent ChromaDB instances.

    Parameters
    ----------
    embedding_model : callable
        Function to generate embeddings
    persist_directory : str, optional
        Directory to persist the database. If None, uses in-memory storage.
    **config
        Additional ChromaDB settings

    Examples
    --------
    >>> import vd  # doctest: +SKIP
    >>> # In-memory
    >>> client = vd.connect('chroma')  # doctest: +SKIP
    >>>
    >>> # Persistent
    >>> client = vd.connect('chroma', persist_directory='./my_db')  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedding_model: Callable[[str], Vector],
        persist_directory: Optional[str] = None,
        **config,
    ):
        """Initialize the ChromaDB backend."""
        super().__init__(embedding_model=embedding_model, **config)

        # Configure ChromaDB settings
        settings = Settings()
        if persist_directory:
            settings = Settings(
                persist_directory=persist_directory,
                is_persistent=True,
            )

        # Create ChromaDB client
        self._client = chromadb.Client(settings)

    def create_collection(
        self,
        name: str,
        *,
        schema: Optional[dict] = None,
        **kwargs,
    ) -> Collection:
        """Create a new collection."""
        # ChromaDB will raise if collection already exists with get_or_create=False
        try:
            chroma_collection = self._client.create_collection(name=name, **kwargs)
        except Exception as e:
            if 'already exists' in str(e).lower():
                raise ValueError(f"Collection '{name}' already exists") from e
            raise

        return ChromaCollection(
            chroma_collection,
            embedding_model=self.embedding_model,
        )

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection."""
        try:
            chroma_collection = self._client.get_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection '{name}' does not exist") from e

        return ChromaCollection(
            chroma_collection,
            embedding_model=self.embedding_model,
        )

    def list_collections(self) -> Iterator[str]:
        """List all collection names."""
        collections = self._client.list_collections()
        return (c.name for c in collections)

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self._client.delete_collection(name=name)
        except Exception as e:
            if 'does not exist' in str(e).lower():
                raise KeyError(f"Collection '{name}' does not exist") from e
            raise
