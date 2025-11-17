"""
In-memory vector database backend.

This backend stores all documents and vectors in memory, making it suitable
for prototyping, testing, and small datasets. It provides a simple reference
implementation of the backend protocol.
"""

from collections.abc import MutableMapping
from typing import Any, Callable, Iterator, Optional, Union

from vd.base import (
    BaseBackend,
    Collection,
    Document,
    DocumentInput,
    Filter,
    SearchResult,
    Vector,
)
from vd.util import (
    cosine_similarity,
    normalize_document_input,
    register_backend,
)


class MemoryCollection(MutableMapping):
    """
    In-memory collection implementation.

    Stores documents in a dictionary and performs brute-force similarity
    search. Suitable for small datasets and testing.

    Parameters
    ----------
    name : str
        Collection name
    embedding_model : callable
        Function to generate embeddings from text
    """

    def __init__(self, name: str, *, embedding_model: Callable[[str], Vector]):
        """Initialize the memory collection."""
        self.name = name
        self.embedding_model = embedding_model
        self._documents: dict[str, Document] = {}

    def __setitem__(self, key: str, value: Union[str, Document]) -> None:
        """Add or update a document."""
        if isinstance(value, str):
            # Convert string to Document
            doc = Document(id=key, text=value)
        elif isinstance(value, tuple):
            # Handle tuple formats
            doc = normalize_document_input(value, auto_id=False)
            doc.id = key  # Override with the provided key
        else:
            doc = value

        # Generate embedding if not provided
        if doc.vector is None:
            doc.vector = self.embedding_model(doc.text)

        self._documents[key] = doc

    def __getitem__(self, key: str) -> Document:
        """Retrieve a document by ID."""
        return self._documents[key]

    def __delitem__(self, key: str) -> None:
        """Delete a document."""
        del self._documents[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over document IDs."""
        return iter(self._documents)

    def __len__(self) -> int:
        """Get number of documents."""
        return len(self._documents)

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
        Search the collection using brute-force similarity.

        Parameters
        ----------
        query : str or list of float
            Query text or pre-computed vector
        limit : int, default 10
            Maximum number of results
        filter : dict, optional
            Metadata filter (basic support for equality filters)
        egress : callable, optional
            Transform function for results
        **kwargs
            Additional options (ignored by memory backend)

        Yields
        ------
        dict or transformed result
            Search results sorted by similarity score
        """
        # Get query vector
        if isinstance(query, str):
            query_vector = self.embedding_model(query)
        else:
            query_vector = query

        # Compute similarities
        results = []
        for doc_id, doc in self._documents.items():
            # Apply filter if provided
            if filter and not self._matches_filter(doc, filter):
                continue

            # Compute similarity
            if doc.vector is not None:
                score = cosine_similarity(query_vector, doc.vector)
                results.append(
                    {
                        'id': doc.id,
                        'text': doc.text,
                        'score': score,
                        'metadata': doc.metadata,
                    }
                )

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Limit results
        results = results[:limit]

        # Apply egress if provided
        if egress is not None:
            results = [egress(r) for r in results]

        yield from results

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
            Batch size (not used in memory backend, but kept for API consistency)
        """
        for doc_input in documents:
            doc = normalize_document_input(doc_input, auto_id=True)
            self[doc.id] = doc

    def upsert(self, document: Document) -> None:
        """
        Insert or update a document (idempotent operation).

        Parameters
        ----------
        document : Document
            Document to insert or update
        """
        self[document.id] = document

    def _matches_filter(self, doc: Document, filter: Filter) -> bool:
        """
        Check if a document matches the filter.

        This is a basic implementation supporting:
        - Simple equality: {'key': value}
        - $gte, $lte, $gt, $lt: {'key': {'$gte': value}}
        - $in: {'key': {'$in': [value1, value2]}}
        - $and: {'$and': [filter1, filter2]}

        Parameters
        ----------
        doc : Document
            Document to check
        filter : dict
            Filter specification

        Returns
        -------
        bool
            True if document matches filter
        """
        for key, condition in filter.items():
            if key == '$and':
                # Logical AND
                if not all(self._matches_filter(doc, f) for f in condition):
                    return False
            elif key == '$or':
                # Logical OR
                if not any(self._matches_filter(doc, f) for f in condition):
                    return False
            elif isinstance(condition, dict):
                # Operator-based filter
                value = doc.metadata.get(key)
                # If the field doesn't exist, the filter doesn't match
                if value is None:
                    return False
                for op, op_value in condition.items():
                    if op == '$gte' and not (value >= op_value):
                        return False
                    elif op == '$lte' and not (value <= op_value):
                        return False
                    elif op == '$gt' and not (value > op_value):
                        return False
                    elif op == '$lt' and not (value < op_value):
                        return False
                    elif op == '$eq' and not (value == op_value):
                        return False
                    elif op == '$ne' and not (value != op_value):
                        return False
                    elif op == '$in':
                        # Handle both scalar and list values
                        if isinstance(value, list):
                            # If value is a list, check if any element is in op_value
                            if not any(v in op_value for v in value):
                                return False
                        else:
                            # If value is scalar, check if it's in op_value
                            if value not in op_value:
                                return False
            else:
                # Simple equality
                if doc.metadata.get(key) != condition:
                    return False

        return True


@register_backend('memory')
class MemoryBackend(BaseBackend):
    """
    In-memory vector database backend.

    Stores all collections and documents in memory. Perfect for testing,
    prototyping, and small datasets.

    Examples
    --------
    >>> import vd  # doctest: +SKIP
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> collection = client.create_collection('test')  # doctest: +SKIP
    >>> collection['doc1'] = "Hello world"  # doctest: +SKIP
    >>> results = list(collection.search("hello"))  # doctest: +SKIP
    """

    def __init__(self, *, embedding_model: Callable[[str], Vector], **config):
        """Initialize the memory backend."""
        super().__init__(embedding_model=embedding_model, **config)
        self._collections: dict[str, MemoryCollection] = {}

    def create_collection(
        self,
        name: str,
        *,
        schema: Optional[dict] = None,
        **kwargs,
    ) -> Collection:
        """Create a new collection."""
        if name in self._collections:
            raise ValueError(f"Collection '{name}' already exists")

        collection = MemoryCollection(name, embedding_model=self.embedding_model)
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> Collection:
        """Get an existing collection."""
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return self._collections[name]

    def list_collections(self) -> Iterator[str]:
        """List all collection names."""
        return iter(self._collections.keys())

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        del self._collections[name]
