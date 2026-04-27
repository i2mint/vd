# VD: Vector Database Facade - Package Specification

**Version:** 1.0  
**Date:** 2025-11-16  
**Author:** Based on project conversations and research

---

## Executive Summary

The `vd` package provides a unified, Pythonic facade for interacting with various vector databases. It abstracts away the specifics of each database's API to offer a consistent, database-agnostic interface for semantic search operations.

### Core Design Philosophy

Following your architectural preferences:
- **Favor functional over object-oriented** where appropriate
- **Use Mapping/MutableMapping abstractions** for storage interfaces
- **Leverage dol and imbed packages** for core functionality
- **Optional LangChain integration** through dependency injection
- **Separation of concerns**: segmentation, embedding, storage, and search are independent
- **Progressive enhancement**: scale from in-memory to distributed databases seamlessly

---

## 1. Key Packages to Leverage

Based on your knowledge base, the following packages should be used:

### 1.1 Core Dependencies

- **`dol`** (Dictionary-of-Locations): Provides the `Mapping`/`MutableMapping` base abstractions, Store patterns, and key-value codec system
- **`imbed`**: Handles embedding generation, segmentation utilities, and vector operations
- **`i2`**: Provides signature manipulation and wrapper utilities for consistent interfaces
- **`oa`**: OpenAI API integration for embeddings

### 1.2 Optional Dependencies

- **`langchain`**: For broader vector database backend support (Chroma, Pinecone, Weaviate, Qdrant, FAISS, etc.)
- Various vector database clients (installed on-demand based on backend choice)

---

## 2. Architecture Overview

### 2.1 Core Components

```
vd/
├── base.py          # Core protocols and base classes
├── util.py          # Utility functions and facades
├── backends/        # Backend implementations
│   ├── memory.py    # In-memory (for small datasets)
│   ├── chroma.py    # ChromaDB backend
│   ├── pinecone.py  # Pinecone backend
│   └── ...          # Other backends
├── stores.py        # Store implementations (Mapping-based)
└── __init__.py      # Public API
```

### 2.2 Hierarchical Structure

Following the research findings, adopt a clean Client → Collection hierarchy:

```python
client = vd.connect(backend='chroma', **config)
collection = client.get_collection('my_docs')
# or
collection = client.create_collection('my_docs', schema=...)
```

---

## 3. Core API Design

### 3.1 Connection Factory

```python
def connect(
    backend: str,
    *,
    embedding_model: str | Callable = 'text-embedding-3-small',
    **backend_kwargs
) -> Client:
    """
    Connect to a vector database backend.
    
    Parameters
    ----------
    backend : str
        Backend identifier ('chroma', 'pinecone', 'weaviate', 'memory', etc.)
    embedding_model : str | Callable
        Embedding model specification (from oa/imbed) or custom callable
    **backend_kwargs
        Backend-specific configuration (API keys, URLs, etc.)
        
    Returns
    -------
    Client
        A client instance for the specified backend
        
    Examples
    --------
    >>> client = vd.connect('chroma', persist_directory='./data')
    >>> client = vd.connect('pinecone', api_key=os.environ['PINECONE_KEY'])
    """
```

### 3.2 Client Interface (Protocol)

```python
from typing import Protocol, Iterator, Optional

class Client(Protocol):
    """Protocol for vector database clients."""
    
    def create_collection(
        self,
        name: str,
        *,
        schema: Optional[dict] = None,
        **kwargs
    ) -> 'Collection':
        """Create a new collection."""
        
    def get_collection(self, name: str) -> 'Collection':
        """Get an existing collection."""
        
    def list_collections(self) -> Iterator[str]:
        """List all collection names."""
        
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
```

### 3.3 Collection Interface (MutableMapping-based)

Following your "Stores" pattern, a Collection should behave like a MutableMapping:

```python
from typing import MutableMapping, Union, Iterable, Any
from dataclasses import dataclass

@dataclass
class Document:
    """Standardized document representation."""
    id: str
    text: str
    vector: Optional[list[float]] = None  # Auto-generated if None
    metadata: dict[str, Any] = field(default_factory=dict)

class Collection(MutableMapping[str, Document]):
    """
    A collection of searchable documents.
    
    Implements MutableMapping for intuitive CRUD operations:
    - collection[doc_id] = document  # add/update
    - doc = collection[doc_id]        # retrieve
    - del collection[doc_id]          # delete
    - list(collection)                # iterate over doc IDs
    """
    
    # MutableMapping methods
    def __setitem__(self, key: str, value: Union[str, Document]) -> None:
        """Add or update a document."""
        
    def __getitem__(self, key: str) -> Document:
        """Retrieve a document by ID."""
        
    def __delitem__(self, key: str) -> None:
        """Delete a document."""
        
    def __iter__(self) -> Iterator[str]:
        """Iterate over document IDs."""
        
    def __len__(self) -> int:
        """Number of documents in collection."""
    
    # Enhanced retrieval (supporting multiple formats)
    def __getitem__(
        self, 
        key: Union[str, slice, list, Callable]
    ) -> Union[Document, list[Document]]:
        """
        Enhanced retrieval supporting:
        - Single ID: collection['doc1']
        - Multiple IDs: collection[['doc1', 'doc2']]
        - Slice: collection[10:20]  # documents 10-20
        - Filter callable: collection[lambda d: d.metadata['type'] == 'article']
        """
    
    # Search methods
    def search(
        self,
        query: Union[str, list[float]],
        *,
        limit: int = 10,
        filter: Optional[dict] = None,
        egress: Optional[Callable] = None,
        **kwargs
    ) -> Iterator[dict]:
        """
        Search the collection.
        
        Parameters
        ----------
        query : str | list[float]
            Query text (auto-embedded) or pre-computed vector
        limit : int
            Maximum number of results
        filter : dict, optional
            Metadata filter (unified filter syntax)
        egress : Callable, optional
            Transform search results (default returns all info)
        **kwargs
            Backend-specific options (alpha for hybrid search, etc.)
            
        Yields
        ------
        dict
            Search results with keys: id, text, score, metadata
            (transformed by egress if provided)
        """
    
    # Batch operations
    def add_documents(
        self,
        documents: Iterable[Union[str, tuple, Document]],
        *,
        batch_size: int = 100
    ) -> None:
        """
        Batch add documents.
        
        Supports flexible input formats:
        - "text" → auto-generated ID
        - ("text", "id") → specified ID  
        - ("text", {"meta": "data"}) → auto-generated ID with metadata
        - ("text", "id", {"meta": "data"}) → full specification
        - Document(...) → full control
        """
    
    def upsert(self, document: Document) -> None:
        """Insert or update a single document (idempotent)."""
```

---

## 4. Data Models

### 4.1 Core Types

```python
from typing import TypeAlias, Union, Mapping

# Text segment representations
Text: TypeAlias = str
TextKey: TypeAlias = str  # Document ID / URI
Metadata: TypeAlias = dict[str, Any]

# Document can be specified in multiple ways
DocumentInput: TypeAlias = Union[
    str,                           # Just text
    tuple[str, str],              # (text, id)
    tuple[str, Metadata],         # (text, metadata)
    tuple[str, str, Metadata],    # (text, id, metadata)
    'Document'                     # Full document object
]

# Vector representations
Vector: TypeAlias = list[float]
VectorMapping: TypeAlias = Mapping[TextKey, Vector]

# Search results
SearchResult: TypeAlias = dict[str, Any]  # {id, text, score, metadata, ...}
```

### 4.2 Document Model

```python
from dataclasses import dataclass, field

@dataclass
class Document:
    """
    Standardized document representation.
    
    Attributes
    ----------
    id : str
        Unique identifier (URI to source)
    text : str
        The text content
    vector : list[float], optional
        Embedding vector (auto-generated if None)
    metadata : dict
        Associated metadata
    """
    id: str
    text: str
    vector: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.3 Filter Model

Unified filter syntax inspired by MongoDB/Pinecone:

```python
Filter: TypeAlias = dict[str, Any]

# Examples:
# {'category': 'article'}  # Equality
# {'views': {'$gte': 1000}}  # Greater than or equal
# {'$and': [{'type': 'blog'}, {'published': True}]}  # Logical AND
# {'tags': {'$in': ['python', 'ai']}}  # In list
```

---

## 5. Backend Architecture

### 5.1 Pluggable Backend System

Use registry pattern for backend registration:

```python
# In vd/util.py
_backends = {}

def register_backend(name: str):
    """Decorator to register a backend implementation."""
    def decorator(backend_class):
        _backends[name] = backend_class
        return backend_class
    return decorator

# In vd/backends/chroma.py
@register_backend('chroma')
class ChromaBackend(BaseBackend):
    """ChromaDB backend implementation."""
    pass

# In vd/backends/memory.py  
@register_backend('memory')
class MemoryBackend(BaseBackend):
    """In-memory backend for small datasets."""
    pass
```

### 5.2 Backend Base Class

```python
from abc import ABC, abstractmethod

class BaseBackend(ABC):
    """Base class for vector database backends."""
    
    def __init__(
        self,
        *,
        embedding_model: Callable,
        **config
    ):
        self.embedding_model = embedding_model
        self.config = config
    
    @abstractmethod
    def create_collection(self, name: str, **kwargs) -> 'Collection':
        """Create a new collection."""
    
    @abstractmethod  
    def get_collection(self, name: str) -> 'Collection':
        """Get existing collection."""
    
    @abstractmethod
    def list_collections(self) -> Iterator[str]:
        """List collection names."""
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
```

### 5.3 Handling Static Backends (FAISS, Annoy)

For read-only/static backends:

```python
class StaticIndexError(Exception):
    """Raised when attempting write operations on static index."""
    pass

class FAISSBackend(BaseBackend):
    """FAISS backend - static index."""
    
    def add_documents(self, documents):
        """Build index from documents (one-time operation)."""
        if self._index_built:
            raise StaticIndexError(
                "FAISS index is static. Rebuild index to add documents."
            )
        # Build index...
        self._index_built = True
    
    def __setitem__(self, key, value):
        raise StaticIndexError(
            "Cannot modify FAISS index after build. "
            "Create new index instead."
        )
```

---

## 6. Integration with Existing Packages

### 6.1 Using `imbed` for Embeddings

```python
from imbed import Embed
from imbed.base import DFLT_EMBEDDING_MODEL

class Collection:
    def __init__(self, embedding_model=None, **kwargs):
        # Use imbed's Embed class
        self.embed = Embed(model=embedding_model or DFLT_EMBEDDING_MODEL)
    
    def _get_vector(self, text: str) -> Vector:
        """Get embedding for text using imbed."""
        return self.embed(text)
    
    def _get_vectors(self, texts: list[str]) -> list[Vector]:
        """Batch embed texts."""
        return list(self.embed(texts))
```

### 6.2 Using `dol` for Storage Patterns

```python
from dol import KvReader, Store, wrap_kvs

class Collection(Store):
    """Collection built on dol.Store."""
    
    def __init__(self, store, embedding_model, **kwargs):
        # Wrap underlying store with key/value transforms
        super().__init__(store)
        self.embedding_model = embedding_model
    
    # Implement key/value transforms as needed
    def _id_of_key(self, k):
        """Transform store keys to document IDs."""
        return k
    
    def _key_of_id(self, doc_id):
        """Transform document IDs to store keys."""
        return doc_id
    
    def _obj_of_data(self, data):
        """Transform stored data to Document objects."""
        return Document(**data)
    
    def _data_of_obj(self, doc):
        """Transform Document objects to storable data."""
        return {'id': doc.id, 'text': doc.text, ...}
```

### 6.3 Using `i2` for Signature Consistency

```python
from i2 import Sig

# Ensure consistent signatures across backends
def normalize_search_signature(backend_search):
    """Wrap backend search with consistent signature."""
    sig = Sig(backend_search)
    # Modify signature to match vd standard
    return sig(backend_search)
```

---

## 7. Implementation Phases

### Phase 1: Core Functionality (MVP)
**Target Backends:** Memory, ChromaDB, Pinecone

**Deliverables:**
- `vd.connect()` factory function
- `Client` protocol and base implementation
- `Collection` MutableMapping implementation
- Core methods: `add`, `search`, `delete`
- Integration with `imbed` for embeddings
- Basic filtering support

### Phase 2: Advanced Features
**Target Backends:** Add Weaviate, Qdrant, Milvus

**Deliverables:**
- Unified filter translation layer
- Hybrid search support (alpha parameter)
- Schema-first vs schema-flexible unification
- Batch operations optimization
- Multi-collection search

### Phase 3: Extended Support
**Target Backends:** pgvector, Elasticsearch, FAISS, Annoy

**Deliverables:**
- Database extension backends (SQL translation)
- Static library backends with graceful error handling
- Advanced metadata filtering
- Performance optimizations

---

## 8. Example Usage Patterns

### 8.1 Basic Usage

```python
import vd

# Connect to backend
client = vd.connect('chroma', persist_directory='./my_data')

# Create/get collection
docs = client.create_collection('articles')

# Add documents (multiple formats supported)
docs['doc1'] = "This is a test document"
docs['doc2'] = ("Another document", {'category': 'test'})

# Batch add
docs.add_documents([
    "First article about AI",
    ("Second article", "doc3"),
    ("Third article", {'category': 'tech'})
])

# Search
results = docs.search(
    "articles about artificial intelligence",
    limit=5,
    filter={'category': 'tech'}
)

for result in results:
    print(f"{result['id']}: {result['text'][:50]}... (score: {result['score']})")
```

### 8.2 Advanced Usage with Custom Egress

```python
# Custom result transformation
def extract_text_only(result):
    return result['text']

# Get just the text
texts = list(docs.search(
    "machine learning",
    limit=10,
    egress=extract_text_only
))

# Or use predefined egress functions
from vd.util import text_only, id_and_score

ids_and_scores = list(docs.search(query, egress=id_and_score))
```

### 8.3 Mall Pattern (Collections of Collections)

```python
# Create a "mall" of collections
db_mall = {
    'articles': client.get_collection('articles'),
    'papers': client.get_collection('research_papers'),
    'docs': client.get_collection('documentation')
}

# Access
db_mall['articles']['doc1']

# Or with custom Mapping that accepts tuples
class VectorMall(Mapping):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            collection_name, doc_id = key
            return self._collections[collection_name][doc_id]
        return self._collections[key]

mall = VectorMall(collections=db_mall)
doc = mall['articles', 'doc1']  # Direct nested access
```

### 8.4 Working with Pre-computed Vectors

```python
# When you already have embeddings
from imbed import Embed

embedder = Embed(model='text-embedding-3-small')
texts = ["doc1 text", "doc2 text"]
vectors = list(embedder(texts))

# Add with pre-computed vectors
for i, (text, vector) in enumerate(zip(texts, vectors)):
    docs[f'doc{i}'] = Document(
        id=f'doc{i}',
        text=text,
        vector=vector
    )
```

---

## 9. Testing Strategy

### 9.1 Test Structure

```
tests/
├── test_core.py           # Core API tests
├── test_backends/         # Backend-specific tests
│   ├── test_memory.py
│   ├── test_chroma.py
│   └── ...
├── test_integration.py    # Integration tests
└── test_util.py          # Utility function tests
```

### 9.2 Key Test Cases

1. **Interface Compliance**: All backends implement Client/Collection protocols
2. **Data Formats**: All input formats work correctly
3. **Search Quality**: Semantic search returns relevant results
4. **Filter Translation**: Filters work across backends
5. **Batch Operations**: Efficient bulk inserts
6. **Error Handling**: Graceful failures (e.g., StaticIndexError for FAISS)
7. **Idempotency**: Upsert operations are truly idempotent

---

## 10. Configuration and Conventions

### 10.1 Configuration File

Support optional configuration via `~/.config/vd/config.py`:

```python
# ~/.config/vd/config.py
DEFAULT_BACKEND = 'chroma'
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'

BACKEND_CONFIGS = {
    'chroma': {
        'persist_directory': '~/.vd/chroma_data'
    },
    'pinecone': {
        'api_key': 'env:PINECONE_API_KEY',
        'environment': 'us-east1-gcp'
    }
}
```

### 10.2 Environment Variables

```bash
VD_DEFAULT_BACKEND=chroma
VD_CHROMA_PERSIST_DIR=./data
VD_PINECONE_API_KEY=your_key_here
```

---

## 11. Documentation Requirements

### 11.1 Docstring Standards

- All functions/classes MUST have docstrings
- Include minimal doctests where practical
- Use NumPy-style docstring format

### 11.2 Example from Specification

```python
def search(
    self,
    query: Union[str, list[float]],
    *,
    limit: int = 10,
    filter: Optional[dict] = None,
    egress: Optional[Callable] = None,
    **kwargs
) -> Iterator[dict]:
    """
    Search the collection for similar documents.
    
    Parameters
    ----------
    query : str or list of float
        Query text (will be embedded) or pre-computed query vector
    limit : int, default 10
        Maximum number of results to return
    filter : dict, optional
        Metadata filter using unified filter syntax
    egress : callable, optional
        Function to transform results. If None, returns full result dict.
    **kwargs
        Backend-specific options (e.g., alpha=0.5 for hybrid search)
    
    Yields
    ------
    dict
        Search results with keys: 'id', 'text', 'score', 'metadata'
        (or transformed by egress function)
    
    Examples
    --------
    >>> docs = collection.search("machine learning", limit=5)
    >>> for doc in docs:
    ...     print(doc['id'], doc['score'])
    
    >>> # With metadata filter
    >>> docs = collection.search(
    ...     "AI research",
    ...     filter={'year': {'$gte': 2020}}
    ... )
    
    >>> # With custom egress
    >>> texts = collection.search(
    ...     "neural networks",
    ...     egress=lambda r: r['text']
    ... )
    """
```

---

## 12. Future Enhancements

### 12.1 Multi-Embedding Support

Allow collections to use multiple embedding models:

```python
collection = client.create_collection(
    'multi_model',
    embeddings={
        'semantic': 'text-embedding-3-small',
        'code': 'code-embedding-model'
    }
)

# Search with specific embedding
results = collection.search(query, embedding='code')
```

### 12.2 Async Support

```python
async def search_async(
    self,
    query: str,
    **kwargs
) -> AsyncIterator[dict]:
    """Async version of search."""
```

### 12.3 Distributed Collections

Support searching across multiple collections:

```python
results = vd.search_all(
    collections=['docs', 'articles', 'papers'],
    query="machine learning",
    aggregation='merge'  # or 'interleave'
)
```

---

## 13. Key Design Decisions Summary

1. **MutableMapping Interface**: Collections behave like dictionaries
2. **Pluggable Backends**: Registry-based backend system
3. **Unified Filter Syntax**: MongoDB-style query language
4. **Auto-embedding**: Text automatically embedded unless vector provided
5. **Lazy Evaluation**: Search returns iterators, not lists
6. **Flexible Input**: Support multiple document input formats
7. **Separation of Concerns**: Embedding, storage, and search are independent
8. **dol Integration**: Leverage existing Store patterns
9. **imbed Integration**: Use for all embedding operations
10. **Optional LangChain**: Can wrap LangChain VectorStores as backends

---

## 14. Success Criteria

The `vd` package will be considered successful if:

1. ✓ A user can switch backends with a single parameter change
2. ✓ The API feels Pythonic (uses Mapping abstractions, iterators, etc.)
3. ✓ Common operations (add, search, delete) work identically across backends
4. ✓ Performance is comparable to using backend clients directly
5. ✓ Documentation with examples covers 80% of use cases
6. ✓ Test coverage >90% for core functionality
7. ✓ Integration with existing ecosystem (dol, imbed) is seamless
8. ✓ Static backends (FAISS) fail gracefully with clear error messages

---

## Appendix A: Comparison Table of Vector Databases

From the research findings:

| Database | Architecture | Data Model | Upsert Pattern | Search API |
|----------|-------------|------------|----------------|------------|
| Pinecone | Dedicated, managed | Vector + Meta | Single `upsert()` | `query()` with `$` filter operators |
| Weaviate | Dedicated, self-hosted | Schema-first Objects | Separate `insert()`/`update()` | `hybrid()` with alpha |
| Milvus | Dedicated, open-source | Schema-first Collections | Separate `insert()`/`upsert()` | `search()` + `query()` |
| Qdrant | Dedicated, self-hosted | Vector + Payload | Single `upsert()` | `search()` + `scroll()` |
| ChromaDB | Client/Server | Document-first | `add()` (text or vectors) | `query()` with where filters |
| FAISS/Annoy | Static Library | In-memory/File | One-time `add()`/`build()` | No updates after build |
| pgvector | DB Extension | SQL table | SQL `UPDATE`/`INSERT` | `ORDER BY` with distance operator |

---

## Appendix B: References

1. LangChain VectorStores Documentation
2. ChromaDB Documentation
3. Pinecone Python Client
4. Project conversations on vd design
5. imbed package documentation
6. dol package documentation

---

**END OF SPECIFICATION**
