# vd - Vector Database Facades

A unified, Pythonic interface for interacting with various vector databases. The `vd` package abstracts away the specifics of each database's API to offer a consistent, database-agnostic interface for semantic search operations.

## Features

- **Unified API**: Single interface for multiple vector database backends
- **Pythonic Design**: Collections behave like MutableMapping (dict-like)
- **Flexible Document Input**: Support for strings, tuples, and Document objects
- **Powerful Filtering**: MongoDB-style query syntax for metadata filtering
- **Automatic Embeddings**: Seamless integration with embedding models via `imbed`
- **Pluggable Backends**: Easy to add new vector database backends
- **Type-Safe**: Full type hints and protocol-based design
- **Well-Tested**: Comprehensive test suite with >90% coverage

## Installation

```bash
# Basic installation (includes memory backend)
pip install vd

# With ChromaDB support
pip install vd[chromadb]

# With all optional dependencies
pip install vd[all]
```

## Quick Start

```python
import vd

# Connect to a backend (memory backend for quick prototyping)
client = vd.connect('memory')

# Create a collection
docs = client.create_collection('my_documents')

# Add documents (simple!)
docs['doc1'] = "Machine learning is a subset of AI"
docs['doc2'] = "Deep learning uses neural networks"
docs['doc3'] = "Python is great for data science"

# Search with semantic similarity
results = docs.search("artificial intelligence", limit=2)
for result in results:
    print(f"{result['id']}: {result['text']} (score: {result['score']:.3f})")
```

## Core Concepts

### Backends

`vd` supports multiple vector database backends:

- **`memory`**: In-memory storage (always available, great for testing)
- **`chroma`**: ChromaDB (requires `pip install chromadb`)

More backends coming soon!

```python
# List available backends
print(vd.list_backends())

# Connect to different backends
memory_client = vd.connect('memory')
chroma_client = vd.connect('chroma', persist_directory='./data')
```

### Collections

Collections are MutableMapping objects that store searchable documents:

```python
# Create a collection
docs = client.create_collection('articles')

# Dict-like operations
docs['doc1'] = "Some text"              # Add
doc = docs['doc1']                       # Retrieve
del docs['doc1']                         # Delete
len(docs)                                # Count
for doc_id in docs:                      # Iterate
    print(doc_id)
```

### Documents

Multiple ways to specify documents:

```python
# String (simple text)
docs['id1'] = "Just some text"

# Tuple: (text, metadata)
docs['id2'] = ("Article text", {'category': 'tech', 'year': 2024})

# Tuple: (text, id) - for batch operations
docs.add_documents([
    ("First article", "custom_id_1"),
    ("Second article", {'author': 'Alice'}),
])

# Document object (full control)
doc = vd.Document(
    id='id3',
    text='Article text',
    metadata={'category': 'science'},
    vector=[0.1, 0.2, ...]  # Optional pre-computed embedding
)
docs.upsert(doc)
```

### Searching

Powerful search with filtering and transformation:

```python
# Basic search
results = docs.search("machine learning", limit=5)

# With metadata filter
results = docs.search(
    "neural networks",
    filter={'category': 'AI', 'year': {'$gte': 2020}}
)

# With egress function (transform results)
texts = docs.search(
    "data science",
    limit=10,
    egress=vd.text_only  # Just return the text
)

# Available egress functions
vd.text_only(result)        # Returns just the text
vd.id_only(result)          # Returns just the ID
vd.id_and_score(result)     # Returns (id, score)
vd.id_text_score(result)    # Returns (id, text, score)
```

### Filtering

MongoDB-style filter syntax:

```python
# Equality
docs.search("query", filter={'category': 'tech'})

# Comparison operators
docs.search("query", filter={'year': {'$gte': 2020}})
docs.search("query", filter={'views': {'$lt': 1000}})

# List membership
docs.search("query", filter={'tags': {'$in': ['python', 'ai']}})

# Logical operators
docs.search("query", filter={
    '$and': [
        {'year': {'$gte': 2020}},
        {'category': 'tech'}
    ]
})
```

Supported operators:
- `$eq`: Equal
- `$ne`: Not equal
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$lt`: Less than
- `$lte`: Less than or equal
- `$in`: In list
- `$and`: Logical AND
- `$or`: Logical OR

## Advanced Usage

### Custom Embedding Models

```python
# Use a specific embedding model
client = vd.connect('memory', embedding_model='text-embedding-3-large')

# Use a custom embedding function
def my_embedder(text: str) -> list[float]:
    # Your embedding logic here
    return [...]

client = vd.connect('memory', embedding_model=my_embedder)
```

### Batch Operations

```python
# Batch add for efficiency
docs.add_documents([
    "Document 1",
    ("Document 2", {'category': 'tech'}),
    ("Document 3", "custom_id", {'year': 2024}),
], batch_size=100)
```

### Collection Management

```python
# List collections
for name in client.list_collections():
    print(name)

# Get existing collection
docs = client.get_collection('my_docs')

# Delete collection
client.delete_collection('old_docs')
```

### Pre-computed Vectors

```python
# If you already have embeddings
doc = vd.Document(
    id='doc1',
    text='Some text',
    vector=[0.1, 0.2, 0.3, ...],  # Your pre-computed embedding
)
docs['doc1'] = doc

# Search with pre-computed query vector
query_vector = [0.15, 0.25, 0.35, ...]
results = docs.search(query_vector, limit=5)
```

## Architecture

The `vd` package is designed with several key principles:

1. **Protocol-based**: Uses Python protocols for type safety without tight coupling
2. **Separation of Concerns**: Embedding, storage, and search are independent
3. **Progressive Enhancement**: Same code works from in-memory to distributed databases
4. **Facade Pattern**: Provides a consistent interface across different backends

### Project Structure

```
vd/
├── __init__.py          # Public API
├── base.py              # Core protocols and types
├── util.py              # Utility functions and factory
├── backends/            # Backend implementations
│   ├── __init__.py
│   ├── memory.py        # In-memory backend
│   └── chroma.py        # ChromaDB backend
└── tests/               # Comprehensive test suite
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=vd --cov-report=html
```

### Adding a New Backend

1. Create a new file in `vd/backends/`
2. Implement the backend class inheriting from `BaseBackend`
3. Implement a collection class with the MutableMapping interface
4. Register the backend with `@register_backend('backend_name')`
5. Add tests in `tests/`

Example:

```python
from vd.base import BaseBackend
from vd.util import register_backend

@register_backend('mydb')
class MyDBBackend(BaseBackend):
    def create_collection(self, name, **kwargs):
        # Implementation
        pass
    # ... other methods
```

## Design Philosophy

The `vd` package follows these design principles:

- **Favor functional over object-oriented** where appropriate
- **Use Mapping/MutableMapping abstractions** for intuitive interfaces
- **Leverage existing packages** (dol, imbed) for core functionality
- **Optional dependencies** for backends (graceful degradation)
- **Progressive enhancement**: Scale from prototypes to production seamlessly

## Integration with i2mint Ecosystem

`vd` is designed to work seamlessly with the i2mint ecosystem:

- **`dol`**: Provides the underlying Mapping/Store patterns
- **`imbed`**: Handles embedding generation and management
- **`i2`**: Signature manipulation for consistent interfaces
- **`oa`**: OpenAI API integration for embeddings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Links

- **GitHub**: https://github.com/i2mint/vd
- **Documentation**: Coming soon
- **PyPI**: Coming soon

## Roadmap

- [ ] Additional backends (Pinecone, Weaviate, Qdrant, FAISS)
- [ ] Async support
- [ ] Multi-collection search
- [ ] Hybrid search (vector + keyword)
- [ ] Advanced filtering syntax
- [ ] Performance optimizations
- [ ] Comprehensive documentation site

## Examples

See `example_usage.py` for a complete working example demonstrating all major features.
