# vd - Vector Database Facades

A unified, Pythonic interface for interacting with various vector databases. The `vd` package abstracts away the specifics of each database's API to offer a consistent, database-agnostic interface for semantic search operations.

## Features

### Core Features
- **Unified API**: Single interface for multiple vector database backends
- **Backend Discovery**: Easy-to-use tools to find, install, and use different vector databases
- **Pythonic Design**: Collections behave like MutableMapping (dict-like)
- **Flexible Document Input**: Support for strings, tuples, and Document objects
- **Powerful Filtering**: MongoDB-style query syntax for metadata filtering
- **Automatic Embeddings**: Seamless integration with embedding models via `imbed`
- **Pluggable Backends**: Easy to add new vector database backends
- **Helpful Error Messages**: Get installation instructions when backends aren't available
- **Type-Safe**: Full type hints and protocol-based design
- **Well-Tested**: Comprehensive test suite with >90% coverage

### Extended Features
- **Command-Line Interface**: Full-featured CLI for common operations
- **Configuration Management**: YAML/TOML config files with profiles and environment variables
- **Backend Comparison**: Compare and get recommendations for backends based on your needs
- **Import/Export**: Support for JSONL, JSON, and directory formats
- **Migration**: Move collections between backends with progress tracking
- **Analytics**: Collection statistics, validation, duplicate detection, outlier analysis
- **Text Preprocessing**: Clean and chunk text with multiple strategies
- **Health Checks**: Monitor backend health and benchmark performance
- **Advanced Search**: Multi-query search, similarity search, reciprocal rank fusion

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

More backends coming soon (Pinecone, Weaviate, Qdrant, Milvus, FAISS)!

```python
# List currently registered backends
print(vd.list_backends())

# Connect to different backends
memory_client = vd.connect('memory')
chroma_client = vd.connect('chroma', persist_directory='./data')
```

### Backend Discovery

`vd` makes it easy to discover and install vector database backends:

```python
import vd

# View all backends with a nicely formatted table
vd.print_backends_table()

# List only backends that are currently available (installed)
available = vd.list_available_backends()
print(f"Available: {available}")

# Get detailed information about a specific backend
info = vd.get_backend_info('chroma')
print(info['description'])
print(info['features'])

# Get installation instructions
instructions = vd.get_install_instructions('chroma')
print(instructions)

# List ALL possible backends (including planned ones)
all_backends = vd.list_all_backends(include_planned=True)
```

When you try to connect to a backend that's not installed, you'll get helpful error messages:

```python
>>> vd.connect('chroma')
ValueError: Backend 'chroma' is not available.

To install it:
  pip install vd[chromadb]

Or run: vd.get_install_instructions('chroma') for more details.
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

## Command-Line Interface

`vd` includes a comprehensive CLI for common operations:

```bash
# List available backends
vd backends
vd backends --planned  # Include planned backends

# Get installation instructions
vd install chroma

# Check backend health
vd health memory

# Export a collection
vd export memory my_docs -o backup.jsonl
vd export memory my_docs -o backup.json -f json

# Import a collection
vd import chroma my_docs -i backup.jsonl

# View collection statistics
vd stats memory my_docs
vd stats memory my_docs -v  # Verbose output

# Validate a collection
vd validate memory my_docs

# Migrate between backends
vd migrate memory source_docs chroma target_docs

# Benchmark search performance
vd benchmark memory my_docs -q "test query" --queries 100
```

## Configuration Management

Manage backend configurations with YAML or TOML files:

```python
import vd

# Connect using a configuration file
client = vd.connect_from_config('vd.yaml')

# Use a specific profile
client = vd.connect_from_config('vd.yaml', profile='production')

# Create example configuration
config_yaml = vd.create_example_config('yaml')
vd.save_config(config, 'vd.yaml')
```

Example `vd.yaml`:
```yaml
profiles:
  default:
    backend: memory
  dev:
    backend: memory
  prod:
    backend: chroma
    persist_directory: ./vector_db
```

Environment variable overrides:
- `VD_PROFILE`: Select profile (default: 'default')
- `VD_BACKEND`: Override backend name
- `VD_EMBEDDING_MODEL`: Override embedding model

## Backend Comparison and Recommendation

Choose the right backend for your needs:

```python
import vd

# Compare backends
vd.print_comparison(['memory', 'chroma', 'pinecone'])

# Get recommendations based on requirements
vd.print_recommendation(
    dataset_size='medium',      # small, medium, large, very_large
    persistence_required=True,
    cloud_required=False,
    budget='free',              # free, low, medium, high
    performance_priority='balanced'  # speed, scalability, balanced
)

# Get backend characteristics
chars = vd.get_backend_characteristics()
print(chars['chroma']['use_cases'])
```

## Import/Export

Export and import collections in multiple formats:

```python
import vd

# Export to JSONL (recommended for large collections)
vd.export_collection(docs, 'backup.jsonl', format='jsonl')

# Export to JSON
vd.export_collection(docs, 'backup.json', format='json')

# Export to directory (one file per document)
vd.export_collection(docs, './backup_dir', format='directory')

# Import from file
vd.import_collection(docs, 'backup.jsonl')
vd.import_collection(docs, 'backup.jsonl', skip_existing=True)
```

## Migration

Move collections between backends:

```python
import vd

# Migrate a collection
source = source_client.get_collection('docs')
target = target_client.create_collection('docs')

stats = vd.migrate_collection(
    source,
    target,
    batch_size=100,
    preserve_vectors=True,  # Keep existing embeddings
    progress_callback=lambda cur, tot: print(f"{cur}/{tot}")
)

# Migrate entire client (all collections)
vd.migrate_client(
    source_client,
    target_client,
    collection_names=['docs1', 'docs2']  # Optional filter
)
```

## Collection Analytics

Analyze and validate collections:

```python
import vd

# Get collection statistics
stats = vd.collection_stats(docs)
print(f"Total: {stats['total_documents']}")
print(f"Avg length: {stats['avg_text_length']}")
print(f"Metadata fields: {stats['metadata_fields']}")

# Metadata distribution
dist = vd.metadata_distribution(docs, 'category')

# Find duplicate or near-duplicate documents
duplicates = vd.find_duplicates(docs, threshold=0.95)

# Find outliers (dissimilar documents)
outliers = vd.find_outliers(docs, threshold=0.3)

# Sample collection
random_sample = vd.sample_collection(docs, n=10, method='random')
diverse_sample = vd.sample_collection(docs, n=10, method='diverse')

# Validate collection integrity
report = vd.validate_collection(docs)
if not report['valid']:
    for issue in report['issues']:
        print(f"Issue: {issue}")
```

## Text Preprocessing

Clean and chunk text before adding to collections:

```python
import vd

# Clean text
clean = vd.clean_text(
    text,
    lowercase=True,
    remove_extra_whitespace=True,
    remove_urls=True,
    remove_emails=True
)

# Chunk text
chunks = vd.chunk_text(
    text,
    chunk_size=500,
    overlap=50,
    strategy='sentences'  # chars, words, sentences, paragraphs
)

# Chunk documents with metadata preservation
chunked_docs = vd.chunk_documents(
    documents,
    chunk_size=500,
    id_template='{doc_id}_chunk_{chunk_num}',
    preserve_metadata=True
)

# Extract metadata from text
metadata = vd.extract_metadata(
    text,
    extract_title=True,
    extract_length=True,
    extract_word_count=True
)
```

## Health Checks and Benchmarking

Monitor and benchmark performance:

```python
import vd

# Check backend health
health = vd.health_check_backend('chroma', persist_directory='./data')
print(f"Status: {health['status']}")
print(f"Available: {health['available']}")

# Check collection health
health = vd.health_check_collection(docs)

# Benchmark search performance
results = vd.benchmark_search(
    docs,
    query="test query",
    n_queries=100,
    limit=10
)
print(f"Avg latency: {results['avg_latency']*1000:.2f}ms")
print(f"P95: {results['p95']*1000:.2f}ms")
print(f"Throughput: {results['queries_per_second']:.1f} queries/sec")

# Benchmark insertion
results = vd.benchmark_insert(docs, n_documents=100, batch_size=10)
```

## Advanced Search

Enhanced search capabilities:

```python
import vd

# Multi-query search
results = vd.multi_query_search(
    docs,
    queries=["AI", "machine learning"],
    limit=10,
    combine='best'  # interleave, concatenate, union, best
)

# Find similar documents
similar = vd.search_similar_to_document(
    docs,
    doc_id='doc1',
    limit=10,
    exclude_self=True
)

# Reciprocal Rank Fusion (combine multiple rankings)
results1 = list(docs.search("query1"))
results2 = list(docs.search("query2"))
combined = vd.reciprocal_rank_fusion([results1, results2])

# Deduplicate results
unique = vd.deduplicate_results(results, key='id', keep='first')
```

## Roadmap

- [x] Import/Export (JSONL, JSON, directory)
- [x] Migration between backends
- [x] Collection analytics and validation
- [x] Text preprocessing and chunking
- [x] Health checks and benchmarking
- [x] Advanced search (multi-query, RRF, similarity)
- [x] Configuration file support (YAML, TOML)
- [x] Backend comparison and recommendation
- [x] Command-line interface
- [ ] Additional backends (Pinecone, Weaviate, Qdrant, FAISS)
- [ ] Async support
- [ ] Hybrid search (vector + keyword)
- [ ] Comprehensive documentation site

## Examples

See the demo scripts for comprehensive examples:
- `example_usage.py` - Basic usage and core features
- `demo_backend_discovery.py` - Backend discovery features
- `demo_config.py` - Configuration management
- `demo_comparison.py` - Backend comparison and recommendation
- `demo_utilities.py` - Import/export, migration, analytics, and more
