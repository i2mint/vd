---
name: vd-quickstart
description: >-
  Vector-search tooling for the vd package. Use this skill when the user wants
  to do semantic / vector search in Python with the vd package — connecting to
  a backend, creating a collection, adding documents, and running a query. Also
  trigger on mentions of "vector database", "embeddings + search", or any time
  the user imports `vd` for a new task and needs the basic happy-path setup.
audience: users
---

# vd quickstart

The `vd` package is a unified facade over multiple vector databases (memory,
ChromaDB, more planned). It hides each backend's API behind a single Pythonic
interface: a **client** owns **collections**, a collection is a `MutableMapping`
of document IDs to documents, and `.search()` returns ranked results.

This skill covers the basic happy path. For backend selection, ingestion of
large corpora, advanced search/filtering, or maintenance ops, use the sibling
skills `vd-backend-choose`, `vd-ingest`, `vd-search`, `vd-ops`.

## The happy path

```python
import vd

# 1. Connect to a backend. 'memory' is always available — no install needed.
client = vd.connect('memory')

# 2. Create a collection (or reuse one with client.get_collection(name)).
docs = client.create_collection('my_documents')

# 3. Add documents with dict-like syntax. Embeddings are auto-generated.
docs['doc1'] = "Machine learning is a subset of AI"
docs['doc2'] = "Deep learning uses neural networks"
docs['doc3'] = "Python is great for data science"

# 4. Search by similarity — returns an iterator of result dicts.
for r in docs.search("artificial intelligence", limit=2):
    print(f"{r['id']}: {r['text']} (score: {r['score']:.3f})")
```

A `result` dict has keys: `id`, `text`, `score`, `metadata`.

## Connecting

```python
client = vd.connect('memory')                                # in-memory, no persist
client = vd.connect('chroma', persist_directory='./data')    # ChromaDB on disk
client = vd.connect('memory', embedding_model='text-embedding-3-large')
client = vd.connect('memory', embedding_model=my_callable)   # custom embedder
```

`vd.connect(backend, *, embedding_model=None, **backend_kwargs)` is the single
entry point. Every kwarg after `backend` is keyword-only.

If the backend isn't installed (e.g. `chroma` without `chromadb`), `connect`
raises `ValueError` with install instructions. Don't try/except around it
silently — surface the message. To pick or compare backends, use the
`vd-backend-choose` skill.

## Adding documents

`vd` accepts four input shapes; pick the simplest that carries the info you
have:

```python
docs['id1'] = "Just text"                             # str
docs['id2'] = ("Article body", {'category': 'tech'})  # (text, metadata)
docs['id3'] = vd.Document(                            # full Document
    id='id3',
    text='Article body',
    metadata={'category': 'science'},
    vector=[0.1, 0.2, ...],   # optional — pass to skip embedding
)

# Batch add — preferred for >~10 docs (one embed call per batch):
docs.add_documents([
    "First article",                              # auto-id
    ("Second article", "doc2"),                   # (text, id)
    ("Third article", {'author': 'Alice'}),       # (text, metadata)
    ("Fourth article", "doc4", {'year': 2024}),   # (text, id, metadata)
], batch_size=100)
```

When a `Document` has a pre-computed `vector`, `vd` will not re-embed — useful
when you already paid for embeddings elsewhere.

## Searching

```python
# Basic
results = docs.search("machine learning", limit=5)

# With a metadata filter (MongoDB-style; see vd-search skill for full syntax)
results = docs.search("AI research", filter={'year': {'$gte': 2020}})

# With a pre-computed query vector instead of text
results = docs.search([0.15, 0.25, ...], limit=5)

# With an egress to project results to something simpler
texts  = docs.search("data science", limit=10, egress=vd.text_only)
ids    = docs.search("data science", limit=10, egress=vd.id_only)
pairs  = docs.search("data science", limit=10, egress=vd.id_and_score)
trips  = docs.search("data science", limit=10, egress=vd.id_text_score)
```

`search` returns an **iterator**, not a list. If you need to iterate twice or
index into it, wrap with `list(...)`.

The `egress` argument applies a function to each result before yielding it.
Pass any `Callable[[dict], Any]`, not just the four built-ins.

## Collection management

```python
client.list_collections()              # iterator of names
docs = client.get_collection('name')   # raises KeyError if missing
client.delete_collection('old_docs')   # raises KeyError if missing

# MutableMapping operations on a collection:
len(docs); 'doc1' in docs; del docs['doc1']
for doc_id in docs: ...
```

## Common gotchas

- **Embedding requires `imbed`.** When you pass text (not a pre-computed
  vector), `vd` calls `imbed.Embed(...)` to generate the embedding. If `imbed`
  isn't configured for an embedding provider (e.g. no OpenAI API key), the
  insert/query will fail at embed time, not at `connect` time. Either set up
  `imbed` or pass `embedding_model=my_callable`.
- **`memory` backend is non-persistent.** Data vanishes when the process exits.
  Use `chroma` with `persist_directory=...` (or migrate later — see `vd-ops`)
  if you need persistence.
- **IDs are strings.** If the user passes an int, it'll work for the dict op
  but will surprise on round-trip. Keep IDs as strings.
- **`Document.id` is required** when you build a `Document` directly. The
  `__setitem__` and `add_documents` paths auto-generate IDs when you pass a
  bare string, but the `Document` dataclass does not.
- **`search` is lazy.** `for r in docs.search(...)` is the idiom. `len()` does
  not work on the result.

## Where to go next

- Picking / installing / configuring a backend → `vd-backend-choose`
- Loading a real corpus (chunking, cleaning, batching) → `vd-ingest`
- Filters, multi-query, RRF, similar-to-doc, dedup → `vd-search`
- Export / import / migrate / analytics / benchmarking → `vd-ops`
