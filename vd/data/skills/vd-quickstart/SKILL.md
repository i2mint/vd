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

# vd — quickstart

`vd` is a **facade over vector databases**: one Pythonic interface to ~15
vector DBs. Switching backend is a one-word change to `connect`.

## The mental model (read this first)

`vd` stores and searches **vectors**. Turning text into vectors (embedding) is
*someone else's job* — `vd` never embeds on its own. Two ways to use it:

1. **Vector-first (the real contract).** You hold the embedding model. You give
   `vd` `Document` objects that already carry a `vector`, and you search with a
   pre-computed query vector.
2. **Text convenience.** Pass an `embedder` (a `text -> vector` callable) to
   `connect`. Then `collection["k"] = "raw text"` and `collection.search("a
   query")` work — `vd` calls your embedder for you.

With no embedder, passing raw text raises `EmbeddingRequiredError` — loud, not
silent.

## Happy path — vector-first

```python
import vd

client = vd.connect("memory")              # switch DB = change this one word
col = client.create_collection("docs")

col["a"] = vd.Document(id="a", text="cats and kittens", vector=[0.1, 0.9, 0.0])
col["b"] = vd.Document(id="b", text="fresh pizza",       vector=[0.9, 0.0, 0.1])

for hit in col.search([0.1, 0.8, 0.0], limit=2):
    print(hit["id"], hit["score"], hit["text"])
```

## Happy path — with a text embedder

```python
client = vd.connect("memory", embedder=my_text_to_vector_fn)
col = client.create_collection("docs")
col["a"] = "cats and kittens"                 # embedded for you
col["b"] = ("fresh pizza", {"kind": "food"})  # (text, metadata) tuple
hits = list(col.search("pets", limit=2))      # query text embedded for you
```

## The objects

- `vd.connect(backend, *, embedder=None, **backend_kwargs) -> Client` — the one
  entry point. `backend_kwargs` are backend-specific (`persist_directory`,
  `url`, `api_key`, `path`, …).
- `Client` is a `Mapping[str, Collection]`: `client["name"]`, `name in client`,
  `list(client)`, plus `create_collection`, `get_collection`,
  `delete_collection`, `get_or_create_collection`.
- `Collection` is a `MutableMapping[str, Document]` **plus** `.search(...)`.
- `Document(id, text="", vector=None, metadata={})` — the stored unit.

## Searching

```python
col.search(query, *, limit=10, filter=None, egress=None, **kwargs)
```

- `query` — text (needs an embedder) or a pre-computed vector.
- yields dicts: `{"id", "text", "score", "metadata"}` — `score` is
  higher-is-better.
- `filter` — MongoDB-style metadata filter (see the **vd-search** skill).
- `egress` — transform each result: `vd.id_only`, `vd.id_and_score`,
  `vd.text_only`, `vd.id_text_score`, or your own.

```python
ids = list(col.search(qvec, limit=5, egress=vd.id_only))
```

## Backends

`vd.list_backends()` shows what is installed and ready. `memory` is always
available. To pick and set up another, use the **vd-backend-choose** skill
(`vd.recommend_backend(...)`, `vd.check_requirements(...)`).

## Gotchas

- **No embedder → text fails loud.** Pass `embedder=` to `connect`, or hand
  `vd` `Document`s with vectors and pre-computed query vectors.
- **A collection's dimension is fixed** by its first vector; a mismatched
  vector later raises `ValueError`. Use a fresh collection per embedding model.
- **`memory` is not persistent.** For persistence use `chroma`
  (`persist_directory=`), `lancedb`/`duckdb`/`sqlite_vec` (`path=`), or a
  server/managed backend.
- **ids are strings.**
