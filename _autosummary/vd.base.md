# vd.base

Core contracts, data model, and abstract bases for the `vd` vectorDB facade.

`vd` is a facade over vector databases. This module is the single source of
truth for the contract every backend adapter satisfies and that all of `vd`’s
higher-level tooling (search, io, migration, analytics, …) is written against.

The contract, smallest-first:

- [`Document`](#vd.base.Document) — the unit stored in a collection: `id`, `text`,
  `vector`, `metadata`.
- [`Collection`](#vd.base.Collection) — a `MutableMapping[str, Document]` *plus* a
  [`search()`](#vd.base.Collection.search) method. This is the one retrieval extension.
- [`Client`](#vd.base.Client) — a `Mapping[str, Collection]`: a live connection to one
  backend, through which collections are created, fetched, and dropped.
- [`AbstractCollection`](#vd.base.AbstractCollection) / [`AbstractClient`](#vd.base.AbstractClient) — adapter-author
  conveniences. A backend implements a handful of *raw primitives*; these bases
  supply everything users actually see (flexible inputs, optional text
  embedding, `egress` transforms, batch helpers, dimension checks) uniformly.

Embedding is deliberately **external**. `vd` stores and searches *vectors*;
turning text into vectors is another package’s job (e.g. `ef`). A
[`Client`](#vd.base.Client) may be handed an optional `embedder` callable purely as a
convenience, so `collection["k"] = "some text"` and
`collection.search("query text")` work. With no embedder configured those
text forms raise [`EmbeddingRequiredError`](#vd.base.EmbeddingRequiredError), and the caller must pass
[`Document`](#vd.base.Document) objects (or pre-computed vectors) directly.

### Module Attributes

| [`DocumentInput`](#vd.base.DocumentInput)   | A document may be supplied to batch operations in several flexible shapes.   |
|------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`METRICS`](#vd.base.METRICS)         | Distance metrics the facade understands.                                     |

### Classes

| [`AbstractClient`](#vd.base.AbstractClient)(\*[, embedder])          | Base class implementing the [`Client`](#vd.base.Client) contract for adapters.                                                                                                               |
|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AbstractCollection`](#vd.base.AbstractCollection)()                    | Base class implementing the [`Collection`](#vd.base.Collection) contract for adapters.                                                                                                           |
| [`AsyncClient`](#vd.base.AsyncClient)(\*args, \*\*kwargs)         | The async sibling of [`Client`](#vd.base.Client).                                                                                                                                            |
| [`AsyncCollection`](#vd.base.AsyncCollection)(\*args, \*\*kwargs)     | The async sibling of [`Collection`](#vd.base.Collection).                                                                                                                                        |
| [`Client`](#vd.base.Client)(\*args, \*\*kwargs)              | A live connection to one backend: `Mapping[str, Collection]`.                                                                                                                                                            |
| [`Collection`](#vd.base.Collection)(\*args, \*\*kwargs)          | A collection of documents: `MutableMapping[str, Document]` + `search`.                                                                                                                                                   |
| [`Document`](#vd.base.Document)(id[, text, vector, metadata])  | The unit stored in a [`Collection`](#vd.base.Collection).                                                                                                                                        |
| [`SupportsBatch`](#vd.base.SupportsBatch)(\*args, \*\*kwargs)       | A collection that supports efficient batch insertion.                                                                                                                                                                    |
| [`SupportsHybrid`](#vd.base.SupportsHybrid)(\*args, \*\*kwargs)      | A collection that supports native hybrid (dense + lexical) search.                                                                                                                                                       |
| [`SupportsNativeAsync`](#vd.base.SupportsNativeAsync)(\*args, \*\*kwargs) | Marker protocol set on async clients/collections that use a backend's native async SDK rather than the universal [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread) wrapper. |

### Exceptions

| [`BackendNotInstalledError`](#vd.base.BackendNotInstalledError)   | Raised when a known backend's Python package is not installed.         |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`EmbeddingRequiredError`](#vd.base.EmbeddingRequiredError)     | Raised when text is given but no embedder is configured.               |
| [`StaticIndexError`](#vd.base.StaticIndexError)           | Raised on a write to a static (immutable) index.                       |
| [`UnsupportedCapabilityError`](#vd.base.UnsupportedCapabilityError) | Raised when an operation needs a capability the backend lacks.         |
| [`UnsupportedFilterError`](#vd.base.UnsupportedFilterError)     | Raised when a metadata filter uses an operator a backend cannot honor. |
| [`VdError`](#vd.base.VdError)                    | Base class for every error `vd` raises on its own behalf.              |

### *class* vd.base.AbstractClient(, embedder=None, \*\*config)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Base class implementing the [`Client`](#vd.base.Client) contract for adapters.

A [`Client`](#vd.base.Client) is a `Mapping[str, Collection]`. A backend subclasses
this and implements [`create_collection()`](#vd.base.AbstractClient.create_collection), [`get_collection()`](#vd.base.AbstractClient.get_collection),
[`delete_collection()`](#vd.base.AbstractClient.delete_collection), and [`list_collections()`](#vd.base.AbstractClient.list_collections); the mapping
behavior, the [`get_or_create_collection()`](#vd.base.AbstractClient.get_or_create_collection) convenience, the `client`
escape hatch, and context-manager support come for free.

* **Parameters:**
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – A `text -> vector` function. Passed to every collection so text
    inputs are accepted as a convenience. `None` (the default) makes the
    client vector-only.
  * **\*\*config** – Backend-specific connection configuration.

#### backend_name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= ''*

The registry name of this backend (e.g. `"chroma"`). Adapters set it.

#### *property* client *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The raw backend client — a supported, documented escape hatch.

Drop to it for backend-specific operations the facade does not expose.
Returns `None` for backends with no external client object (e.g. the
in-memory backend).

#### close()

Release backend resources. Default no-op; adapters override as needed.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod* create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection.

* **Parameters:**
  * **name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Collection name.
  * **dimension** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Vector dimension. May be `None` for backends that can infer it
    from the first written vector; required up front by backends that
    cannot.
  * **metric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Distance metric: `"cosine"`, `"dot"`, or `"l2"`.
  * **\*\*index_config** – Backend-specific index tuning (HNSW `M`/`ef`, IVF `nlist`, …).
    Documented per adapter; never abstracted into a common enum.
* **Raises:**
  [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If a collection of that name already exists.
* **Return type:**
  [`Collection`](#vd.base.Collection)

#### *abstractmethod* delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod* get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`Collection`](#vd.base.Collection)

#### get_or_create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Return the collection `name`, creating it if it does not exist.

The common idiom that every consumer otherwise re-implements as a
`try get_collection / except KeyError: create_collection`.

* **Return type:**
  [`Collection`](#vd.base.Collection)

#### *abstractmethod* list_collections()

Iterate collection names.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* vd.base.AbstractCollection

Bases: `_CollectionPolicy`, [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Base class implementing the [`Collection`](#vd.base.Collection) contract for adapters.

A backend subclasses this and implements the *raw primitives* below;
everything users see is provided here, once, uniformly:

- flexible `__setitem__` inputs (text / tuple / [`Document`](#vd.base.Document)),
- optional text embedding when a `Document` arrives without a vector,
- text-query embedding in [`search()`](#vd.base.AbstractCollection.search),
- central filter validation against [`supported_filter_operators`](#vd.base.AbstractCollection.supported_filter_operators),
- `egress` result transforms,
- batch helpers ([`add_documents()`](#vd.base.AbstractCollection.add_documents), [`upsert()`](#vd.base.AbstractCollection.upsert)),
- eager dimension-mismatch detection.

## Subclass responsibilities (raw primitives)

`_write(doc)`
: Upsert one document. Its `vector` is guaranteed non-`None` and
  dimension-checked.

`_read(key) -> Document`
: Fetch one document; raise `KeyError` if absent.

`_drop(key)`
: Delete one document; raise `KeyError` if absent.

`_keys() -> Iterator[str]`
: Iterate document ids.

`_count() -> int`
: Number of documents.

`_query(vector, *, limit, filter, **kwargs) -> Iterable[SearchResult]`
: Raw nearest-neighbor search. `filter` is the canonical AST — the
  adapter translates it. Each result is a dict with at least `id`,
  `text`, `score`, `metadata`.

## Optional overrides

`_write_many(docs)`
: Efficient bulk upsert. Defaults to a loop over `_write`.

`native` (property)
: The raw backend collection handle (escape hatch).

#### add_documents(documents, , batch_size=100)

Add many documents, embedding and writing them in batches.

Each item may be a string, a `(text, ...)` tuple, or a
[`Document`](#vd.base.Document) (see [`DocumentInput`](#vd.base.DocumentInput)). Items without an `id`
get a deterministic auto-generated one.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* native *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The raw backend collection handle — a supported, documented escape hatch.

Use it to reach backend-specific features the facade does not expose,
rather than circumventing `vd`. Returns `None` if the adapter has
no distinct native object.

#### search(query, , limit=10, filter=None, egress=None, \*\*kwargs)

Return the `limit` documents most similar to `query`.

* **Parameters:**
  * **query** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Query text (embedded via the client’s `embedder`) or a
    pre-computed query vector.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of results.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Metadata filter in the canonical `vd` dialect (see
    [`vd.filters`](vd.filters.md#module-vd.filters)). Validated against this backend’s
    [`supported_filter_operators`](#vd.base.AbstractCollection.supported_filter_operators) before the query runs, so an
    unsupported operator fails with a clear [`UnsupportedFilterError`](#vd.base.UnsupportedFilterError).
  * **egress** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Transform applied to each result dict before it is yielded.
  * **\*\*kwargs** – Backend-specific search options, passed through to `_query`.
* **Yields:**
  *dict* – `{"id", "text", "score", "metadata"}` — or whatever `egress`
  returns. `score` is a higher-is-better, per-metric canonical
  similarity (see the “Score semantics” table at the top of
  [`vd.base`](#module-vd.base)): cosine in `[-1, 1]`, dot in `(-inf, +inf)`,
  l2 squashed to `(0, 1]`. Adapters whose backend returns a
  native combined-ranking score on a different scale (e.g.
  Elasticsearch, Atlas, Pinecone) document the deviation in
  their own docstring.
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### supported_filter_operators *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)* *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

the full language.
Adapters narrow this; [`search()`](#vd.base.AbstractCollection.search) validates against it.

* **Type:**
  Filter operators this backend can honor. Default

#### upsert(document)

Insert or replace `document` (equivalent to `self[doc.id] = doc`).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.base.AsyncClient(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The async sibling of [`Client`](#vd.base.Client).

Same operations — collection create / fetch / drop / list — exposed as
awaitables and async iterators. Construct via [`vd.connect_async()`](vd.md#vd.connect_async).

### *class* vd.base.AsyncCollection(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The async sibling of [`Collection`](#vd.base.Collection).

Same conceptual surface — storage + `search` — but every method is
awaitable and iterators are [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator). The mapping
interface is exposed as explicit `get` / `set` / `delete` / `keys`
/ `count` methods (the stdlib’s `MutableMapping` ABC has no async
counterpart; explicit methods are the Motor / aiopg convention).

Construct via [`vd.connect_async()`](vd.md#vd.connect_async); the universal
`AsyncCollectionWrapper` in [`vd.asynchronous`](vd.asynchronous.md#module-vd.asynchronous) adapts every
backend to this protocol by dispatching to the sync API through
[`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). Backends with native async SDKs override the
wrapper and additionally satisfy [`SupportsNativeAsync`](#vd.base.SupportsNativeAsync).

### *exception* vd.base.BackendNotInstalledError

Bases: [`VdError`](#vd.base.VdError), [`ImportError`](https://docs.python.org/3/builtins/exceptions.html#ImportError)

Raised when a known backend’s Python package is not installed.

Distinct from an *unknown* backend name (a plain `ValueError`): the
backend exists in `vd`’s provider registry, but its client library is
missing. The message carries the `pip install` command to fix it.

### *class* vd.base.Client(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A live connection to one backend: `Mapping[str, Collection]`.

Collections are created explicitly (so create-time parameters such as
`dimension` and `metric` can be supplied) and fetched either by
[`get_collection()`](#vd.base.Client.get_collection) or by mapping access `client[name]`.

#### create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection; raise `ValueError` if it exists.

* **Return type:**
  [`Collection`](#vd.base.Collection)

#### delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`Collection`](#vd.base.Collection)

#### list_collections()

Iterate collection names.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* vd.base.Collection(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection of documents: `MutableMapping[str, Document]` + `search`.

The mapping half is storage; [`search()`](#vd.base.Collection.search) is the single retrieval
extension. This minimal surface is everything `vd`’s tooling depends on.
Batch insertion is an *optional* capability — see [`SupportsBatch`](#vd.base.SupportsBatch).

#### search(query, , limit=10, filter=None, egress=None, \*\*kwargs)

Return the `limit` documents most similar to `query`.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *class* vd.base.Document(id, text='', vector=None, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The unit stored in a [`Collection`](#vd.base.Collection).

* **Parameters:**
  * **id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Unique identifier; the key under which the document lives in a
    collection.
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text content. May be empty for vector-first use cases where no
    text is associated with a vector.
  * **vector** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – The embedding. If `None` when written, the collection embeds
    `text` with its client’s `embedder` — or raises
    [`EmbeddingRequiredError`](#vd.base.EmbeddingRequiredError) if none is configured.
  * **metadata** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Arbitrary metadata, used for filtering and carried through search
    results.

### Examples

```pycon
>>> doc = Document(id="doc1", text="Hello world")
>>> doc.id, doc.text, doc.metadata
('doc1', 'Hello world', {})
>>> Document(id="v1", vector=[0.1, 0.2]).text
''
```

### vd.base.DocumentInput

A document may be supplied to batch operations in several flexible shapes.

alias of [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple) | [`Document`](#vd.base.Document)

### *exception* vd.base.EmbeddingRequiredError

Bases: [`VdError`](#vd.base.VdError), [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Raised when text is given but no embedder is configured.

`vd` operates on vectors. Passing raw text to `collection[key] = text`
or `collection.search(text)` only works when the [`Client`](#vd.base.Client) was
created with an `embedder`. Otherwise, pass a [`Document`](#vd.base.Document) with a
`vector` (or a pre-computed query vector) directly.

### vd.base.METRICS *= frozenset({'cosine', 'dot', 'l2'})*

Distance metrics the facade understands. Adapters map these to their own
spellings (e.g. `"l2"` -> Qdrant `Distance.EUCLID`).

### *exception* vd.base.StaticIndexError

Bases: [`VdError`](#vd.base.VdError)

Raised on a write to a static (immutable) index.

Some backends — notably a plain FAISS flat index — build an index that
cannot accept incremental `__setitem__` / `__delitem__` after creation.
Such collections set `AbstractCollection.supports_incremental_writes`
to `False` and raise this on write. Callers branch on that flag *before*
triggering the error, and use the adapter’s documented `rebuild()` path.

### *class* vd.base.SupportsBatch(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection that supports efficient batch insertion.

`add_documents` and `upsert` are *not* part of the minimal
[`Collection`](#vd.base.Collection) contract. Every adapter built on
[`AbstractCollection`](#vd.base.AbstractCollection) happens to provide them, but generic code
should still feature-discover:

```default
if isinstance(collection, SupportsBatch):
    collection.add_documents(many_docs, batch_size=256)
```

### *class* vd.base.SupportsHybrid(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection that supports native hybrid (dense + lexical) search.

Hybrid search has no syntactic convergence across vector databases, so it
is an opt-in capability, never baseline. Prefer the top-level
[`vd.hybrid_search()`](vd.md#vd.hybrid_search) — it dispatches to this protocol when the
collection implements it and falls back to a pure-Python BM25 + RRF
fusion otherwise. Feature-discover directly only when you specifically
need to refuse the fallback path:

```default
if isinstance(collection, SupportsHybrid):
    hits = collection.hybrid_search("query text", limit=20)
```

The portable contract is **Reciprocal Rank Fusion** (every native backend
supports it). Weighted-blend (`alpha`) and other backend-specific
fusion variants are accepted via `**kwargs` and documented per adapter
— they are not portable across backends.

* **Parameters:**
  * **query** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *or* [*list*](https://docs.python.org/3/builtins/stdtypes.html#list) *[*[*float*](https://docs.python.org/3/builtins/functions.html#float) *]*) – Query text (embedded via the collection’s embedder if configured)
    or a pre-computed query vector for the dense side.
  * **query_text** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str) *,* *optional*) – Explicit text for the lexical side. Defaults to `query` when
    `query` is a string. **Required** when `query` is a vector.
  * **limit** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of fused results to return.
  * **filter** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict) *,* *optional*) – Canonical `vd` metadata filter applied to both sub-searches.
  * **k_dense** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *,* *optional*) – How many results to fetch from each sub-search before fusion.
    Both default to `max(4 * limit, 50)`. Widen for higher recall.
  * **k_lexical** ([*int*](https://docs.python.org/3/builtins/functions.html#int) *,* *optional*) – How many results to fetch from each sub-search before fusion.
    Both default to `max(4 * limit, 50)`. Widen for higher recall.
  * **rrf_k** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Reciprocal Rank Fusion constant (typically 60).
  * **egress** (*callable* *,* *optional*) – Transform applied to each fused result before it is yielded.
  * **\*\*kwargs** – Backend-specific knobs (e.g. `alpha=0.7` on weaviate,
    `ranker="weighted"` on milvus). Documented per adapter.

### *class* vd.base.SupportsNativeAsync(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Marker protocol set on async clients/collections that use a backend’s
native async SDK rather than the universal [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
wrapper.

Why care: in high-concurrency event-loop apps (FastAPI, Starlette, etc.),
a `to_thread`-wrapped backend still blocks a worker thread per request.
For real non-blocking I/O, prefer collections that satisfy this protocol.
The wrapper sets this attribute to `False`; native adapters set it to
`True`. `isinstance(c, SupportsNativeAsync)` matches both — check
`c.native_async` for the boolean.

### *exception* vd.base.UnsupportedCapabilityError

Bases: [`VdError`](#vd.base.VdError), [`NotImplementedError`](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError)

Raised when an operation needs a capability the backend lacks.

Prefer feature-discovery — `isinstance(collection, SupportsHybrid)` — over
catching this, but it is the clear, typed fallback when an optional
operation is called on a backend that does not implement it.

### *exception* vd.base.UnsupportedFilterError

Bases: [`VdError`](#vd.base.VdError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised when a metadata filter uses an operator a backend cannot honor.

The canonical, backend-agnostic filter language lives in [`vd.filters`](vd.filters.md#module-vd.filters)
(a MongoDB-style JSON dialect). When a filter uses an operator outside a
backend’s documented subset — or one that does not exist at all — this is
raised, so the caller can simplify the filter or drop to the backend’s
native filter via the escape hatch (`collection.native`).

### *exception* vd.base.VdError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base class for every error `vd` raises on its own behalf.
