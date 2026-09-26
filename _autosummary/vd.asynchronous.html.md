# vd.asynchronous

Async support for `vd`: universal wrapper + opt-in native implementations.

This module gives every `vd` backend an `async`/`await` surface day one,
without forking the adapter hierarchy. Three pieces:

- [`AsyncCollectionWrapper`](#vd.asynchronous.AsyncCollectionWrapper) / [`AsyncClientWrapper`](#vd.asynchronous.AsyncClientWrapper) —
  thin adapters that take any sync [`vd.Collection`](vd.html.md#vd.Collection) / [`vd.Client`](vd.html.md#vd.Client)
  and dispatch every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). This is the
  **universal fallback**: every backend works through it.
- [`AsyncAbstractCollection`](#vd.asynchronous.AsyncAbstractCollection) / [`AsyncAbstractClient`](#vd.asynchronous.AsyncAbstractClient) — bases for
  **native** async adapters, which do real non-blocking I/O through a
  backend’s own async SDK. A backend implements a few `async` raw
  primitives and registers its client with [`register_async_backend()`](#vd.asynchronous.register_async_backend).
  Native today: `qdrant` against a server (`url=`), on
  `qdrant_client.AsyncQdrantClient`.
- [`connect_async()`](#vd.asynchronous.connect_async) — the entry point. Mirrors [`vd.connect()`](vd.html.md#vd.connect). It
  returns the backend’s registered native client when there is one, and the
  wrapper otherwise (or when called with `native=False`).

The asyncio.to_thread wrapper does **not** make I/O non-blocking — it moves
blocking calls off the event loop onto a worker thread. For real
non-blocking I/O against a network backend, use a client whose
`native_async` attribute is `True` (see [`vd.SupportsNativeAsync`](vd.html.md#vd.SupportsNativeAsync)).

The module name is `vd.asynchronous` (not `vd.async`) because `async`
is a Python keyword.

### Functions

| [`register_async_backend`](#vd.asynchronous.register_async_backend)(name[, factory])           | Register a native async client factory for backend `name`.                                             |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [`list_async_backends`](#vd.asynchronous.list_async_backends)()                             | Return the names of backends with a registered native async client.                                    |
| [`connect_async`](#vd.asynchronous.connect_async)(backend, \*[, native])              | Async sibling of [`vd.connect()`](vd.html.md#vd.connect).             |
| [`hybrid_search_async`](#vd.asynchronous.hybrid_search_async)(collection, query, \*[, ...]) | Async sibling of [`vd.hybrid_search()`](vd.html.md#vd.hybrid_search). |

### Classes

| [`AsyncCollectionWrapper`](#vd.asynchronous.AsyncCollectionWrapper)(sync_collection)   | Adapt a sync [`Collection`](vd.html.md#vd.Collection) to the [`AsyncCollection`](vd.html.md#vd.AsyncCollection) contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).   |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AsyncClientWrapper`](#vd.asynchronous.AsyncClientWrapper)(sync_client)           | Adapt a sync [`Client`](vd.html.md#vd.Client) to the [`AsyncClient`](vd.html.md#vd.AsyncClient) contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).                   |
| [`AsyncAbstractCollection`](#vd.asynchronous.AsyncAbstractCollection)()                 | Base class for **native** async collections (the async sibling of [`vd.AbstractCollection`](vd.html.md#vd.AbstractCollection)).                                                                                                                                                            |
| [`AsyncAbstractClient`](#vd.asynchronous.AsyncAbstractClient)(\*[, embedder])       | Base class for **native** async clients (the async sibling of [`vd.AbstractClient`](vd.html.md#vd.AbstractClient)).                                                                                                                                                                        |
| [`SupportsHybrid`](#vd.asynchronous.SupportsHybrid)(\*args, \*\*kwargs)        | A collection that supports native hybrid (dense + lexical) search.                                                                                                                                                                                                                                                          |

### *class* vd.asynchronous.AsyncAbstractClient(, embedder=None, \*\*config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for **native** async clients (the async sibling of
[`vd.AbstractClient`](vd.html.md#vd.AbstractClient)).

A backend implements [`create_collection()`](#vd.asynchronous.AsyncAbstractClient.create_collection), [`get_collection()`](#vd.asynchronous.AsyncAbstractClient.get_collection),
[`delete_collection()`](#vd.asynchronous.AsyncAbstractClient.delete_collection) and [`list_collections()`](#vd.asynchronous.AsyncAbstractClient.list_collections) as coroutines /
async generators; [`get_or_create_collection()`](#vd.asynchronous.AsyncAbstractClient.get_or_create_collection), the `client` escape
hatch, [`close()`](#vd.asynchronous.AsyncAbstractClient.close) and `async with` support come for free. Register
the class with [`register_async_backend()`](#vd.asynchronous.register_async_backend) so [`connect_async()`](#vd.asynchronous.connect_async)
returns it.

* **Parameters:**
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – A `text -> vector` function handed to every collection.
  * **\*\*config** – Backend-specific connection configuration.

#### backend_name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= ''*

The registry name of this backend (set by [`register_async_backend()`](#vd.asynchronous.register_async_backend)).

#### *property* client *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The raw async backend client — a supported, documented escape hatch.

#### *async* close()

Release backend resources (closes the raw client if it can).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod async* create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection; raise `ValueError` if it exists.

* **Return type:**
  [`AsyncAbstractCollection`](#vd.asynchronous.AsyncAbstractCollection)

#### *abstractmethod async* delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod async* get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`AsyncAbstractCollection`](#vd.asynchronous.AsyncAbstractCollection)

#### *async* get_or_create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Return collection `name`, creating it if missing.

* **Return type:**
  [`AsyncAbstractCollection`](#vd.asynchronous.AsyncAbstractCollection)

#### *abstractmethod* list_collections()

Async-iterate collection names.

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### native_async *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= True*

Real non-blocking I/O through the backend’s async SDK.

### *class* vd.asynchronous.AsyncAbstractCollection

Bases: `_CollectionPolicy`

Base class for **native** async collections (the async sibling of
[`vd.AbstractCollection`](vd.html.md#vd.AbstractCollection)).

A backend subclasses this and implements `async` raw primitives; the
user-facing [`AsyncCollection`](vd.html.md#vd.AsyncCollection) surface is provided here, with
the same input coercion, embedding, dimension checks, filter validation
and `egress` handling as the sync base (both share one policy mixin).

## Subclass responsibilities (async raw primitives)

`async _write_many(docs)`
: Upsert documents; each `vector` is set and dimension-checked.

`async _read(key) -> Document`
: Fetch one document; raise `KeyError` if absent.

`async _drop(key)`
: Delete one document; raise `KeyError` if absent.

`_keys() -> AsyncIterator[str]`
: An async generator of document ids.

`async _count() -> int`
: Number of documents.

`async _query(vector, *, limit, filter, **kwargs) -> list[SearchResult]`
: Raw nearest-neighbor search; `filter` is the canonical AST.

#### *async* add_documents(documents, , batch_size=100)

Batch upsert — mirrors [`vd.AbstractCollection.add_documents()`](vd.html.md#vd.AbstractCollection.add_documents).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* count()

Return the number of documents.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### *async* delete(key)

Delete a document; raises `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* get(key)

Fetch one document; raises `KeyError` if absent.

* **Return type:**
  [`Document`](vd.base.html.md#vd.base.Document)

#### *async* keys()

Yield document ids.

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### *property* native *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The raw backend handle (escape hatch), or `None`.

#### native_async *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= True*

Real non-blocking I/O through the backend’s async SDK.

#### *async* search(query, , limit=10, filter=None, egress=None, \*\*kwargs)

Yield the `limit` documents most similar to `query`.

Same contract as [`vd.AbstractCollection.search()`](vd.html.md#vd.AbstractCollection.search).

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### *async* set(key, value)

Insert or replace a document (idempotent upsert).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### supported_filter_operators *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)* *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

the full language.
Adapters narrow this; [`search()`](#vd.asynchronous.AsyncAbstractCollection.search) validates against it.

* **Type:**
  Filter operators this backend can honor. Default

#### *async* upsert(document)

Insert or replace `document`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.asynchronous.AsyncClientWrapper(sync_client)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Adapt a sync [`Client`](vd.html.md#vd.Client) to the [`AsyncClient`](vd.html.md#vd.AsyncClient)
contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).

Use [`connect_async()`](#vd.asynchronous.connect_async) rather than instantiating this directly.

* **Parameters:**
  **sync_client** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A live [`Client`](vd.html.md#vd.Client) (typically obtained from [`vd.connect()`](vd.html.md#vd.connect)).

#### native_async

Always `False` for this wrapper.

* **Type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### *property* client *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

Pass through to the wrapped client’s `client`.

#### *async* close()

Release backend resources. Calls `close()` on the sync client if present.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection; raise `ValueError` if it exists.

* **Return type:**
  [`AsyncCollection`](vd.base.html.md#vd.base.AsyncCollection)

#### *async* delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`AsyncCollection`](vd.base.html.md#vd.base.AsyncCollection)

#### *async* get_or_create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Return collection `name`, creating it if missing.

* **Return type:**
  [`AsyncCollection`](vd.base.html.md#vd.base.AsyncCollection)

#### *async* list_collections()

Yield collection names.

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### *property* sync *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The underlying sync [`Client`](vd.html.md#vd.Client) — a documented escape hatch.

### *class* vd.asynchronous.AsyncCollectionWrapper(sync_collection)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Adapt a sync [`Collection`](vd.html.md#vd.Collection) to the [`AsyncCollection`](vd.html.md#vd.AsyncCollection)
contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).

Use [`connect_async()`](#vd.asynchronous.connect_async) rather than instantiating this directly — it
will pick this wrapper or a native async adapter as appropriate.

* **Parameters:**
  **::** (*sync_collection*) – A live [`Collection`](vd.html.md#vd.Collection) (typically obtained from a
  [`Client`](vd.html.md#vd.Client)).

#### native_async

Always `False` for this wrapper. The wrapper still satisfies
[`SupportsNativeAsync`](vd.html.md#vd.SupportsNativeAsync) structurally (the attribute is
present), but the boolean tells callers that I/O is happening in a
thread pool rather than on the event loop. Prefer a native
implementation for high-concurrency workloads.

* **Type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### *async* add_documents(documents, , batch_size=100)

Batch upsert — mirrors [`add_documents()`](vd.html.md#vd.AbstractCollection.add_documents).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* count()

Return the number of documents.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

#### *async* delete(key)

Delete a document; raises `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *async* get(key)

Fetch one document; raises `KeyError` if absent.

* **Return type:**
  [`Document`](vd.base.html.md#vd.base.Document)

#### *async* keys()

Yield document ids.

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### *property* native *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

Pass through to the wrapped collection’s `native`.

#### native_async *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= False*

This wrapper offloads to a thread pool; it doesn’t do non-blocking I/O.

#### *async* search(query, , limit=10, filter=None, egress=None, \*\*kwargs)

Yield the `limit` documents most similar to `query`.

The underlying search runs once on a worker thread; results stream
from memory. (Most backends’ sync `search` already returns a list
or a fully-realized iterator under the hood.)

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### *async* set(key, value)

Insert or replace a document (idempotent upsert).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *property* sync *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The underlying sync [`Collection`](vd.html.md#vd.Collection) — a documented escape hatch.

#### *async* upsert(document)

Insert or replace `document`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.asynchronous.SupportsHybrid(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection that supports native hybrid (dense + lexical) search.

Hybrid search has no syntactic convergence across vector databases, so it
is an opt-in capability, never baseline. Prefer the top-level
[`vd.hybrid_search()`](vd.html.md#vd.hybrid_search) — it dispatches to this protocol when the
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

### *async* vd.asynchronous.connect_async(backend, , native=True, \*\*kwargs)

Async sibling of [`vd.connect()`](vd.html.md#vd.connect).

Returns an [`AsyncClient`](vd.html.md#vd.AsyncClient). When the backend has a native async
client (see [`list_async_backends()`](#vd.asynchronous.list_async_backends); today `qdrant`), its registered
factory decides: it may return a native client doing real non-blocking
I/O (qdrant with `url=`) or the wrapper (embedded qdrant, whose async
client would block the loop). Every other backend goes
through the universal [`AsyncClientWrapper`](#vd.asynchronous.AsyncClientWrapper), built on
[`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). Check `client.native_async` to tell them
apart.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend name — same vocabulary as [`vd.connect()`](vd.html.md#vd.connect).
  * **native** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use the backend’s native async client when one is registered
    (default). `False` forces the `to_thread` wrapper around the
    sync adapter.
  * **\*\*kwargs** – Forwarded to the native client’s constructor, or to
    [`vd.connect()`](vd.html.md#vd.connect) for the wrapper. Both take the same arguments
    (`embedder`, `url`, `path`, …).
* **Returns:**
  A live async client. `await` once at session start:
  ```default
  client = await vd.connect_async("memory")
  ```
* **Return type:**
  [`AsyncClient`](vd.base.html.md#vd.base.AsyncClient)

### Examples

```pycon
>>> import asyncio, vd
>>> async def go():
...     client = await vd.connect_async("memory")
...     col = await client.create_collection("docs", dimension=2)
...     await col.set("a", vd.Document(id="a", text="x", vector=[1.0, 0.0]))
...     return await col.count()
>>> asyncio.run(go())
1
```

### *async* vd.asynchronous.hybrid_search_async(collection, query, , query_text=None, limit=10, filter=None, k_dense=None, k_lexical=None, rrf_k=60, lexical_search=None, egress=None, \*\*kwargs)

Async sibling of [`vd.hybrid_search()`](vd.html.md#vd.hybrid_search).

For a wrapped sync collection, runs [`vd.hybrid_search()`](vd.html.md#vd.hybrid_search) (native
hybrid if the backend has it, else the client-side BM25 + RRF fallback)
on a worker thread. For a native async collection it awaits the
collection’s own `hybrid_search` if it has one, and otherwise fuses the
collection’s async dense search with a client-side BM25 scan (O(N): it
reads every document) via RRF. On a native collection, an `async def`
`lexical_search` receives the async collection; a sync one receives a
materialized `{id: Document}` dict (every document is read per call)
and runs on a worker thread. Either way the awaitable + async iterator
interface stays uniform.

Parameters mirror [`vd.hybrid_search()`](vd.html.md#vd.hybrid_search) exactly; see that function for
the full docs.

* **Yields:**
  *dict* – Fused result dicts.
* **Return type:**
  [*AsyncIterator*](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import asyncio, vd
>>> async def go():
...     client = await vd.connect_async("memory")
...     col = await client.create_collection("docs", dimension=2)
...     await col.set("a", vd.Document(id="a", text="cats",
...                                    vector=[1.0, 0.0]))
...     await col.set("b", vd.Document(id="b", text="dogs",
...                                    vector=[0.0, 1.0]))
...     hits = []
...     async for h in vd.hybrid_search_async(col, [0.9, 0.1],
...                                           query_text="cats", limit=1):
...         hits.append(h["id"])
...     return hits
>>> asyncio.run(go())
['a']
```

### vd.asynchronous.list_async_backends()

Return the names of backends with a registered native async client.

Every other backend still works with [`connect_async()`](#vd.asynchronous.connect_async), through the
universal `to_thread` wrapper.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### vd.asynchronous.register_async_backend(name, factory=None)

Register a native async client factory for backend `name`.

Use as a class decorator on an [`AsyncAbstractClient`](#vd.asynchronous.AsyncAbstractClient) subclass, or
call it with a `factory` (a class or a function, sync or `async`,
taking the [`connect_async()`](#vd.asynchronous.connect_async) keyword arguments). Once registered,
[`connect_async()`](#vd.asynchronous.connect_async) returns the factory’s client instead of the
`to_thread` wrapper.

### Examples

```pycon
>>> @register_async_backend('example')
... class ExampleAsyncClient(AsyncAbstractClient):
...     ...
```
