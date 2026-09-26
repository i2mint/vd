# vd

`vd` — one Pythonic interface to every vector database.

`vd` is a **facade over vector databases**. Its purpose is to let you operate
on any vectorDB, and switch between them with a one-argument change, while
keeping each backend’s particular power one escape hatch away. It does three
things:

1. **Choose** — [`recommend_backend()`](#vd.recommend_backend), [`print_backends_table()`](#vd.print_backends_table) and the
   provider registry help you (or an AI agent) pick the right backend.
2. **Set up** — [`check_requirements()`](#vd.check_requirements) and [`setup_guide()`](#vd.setup_guide) diagnose and
   walk you through installing and starting a backend.
3. **Operate** — [`connect()`](#vd.connect) returns a uniform client; collections behave
   as `MutableMapping` of [`Document`](#vd.Document) plus a [`search()`](vd.search.html.md#module-vd.search) method.

## Quick start

```pycon
>>> import vd
>>> client = vd.connect('memory')          # switch DB = change this one word
>>> col = client.create_collection('docs')
>>> col['a'] = vd.Document(id='a', text='cats', vector=[1.0, 0.0])
>>> col['b'] = vd.Document(id='b', text='dogs', vector=[0.0, 1.0])
>>> [hit['id'] for hit in col.search([0.9, 0.1], limit=1)]
['a']
```

## Embedding is external

`vd` stores and searches *vectors*. Turning text into vectors is another
package’s job (e.g. `ef`). Pass an `embedder` to [`connect()`](#vd.connect) only for
the *convenience* of writing/searching raw text; otherwise pass
[`Document`](#vd.Document) objects carrying vectors, and pre-computed query vectors.

### Functions

| [`skills_dir`](#vd.skills_dir)()                                       | Return the path to the bundled AI-agent skills directory.                                                                                                                             |
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`connect`](#vd.connect)(backend, \*[, embedder])                   | Connect to a vector database backend and return its [`Client`](vd.base.html.md#vd.base.Client).                                                           |
| [`register_backend`](#vd.register_backend)(name)                             | Class decorator: register an adapter [`Client`](vd.base.html.md#vd.base.Client) under `name`.                                                             |
| [`connect_async`](#vd.connect_async)(backend, \*[, native])               | Async sibling of [`vd.connect()`](#vd.connect).                                                                                                       |
| [`hybrid_search_async`](#vd.hybrid_search_async)(collection, query, \*[, ...])  | Async sibling of [`vd.hybrid_search()`](#vd.hybrid_search).                                                                                                 |
| [`register_async_backend`](#vd.register_async_backend)(name[, factory])            | Register a native async client factory for backend `name`.                                                                                                                            |
| [`list_async_backends`](#vd.list_async_backends)()                              | Return the names of backends with a registered native async client.                                                                                                                   |
| [`matches_filter`](#vd.matches_filter)(metadata, filter)                   | Return `True` if `metadata` satisfies the MongoDB-style `filter`.                                                                                                                     |
| [`validate_filter`](#vd.validate_filter)(filter, \*[, supported])           | Walk `filter` and raise [`UnsupportedFilterError`](vd.base.html.md#vd.base.UnsupportedFilterError) on any operator that is unknown or not in `supported`. |
| [`list_backends`](#vd.list_backends)()                                    | Return the names of all backends with a registered (importable) adapter.                                                                                                              |
| [`list_available_backends`](#vd.list_available_backends)()                          | Return providers `vd` can [`connect()`](#vd.connect) *right now*.                                                                                     |
| [`list_all_backends`](#vd.list_all_backends)()                                | Return every provider with live `installed` / `has_adapter` flags added.                                                                                                              |
| [`print_backends_table`](#vd.print_backends_table)()                             | Print every known vector database, grouped by deployment archetype.                                                                                                                   |
| [`providers`](#vd.providers)()                                        | Return the full provider registry as `{name: metadata}`.                                                                                                                              |
| [`provider`](#vd.provider)(name)                                     | Return one provider's metadata, or `None` if `name` is unknown.                                                                                                                       |
| [`get_backend_info`](#vd.get_backend_info)(name)                             | Return one provider's metadata with `installed`/`has_adapter` flags.                                                                                                                  |
| [`get_backend_characteristics`](#vd.get_backend_characteristics)()                      | Return a compact `{name: characteristics}` map for comparison tooling.                                                                                                                |
| [`get_install_instructions`](#vd.get_install_instructions)(name)                     | Return a human-readable setup blurb for one provider.                                                                                                                                 |
| [`install_command`](#vd.install_command)(name)                              | Return the `pip install` command that makes `name` usable.                                                                                                                            |
| [`compare_backends`](#vd.compare_backends)(names, \*[, characteristics])     | Return a `{name: {characteristic: value}}` table for the given providers.                                                                                                             |
| [`print_comparison`](#vd.print_comparison)(names)                            | Print a side-by-side comparison table of the given providers.                                                                                                                         |
| [`recommend_backend`](#vd.recommend_backend)(\*[, corpus_size, ...])          | Recommend a vector database from a few yes/no facts about the situation.                                                                                                              |
| [`print_recommendation`](#vd.print_recommendation)(\*\*kwargs)                   | Run [`recommend_backend()`](#vd.recommend_backend) and print the recommendation readably.                                                                       |
| [`check_requirements`](#vd.check_requirements)(backend, \*[, verbose])         | Diagnose whether `backend` is ready to use, and say what to do if not.                                                                                                                |
| [`setup_guide`](#vd.setup_guide)(backend)                               | Return a full, copy-pasteable setup playbook for `backend`.                                                                                                                           |
| [`install_backend`](#vd.install_backend)(backend, \*[, run])                | Return (and optionally run) the `pip install` command for `backend`.                                                                                                                  |
| [`text_only`](#vd.text_only)(result)                                  | Egress: keep only the text.                                                                                                                                                           |
| [`id_only`](#vd.id_only)(result)                                    | Egress: keep only the document id.                                                                                                                                                    |
| [`id_and_score`](#vd.id_and_score)(result)                               | Egress: keep `(id, score)`.                                                                                                                                                           |
| [`id_text_score`](#vd.id_text_score)(result)                              | Egress: keep `(id, text, score)`.                                                                                                                                                     |
| [`cosine_similarity`](#vd.cosine_similarity)(vec1, vec2)                      | Cosine similarity of two vectors (1.0 identical, 0.0 orthogonal).                                                                                                                     |
| [`euclidean_distance`](#vd.euclidean_distance)(vec1, vec2)                     | Euclidean (L2) distance between two vectors.                                                                                                                                          |
| [`normalize_document_input`](#vd.normalize_document_input)(doc_input, \*[, auto_id]) | Normalize a flexible document input to a [`Document`](vd.base.html.md#vd.base.Document).                                                                  |
| [`export_collection`](#vd.export_collection)(collection, output_path, \*)     | Export a collection to a file in the specified format.                                                                                                                                |
| [`import_collection`](#vd.import_collection)(collection, input_path, \*)      | Import documents into a collection from a file.                                                                                                                                       |
| [`export_to_jsonl`](#vd.export_to_jsonl)(collection, output_path, \*)       | Export a collection to JSONL (JSON Lines) format.                                                                                                                                     |
| [`import_from_jsonl`](#vd.import_from_jsonl)(collection, input_path, \*)      | Import documents from JSONL format into a collection.                                                                                                                                 |
| [`export_to_json`](#vd.export_to_json)(collection, output_path, \*[, ...]) | Export a collection to JSON format.                                                                                                                                                   |
| [`import_from_json`](#vd.import_from_json)(collection, input_path, \*)       | Import documents from JSON format into a collection.                                                                                                                                  |
| [`export_to_directory`](#vd.export_to_directory)(collection, output_dir, \*)    | Export collection as a directory with one JSON file per document.                                                                                                                     |
| [`import_from_directory`](#vd.import_from_directory)(collection, input_dir, \*)   | Import documents from a directory of JSON files.                                                                                                                                      |
| [`migrate_collection`](#vd.migrate_collection)(source_collection, ...[, ...])  | Migrate a collection from one backend to another.                                                                                                                                     |
| [`migrate_client`](#vd.migrate_client)(source_client, target_client, \*)   | Migrate all (or selected) collections from one client to another.                                                                                                                     |
| [`copy_collection`](#vd.copy_collection)(source, target, \*[, ...])         | Copy a collection with flexible source/target specification.                                                                                                                          |
| [`collection_stats`](#vd.collection_stats)(collection)                       | Compute comprehensive statistics for a collection.                                                                                                                                    |
| [`metadata_distribution`](#vd.metadata_distribution)(collection, field, \*)       | Get the distribution of values for a metadata field.                                                                                                                                  |
| [`find_duplicates`](#vd.find_duplicates)(collection, \*[, threshold, ...])  | Find near-duplicate documents in a collection.                                                                                                                                        |
| [`find_outliers`](#vd.find_outliers)(collection, \*[, n_neighbors, ...])  | Find outlier documents (those dissimilar to their neighbors).                                                                                                                         |
| [`sample_collection`](#vd.sample_collection)(collection, n, \*[, ...])        | Sample document IDs from a collection.                                                                                                                                                |
| [`validate_collection`](#vd.validate_collection)(collection)                    | Validate collection integrity and identify issues.                                                                                                                                    |
| [`chunk_text`](#vd.chunk_text)(text[, chunk_size, overlap, ...])       | Chunk text into smaller pieces.                                                                                                                                                       |
| [`chunk_documents`](#vd.chunk_documents)(documents[, chunk_size, ...])      | Chunk multiple documents while preserving metadata.                                                                                                                                   |
| [`clean_text`](#vd.clean_text)(text, \*[, lowercase, ...])             | Clean and normalize text.                                                                                                                                                             |
| [`normalize_whitespace`](#vd.normalize_whitespace)(text)                         | Normalize whitespace in text.                                                                                                                                                         |
| [`truncate_text`](#vd.truncate_text)(text, max_length, \*[, suffix])      | Truncate text to maximum length.                                                                                                                                                      |
| [`extract_metadata`](#vd.extract_metadata)(text, \*[, extract_title, ...])   | Extract metadata from text.                                                                                                                                                           |
| [`health_check_backend`](#vd.health_check_backend)(backend_name, \*\*config)     | Check if a backend is healthy and accessible.                                                                                                                                         |
| [`health_check_collection`](#vd.health_check_collection)(collection)                | Check collection health and compute basic stats.                                                                                                                                      |
| [`benchmark_search`](#vd.benchmark_search)(collection, query, \*[, ...])     | Benchmark search performance on a collection.                                                                                                                                         |
| [`benchmark_insert`](#vd.benchmark_insert)(collection[, n_documents, ...])   | Benchmark document insertion performance.                                                                                                                                             |
| [`multi_query_search`](#vd.multi_query_search)(collection, queries, \*[, ...]) | Search with multiple queries and combine results.                                                                                                                                     |
| [`search_similar_to_document`](#vd.search_similar_to_document)(collection, doc_id, \*) | Find documents similar to a specific document.                                                                                                                                        |
| [`reciprocal_rank_fusion`](#vd.reciprocal_rank_fusion)(result_lists, \*[, k])      | Combine multiple result lists using Reciprocal Rank Fusion.                                                                                                                           |
| [`deduplicate_results`](#vd.deduplicate_results)(results, \*[, key, keep])      | Remove duplicate results.                                                                                                                                                             |
| [`hybrid_search`](#vd.hybrid_search)(collection, query, \*[, ...])        | Hybrid (dense + lexical) search that works on any vd Collection.                                                                                                                      |
| [`bm25_lexical_search`](#vd.bm25_lexical_search)(collection, query_text, \*)    | Brute-force BM25 lexical search over a vd collection's stored `text`.                                                                                                                 |
| [`load_config`](#vd.load_config)([path, format])                        | Load configuration from a file.                                                                                                                                                       |
| [`save_config`](#vd.save_config)(config, path, \*[, format])            | Save configuration to a file.                                                                                                                                                         |
| [`connect_from_config`](#vd.connect_from_config)([path, profile, ...])          | Connect to a backend using configuration from a file.                                                                                                                                 |
| [`create_example_config`](#vd.create_example_config)([format])                    | Generate an example configuration file content.                                                                                                                                       |
| [`count_docs`](#vd.count_docs)(docs)                                   | `len` reducer that also handles generator inputs.                                                                                                                                     |
| [`mean_vector`](#vd.mean_vector)(docs)                                  | Element-wise mean of document embeddings.                                                                                                                                             |
| [`parse_window`](#vd.parse_window)(window)                               | Parse a window spec into a `timedelta`.                                                                                                                                               |
| [`to_datetime`](#vd.to_datetime)(ts)                                    | Coerce a timestamp-like value into a tz-aware UTC `datetime`.                                                                                                                         |
| [`to_iso`](#vd.to_iso)(ts)                                         | ISO-8601 (UTC) string suitable for cross-backend metadata storage.                                                                                                                    |

### Classes

| [`Document`](#vd.Document)(id[, text, vector, metadata])        | The unit stored in a [`Collection`](#vd.Collection).                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`Client`](#vd.Client)(\*args, \*\*kwargs)                    | A live connection to one backend: `Mapping[str, Collection]`.                                                                                                                                                                                                                            |
| [`Collection`](#vd.Collection)(\*args, \*\*kwargs)                | A collection of documents: `MutableMapping[str, Document]` + `search`.                                                                                                                                                                                                                   |
| [`AbstractClient`](#vd.AbstractClient)(\*[, embedder])                | Base class implementing the [`Client`](#vd.Client) contract for adapters.                                                                                                                                                                               |
| [`AbstractCollection`](#vd.AbstractCollection)()                          | Base class implementing the [`Collection`](#vd.Collection) contract for adapters.                                                                                                                                                                           |
| [`SupportsBatch`](#vd.SupportsBatch)(\*args, \*\*kwargs)             | A collection that supports efficient batch insertion.                                                                                                                                                                                                                                    |
| [`SupportsHybrid`](#vd.SupportsHybrid)(\*args, \*\*kwargs)            | A collection that supports native hybrid (dense + lexical) search.                                                                                                                                                                                                                       |
| [`SupportsNativeAsync`](#vd.SupportsNativeAsync)(\*args, \*\*kwargs)       | Marker protocol set on async clients/collections that use a backend's native async SDK rather than the universal [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread) wrapper.                                                                 |
| [`AsyncClient`](#vd.AsyncClient)(\*args, \*\*kwargs)               | The async sibling of [`Client`](#vd.Client).                                                                                                                                                                                                            |
| [`AsyncCollection`](#vd.AsyncCollection)(\*args, \*\*kwargs)           | The async sibling of [`Collection`](#vd.Collection).                                                                                                                                                                                                        |
| [`AsyncClientWrapper`](#vd.AsyncClientWrapper)(sync_client)               | Adapt a sync [`Client`](#vd.Client) to the [`AsyncClient`](#vd.AsyncClient) contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).         |
| [`AsyncCollectionWrapper`](#vd.AsyncCollectionWrapper)(sync_collection)       | Adapt a sync [`Collection`](#vd.Collection) to the [`AsyncCollection`](#vd.AsyncCollection) contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). |
| [`AsyncAbstractClient`](#vd.AsyncAbstractClient)(\*[, embedder])           | Base class for **native** async clients (the async sibling of [`vd.AbstractClient`](#vd.AbstractClient)).                                                                                                                                                       |
| [`AsyncAbstractCollection`](#vd.AsyncAbstractCollection)()                     | Base class for **native** async collections (the async sibling of [`vd.AbstractCollection`](#vd.AbstractCollection)).                                                                                                                                               |
| [`BM25Index`](#vd.BM25Index)(collection, \*[, filter, tokenize]) | A reusable Okapi BM25 index over a vd collection's stored `text`.                                                                                                                                                                                                                        |
| [`TimeIndexedCollection`](#vd.TimeIndexedCollection)(collection, \*[, ...])  | Time-indexed wrapper over any vd `Collection`.                                                                                                                                                                                                                                           |
| [`WindowSlice`](#vd.WindowSlice)(start, end)                       | A time window: `[start, end)`.                                                                                                                                                                                                                                                           |

### Exceptions

| [`VdError`](#vd.VdError)                    | Base class for every error `vd` raises on its own behalf.              |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`StaticIndexError`](#vd.StaticIndexError)           | Raised on a write to a static (immutable) index.                       |
| [`UnsupportedFilterError`](#vd.UnsupportedFilterError)     | Raised when a metadata filter uses an operator a backend cannot honor. |
| [`UnsupportedCapabilityError`](#vd.UnsupportedCapabilityError) | Raised when an operation needs a capability the backend lacks.         |
| [`EmbeddingRequiredError`](#vd.EmbeddingRequiredError)     | Raised when text is given but no embedder is configured.               |
| [`BackendNotInstalledError`](#vd.BackendNotInstalledError)   | Raised when a known backend's Python package is not installed.         |

### *class* vd.AbstractClient(, embedder=None, \*\*config)

Bases: [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

Base class implementing the [`Client`](#vd.Client) contract for adapters.

A [`Client`](#vd.Client) is a `Mapping[str, Collection]`. A backend subclasses
this and implements [`create_collection()`](#vd.AbstractClient.create_collection), [`get_collection()`](#vd.AbstractClient.get_collection),
[`delete_collection()`](#vd.AbstractClient.delete_collection), and [`list_collections()`](#vd.AbstractClient.list_collections); the mapping
behavior, the [`get_or_create_collection()`](#vd.AbstractClient.get_or_create_collection) convenience, the `client`
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
  [`Collection`](vd.base.html.md#vd.base.Collection)

#### *abstractmethod* delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod* get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`Collection`](vd.base.html.md#vd.base.Collection)

#### get_or_create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Return the collection `name`, creating it if it does not exist.

The common idiom that every consumer otherwise re-implements as a
`try get_collection / except KeyError: create_collection`.

* **Return type:**
  [`Collection`](vd.base.html.md#vd.base.Collection)

#### *abstractmethod* list_collections()

Iterate collection names.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* vd.AbstractCollection

Bases: `_CollectionPolicy`, [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Base class implementing the [`Collection`](#vd.Collection) contract for adapters.

A backend subclasses this and implements the *raw primitives* below;
everything users see is provided here, once, uniformly:

- flexible `__setitem__` inputs (text / tuple / [`Document`](#vd.Document)),
- optional text embedding when a `Document` arrives without a vector,
- text-query embedding in [`search()`](vd.search.html.md#module-vd.search),
- central filter validation against [`supported_filter_operators`](#vd.AbstractCollection.supported_filter_operators),
- `egress` result transforms,
- batch helpers ([`add_documents()`](#vd.AbstractCollection.add_documents), [`upsert()`](#vd.AbstractCollection.upsert)),
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
[`Document`](#vd.Document) (see `DocumentInput`). Items without an `id`
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
    [`vd.filters`](vd.filters.html.md#module-vd.filters)). Validated against this backend’s
    [`supported_filter_operators`](#vd.AbstractCollection.supported_filter_operators) before the query runs, so an
    unsupported operator fails with a clear [`UnsupportedFilterError`](#vd.UnsupportedFilterError).
  * **egress** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Transform applied to each result dict before it is yielded.
  * **\*\*kwargs** – Backend-specific search options, passed through to `_query`.
* **Yields:**
  *dict* – `{"id", "text", "score", "metadata"}` — or whatever `egress`
  returns. `score` is a higher-is-better, per-metric canonical
  similarity (see the “Score semantics” table at the top of
  [`vd.base`](vd.base.html.md#module-vd.base)): cosine in `[-1, 1]`, dot in `(-inf, +inf)`,
  l2 squashed to `(0, 1]`. Adapters whose backend returns a
  native combined-ranking score on a different scale (e.g.
  Elasticsearch, Atlas, Pinecone) document the deviation in
  their own docstring.
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### supported_filter_operators *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)* *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

the full language.
Adapters narrow this; [`search()`](vd.search.html.md#module-vd.search) validates against it.

* **Type:**
  Filter operators this backend can honor. Default

#### upsert(document)

Insert or replace `document` (equivalent to `self[doc.id] = doc`).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.AsyncAbstractClient(, embedder=None, \*\*config)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Base class for **native** async clients (the async sibling of
[`vd.AbstractClient`](#vd.AbstractClient)).

A backend implements [`create_collection()`](#vd.AsyncAbstractClient.create_collection), [`get_collection()`](#vd.AsyncAbstractClient.get_collection),
[`delete_collection()`](#vd.AsyncAbstractClient.delete_collection) and [`list_collections()`](#vd.AsyncAbstractClient.list_collections) as coroutines /
async generators; [`get_or_create_collection()`](#vd.AsyncAbstractClient.get_or_create_collection), the `client` escape
hatch, [`close()`](#vd.AsyncAbstractClient.close) and `async with` support come for free. Register
the class with [`register_async_backend()`](#vd.register_async_backend) so [`connect_async()`](#vd.connect_async)
returns it.

* **Parameters:**
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – A `text -> vector` function handed to every collection.
  * **\*\*config** – Backend-specific connection configuration.

#### backend_name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= ''*

The registry name of this backend (set by [`register_async_backend()`](#vd.register_async_backend)).

#### *property* client *: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The raw async backend client — a supported, documented escape hatch.

#### *async* close()

Release backend resources (closes the raw client if it can).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod async* create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection; raise `ValueError` if it exists.

* **Return type:**
  [`AsyncAbstractCollection`](vd.asynchronous.html.md#vd.asynchronous.AsyncAbstractCollection)

#### *abstractmethod async* delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### *abstractmethod async* get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`AsyncAbstractCollection`](vd.asynchronous.html.md#vd.asynchronous.AsyncAbstractCollection)

#### *async* get_or_create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Return collection `name`, creating it if missing.

* **Return type:**
  [`AsyncAbstractCollection`](vd.asynchronous.html.md#vd.asynchronous.AsyncAbstractCollection)

#### *abstractmethod* list_collections()

Async-iterate collection names.

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### native_async *: [bool](https://docs.python.org/3/builtins/functions.html#bool)* *= True*

Real non-blocking I/O through the backend’s async SDK.

### *class* vd.AsyncAbstractCollection

Bases: `_CollectionPolicy`

Base class for **native** async collections (the async sibling of
[`vd.AbstractCollection`](#vd.AbstractCollection)).

A backend subclasses this and implements `async` raw primitives; the
user-facing [`AsyncCollection`](#vd.AsyncCollection) surface is provided here, with
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

Batch upsert — mirrors [`vd.AbstractCollection.add_documents()`](#vd.AbstractCollection.add_documents).

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

Same contract as [`vd.AbstractCollection.search()`](#vd.AbstractCollection.search).

* **Return type:**
  [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

#### *async* set(key, value)

Insert or replace a document (idempotent upsert).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### supported_filter_operators *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)* *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

the full language.
Adapters narrow this; [`search()`](vd.search.html.md#module-vd.search) validates against it.

* **Type:**
  Filter operators this backend can honor. Default

#### *async* upsert(document)

Insert or replace `document`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.AsyncClient(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The async sibling of [`Client`](#vd.Client).

Same operations — collection create / fetch / drop / list — exposed as
awaitables and async iterators. Construct via [`vd.connect_async()`](#vd.connect_async).

### *class* vd.AsyncClientWrapper(sync_client)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Adapt a sync [`Client`](#vd.Client) to the [`AsyncClient`](#vd.AsyncClient)
contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).

Use [`connect_async()`](#vd.connect_async) rather than instantiating this directly.

* **Parameters:**
  **sync_client** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – A live [`Client`](#vd.Client) (typically obtained from [`vd.connect()`](#vd.connect)).

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

The underlying sync [`Client`](#vd.Client) — a documented escape hatch.

### *class* vd.AsyncCollection(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

The async sibling of [`Collection`](#vd.Collection).

Same conceptual surface — storage + `search` — but every method is
awaitable and iterators are [`AsyncIterator`](https://docs.python.org/3/library/typing.html#typing.AsyncIterator). The mapping
interface is exposed as explicit `get` / `set` / `delete` / `keys`
/ `count` methods (the stdlib’s `MutableMapping` ABC has no async
counterpart; explicit methods are the Motor / aiopg convention).

Construct via [`vd.connect_async()`](#vd.connect_async); the universal
[`AsyncCollectionWrapper`](#vd.AsyncCollectionWrapper) in [`vd.asynchronous`](vd.asynchronous.html.md#module-vd.asynchronous) adapts every
backend to this protocol by dispatching to the sync API through
[`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). Backends with native async SDKs override the
wrapper and additionally satisfy [`SupportsNativeAsync`](#vd.SupportsNativeAsync).

### *class* vd.AsyncCollectionWrapper(sync_collection)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Adapt a sync [`Collection`](#vd.Collection) to the [`AsyncCollection`](#vd.AsyncCollection)
contract by dispatching every method to [`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread).

Use [`connect_async()`](#vd.connect_async) rather than instantiating this directly — it
will pick this wrapper or a native async adapter as appropriate.

* **Parameters:**
  **::** (*sync_collection*) – A live [`Collection`](#vd.Collection) (typically obtained from a
  [`Client`](#vd.Client)).

#### native_async

Always `False` for this wrapper. The wrapper still satisfies
[`SupportsNativeAsync`](#vd.SupportsNativeAsync) structurally (the attribute is
present), but the boolean tells callers that I/O is happening in a
thread pool rather than on the event loop. Prefer a native
implementation for high-concurrency workloads.

* **Type:**
  [*bool*](https://docs.python.org/3/builtins/functions.html#bool)

#### *async* add_documents(documents, , batch_size=100)

Batch upsert — mirrors [`add_documents()`](#vd.AbstractCollection.add_documents).

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

The underlying sync [`Collection`](#vd.Collection) — a documented escape hatch.

#### *async* upsert(document)

Insert or replace `document`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### *class* vd.BM25Index(collection, \*, filter=None, tokenize=<function \_tokenize>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A reusable Okapi BM25 index over a vd collection’s stored `text`.

Builds the **query-independent** term statistics — per-document token
lists, document frequencies, document lengths, and the mean length — once
in `__init__()`, then answers many queries against them via
[`search()`](vd.search.html.md#module-vd.search). This is the build-once / query-many companion to
[`bm25_lexical_search()`](#vd.bm25_lexical_search) (which builds a throwaway index for a single
query): for batch evaluation or any repeated querying of the same
collection it turns an O(N · Q) workload (re-tokenizing every document on
every query) into O(N + scoring · Q).

Construction is **O(N)** in the collection size; each [`search()`](vd.search.html.md#module-vd.search) is
O(matching documents). Fine for prototypes and collections up to ~100k
documents; for larger workloads switch to a backend with a native text
index (weaviate, elasticsearch, redis, lancedb, …).

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection (or mapping-like `id -> obj` exposing `.text` and
    `.metadata`). Documents whose `text` is empty contribute zero score
    and are dropped at build time.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter, applied **once** at build time (via
    [`vd.filters.matches_filter()`](vd.filters.html.md#vd.filters.matches_filter)) so the index covers only the
    matching documents and its statistics reflect that subset.
  * **tokenize** ([`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Tokenizer (default: lowercased `\w+` tokens). Pass a custom one for
    stemming, CJK, etc.

### Examples

```pycon
>>> import vd
>>> c = vd.connect('memory').create_collection('t', dimension=2)
>>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
>>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
>>> index = vd.BM25Index(c)
>>> index.search('quick fox', limit=1)[0]['id']
'a'
```

#### search(query_text, , limit=10, k1=1.5, b=0.75)

Okapi BM25 scores for `query_text` over the indexed documents.

Returns result dicts in the same shape as [`Collection.search()`](#vd.Collection.search) —
`{"id", "text", "score", "metadata"}` — sorted by descending score.
`k1` / `b` are the standard Okapi hyperparameters (scoring-time, so
one index can be queried with different settings).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *exception* vd.BackendNotInstalledError

Bases: [`VdError`](vd.base.html.md#vd.base.VdError), [`ImportError`](https://docs.python.org/3/builtins/exceptions.html#ImportError)

Raised when a known backend’s Python package is not installed.

Distinct from an *unknown* backend name (a plain `ValueError`): the
backend exists in `vd`’s provider registry, but its client library is
missing. The message carries the `pip install` command to fix it.

### *class* vd.Client(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A live connection to one backend: `Mapping[str, Collection]`.

Collections are created explicitly (so create-time parameters such as
`dimension` and `metric` can be supplied) and fetched either by
[`get_collection()`](#vd.Client.get_collection) or by mapping access `client[name]`.

#### create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

Create a new collection; raise `ValueError` if it exists.

* **Return type:**
  [`Collection`](vd.base.html.md#vd.base.Collection)

#### delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`Collection`](vd.base.html.md#vd.base.Collection)

#### list_collections()

Iterate collection names.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* vd.Collection(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection of documents: `MutableMapping[str, Document]` + `search`.

The mapping half is storage; [`search()`](vd.search.html.md#module-vd.search) is the single retrieval
extension. This minimal surface is everything `vd`’s tooling depends on.
Batch insertion is an *optional* capability — see [`SupportsBatch`](#vd.SupportsBatch).

#### search(query, , limit=10, filter=None, egress=None, \*\*kwargs)

Return the `limit` documents most similar to `query`.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *class* vd.Document(id, text='', vector=None, metadata=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The unit stored in a [`Collection`](#vd.Collection).

* **Parameters:**
  * **id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Unique identifier; the key under which the document lives in a
    collection.
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The text content. May be empty for vector-first use cases where no
    text is associated with a vector.
  * **vector** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – The embedding. If `None` when written, the collection embeds
    `text` with its client’s `embedder` — or raises
    [`EmbeddingRequiredError`](#vd.EmbeddingRequiredError) if none is configured.
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

### *exception* vd.EmbeddingRequiredError

Bases: [`VdError`](vd.base.html.md#vd.base.VdError), [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Raised when text is given but no embedder is configured.

`vd` operates on vectors. Passing raw text to `collection[key] = text`
or `collection.search(text)` only works when the [`Client`](#vd.Client) was
created with an `embedder`. Otherwise, pass a [`Document`](#vd.Document) with a
`vector` (or a pre-computed query vector) directly.

### *exception* vd.StaticIndexError

Bases: [`VdError`](vd.base.html.md#vd.base.VdError)

Raised on a write to a static (immutable) index.

Some backends — notably a plain FAISS flat index — build an index that
cannot accept incremental `__setitem__` / `__delitem__` after creation.
Such collections set `AbstractCollection.supports_incremental_writes`
to `False` and raise this on write. Callers branch on that flag *before*
triggering the error, and use the adapter’s documented `rebuild()` path.

### *class* vd.SupportsBatch(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection that supports efficient batch insertion.

`add_documents` and `upsert` are *not* part of the minimal
[`Collection`](#vd.Collection) contract. Every adapter built on
[`AbstractCollection`](#vd.AbstractCollection) happens to provide them, but generic code
should still feature-discover:

```default
if isinstance(collection, SupportsBatch):
    collection.add_documents(many_docs, batch_size=256)
```

### *class* vd.SupportsHybrid(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A collection that supports native hybrid (dense + lexical) search.

Hybrid search has no syntactic convergence across vector databases, so it
is an opt-in capability, never baseline. Prefer the top-level
[`vd.hybrid_search()`](#vd.hybrid_search) — it dispatches to this protocol when the
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

### *class* vd.SupportsNativeAsync(\*args, \*\*kwargs)

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

### *class* vd.TimeIndexedCollection(collection, \*, ts_field='ts', ts_parser=<function to_datetime>)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

Time-indexed wrapper over any vd `Collection`.

Maintains a sorted `(ts_epoch, id)` index alongside the underlying
collection. Each stored document MUST carry a timestamp in its metadata
under `ts_field` (default `"ts"`). The stored value is normalized to
an ISO-8601 string so backend-side filtering remains usable.

* **Parameters:**
  * **collection** ([`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)) – Any vd Collection (MutableMapping + `search`).
  * **ts_field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Metadata key holding the timestamp.
  * **ts_parser** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime)]) – Optional custom parser `Any -> datetime`. Defaults to
    [`to_datetime()`](#vd.to_datetime).

### Notes

The index is rebuilt on construction from whatever the underlying
collection already contains (so the wrapper is safe to re-wrap a persisted
collection across process restarts).

#### *property* base *: [MutableMapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)*

The wrapped underlying collection.

#### query_window(start=None, end=None, , filt=None)

Yield documents with `start <= ts < end`, in chronological order.

`start` / `end` may be `None` for half-open infinity. `filt` is
an optional MongoDB-style predicate applied to document metadata,
evaluated client-side (so it works on any backend).

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`Document`](vd.base.html.md#vd.base.Document)]

#### reindex()

Force-rebuild the in-memory time index from the underlying collection.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### search_window(query, , start=None, end=None, limit=10, filt=None, \*\*kwargs)

Semantic search restricted to a time window.

Builds a metadata filter on `ts_field` and delegates to the
underlying collection’s `search`. Falls back to a client-side
post-filter for backends that don’t honor the filter.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]

#### time_range()

Return `(min_ts, max_ts)` as aware datetimes, or None if empty.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime)]]

```pycon
>>> from vd import connect, Document
>>> import hashlib
>>> emb = lambda t: [b/128.0-1.0 for b in hashlib.md5(t.encode()).digest()[:4]]
>>> col = connect('memory', embedder=emb).create_collection('t')
>>> t = TimeIndexedCollection(col)
>>> t['a'] = Document(id='a', text='x', metadata={'ts': '2025-01-01'})
>>> t['b'] = Document(id='b', text='y', metadata={'ts': '2025-03-01'})
>>> [d.isoformat() for d in t.time_range()]
['2025-01-01T00:00:00+00:00', '2025-03-01T00:00:00+00:00']
```

#### window_iter(window='1d', \*, start=None, end=None, reducer=<function count_docs>, skip_empty=False, align=True)

Yield `(window_start, window_end, reducer_value)` over fixed windows.

* **Parameters:**
  * **window** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`timedelta`](https://docs.python.org/3/library/datetime.html#datetime.timedelta), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Window size. See [`parse_window()`](#vd.parse_window) for accepted forms.
  * **start** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Override the data range. Default: actual min/max ts in the index.
  * **end** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Override the data range. Default: actual min/max ts in the index.
  * **reducer** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Document`](vd.base.html.md#vd.base.Document)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Callable taking the iterable of in-window 

    ```
    ``
    ```

    Document\`\`s. Default
    is [`count_docs()`](#vd.count_docs). See also [`mean_vector()`](#vd.mean_vector).
  * **skip_empty** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, omit windows that contained zero documents.
  * **align** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), align `start` to the previous midnight (for
    daily windows) or to `window`-rounded boundary so downstream
    joins are clean. If False, use the literal `start`.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *exception* vd.UnsupportedCapabilityError

Bases: [`VdError`](vd.base.html.md#vd.base.VdError), [`NotImplementedError`](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError)

Raised when an operation needs a capability the backend lacks.

Prefer feature-discovery — `isinstance(collection, SupportsHybrid)` — over
catching this, but it is the clear, typed fallback when an optional
operation is called on a backend that does not implement it.

### *exception* vd.UnsupportedFilterError

Bases: [`VdError`](vd.base.html.md#vd.base.VdError), [`ValueError`](https://docs.python.org/3/builtins/exceptions.html#ValueError)

Raised when a metadata filter uses an operator a backend cannot honor.

The canonical, backend-agnostic filter language lives in [`vd.filters`](vd.filters.html.md#module-vd.filters)
(a MongoDB-style JSON dialect). When a filter uses an operator outside a
backend’s documented subset — or one that does not exist at all — this is
raised, so the caller can simplify the filter or drop to the backend’s
native filter via the escape hatch (`collection.native`).

### *exception* vd.VdError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Base class for every error `vd` raises on its own behalf.

### *class* vd.WindowSlice(start, end)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A time window: `[start, end)`.

### vd.benchmark_insert(collection, n_documents=100, , text_length=100, batch_size=10)

Benchmark document insertion performance.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to benchmark
  * **n_documents** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of documents to insert
  * **text_length** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Length of test documents
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for insertion
* **Returns:**
  Benchmark results
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> results = vd.benchmark_insert(docs, n_documents=50)
```

### vd.benchmark_search(collection, query, , n_queries=100, limit=10)

Benchmark search performance on a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to benchmark
  * **query** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Query text to use
  * **n_queries** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of queries to run
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of results per query
* **Returns:**
  Benchmark results with:
  - total_time: Total time for all queries
  - avg_latency: Average query latency
  - min_latency: Minimum latency
  - max_latency: Maximum latency
  - p50, p95, p99: Latency percentiles
  - queries_per_second: Throughput
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> # Add some documents...
>>> results = vd.benchmark_search(docs, "test query", n_queries=50)
```

### vd.bm25_lexical_search(collection, query_text, , limit=10, filter=None, k1=1.5, b=0.75)

Brute-force BM25 lexical search over a vd collection’s stored `text`.

Builds a throwaway [`BM25Index`](#vd.BM25Index) over `collection` and runs a single
query against it. Used as the default lexical side of [`hybrid_search()`](#vd.hybrid_search)
when a collection does not implement [`SupportsHybrid`](#vd.SupportsHybrid).

Cost is **O(N)** in the collection size — fine for prototypes and
collections up to ~100k documents. \*\*For repeated queries over the same
collection, build a\*\* [`BM25Index`](#vd.BM25Index) **once and call**
[`BM25Index.search()`](#vd.BM25Index.search) **per query** instead of calling this function in a
loop — the term statistics are then computed once rather than on every call.
For larger workloads, switch to a backend with native hybrid search
(weaviate, elasticsearch, redis, lancedb, …) or pass a custom `lexical_search`
callable to [`hybrid_search()`](#vd.hybrid_search) that consults a real text index.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection. Documents whose `text` is empty contribute zero
    score and are filtered out of the result.
  * **query_text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The lexical query.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum number of results.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter. Applied client-side via
    [`vd.filters.matches_filter()`](vd.filters.html.md#vd.filters.matches_filter).
  * **k1** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – BM25 hyperparameters. Defaults match the standard Okapi BM25.
  * **b** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – BM25 hyperparameters. Defaults match the standard Okapi BM25.
* **Returns:**
  Result dicts in the same shape as [`Collection.search()`](#vd.Collection.search) —
  `{"id", "text", "score", "metadata"}` — sorted by descending score.
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> c = vd.connect('memory').create_collection('t', dimension=2)
>>> c['a'] = vd.Document(id='a', text='the quick brown fox', vector=[1.0, 0.0])
>>> c['b'] = vd.Document(id='b', text='lazy dog sleeps', vector=[0.0, 1.0])
>>> hits = bm25_lexical_search(c, 'quick fox', limit=1)
>>> hits[0]['id']
'a'
```

### vd.check_requirements(backend, , verbose=True)

Diagnose whether `backend` is ready to use, and say what to do if not.

Runs an installed-check plus archetype-specific checks (embedded / server /
managed), then computes the single most useful *next step*.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – A provider name (see [`vd.list_all_backends()`](#vd.list_all_backends)).
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Print a human-readable report (in addition to returning the dict).
* **Returns:**
  `{"backend", "archetype", "ok", "checks", "next_step"}` where
  `checks` is a list of `{"name", "ok", "detail"}` records.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> report = check_requirements('memory', verbose=False)
>>> report['ok']
True
```

### vd.chunk_documents(documents, chunk_size=500, , overlap=50, strategy='chars', id_template='{doc_id}_chunk_{chunk_num}', preserve_metadata=True)

Chunk multiple documents while preserving metadata.

* **Parameters:**
  * **documents** ([`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – Iterator of (doc_id, text) or (doc_id, text, metadata) tuples
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Size of each chunk
  * **overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Overlap between chunks
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Chunking strategy (see chunk_text)
  * **id_template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Template for chunk IDs. Can use {doc_id} and {chunk_num}
  * **preserve_metadata** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to copy metadata to all chunks
* **Yields:**
  *tuple* – (chunk_id, chunk_text, metadata) tuples
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### Examples

```pycon
>>> docs = [('doc1', 'Long text...', {'author': 'Alice'})]
>>> chunks = list(chunk_documents(docs, chunk_size=20))
>>> len(chunks) >= 1
True
```

### vd.chunk_text(text, chunk_size=500, , overlap=50, strategy='chars', preserve_sentences=True)

Chunk text into smaller pieces.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to chunk
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target size of each chunk (in characters or tokens depending on strategy)
  * **overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of characters/tokens to overlap between chunks
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – 

    Chunking strategy:
    - ’chars’: Character-based chunking
    - ’words’: Word-based chunking
    - ’sentences’: Sentence-based chunking
    - ’paragraphs’: Paragraph-based chunking
  * **preserve_sentences** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Try to avoid breaking sentences when using chars/words strategy
* **Returns:**
  List of text chunks
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> text = "This is sentence one. This is sentence two. This is sentence three."
>>> chunks = chunk_text(text, chunk_size=30, strategy='chars')
>>> len(chunks) >= 2
True
```

```pycon
>>> chunks = chunk_text(text, strategy='sentences')
>>> len(chunks)
3
```

### vd.clean_text(text, , lowercase=False, remove_extra_whitespace=True, remove_urls=False, remove_emails=False, remove_numbers=False, remove_punctuation=False)

Clean and normalize text.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to clean
  * **lowercase** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Convert to lowercase
  * **remove_extra_whitespace** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Collapse multiple spaces/newlines
  * **remove_urls** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove URLs
  * **remove_emails** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove email addresses
  * **remove_numbers** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove numbers
  * **remove_punctuation** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove punctuation
* **Returns:**
  Cleaned text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> text = "Hello   World!  Visit https://example.com"
>>> clean_text(text, remove_urls=True)
'Hello World! Visit'
>>> clean_text(text, lowercase=True, remove_punctuation=True)
'hello world visit httpsexamplecom'
```

### vd.collection_stats(collection)

Compute comprehensive statistics for a collection.

* **Parameters:**
  **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to analyze
* **Returns:**
  Statistics including:
  - total_documents: Number of documents
  - avg_text_length: Average text length in characters
  - min_text_length: Minimum text length
  - max_text_length: Maximum text length
  - total_chars: Total characters across all documents
  - metadata_fields: Set of all metadata fields used
  - metadata_field_counts: Count of documents with each metadata field
  - embedding_dimension: Dimension of embeddings (if available)
  - has_vectors: Number of documents with vectors
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> docs['doc1'] = ("Hello", {'category': 'greeting'})
>>> stats = vd.collection_stats(docs)
>>> print(stats['total_documents'])
1
```

### vd.compare_backends(names, , characteristics=None)

Return a `{name: {characteristic: value}}` table for the given providers.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.connect(backend, , embedder=None, \*\*backend_kwargs)

Connect to a vector database backend and return its [`Client`](vd.base.html.md#vd.base.Client).

This is the single entry point of `vd`. Switching vector databases is a
one-argument change here.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend name: `"memory"`, `"chroma"`, `"qdrant"`, `"faiss"`,
    `"lancedb"`, `"sqlite_vec"`, `"duckdb"`, `"pgvector"`,
    `"pinecone"`, … Run [`vd.list_backends()`](#vd.list_backends) for what is installed.
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – A `text -> vector` function. Supply it only if you want the
    *convenience* of passing raw text to `collection[key] = "text"` and
    `collection.search("query text")`. `vd` never embeds on its own —
    with no embedder, pass [`Document`](vd.base.html.md#vd.base.Document) objects with vectors
    and pre-computed query vectors.
  * **\*\*backend_kwargs** – Backend-specific connection options (`persist_directory`, `url`,
    `api_key`, `path`, …). See each adapter’s docstring.
* **Returns:**
  A connected client — a `Mapping` of collection name to collection.
* **Return type:**
  [`Client`](vd.base.html.md#vd.base.Client)

### Examples

```pycon
>>> client = connect('memory')
>>> client = connect('chroma', persist_directory='./db')
>>> client = connect('qdrant', url='http://localhost:6333')
```

### *async* vd.connect_async(backend, , native=True, \*\*kwargs)

Async sibling of [`vd.connect()`](#vd.connect).

Returns an [`AsyncClient`](#vd.AsyncClient). When the backend has a native async
client (see [`list_async_backends()`](#vd.list_async_backends); today `qdrant`), its registered
factory decides: it may return a native client doing real non-blocking
I/O (qdrant with `url=`) or the wrapper (embedded qdrant, whose async
client would block the loop). Every other backend goes
through the universal [`AsyncClientWrapper`](#vd.AsyncClientWrapper), built on
[`asyncio.to_thread()`](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread). Check `client.native_async` to tell them
apart.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend name — same vocabulary as [`vd.connect()`](#vd.connect).
  * **native** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Use the backend’s native async client when one is registered
    (default). `False` forces the `to_thread` wrapper around the
    sync adapter.
  * **\*\*kwargs** – Forwarded to the native client’s constructor, or to
    [`vd.connect()`](#vd.connect) for the wrapper. Both take the same arguments
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

### vd.connect_from_config(path=None, , profile=None, apply_env=True, embedder=None, \*\*overrides)

Connect to a backend using configuration from a file.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Path to configuration file. If not provided, searches for default
    config files.
  * **profile** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Profile name to use from configuration. Defaults to ‘default’ or
    the VD_PROFILE environment variable.
  * **apply_env** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to apply environment variable overrides
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Optional `text -> vector` convenience embedder, passed to
    [`vd.connect()`](#vd.connect). A vd config file describes the \*backend
    connection\*, not embedding — embedding stays the caller’s concern.
  * **\*\*overrides** – Additional keyword arguments to override configuration values
* **Returns:**
  Connected client instance
* **Return type:**
  [`Client`](vd.base.html.md#vd.base.Client)

### Examples

```pycon
>>> # With a config file
>>> client = connect_from_config('vd.yaml')
```

```pycon
>>> # With a specific profile
>>> client = connect_from_config('vd.yaml', profile='production')
```

```pycon
>>> # With environment variable VD_PROFILE=dev
>>> client = connect_from_config()
```

```pycon
>>> # With overrides
>>> client = connect_from_config('vd.yaml', persist_directory='./data')
```

### vd.copy_collection(source, target, , batch_size=100, preserve_vectors=True)

Copy a collection with flexible source/target specification.

* **Parameters:**
  * **source** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`Collection`](vd.base.html.md#vd.base.Collection)) – Either a Collection object or (backend_name, collection_name) tuple
  * **target** ([`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)] | [`Collection`](vd.base.html.md#vd.base.Collection)) – Either a Collection object or (backend_name, collection_name, config) tuple
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for copying
  * **preserve_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to preserve vectors
* **Returns:**
  Migration statistics
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> # Copy between backends
>>> stats = vd.copy_collection(
...     source=('memory', 'docs'),
...     target=('chroma', 'docs', {'persist_directory': './data'}),
... )
```

### vd.cosine_similarity(vec1, vec2)

Cosine similarity of two vectors (1.0 identical, 0.0 orthogonal).

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Examples

```pycon
>>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
1.0
>>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
0.0
```

### vd.count_docs(docs)

`len` reducer that also handles generator inputs.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> count_docs(iter([1, 2, 3]))
3
```

### vd.create_example_config(format='yaml')

Generate an example configuration file content.

* **Parameters:**
  **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of configuration: ‘yaml’ or ‘toml’
* **Returns:**
  Example configuration as a string
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> yaml_config = create_example_config('yaml')
>>> print(yaml_config)
>>> toml_config = create_example_config('toml')
```

### vd.deduplicate_results(results, , key='id', keep='first')

Remove duplicate results.

* **Parameters:**
  * **results** ([`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Search results
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Field to check for duplicates
  * **keep** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Which duplicate to keep: ‘first’ or ‘highest_score’
* **Yields:**
  *dict* – Deduplicated results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> results = [
...     {'id': 'doc1', 'score': 0.9},
...     {'id': 'doc1', 'score': 0.8},
...     {'id': 'doc2', 'score': 0.7}
... ]
>>> unique = list(deduplicate_results(iter(results)))
>>> len(unique)
2
```

### vd.euclidean_distance(vec1, vec2)

Euclidean (L2) distance between two vectors.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Examples

```pycon
>>> euclidean_distance([1.0, 0.0], [1.0, 0.0])
0.0
```

### vd.export_collection(collection, output_path, , format='jsonl', \*\*kwargs)

Export a collection to a file in the specified format.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to export
  * **output_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file/directory path
  * **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Export format: ‘jsonl’, ‘json’, ‘directory’
  * **\*\*kwargs** – Additional format-specific options
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> vd.export_collection(docs, 'backup.jsonl')
```

### vd.export_to_directory(collection, output_dir, , include_vectors=True)

Export collection as a directory with one JSON file per document.

Useful for version control and easy browsing.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to export
  * **output_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output directory path
  * **include_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include vectors
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.export_to_json(collection, output_path, , include_vectors=True, indent=2)

Export a collection to JSON format.

Creates a JSON array of all documents.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to export
  * **output_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path
  * **include_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include embedding vectors
  * **indent** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – JSON indentation (None for compact)
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.export_to_jsonl(collection, output_path, , include_vectors=True)

Export a collection to JSONL (JSON Lines) format.

Each line is a JSON object representing a document.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to export
  * **output_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path
  * **include_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include embedding vectors in the export
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> docs['doc1'] = "Hello"
>>> vd.export_to_jsonl(docs, 'backup.jsonl')
1
```

### vd.extract_metadata(text, , extract_title=True, extract_length=True, extract_word_count=True, extract_language=False)

Extract metadata from text.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to analyze
  * **extract_title** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Extract first line as title
  * **extract_length** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add text length
  * **extract_word_count** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add word count
  * **extract_language** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Detect language (requires langdetect)
* **Returns:**
  Extracted metadata
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> text = "My Title\n\nThis is the content."
>>> meta = extract_metadata(text)
>>> meta['title']
'My Title'
>>> meta['char_count']
30
```

### vd.find_duplicates(collection, , threshold=0.95, method='cosine')

Find near-duplicate documents in a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to analyze
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Similarity threshold above which documents are considered duplicates
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Similarity method: ‘cosine’ or ‘exact’
* **Returns:**
  List of (doc_id1, doc_id2, similarity) tuples for duplicates
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> docs['doc1'] = "Hello world"
>>> docs['doc2'] = "Hello world"
>>> duplicates = vd.find_duplicates(docs)
>>> len(duplicates) > 0
True
```

### vd.find_outliers(collection, , n_neighbors=5, threshold=0.3)

Find outlier documents (those dissimilar to their neighbors).

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to analyze
  * **n_neighbors** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of neighbors to consider
  * **threshold** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Average similarity threshold below which a document is an outlier
* **Returns:**
  List of (doc_id, avg_similarity) for outliers
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> # Add some documents...
>>> outliers = vd.find_outliers(docs)
```

### vd.get_backend_characteristics()

Return a compact `{name: characteristics}` map for comparison tooling.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.get_backend_info(name)

Return one provider’s metadata with `installed`/`has_adapter` flags.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### vd.get_install_instructions(name)

Return a human-readable setup blurb for one provider.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.health_check_backend(backend_name, \*\*config)

Check if a backend is healthy and accessible.

* **Parameters:**
  * **backend_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend name to check
  * **\*\*config** – Backend-specific configuration
* **Returns:**
  Health report with keys:
  - status: ‘healthy’, ‘unhealthy’, or ‘unavailable’
  - available: Whether backend is installed
  - registered: Whether backend is registered
  - message: Status message
  - details: Additional details (if connected successfully)
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> status = vd.health_check_backend('memory')
>>> print(status['status'])
'healthy'
```

### vd.health_check_collection(collection)

Check collection health and compute basic stats.

* **Parameters:**
  **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to check
* **Returns:**
  Health report
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> status = vd.health_check_collection(docs)
```

### vd.hybrid_search(collection, query, , query_text=None, limit=10, filter=None, k_dense=None, k_lexical=None, rrf_k=60, lexical_search=None, egress=None, \*\*kwargs)

Hybrid (dense + lexical) search that works on any vd Collection.

Dispatches to the collection’s native `hybrid_search` when it implements
[`SupportsHybrid`](#vd.SupportsHybrid) (efficient, server-side). Otherwise fuses the
collection’s own dense [`search()`](#vd.Collection.search) with a client-side
lexical scan (default: [`bm25_lexical_search()`](#vd.bm25_lexical_search)) via \*\*Reciprocal Rank
Fusion\*\*.

The portable contract is RRF. Backend-specific knobs (weighted blend
`alpha`, fusion-type variants, native ranker choices) are accepted via
`**kwargs` and forwarded to the adapter when it has a native
implementation; they are ignored by the client-side fallback.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Any vd Collection — native-hybrid or not.
  * **query** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]) – Query text (embedded by the collection if it has an embedder) or a
    pre-computed query vector. When `query` is a vector, `query_text`
    is **required**.
  * **query_text** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Explicit text for the lexical side. Defaults to `query` when
    `query` is a string.
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of fused results to return.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Canonical `vd` metadata filter, applied to both sub-searches.
  * **k_dense** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – How many results to fetch from each sub-search before fusion. Default
    is `max(4 * limit, 50)` for each side. Widen for higher recall.
  * **k_lexical** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – How many results to fetch from each sub-search before fusion. Default
    is `max(4 * limit, 50)` for each side. Widen for higher recall.
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Reciprocal Rank Fusion constant (typically 60).
  * **lexical_search** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]]) – Custom `lexical_search(collection, query_text, *, limit, filter,
    \*\*kwargs) -> list[SearchResult]`. Defaults to
    [`bm25_lexical_search()`](#vd.bm25_lexical_search). Used only on the fallback path; on the
    native path it is ignored with a `UserWarning`.
  * **egress** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Per-result transform applied before yielding.
  * **\*\*kwargs** – Extra options. On the native path they are forwarded to the adapter
    (e.g. `alpha=0.7` on weaviate). On the fallback path they are
    ignored.
* **Yields:**
  *dict* – Fused result dicts. `score` is the RRF score on the fallback path,
  or the adapter’s fused score on the native path.
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> col = client.create_collection('docs', dimension=2)
>>> col['a'] = vd.Document(id='a', text='cats purr',
...                        vector=[1.0, 0.0])
>>> col['b'] = vd.Document(id='b', text='dogs bark',
...                        vector=[0.0, 1.0])
>>> hits = list(vd.hybrid_search(col, [0.9, 0.1], query_text='cats',
...                              limit=1))
>>> hits[0]['id']
'a'
```

### *async* vd.hybrid_search_async(collection, query, , query_text=None, limit=10, filter=None, k_dense=None, k_lexical=None, rrf_k=60, lexical_search=None, egress=None, \*\*kwargs)

Async sibling of [`vd.hybrid_search()`](#vd.hybrid_search).

For a wrapped sync collection, runs [`vd.hybrid_search()`](#vd.hybrid_search) (native
hybrid if the backend has it, else the client-side BM25 + RRF fallback)
on a worker thread. For a native async collection it awaits the
collection’s own `hybrid_search` if it has one, and otherwise fuses the
collection’s async dense search with a client-side BM25 scan (O(N): it
reads every document) via RRF. On a native collection, an `async def`
`lexical_search` receives the async collection; a sync one receives a
materialized `{id: Document}` dict (every document is read per call)
and runs on a worker thread. Either way the awaitable + async iterator
interface stays uniform.

Parameters mirror [`vd.hybrid_search()`](#vd.hybrid_search) exactly; see that function for
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

### vd.id_and_score(result)

Egress: keep `(id, score)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### vd.id_only(result)

Egress: keep only the document id.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.id_text_score(result)

Egress: keep `(id, text, score)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### vd.import_collection(collection, input_path, , format=None, \*\*kwargs)

Import documents into a collection from a file.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to import into
  * **input_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input file/directory path
  * **format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Import format: ‘jsonl’, ‘json’, ‘directory’
    If None, inferred from file extension
  * **\*\*kwargs** – Additional format-specific options
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> vd.import_collection(docs, 'backup.jsonl')
```

### vd.import_from_directory(collection, input_dir, , batch_size=100, skip_existing=False, pattern='\*.json')

Import documents from a directory of JSON files.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to import into
  * **input_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input directory path
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for adding documents
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents with IDs that already exist
  * **pattern** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – File pattern to match
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.import_from_json(collection, input_path, , batch_size=100, skip_existing=False)

Import documents from JSON format into a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to import into
  * **input_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input file path
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for adding documents
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents with IDs that already exist
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.import_from_jsonl(collection, input_path, , batch_size=100, skip_existing=False)

Import documents from JSONL format into a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to import into
  * **input_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input file path
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for adding documents
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents with IDs that already exist
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> vd.import_from_jsonl(docs, 'backup.jsonl')
1
```

### vd.install_backend(backend, , run=False)

Return (and optionally run) the `pip install` command for `backend`.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Provider name.
  * **run** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If `True`, actually invoke pip in the current interpreter. If
    `False` (the default), only return the command — the caller decides.
* **Returns:**
  The pip command (or a note that nothing is needed).
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.install_command(name)

Return the `pip install` command that makes `name` usable.

A backend `vd` has an adapter for installs through `vd`’s own extra
(`vd[<backend>]`), which pins exactly the client libraries that adapter
imports. The extra is double-quoted so the command is safe to paste into
zsh, bash, PowerShell and cmd. Providers without an adapter get their raw
client package(s).

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> install_command('qdrant')
'pip install "vd[qdrant]"'
>>> install_command('opensearch')
'pip install opensearch-py'
>>> install_command('memory')
'memory needs no installation (built into vd)'
```

### vd.list_all_backends()

Return every provider with live `installed` / `has_adapter` flags added.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.list_async_backends()

Return the names of backends with a registered native async client.

Every other backend still works with [`connect_async()`](#vd.connect_async), through the
universal `to_thread` wrapper.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### vd.list_available_backends()

Return providers `vd` can [`connect()`](#vd.connect) *right now*.

A backend is available iff its adapter module imported successfully — which
happens only when its client library is installed. This is exactly the set
of registered backends.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### vd.list_backends()

Return the names of all backends with a registered (importable) adapter.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### vd.load_config(path=None, , format=None)

Load configuration from a file.

Automatically detects format from file extension if not specified.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Path to configuration file. If not provided, looks for default
    config files in: ./vd.yaml, ./vd.yml, ./vd.toml, ~/.vd/config.yaml, etc.
  * **format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Configuration format: ‘yaml’ or ‘toml’. Auto-detected from extension
    if not provided.
* **Returns:**
  Configuration dictionary
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> config = load_config('vd.yaml')
>>> config = load_config('vd.toml')
>>> config = load_config()  # Looks for default config files
```

### vd.matches_filter(metadata, filter)

Return `True` if `metadata` satisfies the MongoDB-style `filter`.

An empty or `None` filter matches everything. Unknown operators raise
[`UnsupportedFilterError`](vd.base.html.md#vd.base.UnsupportedFilterError) — they never silently match.

* **Parameters:**
  * **metadata** ([`Mapping`](https://docs.python.org/3/library/typing.html#typing.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – A document’s metadata dict.
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – A filter in the canonical `vd` dialect (see the module docstring).
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### Examples

```pycon
>>> matches_filter({'year': 2024}, None)
True
>>> matches_filter({'year': 2024, 'cat': 'tech'},
...                {'year': {'$gte': 2020}, 'cat': 'tech'})
True
>>> matches_filter({'views': 50}, {'views': {'$gte': 10, '$lte': 100}})
True
```

### vd.mean_vector(docs)

Element-wise mean of document embeddings. `None` if empty / no vectors.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

```pycon
>>> from vd.base import Document
>>> mean_vector([
...     Document(id='a', text='', vector=[1.0, 2.0]),
...     Document(id='b', text='', vector=[3.0, 4.0]),
... ])
[2.0, 3.0]
>>> mean_vector([]) is None
True
```

### vd.metadata_distribution(collection, field, , top_n=None)

Get the distribution of values for a metadata field.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to analyze
  * **field** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Metadata field name
  * **top_n** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – If specified, return only the top N most common values
* **Returns:**
  Mapping of field values to their counts
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`int`](https://docs.python.org/3/builtins/functions.html#int)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> docs['doc1'] = ("Hello", {'category': 'A'})
>>> docs['doc2'] = ("World", {'category': 'A'})
>>> docs['doc3'] = ("Test", {'category': 'B'})
>>> dist = vd.metadata_distribution(docs, 'category')
>>> print(dist)
{'A': 2, 'B': 1}
```

### vd.migrate_client(source_client, target_client, , collection_names=None, batch_size=100, preserve_vectors=True, progress_callback=None)

Migrate all (or selected) collections from one client to another.

* **Parameters:**
  * **source_client** ([`Client`](vd.base.html.md#vd.base.Client)) – Source database client
  * **target_client** ([`Client`](vd.base.html.md#vd.base.Client)) – Target database client
  * **collection_names** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Specific collections to migrate. If None, migrates all.
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for migration
  * **preserve_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to preserve vectors
  * **progress_callback** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Function called with (collection_name, current, total)
* **Returns:**
  Overall migration statistics
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> source = vd.connect('memory')
>>> target = vd.connect('chroma', persist_directory='./backup')
>>> stats = vd.migrate_client(source, target)
```

### vd.migrate_collection(source_collection, target_collection, , batch_size=100, preserve_vectors=True, progress_callback=None, skip_existing=False)

Migrate a collection from one backend to another.

* **Parameters:**
  * **source_collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Source collection to migrate from
  * **target_collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Target collection to migrate to
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of documents to migrate per batch
  * **preserve_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to preserve pre-computed vectors
  * **progress_callback** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`int`](https://docs.python.org/3/builtins/functions.html#int), [`int`](https://docs.python.org/3/builtins/functions.html#int)], [`None`](https://docs.python.org/3/builtins/constants.html#None)]]) – Function called with (current, total) to report progress
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents that already exist in target
* **Returns:**
  Migration statistics with keys:
  - total: Total documents in source
  - migrated: Number of documents migrated
  - skipped: Number of documents skipped
  - failed: Number of failures
  - errors: List of error messages
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> # Create source and target
>>> source_client = vd.connect('memory')
>>> target_client = vd.connect('chroma', persist_directory='./data')
>>> source = source_client.get_collection('my_docs')
>>> target = target_client.create_collection('my_docs')
>>>
>>> # Migrate
>>> stats = vd.migrate_collection(source, target)
>>> print(f"Migrated {stats['migrated']} documents")
```

### vd.multi_query_search(collection, queries, , limit=10, combine='interleave', filter=None, \*\*kwargs)

Search with multiple queries and combine results.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to search
  * **queries** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Multiple query strings
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Total number of results to return
  * **combine** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – 

    How to combine results:
    - ’interleave’: Interleave results from each query
    - ’concatenate’: Concatenate all results
    - ’union’: Remove duplicates across queries
    - ’best’: Take best results across all queries
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Metadata filter
  * **\*\*kwargs** – Additional search options
* **Yields:**
  *dict* – Search results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> results = vd.multi_query_search(
...     docs,
...     ["What is AI?", "How does ML work?"],
...     limit=10
... )
```

### vd.normalize_document_input(doc_input, , auto_id=True)

Normalize a flexible document input to a [`Document`](vd.base.html.md#vd.base.Document).

Accepted shapes: a [`Document`](vd.base.html.md#vd.base.Document); a `str` (just text); a
tuple `(text, id)`, `(text, metadata)`, or `(text, id, metadata)`.

* **Parameters:**
  * **doc_input** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple), [`Document`](vd.base.html.md#vd.base.Document)]) – The input to normalize.
  * **auto_id** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – When the input carries no id, generate one (vs. leaving it empty).
* **Return type:**
  [`Document`](vd.base.html.md#vd.base.Document)

### Examples

```pycon
>>> normalize_document_input(("Hello", "doc1")).id
'doc1'
>>> normalize_document_input(("Hello", {"k": "v"})).metadata
{'k': 'v'}
>>> normalize_document_input("Hello world").id.startswith('doc_')
True
```

### vd.normalize_whitespace(text)

Normalize whitespace in text.

Replaces tabs, multiple spaces, and multiple newlines with single versions.

* **Parameters:**
  **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to normalize
* **Returns:**
  Normalized text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> normalize_whitespace("Hello\t\tWorld  \n\n\nTest")
'Hello World \nTest'
```

### vd.parse_window(window)

Parse a window spec into a `timedelta`.

Strings use a trailing unit char: `"1d"`, `"4h"`, `"30m"`, `"15s"`,
`"1w"`. Numbers are treated as seconds. A `timedelta` is returned as-is.

* **Return type:**
  [`timedelta`](https://docs.python.org/3/library/datetime.html#datetime.timedelta)

```pycon
>>> parse_window('1d') == timedelta(days=1)
True
>>> parse_window('4h') == timedelta(hours=4)
True
>>> parse_window(3600) == timedelta(hours=1)
True
```

### vd.print_backends_table()

Print every known vector database, grouped by deployment archetype.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### vd.print_comparison(names)

Print a side-by-side comparison table of the given providers.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### vd.print_recommendation(\*\*kwargs)

Run [`recommend_backend()`](#vd.recommend_backend) and print the recommendation readably.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### vd.provider(name)

Return one provider’s metadata, or `None` if `name` is unknown.

* **Return type:**
  [`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.providers()

Return the full provider registry as `{name: metadata}`.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### vd.reciprocal_rank_fusion(result_lists, , k=60)

Combine multiple result lists using Reciprocal Rank Fusion.

RRF is a simple yet effective way to combine rankings from multiple sources.

* **Parameters:**
  * **result_lists** ([`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]) – Multiple lists of search results
  * **k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Constant for RRF formula (typically 60)
* **Returns:**
  Combined and re-ranked results
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> results1 = [{'id': 'doc1', 'score': 0.9}, {'id': 'doc2', 'score': 0.8}]
>>> results2 = [{'id': 'doc2', 'score': 0.95}, {'id': 'doc3', 'score': 0.7}]
>>> combined = reciprocal_rank_fusion([results1, results2])
```

### vd.recommend_backend(, corpus_size='medium', persistence=True, can_run_docker=True, cloud_ok=True, budget='free', existing_db=None, needs_hybrid=False, air_gapped=False)

Recommend a vector database from a few yes/no facts about the situation.

A direct encoding of the decision framework in the report’s §4. Returns a
primary pick, a runner-up, and the reasoning trail.

* **Parameters:**
  * **corpus_size** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Rough vector count: tiny <100k, small <10M, medium ~10M, large <100M,
    huge >100M.
  * **persistence** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Must data survive a process restart?
  * **can_run_docker** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Can the user run Docker / operate a server process?
  * **cloud_ok** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Is a managed cloud service acceptable (vs. on-prem only)?
  * **budget** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Free-tier-only, or is paid acceptable?
  * **existing_db** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A database the user already operates — strongly biases the pick.
  * **needs_hybrid** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Need keyword + vector ranking fused in one query?
  * **air_gapped** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Must run with zero network / zero telemetry?
* **Returns:**
  `{"primary", "runner_up", "reasoning", "alternatives"}`.
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> rec = recommend_backend(corpus_size='tiny', persistence=False)
>>> rec['primary']
'memory'
>>> rec = recommend_backend(existing_db='postgres')
>>> rec['primary']
'pgvector'
```

### vd.register_async_backend(name, factory=None)

Register a native async client factory for backend `name`.

Use as a class decorator on an [`AsyncAbstractClient`](#vd.AsyncAbstractClient) subclass, or
call it with a `factory` (a class or a function, sync or `async`,
taking the [`connect_async()`](#vd.connect_async) keyword arguments). Once registered,
[`connect_async()`](#vd.connect_async) returns the factory’s client instead of the
`to_thread` wrapper.

### Examples

```pycon
>>> @register_async_backend('example')
... class ExampleAsyncClient(AsyncAbstractClient):
...     ...
```

### vd.register_backend(name)

Class decorator: register an adapter [`Client`](vd.base.html.md#vd.base.Client) under `name`.

* **Return type:**
  [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`type`](https://docs.python.org/3/builtins/functions.html#type)], [`type`](https://docs.python.org/3/builtins/functions.html#type)]

### Examples

```pycon
>>> from vd.base import AbstractClient
>>> @register_backend('example')
... class ExampleClient(AbstractClient):
...     ...
```

### vd.sample_collection(collection, n, , method='random', seed=None)

Sample document IDs from a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to sample from
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of documents to sample
  * **method** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Sampling method: ‘random’, ‘first’, ‘diverse’
  * **seed** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Random seed for reproducibility
* **Returns:**
  Sampled document IDs
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> # Add 100 documents...
>>> sample = vd.sample_collection(docs, 10, method='random')
>>> len(sample)
10
```

### vd.save_config(config, path, , format=None)

Save configuration to a file.

* **Parameters:**
  * **config** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Configuration dictionary to save
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to save configuration file
  * **format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Format to save: ‘yaml’ or ‘toml’. Auto-detected from extension
    if not provided.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Examples

```pycon
>>> config = {
...     'profiles': {
...         'dev': {'backend': 'memory'},
...         'prod': {'backend': 'chroma', 'persist_directory': './data'}
...     }
... }
>>> save_config(config, 'vd.yaml')
```

### vd.search_similar_to_document(collection, doc_id, , limit=10, exclude_self=True, filter=None, \*\*kwargs)

Find documents similar to a specific document.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to search
  * **doc_id** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ID of the reference document
  * **limit** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of similar documents to return
  * **exclude_self** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to exclude the reference document from results
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]) – Metadata filter
  * **\*\*kwargs** – Additional search options
* **Yields:**
  *dict* – Search results
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> similar = vd.search_similar_to_document(docs, 'doc1', limit=5)
```

### vd.setup_guide(backend)

Return a full, copy-pasteable setup playbook for `backend`.

Covers: the pip install, a Docker one-liner for server backends, the
environment variables for managed backends, a verify command, and the
relevant documentation links.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.skills_dir()

Return the path to the bundled AI-agent skills directory.

* **Return type:**
  [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

### vd.text_only(result)

Egress: keep only the text. `>>> text_only({'text': 'hi'})` -> `'hi'`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.to_datetime(ts)

Coerce a timestamp-like value into a tz-aware UTC `datetime`.

Accepts ISO-8601 strings (with or without timezone), date-only strings
(`"2025-03-13"`), epoch seconds (int or float), and `datetime` objects.
Naive datetimes / strings are assumed UTC.

* **Return type:**
  [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime)

```pycon
>>> to_datetime('2025-03-13T09:00:00').isoformat()
'2025-03-13T09:00:00+00:00'
>>> to_datetime('2025-03-13').isoformat()
'2025-03-13T00:00:00+00:00'
>>> to_datetime(1741856400).isoformat()
'2025-03-13T09:00:00+00:00'
```

### vd.to_iso(ts)

ISO-8601 (UTC) string suitable for cross-backend metadata storage.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> to_iso('2025-03-13T09:00:00')
'2025-03-13T09:00:00+00:00'
```

### vd.truncate_text(text, max_length, , suffix='...')

Truncate text to maximum length.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to truncate
  * **max_length** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum length
  * **suffix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Suffix to add to truncated text
* **Returns:**
  Truncated text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> truncate_text("This is a long text", 10)
'This is...'
```

### vd.validate_collection(collection)

Validate collection integrity and identify issues.

* **Parameters:**
  **collection** ([`Collection`](vd.base.html.md#vd.base.Collection)) – Collection to validate
* **Returns:**
  Validation report with:
  - valid: Whether collection is valid
  - issues: List of issue descriptions
  - warnings: List of warning messages
  - stats: Basic stats
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> docs = client.create_collection('test')
>>> report = vd.validate_collection(docs)
>>> print(report['valid'])
True
```

### vd.validate_filter(filter, , supported=frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'}))

Walk `filter` and raise [`UnsupportedFilterError`](vd.base.html.md#vd.base.UnsupportedFilterError) on any
operator that is unknown or not in `supported`.

Backends that translate the canonical filter to a native query call this
with their own (possibly narrower) `supported` subset, so callers get a
clear `vd` error up front instead of an opaque backend error later.

* **Parameters:**
  * **filter** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – A filter in the canonical `vd` dialect. `None` / empty is valid.
  * **supported** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – The operator subset to allow. Defaults to every operator the language
    defines.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Examples

```pycon
>>> validate_filter({'year': {'$gte': 2020}})            # ok, returns None
>>> validate_filter({'a': {'$regex': '.*'}})             # not in the language
Traceback (most recent call last):
    ...
vd.base.UnsupportedFilterError: Unknown filter operator '$regex'. ...
>>> validate_filter({'a': {'$exists': True}}, supported={'$eq'})
Traceback (most recent call last):
    ...
vd.base.UnsupportedFilterError: Filter operator '$exists' is not supported ...
```

### Modules

| [`analytics`](vd.analytics.html.md#module-vd.analytics)       | Analytics and statistics for vd collections.                                                                              |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`asynchronous`](vd.asynchronous.html.md#module-vd.asynchronous) | Async support for `vd`: universal wrapper + opt-in native implementations.                                                |
| [`backends`](vd.backends.html.md#module-vd.backends)         | Backend adapters for the vector databases `vd` supports.                                                                  |
| [`base`](vd.base.html.md#module-vd.base)                 | Core contracts, data model, and abstract bases for the `vd` vectorDB facade.                                              |
| [`cli`](vd.cli.html.md#module-vd.cli)                   | Command-line interface for vd.                                                                                            |
| [`config`](vd.config.html.md#module-vd.config)             | Configuration management for vd.                                                                                          |
| [`filters`](vd.filters.html.md#module-vd.filters)           | The canonical metadata-filter language for `vd`.                                                                          |
| [`health`](vd.health.html.md#module-vd.health)             | Health check and validation utilities for vd.                                                                             |
| [`io`](vd.io.html.md#module-vd.io)                     | Import/export utilities for vd collections.                                                                               |
| [`migration`](vd.migration.html.md#module-vd.migration)       | Migration utilities for moving data between backends.                                                                     |
| [`requirements`](vd.requirements.html.md#module-vd.requirements) | Setup assistance: turning provider metadata into actionable diagnostics.                                                  |
| [`search`](vd.search.html.md#module-vd.search)             | Advanced search utilities for vd.                                                                                         |
| [`text`](vd.text.html.md#module-vd.text)                 | Text preprocessing and chunking utilities for vd.                                                                         |
| [`time_indexed`](vd.time_indexed.html.md#module-vd.time_indexed) | Time-indexed vector collection wrapper.                                                                                   |
| [`util`](vd.util.html.md#module-vd.util)                 | The backend registry, the [`connect()`](#vd.connect) factory, and small shared utilities. |
