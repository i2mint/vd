# vd.backends

Backend adapters for the vector databases `vd` supports.

Adapters are registered **lazily**: `import vd` imports only the always-available
`memory` adapter, because importing every installed client SDK (chromadb, lancedb,
…) costs seconds. An adapter module is imported the first time its backend is
resolved by name ([`vd.connect()`](vd.html.md#vd.connect), [`vd.util.get_backend()`](vd.util.html.md#vd.util.get_backend),
[`vd.connect_async()`](vd.html.md#vd.connect_async)), or all at once by [`load_all()`](#vd.backends.load_all), which
[`vd.list_backends()`](vd.html.md#vd.list_backends) and [`vd.list_async_backends()`](vd.html.md#vd.list_async_backends) call. Each adapter lives
in its own module and registers itself with the [`register_backend()`](vd.util.html.md#vd.util.register_backend)
decorator; a module whose third-party client is not installed fails to import and
is skipped silently, so that backend does not appear in [`vd.list_backends()`](vd.html.md#vd.list_backends).

### Module Attributes

| [`BACKEND_MODULES`](#vd.backends.BACKEND_MODULES)   | Every backend module vd ships; the module's name is the backend's name.   |
|--------------------------------------------------------------------|---------------------------------------------------------------------------|

### Functions

| [`load`](#vd.backends.load)(name)   | Import the adapter module for backend `name`; whether it is now available.                     |
|---------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [`load_all`](#vd.backends.load_all)()   | Import every adapter module whose client library is installed (the old import-time behaviour). |

### vd.backends.BACKEND_MODULES *= ('memory', 'chroma', 'faiss', 'sqlite_vec', 'duckdb', 'lancedb', 'qdrant', 'pgvector', 'pinecone', 'weaviate', 'milvus', 'redis', 'elasticsearch', 'mongodb', 'turbopuffer')*

Every backend module vd ships; the module’s name is the backend’s name.

### vd.backends.load(name)

Import the adapter module for backend `name`; whether it is now available.

`False` for a name that is not a shipped adapter and for one whose third-party
client is not installed (the module then fails to import and is skipped).

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### vd.backends.load_all()

Import every adapter module whose client library is installed (the old import-time behaviour).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Modules

| [`memory`](vd.backends.memory.html.md#module-vd.backends.memory)   | In-memory backend — the reference adapter.   |
|-------------------------------------------------------------------------------------|----------------------------------------------|
