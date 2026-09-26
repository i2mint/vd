# vd.health

Health check and validation utilities for vd.

Provides functions to check backend health, validate configurations,
and benchmark performance.

### Functions

| [`benchmark_insert`](#vd.health.benchmark_insert)(collection[, n_documents, ...])   | Benchmark document insertion performance.        |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`benchmark_search`](#vd.health.benchmark_search)(collection, query, \*[, ...])     | Benchmark search performance on a collection.    |
| [`health_check_backend`](#vd.health.health_check_backend)(backend_name, \*\*config)     | Check if a backend is healthy and accessible.    |
| [`health_check_collection`](#vd.health.health_check_collection)(collection)                | Check collection health and compute basic stats. |

### vd.health.benchmark_insert(collection, n_documents=100, , text_length=100, batch_size=10)

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

### vd.health.benchmark_search(collection, query, , n_queries=100, limit=10)

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

### vd.health.health_check_backend(backend_name, \*\*config)

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

### vd.health.health_check_collection(collection)

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
