# vd.util

The backend registry, the [`connect()`](#vd.util.connect) factory, and small shared utilities.

This module is intentionally thin. It owns three things:

- the **registry** mapping a backend name (`"chroma"`, `"qdrant"`, …) to
  its adapter [`Client`](vd.base.html.md#vd.base.Client) class, via the [`register_backend()`](#vd.util.register_backend)
  decorator;
- [`connect()`](#vd.util.connect), the one entry point users call to get a client;
- backend-agnostic helpers: document-input normalization, search-result
  `egress` functions, and vector math.

Everything *descriptive* about a backend (pip package, license, docs URLs,
deployment archetype, setup checks) lives in [`vd.providers`](vd.html.md#vd.providers) and its
data file, not here.

### Functions

| [`connect`](#vd.util.connect)(backend, \*[, embedder])                   | Connect to a vector database backend and return its [`Client`](vd.base.html.md#vd.base.Client).   |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [`cosine_similarity`](#vd.util.cosine_similarity)(vec1, vec2)                      | Cosine similarity of two vectors (1.0 identical, 0.0 orthogonal).                                                             |
| [`euclidean_distance`](#vd.util.euclidean_distance)(vec1, vec2)                     | Euclidean (L2) distance between two vectors.                                                                                  |
| [`get_backend`](#vd.util.get_backend)(name)                                  | Return the adapter [`Client`](vd.base.html.md#vd.base.Client) class registered under `name`.      |
| [`id_and_score`](#vd.util.id_and_score)(result)                               | Egress: keep `(id, score)`.                                                                                                   |
| [`id_only`](#vd.util.id_only)(result)                                    | Egress: keep only the document id.                                                                                            |
| [`id_text_score`](#vd.util.id_text_score)(result)                              | Egress: keep `(id, text, score)`.                                                                                             |
| [`list_backends`](#vd.util.list_backends)()                                    | Return the names of all backends with a registered (importable) adapter.                                                      |
| [`mean_vector`](#vd.util.mean_vector)(vectors)                               | Component-wise mean of a non-empty iterable of equal-length vectors.                                                          |
| [`normalize_document_input`](#vd.util.normalize_document_input)(doc_input, \*[, auto_id]) | Normalize a flexible document input to a [`Document`](vd.base.html.md#vd.base.Document).          |
| [`register_backend`](#vd.util.register_backend)(name)                             | Class decorator: register an adapter [`Client`](vd.base.html.md#vd.base.Client) under `name`.     |
| [`text_only`](#vd.util.text_only)(result)                                  | Egress: keep only the text.                                                                                                   |

### vd.util.connect(backend, , embedder=None, \*\*backend_kwargs)

Connect to a vector database backend and return its [`Client`](vd.base.html.md#vd.base.Client).

This is the single entry point of `vd`. Switching vector databases is a
one-argument change here.

* **Parameters:**
  * **backend** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Backend name: `"memory"`, `"chroma"`, `"qdrant"`, `"faiss"`,
    `"lancedb"`, `"sqlite_vec"`, `"duckdb"`, `"pgvector"`,
    `"pinecone"`, … Run [`vd.list_backends()`](vd.html.md#vd.list_backends) for what is installed.
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

### vd.util.cosine_similarity(vec1, vec2)

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

### vd.util.euclidean_distance(vec1, vec2)

Euclidean (L2) distance between two vectors.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### Examples

```pycon
>>> euclidean_distance([1.0, 0.0], [1.0, 0.0])
0.0
```

### vd.util.get_backend(name)

Return the adapter [`Client`](vd.base.html.md#vd.base.Client) class registered under `name`.

* **Raises:**
  * [**BackendNotInstalledError**](vd.html.md#vd.BackendNotInstalledError) – If `name` is a known backend whose client library is not installed.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – If `name` is not a known backend at all.
* **Return type:**
  [`type`](https://docs.python.org/3/builtins/functions.html#type)

### vd.util.id_and_score(result)

Egress: keep `(id, score)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### vd.util.id_only(result)

Egress: keep only the document id.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### vd.util.id_text_score(result)

Egress: keep `(id, text, score)`.

* **Return type:**
  [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)]

### vd.util.list_backends()

Return the names of all backends with a registered (importable) adapter.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### vd.util.mean_vector(vectors)

Component-wise mean of a non-empty iterable of equal-length vectors.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]

### Examples

```pycon
>>> mean_vector([[0.0, 2.0], [2.0, 4.0]])
[1.0, 3.0]
```

### vd.util.normalize_document_input(doc_input, , auto_id=True)

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

### vd.util.register_backend(name)

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

### vd.util.text_only(result)

Egress: keep only the text. `>>> text_only({'text': 'hi'})` -> `'hi'`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
