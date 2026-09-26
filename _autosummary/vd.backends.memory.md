# vd.backends.memory

In-memory backend — the reference adapter.

Stores documents in a plain `dict` and answers queries with brute-force
similarity in pure Python. Always available (no third-party dependency), and
the canonical example of how a backend implements the small set of raw
primitives that [`AbstractCollection`](vd.base.md#vd.base.AbstractCollection) builds on. Use it for
tests, notebooks, and corpora small enough that an ANN index is overkill.

### Classes

| [`MemoryClient`](#vd.backends.memory.MemoryClient)(\*[, embedder])                | In-memory vector database client.                           |
|----------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`MemoryCollection`](#vd.backends.memory.MemoryCollection)(name, \*[, embedder, ...]) | A collection backed by an in-process `dict[str, Document]`. |

### *class* vd.backends.memory.MemoryClient(, embedder=None, \*\*config)

Bases: [`AbstractClient`](vd.base.md#vd.base.AbstractClient)

In-memory vector database client.

### Examples

```pycon
>>> import vd
>>> client = vd.connect('memory')
>>> col = client.create_collection('docs')
>>> col['a'] = vd.Document(id='a', text='cat', vector=[1.0, 0.0])
>>> col['b'] = vd.Document(id='b', text='dog', vector=[0.0, 1.0])
>>> [r['id'] for r in col.search([0.9, 0.1], limit=1)]
['a']
```

#### backend_name *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)* *= 'memory'*

The registry name of this backend (e.g. `"chroma"`). Adapters set it.

#### create_collection(name, , dimension=None, metric='cosine', \*\*index_config)

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
  [`MemoryCollection`](#vd.backends.memory.MemoryCollection)

#### delete_collection(name)

Drop a collection; raise `KeyError` if absent.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### get_collection(name)

Return an existing collection; raise `KeyError` if absent.

* **Return type:**
  [`MemoryCollection`](#vd.backends.memory.MemoryCollection)

#### list_collections()

Iterate collection names.

* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* vd.backends.memory.MemoryCollection(name, , embedder=None, dimension=None, metric='cosine')

Bases: [`AbstractCollection`](vd.base.md#vd.base.AbstractCollection)

A collection backed by an in-process `dict[str, Document]`.

#### *property* native *: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Document](vd.base.md#vd.base.Document)]*

The raw `dict[str, Document]` backing this collection (escape hatch).

#### supported_filter_operators *: [frozenset](https://docs.python.org/3/builtins/stdtypes.html#frozenset)* *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

the full language.
Adapters narrow this; `search()` validates against it.

* **Type:**
  Filter operators this backend can honor. Default
