# vd.migration

Migration utilities for moving data between backends.

Provides functions to migrate collections between different vector database
backends while preserving all data, metadata, and embeddings.

### Functions

| [`copy_collection`](#vd.migration.copy_collection)(source, target, \*[, ...])        | Copy a collection with flexible source/target specification.      |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`migrate_client`](#vd.migration.migrate_client)(source_client, target_client, \*)  | Migrate all (or selected) collections from one client to another. |
| [`migrate_collection`](#vd.migration.migrate_collection)(source_collection, ...[, ...]) | Migrate a collection from one backend to another.                 |

### vd.migration.copy_collection(source, target, , batch_size=100, preserve_vectors=True)

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

### vd.migration.migrate_client(source_client, target_client, , collection_names=None, batch_size=100, preserve_vectors=True, progress_callback=None)

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

### vd.migration.migrate_collection(source_collection, target_collection, , batch_size=100, preserve_vectors=True, progress_callback=None, skip_existing=False)

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
