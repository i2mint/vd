# vd.io

Import/export utilities for vd collections.

This module provides functions to export collections to various formats
and import data from different sources.

### Functions

| [`export_collection`](#vd.io.export_collection)(collection, output_path, \*)     | Export a collection to a file in the specified format.            |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`export_to_directory`](#vd.io.export_to_directory)(collection, output_dir, \*)    | Export collection as a directory with one JSON file per document. |
| [`export_to_json`](#vd.io.export_to_json)(collection, output_path, \*[, ...]) | Export a collection to JSON format.                               |
| [`export_to_jsonl`](#vd.io.export_to_jsonl)(collection, output_path, \*)       | Export a collection to JSONL (JSON Lines) format.                 |
| [`import_collection`](#vd.io.import_collection)(collection, input_path, \*)      | Import documents into a collection from a file.                   |
| [`import_from_directory`](#vd.io.import_from_directory)(collection, input_dir, \*)   | Import documents from a directory of JSON files.                  |
| [`import_from_json`](#vd.io.import_from_json)(collection, input_path, \*)       | Import documents from JSON format into a collection.              |
| [`import_from_jsonl`](#vd.io.import_from_jsonl)(collection, input_path, \*)      | Import documents from JSONL format into a collection.             |

### vd.io.export_collection(collection, output_path, , format='jsonl', \*\*kwargs)

Export a collection to a file in the specified format.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to export
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

### vd.io.export_to_directory(collection, output_dir, , include_vectors=True)

Export collection as a directory with one JSON file per document.

Useful for version control and easy browsing.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to export
  * **output_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output directory path
  * **include_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include vectors
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.io.export_to_json(collection, output_path, , include_vectors=True, indent=2)

Export a collection to JSON format.

Creates a JSON array of all documents.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to export
  * **output_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Output file path
  * **include_vectors** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to include embedding vectors
  * **indent** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – JSON indentation (None for compact)
* **Returns:**
  Number of documents exported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.io.export_to_jsonl(collection, output_path, , include_vectors=True)

Export a collection to JSONL (JSON Lines) format.

Each line is a JSON object representing a document.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to export
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

### vd.io.import_collection(collection, input_path, , format=None, \*\*kwargs)

Import documents into a collection from a file.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to import into
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

### vd.io.import_from_directory(collection, input_dir, , batch_size=100, skip_existing=False, pattern='\*.json')

Import documents from a directory of JSON files.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to import into
  * **input_dir** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input directory path
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for adding documents
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents with IDs that already exist
  * **pattern** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – File pattern to match
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.io.import_from_json(collection, input_path, , batch_size=100, skip_existing=False)

Import documents from JSON format into a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to import into
  * **input_path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Input file path
  * **batch_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Batch size for adding documents
  * **skip_existing** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, skip documents with IDs that already exist
* **Returns:**
  Number of documents imported
* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### vd.io.import_from_jsonl(collection, input_path, , batch_size=100, skip_existing=False)

Import documents from JSONL format into a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to import into
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
