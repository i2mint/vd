# vd.analytics

Analytics and statistics for vd collections.

Provides functions to analyze collections, find duplicates, compute statistics,
and gain insights into your vector database.

### Functions

| [`collection_stats`](#vd.analytics.collection_stats)(collection)                      | Compute comprehensive statistics for a collection.            |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [`find_duplicates`](#vd.analytics.find_duplicates)(collection, \*[, threshold, ...]) | Find near-duplicate documents in a collection.                |
| [`find_outliers`](#vd.analytics.find_outliers)(collection, \*[, n_neighbors, ...]) | Find outlier documents (those dissimilar to their neighbors). |
| [`metadata_distribution`](#vd.analytics.metadata_distribution)(collection, field, \*)      | Get the distribution of values for a metadata field.          |
| [`sample_collection`](#vd.analytics.sample_collection)(collection, n, \*[, ...])       | Sample document IDs from a collection.                        |
| [`validate_collection`](#vd.analytics.validate_collection)(collection)                   | Validate collection integrity and identify issues.            |

### vd.analytics.collection_stats(collection)

Compute comprehensive statistics for a collection.

* **Parameters:**
  **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to analyze
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

### vd.analytics.find_duplicates(collection, , threshold=0.95, method='cosine')

Find near-duplicate documents in a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to analyze
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

### vd.analytics.find_outliers(collection, , n_neighbors=5, threshold=0.3)

Find outlier documents (those dissimilar to their neighbors).

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to analyze
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

### vd.analytics.metadata_distribution(collection, field, , top_n=None)

Get the distribution of values for a metadata field.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to analyze
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

### vd.analytics.sample_collection(collection, n, , method='random', seed=None)

Sample document IDs from a collection.

* **Parameters:**
  * **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to sample from
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

### vd.analytics.validate_collection(collection)

Validate collection integrity and identify issues.

* **Parameters:**
  **collection** ([`Collection`](vd.base.md#vd.base.Collection)) – Collection to validate
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
