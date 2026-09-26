# vd.time_indexed

Time-indexed vector collection wrapper.

Wraps any vd `Collection` and adds an in-memory sorted `(timestamp, id)`
index, enabling efficient retrieval by time window, fixed-window aggregation
(daily/hourly buckets), and time-bounded semantic search.

The wrapper is backend-agnostic: it works with any object that satisfies the
vd `Collection` protocol (`MutableMapping` + `search`). Timestamps are
stored as ISO-8601 strings in document metadata, so backend-side metadata
filtering keeps working (e.g. ChromaDB’s `$gte` filter on a string field
sorts correctly because ISO-8601 is lexicographically ordered).

### Examples

```pycon
>>> from vd import connect, Document
>>> import hashlib
>>> def fake_embed(t):  # 8-dim deterministic toy embedding
...     return [b / 128.0 - 1.0 for b in hashlib.md5(t.encode()).digest()[:8]]
>>> client = connect('memory', embedder=fake_embed)
>>> news = TimeIndexedCollection(client.create_collection('news'))
>>> news['a'] = Document(id='a', text='Earnings miss',
...                      metadata={'ts': '2025-03-13T09:00:00'})
>>> news['b'] = Document(id='b', text='Profit warning',
...                      metadata={'ts': '2025-03-13T15:30:00'})
>>> news['c'] = Document(id='c', text='Tariffs announced',
...                      metadata={'ts': '2025-03-14T08:00:00'})
>>> [d.id for d in news.query_window('2025-03-13', '2025-03-14')]
['a', 'b']
>>> # Count documents per day
>>> [(s.date().isoformat(), v) for s, _, v in
...  news.window_iter(window='1d', reducer=len)]
[('2025-03-13', 2), ('2025-03-14', 1)]
```

### Functions

| [`count_docs`](#vd.time_indexed.count_docs)(docs)     | `len` reducer that also handles generator inputs.                  |
|-----------------------------------------------------------------------|--------------------------------------------------------------------|
| [`mean_vector`](#vd.time_indexed.mean_vector)(docs)    | Element-wise mean of document embeddings.                          |
| [`parse_window`](#vd.time_indexed.parse_window)(window) | Parse a window spec into a `timedelta`.                            |
| [`to_datetime`](#vd.time_indexed.to_datetime)(ts)      | Coerce a timestamp-like value into a tz-aware UTC `datetime`.      |
| [`to_iso`](#vd.time_indexed.to_iso)(ts)           | ISO-8601 (UTC) string suitable for cross-backend metadata storage. |

### Classes

| [`TimeIndexedCollection`](#vd.time_indexed.TimeIndexedCollection)(collection, \*[, ...])   | Time-indexed wrapper over any vd `Collection`.   |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [`WindowSlice`](#vd.time_indexed.WindowSlice)(start, end)                        | A time window: `[start, end)`.                   |

### *class* vd.time_indexed.TimeIndexedCollection(collection, \*, ts_field='ts', ts_parser=<function to_datetime>)

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
    [`to_datetime()`](#vd.time_indexed.to_datetime).

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
  * **window** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`timedelta`](https://docs.python.org/3/library/datetime.html#datetime.timedelta), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float)]) – Window size. See [`parse_window()`](#vd.time_indexed.parse_window) for accepted forms.
  * **start** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Override the data range. Default: actual min/max ts in the index.
  * **end** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`int`](https://docs.python.org/3/builtins/functions.html#int), [`float`](https://docs.python.org/3/builtins/functions.html#float), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Override the data range. Default: actual min/max ts in the index.
  * **reducer** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`Document`](vd.base.html.md#vd.base.Document)]], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Callable taking the iterable of in-window 

    ```
    ``
    ```

    Document\`\`s. Default
    is [`count_docs()`](#vd.time_indexed.count_docs). See also [`mean_vector()`](#vd.time_indexed.mean_vector).
  * **skip_empty** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, omit windows that contained zero documents.
  * **align** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True (default), align `start` to the previous midnight (for
    daily windows) or to `window`-rounded boundary so downstream
    joins are clean. If False, use the literal `start`.
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]

### *class* vd.time_indexed.WindowSlice(start, end)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A time window: `[start, end)`.

### vd.time_indexed.count_docs(docs)

`len` reducer that also handles generator inputs.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

```pycon
>>> count_docs(iter([1, 2, 3]))
3
```

### vd.time_indexed.mean_vector(docs)

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

### vd.time_indexed.parse_window(window)

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

### vd.time_indexed.to_datetime(ts)

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

### vd.time_indexed.to_iso(ts)

ISO-8601 (UTC) string suitable for cross-backend metadata storage.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> to_iso('2025-03-13T09:00:00')
'2025-03-13T09:00:00+00:00'
```
