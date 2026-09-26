# vd.filters

The canonical metadata-filter language for `vd`.

`vd` uses a single, backend-agnostic, MongoDB-style JSON dialect to filter
documents by metadata. This module is the **single source of truth** for that
language:

- [`SUPPORTED_FILTER_OPERATORS`](#vd.filters.SUPPORTED_FILTER_OPERATORS) — every operator the language defines.
- [`matches_filter()`](#vd.filters.matches_filter) — evaluate a filter against a metadata dict in Python.
  Used directly by backends that filter client-side (e.g. the `memory`
  backend), and the reference semantics every backend’s native translation
  must agree with.
- [`validate_filter()`](#vd.filters.validate_filter) — walk a filter and fail loud
  ([`UnsupportedFilterError`](vd.base.md#vd.base.UnsupportedFilterError)) on any unknown / unsupported
  operator. Used by backends that translate the filter to a native query, so
  the caller gets a clear `vd` error instead of an opaque backend error.

## Filter syntax

A filter is a `dict`. Each key is either a **metadata field name** or a
**logical operator** (`$and`, `$or`, `$not`). A bare `{'field': value}`
is sugar for `{'field': {'$eq': value}}`. Multiple top-level fields combine
with an implicit `$and`.

Field operators (inside `{'field': {...}}`): `$eq`, `$ne`, `$gt`,
`$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`.

Logical operators (top-level): `$and` / `$or` take a list of subfilters;
`$not` takes a single subfilter.

### Examples

```pycon
>>> matches_filter({'year': 2024, 'tag': 'ai'}, {'year': 2024})
True
>>> matches_filter({'year': 2024}, {'year': {'$gte': 2025}})
False
>>> matches_filter({'year': 2024}, {'$or': [{'year': 2024}, {'year': 2025}]})
True
>>> matches_filter({'tags': ['python', 'ai']}, {'tags': {'$in': ['ai']}})
True
>>> matches_filter({'a': 1}, {'b': {'$exists': False}})
True
>>> matches_filter({'a': 1}, {'$not': {'a': 1}})
False
```

An unknown operator fails loud rather than silently matching everything:

```pycon
>>> matches_filter({'a': 1}, {'a': {'$bogus': 1}})
Traceback (most recent call last):
    ...
vd.base.UnsupportedFilterError: Unknown filter operator '$bogus'. ...
```

### Module Attributes

| [`LOGICAL_OPERATORS`](#vd.filters.LOGICAL_OPERATORS)          | Operators used at the top level of a filter to combine subfilters.   |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`FIELD_OPERATORS`](#vd.filters.FIELD_OPERATORS)            | Operators used inside a `{'field': {...}}` condition.                |
| [`SUPPORTED_FILTER_OPERATORS`](#vd.filters.SUPPORTED_FILTER_OPERATORS) | Every operator the canonical `vd` filter language defines.           |

### Functions

| [`matches_filter`](#vd.filters.matches_filter)(metadata, filter)         | Return `True` if `metadata` satisfies the MongoDB-style `filter`.                                                                                                                     |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`validate_filter`](#vd.filters.validate_filter)(filter, \*[, supported]) | Walk `filter` and raise [`UnsupportedFilterError`](vd.base.md#vd.base.UnsupportedFilterError) on any operator that is unknown or not in `supported`. |

### vd.filters.FIELD_OPERATORS *= frozenset({'$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin'})*

Operators used inside a `{'field': {...}}` condition.

### vd.filters.LOGICAL_OPERATORS *= frozenset({'$and', '$not', '$or'})*

Operators used at the top level of a filter to combine subfilters.

### vd.filters.SUPPORTED_FILTER_OPERATORS *= frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'})*

Every operator the canonical `vd` filter language defines.

### vd.filters.matches_filter(metadata, filter)

Return `True` if `metadata` satisfies the MongoDB-style `filter`.

An empty or `None` filter matches everything. Unknown operators raise
[`UnsupportedFilterError`](vd.base.md#vd.base.UnsupportedFilterError) — they never silently match.

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

### vd.filters.validate_filter(filter, , supported=frozenset({'$and', '$eq', '$exists', '$gt', '$gte', '$in', '$lt', '$lte', '$ne', '$nin', '$not', '$or'}))

Walk `filter` and raise [`UnsupportedFilterError`](vd.base.md#vd.base.UnsupportedFilterError) on any
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
