"""
The canonical metadata-filter language for ``vd``.

``vd`` uses a single, backend-agnostic, MongoDB-style JSON dialect to filter
documents by metadata. This module is the **single source of truth** for that
language:

- :data:`SUPPORTED_FILTER_OPERATORS` — every operator the language defines.
- :func:`matches_filter` — evaluate a filter against a metadata dict in Python.
  Used directly by backends that filter client-side (e.g. the ``memory``
  backend), and the reference semantics every backend's native translation
  must agree with.
- :func:`validate_filter` — walk a filter and fail loud
  (:class:`~vd.base.UnsupportedFilterError`) on any unknown / unsupported
  operator. Used by backends that translate the filter to a native query, so
  the caller gets a clear ``vd`` error instead of an opaque backend error.

Filter syntax
-------------
A filter is a ``dict``. Each key is either a **metadata field name** or a
**logical operator** (``$and``, ``$or``, ``$not``). A bare ``{'field': value}``
is sugar for ``{'field': {'$eq': value}}``. Multiple top-level fields combine
with an implicit ``$and``.

Field operators (inside ``{'field': {...}}``): ``$eq``, ``$ne``, ``$gt``,
``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``, ``$exists``.

Logical operators (top-level): ``$and`` / ``$or`` take a list of subfilters;
``$not`` takes a single subfilter.

Examples
--------
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

An unknown operator fails loud rather than silently matching everything:

>>> matches_filter({'a': 1}, {'a': {'$bogus': 1}})
Traceback (most recent call last):
    ...
vd.base.UnsupportedFilterError: Unknown filter operator '$bogus'. ...
"""

from typing import Any, Iterable, Mapping, Optional

from vd.base import Filter, UnsupportedFilterError

# ---------------------------------------------------------------------------
# The operator vocabulary (single source of truth)
# ---------------------------------------------------------------------------

#: Operators used at the top level of a filter to combine subfilters.
LOGICAL_OPERATORS = frozenset({"$and", "$or", "$not"})

#: Operators used inside a ``{'field': {...}}`` condition.
FIELD_OPERATORS = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists"}
)

#: Every operator the canonical ``vd`` filter language defines.
SUPPORTED_FILTER_OPERATORS = LOGICAL_OPERATORS | FIELD_OPERATORS

# Sentinel distinguishing "field absent from metadata" from "field present and None".
_MISSING = object()


def _in(actual: Any, operand: Any) -> bool:
    """True if ``actual`` is in ``operand`` (list-valued ``actual`` matches on overlap)."""
    if not isinstance(operand, (list, tuple, set)):
        raise UnsupportedFilterError(
            f"$in / $nin operand must be a list, got {type(operand).__name__}"
        )
    if isinstance(actual, (list, tuple, set)):
        return any(a in operand for a in actual)
    return actual in operand


def _apply_field_operator(op: str, actual: Any, operand: Any) -> bool:
    """Evaluate a single field operator. ``actual`` may be the ``_MISSING`` sentinel."""
    if op == "$exists":
        return (actual is not _MISSING) is bool(operand)

    if actual is _MISSING:
        # A missing field can only satisfy "not equal" / "not in".
        return op in ("$ne", "$nin")

    if op == "$eq":
        return actual == operand
    if op == "$ne":
        return actual != operand
    if op == "$in":
        return _in(actual, operand)
    if op == "$nin":
        return not _in(actual, operand)

    # Ordered comparisons — incomparable types never match (rather than erroring).
    try:
        if op == "$gt":
            return actual > operand
        if op == "$gte":
            return actual >= operand
        if op == "$lt":
            return actual < operand
        if op == "$lte":
            return actual <= operand
    except TypeError:
        return False

    # Unreachable when the caller validated `op` against FIELD_OPERATORS first.
    raise UnsupportedFilterError(f"Unknown filter operator {op!r}")


def _match_field(actual: Any, condition: Any) -> bool:
    """Match one field. ``condition`` is a bare value (``$eq`` sugar) or an operator dict."""
    if not isinstance(condition, dict):
        # Bare-value equality. A missing field never equals a concrete value.
        return actual is not _MISSING and actual == condition
    for op, operand in condition.items():
        if op not in FIELD_OPERATORS:
            raise UnsupportedFilterError(
                f"Unknown filter operator {op!r}. "
                f"Supported field operators: {sorted(FIELD_OPERATORS)}"
            )
        if not _apply_field_operator(op, actual, operand):
            return False
    return True


def matches_filter(metadata: Mapping[str, Any], filter: Optional[Filter]) -> bool:
    """
    Return ``True`` if ``metadata`` satisfies the MongoDB-style ``filter``.

    An empty or ``None`` filter matches everything. Unknown operators raise
    :class:`~vd.base.UnsupportedFilterError` — they never silently match.

    Parameters
    ----------
    metadata : Mapping
        A document's metadata dict.
    filter : dict or None
        A filter in the canonical ``vd`` dialect (see the module docstring).

    Examples
    --------
    >>> matches_filter({'year': 2024}, None)
    True
    >>> matches_filter({'year': 2024, 'cat': 'tech'},
    ...                {'year': {'$gte': 2020}, 'cat': 'tech'})
    True
    >>> matches_filter({'views': 50}, {'views': {'$gte': 10, '$lte': 100}})
    True
    """
    if not filter:
        return True
    for key, condition in filter.items():
        if key == "$and":
            if not all(matches_filter(metadata, f) for f in condition):
                return False
        elif key == "$or":
            if not any(matches_filter(metadata, f) for f in condition):
                return False
        elif key == "$not":
            if matches_filter(metadata, condition):
                return False
        elif key.startswith("$"):
            raise UnsupportedFilterError(
                f"Unknown logical operator {key!r}. "
                f"Supported logical operators: {sorted(LOGICAL_OPERATORS)}"
            )
        else:
            if not _match_field(metadata.get(key, _MISSING), condition):
                return False
    return True


def validate_filter(
    filter: Optional[Filter],
    *,
    supported: Iterable[str] = SUPPORTED_FILTER_OPERATORS,
) -> None:
    """
    Walk ``filter`` and raise :class:`~vd.base.UnsupportedFilterError` on any
    operator that is unknown or not in ``supported``.

    Backends that translate the canonical filter to a native query call this
    with their own (possibly narrower) ``supported`` subset, so callers get a
    clear ``vd`` error up front instead of an opaque backend error later.

    Parameters
    ----------
    filter : dict or None
        A filter in the canonical ``vd`` dialect. ``None`` / empty is valid.
    supported : iterable of str, optional
        The operator subset to allow. Defaults to every operator the language
        defines.

    Examples
    --------
    >>> validate_filter({'year': {'$gte': 2020}})            # ok, returns None
    >>> validate_filter({'a': {'$regex': '.*'}})             # not in the language
    Traceback (most recent call last):
        ...
    vd.base.UnsupportedFilterError: Unknown filter operator '$regex'. ...
    >>> validate_filter({'a': {'$exists': True}}, supported={'$eq'})
    Traceback (most recent call last):
        ...
    vd.base.UnsupportedFilterError: Filter operator '$exists' is not supported ...
    """
    if not filter:
        return
    supported = frozenset(supported)
    for key, condition in filter.items():
        if key in LOGICAL_OPERATORS:
            if key not in supported:
                raise UnsupportedFilterError(
                    f"Filter operator {key!r} is not supported by this backend. "
                    f"Supported here: {sorted(supported)}"
                )
            subfilters = condition if key in ("$and", "$or") else [condition]
            for sub in subfilters:
                validate_filter(sub, supported=supported)
        elif key.startswith("$"):
            raise UnsupportedFilterError(
                f"Unknown logical operator {key!r}. "
                f"Supported logical operators: {sorted(LOGICAL_OPERATORS)}"
            )
        elif isinstance(condition, dict):
            for op in condition:
                if op not in FIELD_OPERATORS:
                    raise UnsupportedFilterError(
                        f"Unknown filter operator {op!r}. "
                        f"Supported field operators: {sorted(FIELD_OPERATORS)}"
                    )
                if op not in supported:
                    raise UnsupportedFilterError(
                        f"Filter operator {op!r} is not supported by this backend. "
                        f"Supported here: {sorted(supported)}"
                    )
        # else: bare-value equality — always supported, nothing to validate.
