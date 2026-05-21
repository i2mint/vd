"""
Time-indexed vector collection wrapper.

Wraps any vd ``Collection`` and adds an in-memory sorted ``(timestamp, id)``
index, enabling efficient retrieval by time window, fixed-window aggregation
(daily/hourly buckets), and time-bounded semantic search.

The wrapper is backend-agnostic: it works with any object that satisfies the
vd ``Collection`` protocol (``MutableMapping`` + ``search``). Timestamps are
stored as ISO-8601 strings in document metadata, so backend-side metadata
filtering keeps working (e.g. ChromaDB's ``$gte`` filter on a string field
sorts correctly because ISO-8601 is lexicographically ordered).

Examples
--------
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
"""

import bisect
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence, Union

from vd.base import Document, Filter, Vector


# Public type alias for anything we accept as a timestamp.
TimestampLike = Union[str, datetime, int, float]


def to_datetime(ts: TimestampLike) -> datetime:
    """Coerce a timestamp-like value into a tz-aware UTC ``datetime``.

    Accepts ISO-8601 strings (with or without timezone), date-only strings
    (``"2025-03-13"``), epoch seconds (int or float), and ``datetime`` objects.
    Naive datetimes / strings are assumed UTC.

    >>> to_datetime('2025-03-13T09:00:00').isoformat()
    '2025-03-13T09:00:00+00:00'
    >>> to_datetime('2025-03-13').isoformat()
    '2025-03-13T00:00:00+00:00'
    >>> to_datetime(1741856400).isoformat()
    '2025-03-13T09:00:00+00:00'
    """
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if isinstance(ts, str):
        s = ts.strip()
        # Allow trailing 'Z' (Python <3.11 doesn't parse it)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # date-only fallback
            dt = datetime.strptime(s, "%Y-%m-%d")
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    raise TypeError(f"Cannot coerce {ts!r} ({type(ts).__name__}) to datetime")


def to_iso(ts: TimestampLike) -> str:
    """ISO-8601 (UTC) string suitable for cross-backend metadata storage.

    >>> to_iso('2025-03-13T09:00:00')
    '2025-03-13T09:00:00+00:00'
    """
    return to_datetime(ts).isoformat()


# Window-step parser ----------------------------------------------------------

_WINDOW_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_window(window: Union[str, timedelta, int, float]) -> timedelta:
    """Parse a window spec into a ``timedelta``.

    Strings use a trailing unit char: ``"1d"``, ``"4h"``, ``"30m"``, ``"15s"``,
    ``"1w"``. Numbers are treated as seconds. A ``timedelta`` is returned as-is.

    >>> parse_window('1d') == timedelta(days=1)
    True
    >>> parse_window('4h') == timedelta(hours=4)
    True
    >>> parse_window(3600) == timedelta(hours=1)
    True
    """
    if isinstance(window, timedelta):
        return window
    if isinstance(window, (int, float)):
        return timedelta(seconds=float(window))
    if isinstance(window, str):
        s = window.strip().lower()
        if not s:
            raise ValueError("empty window spec")
        unit = s[-1]
        if unit not in _WINDOW_UNITS:
            raise ValueError(f"unknown window unit {unit!r} in {window!r}")
        n = float(s[:-1])
        return timedelta(**{_WINDOW_UNITS[unit]: n})
    raise TypeError(f"Cannot parse window spec {window!r}")


# Reducers --------------------------------------------------------------------


def _vec_or_none(doc: Document) -> Optional[Vector]:
    return doc.vector


def mean_vector(docs: Iterable[Document]) -> Optional[list[float]]:
    """Element-wise mean of document embeddings. ``None`` if empty / no vectors.

    >>> from vd.base import Document
    >>> mean_vector([
    ...     Document(id='a', text='', vector=[1.0, 2.0]),
    ...     Document(id='b', text='', vector=[3.0, 4.0]),
    ... ])
    [2.0, 3.0]
    >>> mean_vector([]) is None
    True
    """
    total: list[float] = []
    n = 0
    for d in docs:
        v = d.vector
        if v is None:
            continue
        if not total:
            total = list(v)
        else:
            for i, x in enumerate(v):
                total[i] += x
        n += 1
    if n == 0:
        return None
    return [x / n for x in total]


def count_docs(docs: Iterable[Document]) -> int:
    """``len`` reducer that also handles generator inputs.

    >>> count_docs(iter([1, 2, 3]))
    3
    """
    return sum(1 for _ in docs)


# Core wrapper ---------------------------------------------------------------


@dataclass
class WindowSlice:
    """A time window: ``[start, end)``."""

    start: datetime
    end: datetime


class TimeIndexedCollection(MutableMapping):
    """Time-indexed wrapper over any vd ``Collection``.

    Maintains a sorted ``(ts_epoch, id)`` index alongside the underlying
    collection. Each stored document MUST carry a timestamp in its metadata
    under ``ts_field`` (default ``"ts"``). The stored value is normalized to
    an ISO-8601 string so backend-side filtering remains usable.

    Parameters
    ----------
    collection
        Any vd Collection (MutableMapping + ``search``).
    ts_field
        Metadata key holding the timestamp.
    ts_parser
        Optional custom parser ``Any -> datetime``. Defaults to
        :func:`to_datetime`.

    Notes
    -----
    The index is rebuilt on construction from whatever the underlying
    collection already contains (so the wrapper is safe to re-wrap a persisted
    collection across process restarts).
    """

    def __init__(
        self,
        collection: MutableMapping,
        *,
        ts_field: str = "ts",
        ts_parser: Callable[[Any], datetime] = to_datetime,
    ):
        self._coll = collection
        self._ts_field = ts_field
        self._parse = ts_parser
        self._index: list[tuple[float, str]] = []
        self._ts_by_id: dict[str, float] = {}
        self._reindex_from_collection()

    # --- index lifecycle ----------------------------------------------------

    def _reindex_from_collection(self) -> None:
        items: list[tuple[float, str]] = []
        for doc_id in self._coll:
            doc = self._coll[doc_id]
            ts_raw = (
                doc.metadata.get(self._ts_field) if hasattr(doc, "metadata") else None
            )
            if ts_raw is None:
                continue
            try:
                ts_epoch = self._parse(ts_raw).timestamp()
            except Exception:
                continue
            items.append((ts_epoch, doc_id))
        items.sort()
        self._index = items
        self._ts_by_id = {doc_id: ts for ts, doc_id in items}

    def reindex(self) -> None:
        """Force-rebuild the in-memory time index from the underlying collection."""
        self._reindex_from_collection()

    def _index_insert(self, ts: float, doc_id: str) -> None:
        bisect.insort(self._index, (ts, doc_id))
        self._ts_by_id[doc_id] = ts

    def _index_remove(self, doc_id: str) -> None:
        ts = self._ts_by_id.pop(doc_id, None)
        if ts is None:
            return
        idx = bisect.bisect_left(self._index, (ts, doc_id))
        if idx < len(self._index) and self._index[idx] == (ts, doc_id):
            self._index.pop(idx)

    # --- MutableMapping interface ------------------------------------------

    def __setitem__(self, key: str, value: Union[str, Document, tuple]) -> None:
        # Normalize string-valued writes (text only) — caller must put ts in metadata.
        if isinstance(value, str):
            raise ValueError(
                f"TimeIndexedCollection needs metadata['{self._ts_field}']; "
                f"pass a Document or include ts metadata explicitly."
            )
        # Extract ts from incoming value's metadata (the underlying collection
        # may convert tuple→Document, so we extract first).
        if isinstance(value, Document):
            metadata = dict(value.metadata)
        elif (
            isinstance(value, tuple) and len(value) >= 2 and isinstance(value[-1], dict)
        ):
            metadata = dict(value[-1])
        else:
            raise ValueError(
                f"Unsupported value type for TimeIndexedCollection: {type(value)!r}"
            )

        ts_raw = metadata.get(self._ts_field)
        if ts_raw is None:
            raise ValueError(
                f"Document {key!r} is missing required metadata field "
                f"{self._ts_field!r}"
            )
        ts_dt = self._parse(ts_raw)
        # Canonicalize the metadata field to ISO so backend filters are stable.
        if isinstance(value, Document):
            value.metadata = {**value.metadata, self._ts_field: ts_dt.isoformat()}
        elif isinstance(value, tuple):
            md = {**value[-1], self._ts_field: ts_dt.isoformat()}
            value = value[:-1] + (md,)

        self._index_remove(key)  # in case of update
        self._coll[key] = value
        self._index_insert(ts_dt.timestamp(), key)

    def __getitem__(self, key: str) -> Document:
        return self._coll[key]

    def __delitem__(self, key: str) -> None:
        self._index_remove(key)
        del self._coll[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._coll)

    def __len__(self) -> int:
        return len(self._coll)

    @property
    def name(self) -> str:
        return getattr(self._coll, "name", "<unnamed>")

    @property
    def base(self) -> MutableMapping:
        """The wrapped underlying collection."""
        return self._coll

    # --- time-window operations --------------------------------------------

    def time_range(self) -> Optional[tuple[datetime, datetime]]:
        """Return ``(min_ts, max_ts)`` as aware datetimes, or None if empty.

        >>> from vd import connect, Document
        >>> import hashlib
        >>> emb = lambda t: [b/128.0-1.0 for b in hashlib.md5(t.encode()).digest()[:4]]
        >>> col = connect('memory', embedder=emb).create_collection('t')
        >>> t = TimeIndexedCollection(col)
        >>> t['a'] = Document(id='a', text='x', metadata={'ts': '2025-01-01'})
        >>> t['b'] = Document(id='b', text='y', metadata={'ts': '2025-03-01'})
        >>> [d.isoformat() for d in t.time_range()]
        ['2025-01-01T00:00:00+00:00', '2025-03-01T00:00:00+00:00']
        """
        if not self._index:
            return None
        lo = datetime.fromtimestamp(self._index[0][0], tz=timezone.utc)
        hi = datetime.fromtimestamp(self._index[-1][0], tz=timezone.utc)
        return lo, hi

    def query_window(
        self,
        start: Optional[TimestampLike] = None,
        end: Optional[TimestampLike] = None,
        *,
        filt: Optional[Filter] = None,
    ) -> Iterator[Document]:
        """Yield documents with ``start <= ts < end``, in chronological order.

        ``start`` / ``end`` may be ``None`` for half-open infinity. ``filt`` is
        an optional MongoDB-style predicate applied to document metadata,
        evaluated client-side (so it works on any backend).
        """
        lo = to_datetime(start).timestamp() if start is not None else float("-inf")
        hi = to_datetime(end).timestamp() if end is not None else float("inf")
        i_lo = bisect.bisect_left(self._index, (lo, ""))
        i_hi = bisect.bisect_left(self._index, (hi, ""))
        for _, doc_id in self._index[i_lo:i_hi]:
            doc = self._coll[doc_id]
            if filt is None or _match_filter(doc.metadata, filt):
                yield doc

    def window_iter(
        self,
        window: Union[str, timedelta, int, float] = "1d",
        *,
        start: Optional[TimestampLike] = None,
        end: Optional[TimestampLike] = None,
        reducer: Callable[[Iterable[Document]], Any] = count_docs,
        skip_empty: bool = False,
        align: bool = True,
    ) -> Iterator[tuple[datetime, datetime, Any]]:
        """Yield ``(window_start, window_end, reducer_value)`` over fixed windows.

        Parameters
        ----------
        window
            Window size. See :func:`parse_window` for accepted forms.
        start, end
            Override the data range. Default: actual min/max ts in the index.
        reducer
            Callable taking the iterable of in-window ``Document``s. Default
            is :func:`count_docs`. See also :func:`mean_vector`.
        skip_empty
            If True, omit windows that contained zero documents.
        align
            If True (default), align ``start`` to the previous midnight (for
            daily windows) or to ``window``-rounded boundary so downstream
            joins are clean. If False, use the literal ``start``.
        """
        step = parse_window(window)
        tr = self.time_range()
        if tr is None and (start is None or end is None):
            return
        data_lo, data_hi = tr if tr else (None, None)
        win_start = to_datetime(start) if start is not None else data_lo
        win_end = (
            to_datetime(end)
            if end is not None
            else (data_hi + timedelta(seconds=1) if data_hi else win_start + step)
        )
        if align:
            win_start = _align_floor(win_start, step)
        cur = win_start
        while cur < win_end:
            nxt = cur + step
            docs = list(self.query_window(cur, nxt))
            if skip_empty and not docs:
                cur = nxt
                continue
            yield cur, nxt, reducer(docs)
            cur = nxt

    def search_window(
        self,
        query: Union[str, Sequence[float]],
        *,
        start: Optional[TimestampLike] = None,
        end: Optional[TimestampLike] = None,
        limit: int = 10,
        filt: Optional[Filter] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """Semantic search restricted to a time window.

        Builds a metadata filter on ``ts_field`` and delegates to the
        underlying collection's ``search``. Falls back to a client-side
        post-filter for backends that don't honor the filter.
        """
        ts_clauses: dict[str, dict] = {}
        if start is not None:
            ts_clauses["$gte"] = to_iso(start)
        if end is not None:
            ts_clauses["$lt"] = to_iso(end)
        ts_filter: Filter = {self._ts_field: ts_clauses} if ts_clauses else {}
        if filt and ts_filter:
            combined: Filter = {"$and": [ts_filter, filt]}
        else:
            combined = ts_filter or filt or {}

        # Over-fetch to allow client-side fallback filtering if a backend
        # silently ignores the metadata filter.
        fetch = max(limit * 4, limit)
        results = list(self._coll.search(query, limit=fetch, filter=combined, **kwargs))
        n_yielded = 0
        s_epoch = to_datetime(start).timestamp() if start is not None else float("-inf")
        e_epoch = to_datetime(end).timestamp() if end is not None else float("inf")
        for r in results:
            ts_raw = (r.get("metadata") or {}).get(self._ts_field)
            if ts_raw is not None:
                try:
                    ts = self._parse(ts_raw).timestamp()
                except Exception:
                    continue
                if not (s_epoch <= ts < e_epoch):
                    continue
            yield r
            n_yielded += 1
            if n_yielded >= limit:
                return


# Window alignment helper -----------------------------------------------------


def _align_floor(dt: datetime, step: timedelta) -> datetime:
    """Floor ``dt`` to the nearest step boundary (UTC epoch-aligned).

    >>> _align_floor(datetime(2025, 3, 13, 14, 30, tzinfo=timezone.utc),
    ...              timedelta(days=1)).isoformat()
    '2025-03-13T00:00:00+00:00'
    >>> _align_floor(datetime(2025, 3, 13, 14, 30, tzinfo=timezone.utc),
    ...              timedelta(hours=4)).isoformat()
    '2025-03-13T12:00:00+00:00'
    """
    epoch_s = dt.timestamp()
    step_s = step.total_seconds()
    aligned = (epoch_s // step_s) * step_s
    return datetime.fromtimestamp(aligned, tz=timezone.utc)


# Lightweight client-side filter (subset of MongoDB syntax) ------------------


def _match_filter(metadata: dict, filt: Filter) -> bool:
    """Subset of the memory-backend filter: equality, $gte/$lte/$gt/$lt/$eq/$ne/$in, $and/$or."""
    for key, cond in filt.items():
        if key == "$and":
            if not all(_match_filter(metadata, f) for f in cond):
                return False
        elif key == "$or":
            if not any(_match_filter(metadata, f) for f in cond):
                return False
        elif isinstance(cond, dict):
            value = metadata.get(key)
            if value is None:
                return False
            for op, op_value in cond.items():
                if op == "$gte" and not (value >= op_value):
                    return False
                elif op == "$lte" and not (value <= op_value):
                    return False
                elif op == "$gt" and not (value > op_value):
                    return False
                elif op == "$lt" and not (value < op_value):
                    return False
                elif op == "$eq" and not (value == op_value):
                    return False
                elif op == "$ne" and not (value != op_value):
                    return False
                elif op == "$in":
                    if isinstance(value, list):
                        if not any(v in op_value for v in value):
                            return False
                    else:
                        if value not in op_value:
                            return False
        else:
            if metadata.get(key) != cond:
                return False
    return True
