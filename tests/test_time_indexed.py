"""Tests for vd.time_indexed.TimeIndexedCollection."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

import vd
from vd import (
    Document,
    TimeIndexedCollection,
    connect,
    count_docs,
    mean_vector,
    parse_window,
    to_datetime,
    to_iso,
)


def fake_embed(text: str) -> list[float]:
    """Deterministic 8-dim toy embedding."""
    return [b / 128.0 - 1.0 for b in hashlib.md5(text.encode()).digest()[:8]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_to_datetime_iso_str():
    dt = to_datetime("2025-03-13T09:00:00")
    assert dt == datetime(2025, 3, 13, 9, tzinfo=timezone.utc)


def test_to_datetime_z_suffix():
    dt = to_datetime("2025-03-13T09:00:00Z")
    assert dt == datetime(2025, 3, 13, 9, tzinfo=timezone.utc)


def test_to_datetime_date_only():
    dt = to_datetime("2025-03-13")
    assert dt == datetime(2025, 3, 13, tzinfo=timezone.utc)


def test_to_datetime_epoch_int():
    dt = to_datetime(1741856400)  # 2025-03-13T09:00:00Z
    assert dt == datetime(2025, 3, 13, 9, tzinfo=timezone.utc)


def test_to_datetime_naive_assumed_utc():
    naive = datetime(2025, 3, 13, 9)
    assert to_datetime(naive).tzinfo == timezone.utc


def test_to_iso_roundtrips():
    s = to_iso("2025-03-13T09:00:00")
    assert s == "2025-03-13T09:00:00+00:00"
    assert to_datetime(s).isoformat() == s


def test_parse_window_strings():
    assert parse_window("1d") == timedelta(days=1)
    assert parse_window("4h") == timedelta(hours=4)
    assert parse_window("30m") == timedelta(minutes=30)
    assert parse_window("15s") == timedelta(seconds=15)
    assert parse_window("2w") == timedelta(weeks=2)


def test_parse_window_seconds_int():
    assert parse_window(3600) == timedelta(hours=1)


def test_parse_window_passthrough():
    td = timedelta(hours=12)
    assert parse_window(td) is td


def test_parse_window_invalid():
    with pytest.raises(ValueError):
        parse_window("1x")
    with pytest.raises(ValueError):
        parse_window("")


def test_mean_vector_basic():
    docs = [
        Document(id="a", text="", vector=[1.0, 2.0]),
        Document(id="b", text="", vector=[3.0, 4.0]),
    ]
    assert mean_vector(docs) == [2.0, 3.0]


def test_mean_vector_empty():
    assert mean_vector([]) is None


def test_mean_vector_skips_none_vectors():
    docs = [
        Document(id="a", text="", vector=[1.0, 2.0]),
        Document(id="b", text="", vector=None),
        Document(id="c", text="", vector=[3.0, 4.0]),
    ]
    assert mean_vector(docs) == [2.0, 3.0]


# ---------------------------------------------------------------------------
# TimeIndexedCollection
# ---------------------------------------------------------------------------


@pytest.fixture
def tic():
    """Empty TimeIndexedCollection over a fresh memory backend."""
    client = connect("memory", embedder=fake_embed)
    base = client.create_collection("news")
    return TimeIndexedCollection(base)


@pytest.fixture
def populated(tic):
    """A TIC pre-populated with three ordered docs across two days."""
    tic["a"] = Document(
        id="a", text="Earnings miss", metadata={"ts": "2025-03-13T09:00:00"}
    )
    tic["b"] = Document(
        id="b", text="Profit warning", metadata={"ts": "2025-03-13T15:30:00"}
    )
    tic["c"] = Document(
        id="c", text="Tariffs announced", metadata={"ts": "2025-03-14T08:00:00"}
    )
    return tic


def test_setitem_indexes_in_order(populated):
    ids = [doc.id for doc in populated.query_window()]
    assert ids == ["a", "b", "c"]


def test_setitem_canonicalizes_ts(populated):
    # Stored back in ISO form
    assert populated["a"].metadata["ts"] == "2025-03-13T09:00:00+00:00"


def test_setitem_rejects_missing_ts(tic):
    with pytest.raises(ValueError, match="missing required metadata"):
        tic["x"] = Document(id="x", text="oops", metadata={})


def test_setitem_rejects_bare_string(tic):
    with pytest.raises(ValueError, match="needs metadata"):
        tic["x"] = "plain text"


def test_delitem_removes_from_index(populated):
    del populated["b"]
    assert "b" not in populated
    ids = [doc.id for doc in populated.query_window()]
    assert ids == ["a", "c"]


def test_update_existing_key_reindexes(populated):
    populated["b"] = Document(
        id="b", text="Profit warning v2", metadata={"ts": "2025-03-14T20:00:00"}
    )
    ids = [doc.id for doc in populated.query_window()]
    assert ids == ["a", "c", "b"]


def test_query_window_inclusive_start_exclusive_end(populated):
    ids = [d.id for d in populated.query_window("2025-03-13", "2025-03-14")]
    assert ids == ["a", "b"]


def test_query_window_open_start(populated):
    ids = [d.id for d in populated.query_window(end="2025-03-14")]
    assert ids == ["a", "b"]


def test_query_window_open_end(populated):
    ids = [d.id for d in populated.query_window(start="2025-03-14")]
    assert ids == ["c"]


def test_query_window_with_filter(populated):
    # Add a tagged doc; filter to only it
    populated["d"] = Document(
        id="d",
        text="Acquisition",
        metadata={"ts": "2025-03-13T11:00:00", "kind": "M&A"},
    )
    ids = [
        d.id
        for d in populated.query_window(
            "2025-03-13", "2025-03-14", filt={"kind": "M&A"}
        )
    ]
    assert ids == ["d"]


def test_time_range(populated):
    lo, hi = populated.time_range()
    assert lo == datetime(2025, 3, 13, 9, tzinfo=timezone.utc)
    assert hi == datetime(2025, 3, 14, 8, tzinfo=timezone.utc)


def test_time_range_empty(tic):
    assert tic.time_range() is None


def test_window_iter_count_default(populated):
    out = [(s.date().isoformat(), v) for s, _, v in populated.window_iter("1d")]
    assert out == [("2025-03-13", 2), ("2025-03-14", 1)]


def test_window_iter_skip_empty(tic):
    tic["a"] = Document(id="a", text="x", metadata={"ts": "2025-03-13T00:00:00"})
    tic["b"] = Document(id="b", text="y", metadata={"ts": "2025-03-20T00:00:00"})
    out = list(tic.window_iter("1d", skip_empty=True))
    # Only the two populated days should appear
    assert [v for _, _, v in out] == [1, 1]
    assert len(out) == 2


def test_window_iter_custom_reducer(populated):
    out = [
        (s.date().isoformat(), v)
        for s, _, v in populated.window_iter("1d", reducer=mean_vector)
    ]
    # Two windows, each has a non-None mean vector of length 8
    assert [k for k, _ in out] == ["2025-03-13", "2025-03-14"]
    for _, v in out:
        assert v is not None
        assert len(v) == 8


def test_window_iter_explicit_bounds(populated):
    out = list(
        populated.window_iter(
            "1d", start="2025-03-13", end="2025-03-15", reducer=count_docs
        )
    )
    assert [v for _, _, v in out] == [2, 1]


def test_search_window_filters_by_time(populated):
    # Search for "earnings"-ish text; restrict to the 13th
    results = list(
        populated.search_window("Earnings", start="2025-03-13", end="2025-03-14")
    )
    ids = [r["id"] for r in results]
    # All results should have ts < 2025-03-14
    assert "c" not in ids
    assert "a" in ids


def test_reindex_recovers_from_external_writes():
    """If something writes directly into the underlying collection, ``reindex`` recovers."""
    client = connect("memory", embedder=fake_embed)
    base = client.create_collection("news")
    tic = TimeIndexedCollection(base)
    # Bypass the wrapper:
    base["a"] = Document(id="a", text="x", metadata={"ts": "2025-03-13T00:00:00"})
    # Wrapper is unaware until reindex
    assert list(tic.query_window()) == []
    tic.reindex()
    assert [d.id for d in tic.query_window()] == ["a"]


def test_reconstruction_across_instances():
    """Rewrapping a populated collection rebuilds the index correctly."""
    client = connect("memory", embedder=fake_embed)
    base = client.create_collection("news")
    tic1 = TimeIndexedCollection(base)
    tic1["a"] = Document(id="a", text="x", metadata={"ts": "2025-03-13T00:00:00"})
    tic1["b"] = Document(id="b", text="y", metadata={"ts": "2025-03-12T00:00:00"})
    # New wrapper over same backing store
    tic2 = TimeIndexedCollection(base)
    assert [d.id for d in tic2.query_window()] == ["b", "a"]


def test_custom_ts_field():
    client = connect("memory", embedder=fake_embed)
    base = client.create_collection("ev")
    tic = TimeIndexedCollection(base, ts_field="event_time")
    tic["a"] = Document(id="a", text="x", metadata={"event_time": "2025-03-13"})
    assert [d.id for d in tic.query_window()] == ["a"]
