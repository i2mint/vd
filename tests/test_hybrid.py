"""
Contract tests for :func:`vd.hybrid_search` and the :class:`vd.SupportsHybrid` protocol.

Every backend the suite can reach runs the same parametrized hybrid contract.
The runtime ``isinstance(collection, SupportsHybrid)`` discovery splits the
sweep into two paths automatically — native adapters use their own
``hybrid_search`` method; everything else uses the client-side BM25 + RRF
fallback. The contract assertions are the same either way.

What's tested
-------------
- ``vd.hybrid_search`` returns a non-empty iterator for a non-trivial query
  on a populated collection.
- Each result is a dict with ``id``, ``text``, ``score`` (fused), ``metadata``.
- Results are ordered by fused score, descending.
- A query that matches *only* lexically still ranks the matching doc above
  noise, proving the lexical signal flows through the fusion. Similarly for
  a query that matches *only* densely.
- The native vs fallback split is observable via ``isinstance(c,
  SupportsHybrid)``; both code paths produce the contract above.
- ``bm25_lexical_search`` ranks documents in expected order on a small
  hand-built corpus (no fusion, no embedder needed).
"""

import pytest

import vd

# ---------- BM25-only unit tests (no fusion, no embedder, no server) ------- #


def test_bm25_basic_ordering():
    """BM25 ranks term-matching docs above non-matching docs."""
    client = vd.connect("memory")
    col = client.create_collection("bm25", dimension=2)
    col["a"] = vd.Document(id="a", text="the quick brown fox", vector=[1.0, 0.0])
    col["b"] = vd.Document(id="b", text="lazy dog sleeps", vector=[0.0, 1.0])
    col["c"] = vd.Document(id="c", text="quick fox runs", vector=[0.5, 0.5])

    hits = vd.bm25_lexical_search(col, "quick fox", limit=3)
    ids = [h["id"] for h in hits]
    assert ids[:2] == ["c", "a"] or ids[:2] == ["a", "c"], (
        f"both 'a' and 'c' contain 'quick fox'; got {ids}"
    )
    assert "b" not in ids[:2]


def test_bm25_empty_query_returns_empty_list():
    client = vd.connect("memory")
    col = client.create_collection("bm25e", dimension=2)
    col["a"] = vd.Document(id="a", text="hello world", vector=[1.0, 0.0])
    assert vd.bm25_lexical_search(col, "", limit=10) == []
    assert vd.bm25_lexical_search(col, "   ", limit=10) == []


def test_bm25_skips_docs_with_empty_text():
    client = vd.connect("memory")
    col = client.create_collection("bm25t", dimension=2)
    col["a"] = vd.Document(id="a", text="", vector=[1.0, 0.0])
    col["b"] = vd.Document(id="b", text="match this", vector=[0.0, 1.0])
    hits = vd.bm25_lexical_search(col, "match", limit=10)
    assert [h["id"] for h in hits] == ["b"]


def test_bm25_filter_is_applied():
    client = vd.connect("memory")
    col = client.create_collection("bm25f", dimension=2)
    col["a"] = vd.Document(
        id="a", text="cats purr softly", vector=[1.0, 0.0], metadata={"kind": "x"}
    )
    col["b"] = vd.Document(
        id="b", text="cats meow loudly", vector=[0.0, 1.0], metadata={"kind": "y"}
    )
    hits = vd.bm25_lexical_search(col, "cats", limit=10, filter={"kind": "x"})
    assert [h["id"] for h in hits] == ["a"]


# ---------- BM25Index: build-once / query-many ----------------------------- #


def _toy_collection():
    client = vd.connect("memory")
    col = client.create_collection("bm25idx", dimension=2)
    col["a"] = vd.Document(id="a", text="the quick brown fox", vector=[1.0, 0.0])
    col["b"] = vd.Document(id="b", text="lazy dog sleeps", vector=[0.0, 1.0])
    col["c"] = vd.Document(id="c", text="quick fox runs", vector=[0.5, 0.5])
    return col


def test_bm25_index_matches_one_shot_function():
    """A prebuilt BM25Index gives identical results to bm25_lexical_search."""
    col = _toy_collection()
    index = vd.BM25Index(col)
    for query in ("quick fox", "lazy dog", "runs", "nonexistentterm"):
        one_shot = vd.bm25_lexical_search(col, query, limit=10)
        from_index = index.search(query, limit=10)
        assert [(r["id"], round(r["score"], 9)) for r in from_index] == [
            (r["id"], round(r["score"], 9)) for r in one_shot
        ], f"mismatch for {query!r}"


def test_bm25_index_reusable_across_queries():
    """The same index answers many queries (build-once / query-many)."""
    index = vd.BM25Index(_toy_collection())
    assert index.search("lazy dog", limit=1)[0]["id"] == "b"
    assert index.search("quick fox", limit=2)[0]["id"] in ("a", "c")
    assert len(index) == 3


def test_bm25_index_empty_query_and_filter():
    col = _toy_collection()
    assert vd.BM25Index(col).search("", limit=5) == []
    # filter applied once at build time → only matching docs are indexed
    col["d"] = vd.Document(
        id="d", text="quick fox flies", vector=[0.2, 0.8], metadata={"kind": "z"}
    )
    index = vd.BM25Index(col, filter={"kind": "z"})
    assert len(index) == 1
    assert [r["id"] for r in index.search("quick fox", limit=10)] == ["d"]


# ---------- Hybrid contract: parametrized over every reachable backend ----- #


# Vectors are chosen so dense-only and lexical-only signals are separable.
# Dimension matches the test embedder (EMBED_DIM=16 from conftest.py).
_DOCS = [
    # Dense-only winner for the query vector below: doc 'dense_top' has a vector
    # close to the query vector but text that has nothing to do with the query.
    {
        "id": "dense_top",
        "text": "completely unrelated text about geology and rocks",
        "vec_seed": 0,  # makes it match the query vector closely
    },
    # Lexical-only winner: text matches but vector is orthogonal.
    {
        "id": "lex_top",
        "text": "machine learning embeddings power retrieval systems",
        "vec_seed": 1,
    },
    # Both — should top the fused ranking.
    {
        "id": "both",
        "text": "machine learning models for retrieval",
        "vec_seed": 0,
    },
    # Noise documents.
    {
        "id": "noise_1",
        "text": "cooking recipes for autumn",
        "vec_seed": 2,
    },
    {
        "id": "noise_2",
        "text": "history of medieval trade routes",
        "vec_seed": 3,
    },
]


def _seeded_vector(seed: int, dim: int) -> list[float]:
    """Deterministic small vector — used so we can craft known-similar pairs."""
    import math

    base = [math.sin(seed + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in base)) or 1.0
    return [x / norm for x in base]


@pytest.fixture
def populated_collection(client):
    """A collection populated with the dense+lexical-separable test docs."""
    dim = 16  # conftest.EMBED_DIM
    col = client.create_collection("hybrid_test", dimension=dim)
    for d in _DOCS:
        col[d["id"]] = vd.Document(
            id=d["id"],
            text=d["text"],
            vector=_seeded_vector(d["vec_seed"], dim),
        )
    return col


def test_hybrid_search_returns_contract_shape(populated_collection):
    """Each hit is a dict with id, text, score, metadata; sorted desc by score."""
    query_vec = _seeded_vector(0, 16)
    hits = list(
        vd.hybrid_search(
            populated_collection,
            query_vec,
            query_text="machine learning retrieval",
            limit=4,
        )
    )
    assert len(hits) > 0, "hybrid_search returned no results"
    assert len(hits) <= 4
    for h in hits:
        assert {"id", "text", "score", "metadata"} <= set(h.keys())
    # Descending score order.
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_fuses_dense_and_lexical(populated_collection):
    """
    The 'both' doc (top dense AND top lexical) ranks above 'dense_top'
    (dense-only) and above 'lex_top' (lexical-only). RRF makes this true on
    both the native and fallback paths.
    """
    query_vec = _seeded_vector(0, 16)
    hits = list(
        vd.hybrid_search(
            populated_collection,
            query_vec,
            query_text="machine learning retrieval",
            limit=5,
        )
    )
    ids = [h["id"] for h in hits]
    assert "both" in ids, f"'both' missing from fused results: {ids}"
    both_idx = ids.index("both")
    # 'both' must be at or near the top — fused signal beats single-signal docs.
    assert both_idx <= 1, (
        f"'both' should rank top-2; got position {both_idx} in {ids}"
    )
    # And both single-signal docs should also appear above noise.
    assert "noise_1" not in ids[:2] and "noise_2" not in ids[:2]


def test_hybrid_search_string_query_uses_text_for_both_sides(
    populated_collection, embedder
):
    """When query is a string, it's auto-used for both dense (via embedder) + lexical."""
    hits = list(
        vd.hybrid_search(populated_collection, "machine learning retrieval", limit=5)
    )
    ids = [h["id"] for h in hits]
    # 'both' has the literal lexical match; should be top-ranked.
    assert "both" in ids[:2]


def test_hybrid_search_requires_query_text_when_query_is_vector(populated_collection):
    """A vector query without query_text is a clear error (no lexical side)."""
    query_vec = _seeded_vector(0, 16)
    with pytest.raises(ValueError, match="query_text"):
        list(vd.hybrid_search(populated_collection, query_vec, limit=3))


def test_hybrid_search_honors_filter(populated_collection):
    """Filter is applied on both sub-searches; results are restricted accordingly."""
    # Re-tag one doc so we can filter it in.
    populated_collection["both"] = vd.Document(
        id="both",
        text="machine learning models for retrieval",
        vector=_seeded_vector(0, 16),
        metadata={"tag": "keep"},
    )
    query_vec = _seeded_vector(0, 16)
    hits = list(
        vd.hybrid_search(
            populated_collection,
            query_vec,
            query_text="machine learning retrieval",
            limit=5,
            filter={"tag": "keep"},
        )
    )
    ids = [h["id"] for h in hits]
    assert ids == ["both"], f"only the tagged doc should pass the filter; got {ids}"


def test_native_vs_fallback_path_is_observable(populated_collection, backend_name):
    """isinstance(c, SupportsHybrid) cleanly splits the two paths."""
    is_native = isinstance(populated_collection, vd.SupportsHybrid)
    if backend_name in {"weaviate", "elasticsearch", "redis", "lancedb"}:
        assert is_native, (
            f"{backend_name} should be SupportsHybrid in this PR; got {is_native}"
        )
    else:
        assert not is_native, (
            f"{backend_name} should NOT be SupportsHybrid yet; got {is_native}"
        )


def test_hybrid_search_custom_lexical_callable(populated_collection):
    """Users can supply their own lexical_search callable (fallback path only)."""
    if isinstance(populated_collection, vd.SupportsHybrid):
        pytest.skip("lexical_search= override only applies on the fallback path")

    calls = {"count": 0}

    def my_lex(collection, query_text, *, limit, filter, **kwargs):
        calls["count"] += 1
        # Trivial: return doc 'noise_1' as a sentinel so we can assert it was called.
        doc = collection["noise_1"]
        return [
            {
                "id": doc.id,
                "text": doc.text,
                "score": 99.0,
                "metadata": dict(doc.metadata),
            }
        ]

    hits = list(
        vd.hybrid_search(
            populated_collection,
            _seeded_vector(0, 16),
            query_text="anything",
            limit=5,
            lexical_search=my_lex,
        )
    )
    assert calls["count"] == 1
    ids = [h["id"] for h in hits]
    assert "noise_1" in ids, "the custom lexical search's hit should fuse in"


# ---------- LanceDB: native lexical side via its built-in FTS index -------- #


@pytest.fixture
def lance_col():
    pytest.importorskip("lancedb")
    col = vd.connect("lancedb").create_collection("lance_hybrid", dimension=2)
    col["a"] = vd.Document(id="a", text="the quick brown fox", vector=[1.0, 0.0])
    col["b"] = vd.Document(id="b", text="lazy dog sleeps", vector=[0.0, 1.0])
    return col


def _fts_indexed_columns(col) -> list:
    return [
        list(i.columns) for i in col.native.list_indices() if i.index_type == "FTS"
    ]


def test_lancedb_hybrid_uses_native_fts_index(lance_col):
    """The first hybrid call builds a LanceDB FTS index on ``text`` and uses it."""
    assert isinstance(lance_col, vd.SupportsHybrid)
    assert _fts_indexed_columns(lance_col) == []
    hits = list(
        vd.hybrid_search(lance_col, [0.0, 1.0], query_text="quick fox", limit=2)
    )
    assert {h["id"] for h in hits} == {"a", "b"}
    assert _fts_indexed_columns(lance_col) == [["text"]]
    # The lexical primitive alone finds only the term match.
    assert [h["id"] for h in lance_col._lexical_query("quick fox", limit=5,
                                                      filter=None)] == ["a"]


def test_lancedb_lexical_sees_writes_after_index_build(lance_col):
    """Upserts and deletes after the FTS index exists are reflected."""
    list(vd.hybrid_search(lance_col, [1.0, 0.0], query_text="fox", limit=1))
    lance_col["c"] = vd.Document(id="c", text="a fox runs", vector=[0.5, 0.5])
    lance_col["a"] = vd.Document(id="a", text="nothing here", vector=[1.0, 0.0])
    ids = [h["id"] for h in lance_col._lexical_query("fox", limit=5, filter=None)]
    assert ids == ["c"]
    del lance_col["c"]
    assert lance_col._lexical_query("fox", limit=5, filter=None) == []


def test_lancedb_lexical_filter_and_odd_queries(lance_col):
    """Metadata filters apply; query-syntax characters don't raise."""
    lance_col["c"] = vd.Document(
        id="c", text="fox on the hill", vector=[0.5, 0.5], metadata={"k": 1}
    )
    hits = lance_col._lexical_query("fox", limit=5, filter={"k": 1})
    assert [h["id"] for h in hits] == ["c"]
    assert {"id", "text", "score", "metadata"} <= set(hits[0])
    for odd in ['a "quote', "(", "title:x", "C++?"]:
        lance_col._lexical_query(odd, limit=5, filter=None)


def test_lancedb_hybrid_before_first_write():
    """A created-but-empty collection returns no hits instead of raising."""
    pytest.importorskip("lancedb")
    col = vd.connect("lancedb").create_collection("empty_hybrid", dimension=2)
    assert list(vd.hybrid_search(col, [1.0, 0.0], query_text="x", limit=3)) == []


def test_lexical_search_on_native_path_warns(lance_col):
    """A custom lexical_search can't be honoured natively; say so instead of ignoring it."""
    calls = []

    def my_lex(collection, text, *, limit, filter, **kwargs):
        calls.append(text)
        return []

    with pytest.warns(UserWarning, match="lexical_search"):
        list(vd.hybrid_search(lance_col, [1.0, 0.0], query_text="fox",
                              limit=1, lexical_search=my_lex))
    assert calls == []


def test_lancedb_first_hybrid_calls_concurrently():
    """Concurrent first hybrid calls must not race on creating the FTS index."""
    from concurrent.futures import ThreadPoolExecutor

    pytest.importorskip("lancedb")
    for round_ in range(3):  # a fresh table each round hits the creation window
        client = vd.connect("lancedb")
        col = client.create_collection(f"race{round_}", dimension=2)
        col["a"] = vd.Document(id="a", text="the quick brown fox", vector=[1.0, 0.0])
        col["b"] = vd.Document(id="b", text="lazy dog sleeps", vector=[0.0, 1.0])
        # Each worker gets its own collection object, as separate requests would.
        cols = [client.get_collection(f"race{round_}") for _ in range(8)]

        def run(c):
            return [h["id"] for h in vd.hybrid_search(
                c, [1.0, 0.0], query_text="quick fox", limit=2)]

        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(run, cols))
        assert all(r and r[0] == "a" for r in results), results


def test_lancedb_hybrid_after_table_dropped_and_recreated(lance_col):
    """A dropped-and-recreated table gets a fresh FTS index (no stale cache)."""
    client_like = lance_col._db
    list(vd.hybrid_search(lance_col, [1.0, 0.0], query_text="fox", limit=1))
    client_like.drop_table(lance_col.name)            # dropped behind vd's back
    lance_col["c"] = vd.Document(id="c", text="a fox again", vector=[1.0, 0.0])
    hits = list(vd.hybrid_search(lance_col, [1.0, 0.0], query_text="fox", limit=1))
    assert [h["id"] for h in hits] == ["c"]
    # and through the client API, with a fresh collection object
    client = vd.connect("lancedb")
    col = client.create_collection("recreated", dimension=2)
    col["a"] = vd.Document(id="a", text="red fox", vector=[1.0, 0.0])
    list(vd.hybrid_search(col, [1.0, 0.0], query_text="fox", limit=1))
    client.delete_collection("recreated")
    col2 = client.create_collection("recreated", dimension=2)
    col2["b"] = vd.Document(id="b", text="blue fox", vector=[1.0, 0.0])
    assert [h["id"] for h in vd.hybrid_search(
        col2, [1.0, 0.0], query_text="fox", limit=1)] == ["b"]
