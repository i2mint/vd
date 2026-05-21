"""
DuckDB-VSS backend.

DuckDB's ``vss`` extension adds HNSW vector search to DuckDB — the right
choice when you want analytics and ANN over the same column in the same SQL
engine. Each ``vd`` collection is one DuckDB table ``(doc_id, text, metadata,
embedding FLOAT[N])``; the optional HNSW index is created best-effort.

``vd`` applies the canonical metadata filter client-side (over-fetching
candidates first) so semantics match every other backend.

Requires: ``pip install duckdb``
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Iterator, Optional

try:
    import duckdb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The duckdb backend needs the 'duckdb' package. "
        "Install it with: pip install duckdb"
    ) from e

from vd.backends._helpers import apply_client_filter, overfetch_limit, score_from_distance
from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.util import register_backend

#: vd metric -> DuckDB-VSS distance function and index metric name.
_DISTANCE_FN = {
    "cosine": "array_cosine_distance",
    "l2": "array_distance",
    "dot": "array_negative_inner_product",
}
_INDEX_METRIC = {"cosine": "cosine", "l2": "l2sq", "dot": "ip"}


def _open(path: str) -> "duckdb.DuckDBPyConnection":
    """Open a DuckDB connection with the vss extension installed and loaded."""
    conn = duckdb.connect(path)
    conn.execute("INSTALL vss; LOAD vss;")
    conn.execute("SET hnsw_enable_experimental_persistence = true")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _vd_collections "
        "(name VARCHAR PRIMARY KEY, dimension INTEGER, metric VARCHAR)"
    )
    return conn


class DuckDBCollection(AbstractCollection):
    """A collection backed by one DuckDB table with a ``FLOAT[N]`` vector column."""

    def __init__(
        self,
        name: str,
        conn: "duckdb.DuckDBPyConnection",
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._conn = conn
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._tbl = f'"{name}"'
        self._created = self._table_exists()

    @property
    def native(self) -> "duckdb.DuckDBPyConnection":
        """The raw DuckDB connection (escape hatch)."""
        return self._conn

    def _table_exists(self) -> bool:
        return bool(
            self._conn.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_name=?",
                [self.name],
            ).fetchone()
        )

    def _ensure_table(self, dimension: int) -> None:
        """Create the backing table (and HNSW index) once the dimension is known."""
        if self._created:
            return
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._tbl} "
            f"(doc_id VARCHAR PRIMARY KEY, text VARCHAR, metadata VARCHAR, "
            f"embedding FLOAT[{dimension}])"
        )
        try:  # HNSW index is an optimization — tolerate its absence
            metric = _INDEX_METRIC.get(self.metric, "cosine")
            self._conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{self.name}_hnsw" ON {self._tbl} '
                f"USING HNSW (embedding) WITH (metric = '{metric}')"
            )
        except Exception:
            pass
        self._created = True

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._ensure_table(self.dimension)
        self._conn.execute(
            f"INSERT INTO {self._tbl}(doc_id, text, metadata, embedding) "
            f"VALUES (?,?,?,?) ON CONFLICT (doc_id) DO UPDATE SET "
            f"text=excluded.text, metadata=excluded.metadata, "
            f"embedding=excluded.embedding",
            [doc.id, doc.text, json.dumps(doc.metadata or {}), doc.vector],
        )

    def _read(self, key: str) -> Document:
        row = self._conn.execute(
            f"SELECT text, metadata, embedding FROM {self._tbl} WHERE doc_id=?",
            [key],
        ).fetchone() if self._created else None
        if row is None:
            raise KeyError(key)
        text, metadata, embedding = row
        return Document(
            id=key,
            text=text or "",
            vector=list(embedding) if embedding is not None else None,
            metadata=json.loads(metadata or "{}"),
        )

    def _drop(self, key: str) -> None:
        if not self._created:
            raise KeyError(key)
        exists = self._conn.execute(
            f"SELECT 1 FROM {self._tbl} WHERE doc_id=?", [key]
        ).fetchone()
        if exists is None:
            raise KeyError(key)
        self._conn.execute(f"DELETE FROM {self._tbl} WHERE doc_id=?", [key])

    def _keys(self) -> Iterator[str]:
        if not self._created:
            return iter(())
        rows = self._conn.execute(f"SELECT doc_id FROM {self._tbl}").fetchall()
        return iter(r[0] for r in rows)

    def _count(self) -> int:
        if not self._created:
            return 0
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._tbl}").fetchone()[0]

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        if not self._created:
            return []
        fn = _DISTANCE_FN.get(self.metric, "array_cosine_distance")
        dim = len(vector)
        rows = self._conn.execute(
            f"SELECT doc_id, text, metadata, "
            f"{fn}(embedding, ?::FLOAT[{dim}]) AS dist "
            f"FROM {self._tbl} ORDER BY dist LIMIT ?",
            [vector, overfetch_limit(limit, filter)],
        ).fetchall()
        results = [
            {
                "id": doc_id,
                "text": text or "",
                "score": score_from_distance(dist, self.metric),
                "metadata": json.loads(metadata or "{}"),
            }
            for doc_id, text, metadata, dist in rows
        ]
        return apply_client_filter(results, filter, limit=limit)


@register_backend("duckdb")
class DuckDBClient(AbstractClient):
    """
    DuckDB-VSS client.

    Parameters
    ----------
    path : str, optional
        Path to the DuckDB database file. Defaults to ``":memory:"``.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        path: str = ":memory:",
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        self._client = _open(path)

    def _row(self, name: str):
        return self._client.execute(
            "SELECT dimension, metric FROM _vd_collections WHERE name=?", [name]
        ).fetchone()

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> DuckDBCollection:
        if self._row(name) is not None:
            raise ValueError(f"Collection {name!r} already exists")
        self._client.execute(
            "INSERT INTO _vd_collections VALUES (?,?,?)", [name, dimension, metric]
        )
        return DuckDBCollection(
            name, self._client, embedder=self._embedder,
            dimension=dimension, metric=metric,
        )

    def get_collection(self, name: str) -> DuckDBCollection:
        row = self._row(name)
        if row is None:
            raise KeyError(f"Collection {name!r} does not exist")
        return DuckDBCollection(
            name, self._client, embedder=self._embedder,
            dimension=row[0], metric=row[1] or "cosine",
        )

    def delete_collection(self, name: str) -> None:
        if self._row(name) is None:
            raise KeyError(f"Collection {name!r} does not exist")
        self._client.execute(f'DROP TABLE IF EXISTS "{name}"')
        self._client.execute("DELETE FROM _vd_collections WHERE name=?", [name])

    def list_collections(self) -> Iterator[str]:
        rows = self._client.execute("SELECT name FROM _vd_collections").fetchall()
        return iter(r[0] for r in rows)

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._client.close()
