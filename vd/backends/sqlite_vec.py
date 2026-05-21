"""
sqlite-vec backend.

sqlite-vec adds brute-force vector search to plain SQLite — the right choice
when an app already uses SQLite, or wants a single-file vector store with zero
server. Each ``vd`` collection is two tables in the SQLite database: a regular
``<name>_docs`` table (id, text, metadata-as-JSON) and a ``vec0`` virtual
table ``<name>_vec`` holding the vectors, joined by ``rowid``.

sqlite-vec's native metadata filtering is limited, so ``vd`` applies the
canonical filter client-side (over-fetching candidates first).

Requires: ``pip install sqlite-vec``  (and SQLite >= 3.41 with loadable
extensions — see :func:`vd.check_requirements`).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Callable, Iterable, Iterator, Optional

try:
    import sqlite_vec
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The sqlite_vec backend needs the 'sqlite-vec' package. "
        "Install it with: pip install sqlite-vec"
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


def _open(path: str) -> sqlite3.Connection:
    """Open a SQLite connection with the sqlite-vec extension loaded."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _vd_collections "
        "(name TEXT PRIMARY KEY, dimension INTEGER, metric TEXT)"
    )
    return conn


class SqliteVecCollection(AbstractCollection):
    """A collection backed by a ``<name>_docs`` table + a ``vec0`` virtual table."""

    def __init__(
        self,
        name: str,
        conn: sqlite3.Connection,
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
        self._docs_tbl = f'"{name}_docs"'
        self._vec_tbl = f'"{name}_vec"'

    @property
    def native(self) -> sqlite3.Connection:
        """The raw ``sqlite3.Connection`` (escape hatch)."""
        return self._conn

    def _ensure_tables(self, dimension: int) -> None:
        """Create the docs + vec0 tables once the dimension is known."""
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._docs_tbl} "
            f"(rowid INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, text TEXT, metadata TEXT)"
        )
        col_def = f"embedding float[{dimension}]"
        if self.metric == "cosine":
            col_def += " distance_metric=cosine"
        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_tbl} "
                f"USING vec0({col_def})"
            )
        except sqlite3.OperationalError:
            # Older sqlite-vec without distance_metric= support.
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._vec_tbl} "
                f"USING vec0(embedding float[{dimension}])"
            )
        self._conn.commit()

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        if self.dimension is not None:
            self._ensure_tables(self.dimension)
        cur = self._conn.execute(
            f"INSERT INTO {self._docs_tbl}(doc_id, text, metadata) VALUES(?,?,?) "
            f"ON CONFLICT(doc_id) DO UPDATE SET text=excluded.text, "
            f"metadata=excluded.metadata RETURNING rowid",
            (doc.id, doc.text, json.dumps(doc.metadata or {})),
        )
        rowid = cur.fetchone()[0]
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._vec_tbl}(rowid, embedding) VALUES(?, ?)",
            (rowid, sqlite_vec.serialize_float32(doc.vector)),
        )
        self._conn.commit()

    def _read(self, key: str) -> Document:
        row = self._conn.execute(
            f"SELECT rowid, text, metadata FROM {self._docs_tbl} WHERE doc_id=?",
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        rowid, text, metadata = row
        vec_row = self._conn.execute(
            f"SELECT vec_to_json(embedding) FROM {self._vec_tbl} WHERE rowid=?",
            (rowid,),
        ).fetchone()
        vector = json.loads(vec_row[0]) if vec_row else None
        return Document(
            id=key, text=text or "", vector=vector, metadata=json.loads(metadata or "{}")
        )

    def _drop(self, key: str) -> None:
        row = self._conn.execute(
            f"SELECT rowid FROM {self._docs_tbl} WHERE doc_id=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        self._conn.execute(f"DELETE FROM {self._docs_tbl} WHERE rowid=?", (row[0],))
        self._conn.execute(f"DELETE FROM {self._vec_tbl} WHERE rowid=?", (row[0],))
        self._conn.commit()

    def _keys(self) -> Iterator[str]:
        try:
            rows = self._conn.execute(
                f"SELECT doc_id FROM {self._docs_tbl}"
            ).fetchall()
        except sqlite3.OperationalError:
            return iter(())  # tables not created yet (empty collection)
        return iter(r[0] for r in rows)

    def _count(self) -> int:
        try:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM {self._docs_tbl}"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        fetch = overfetch_limit(limit, filter)
        try:
            rows = self._conn.execute(
                f"SELECT d.doc_id, d.text, d.metadata, v.distance "
                f"FROM {self._vec_tbl} v JOIN {self._docs_tbl} d ON d.rowid = v.rowid "
                f"WHERE v.embedding MATCH ? ORDER BY v.distance LIMIT ?",
                (sqlite_vec.serialize_float32(vector), fetch),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # empty collection — tables not created yet
        results = [
            {
                "id": doc_id,
                "text": text or "",
                "score": score_from_distance(distance, self.metric),
                "metadata": json.loads(metadata or "{}"),
            }
            for doc_id, text, metadata, distance in rows
        ]
        return apply_client_filter(results, filter, limit=limit)


@register_backend("sqlite_vec")
class SqliteVecClient(AbstractClient):
    """
    sqlite-vec client.

    Parameters
    ----------
    path : str, optional
        Path to the SQLite database file. Defaults to ``":memory:"`` (a
        non-persistent in-process database).
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
            "SELECT dimension, metric FROM _vd_collections WHERE name=?", (name,)
        ).fetchone()

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> SqliteVecCollection:
        if self._row(name) is not None:
            raise ValueError(f"Collection {name!r} already exists")
        self._client.execute(
            "INSERT INTO _vd_collections(name, dimension, metric) VALUES(?,?,?)",
            (name, dimension, metric),
        )
        self._client.commit()
        return SqliteVecCollection(
            name, self._client, embedder=self._embedder,
            dimension=dimension, metric=metric,
        )

    def get_collection(self, name: str) -> SqliteVecCollection:
        row = self._row(name)
        if row is None:
            raise KeyError(f"Collection {name!r} does not exist")
        return SqliteVecCollection(
            name, self._client, embedder=self._embedder,
            dimension=row[0], metric=row[1] or "cosine",
        )

    def delete_collection(self, name: str) -> None:
        if self._row(name) is None:
            raise KeyError(f"Collection {name!r} does not exist")
        self._client.execute(f'DROP TABLE IF EXISTS "{name}_docs"')
        self._client.execute(f'DROP TABLE IF EXISTS "{name}_vec"')
        self._client.execute("DELETE FROM _vd_collections WHERE name=?", (name,))
        self._client.commit()

    def list_collections(self) -> Iterator[str]:
        rows = self._client.execute("SELECT name FROM _vd_collections").fetchall()
        return iter(r[0] for r in rows)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._client.close()
