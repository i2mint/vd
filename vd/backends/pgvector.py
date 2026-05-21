"""
pgvector backend.

pgvector is a PostgreSQL extension that adds a ``vector(N)`` column type and
three distance operators (``<=>``, ``<->``, ``<#>``). This adapter maps the
``vd`` facade onto a running Postgres database with pgvector installed.

**When to use it:** if you already run Postgres, pgvector is the obvious choice
— no second database, full SQL joins against your relational data, battle-tested
backup/replication, and trivial lock-in reversal (``pg_dump``). Choose a
dedicated vector-DB (Qdrant, Milvus …) only if you measure a bottleneck above
~10M vectors or require native hybrid search.

**Mapping to vd:**

- A ``vd`` *collection* maps to one Postgres table
  ``"<name>"(doc_id text primary key, text text, metadata jsonb,
  embedding vector(N))``.
- All ``vd`` collection metadata is in ``_vd_collections(name, dimension,
  metric)`` — the same meta-table pattern used by the DuckDB and sqlite-vec
  adapters.
- Metadata filtering is applied *client-side* via :func:`apply_client_filter`
  (overfetch candidates, then post-filter). This gives the full ``vd`` filter
  language without any SQL injection surface.
- An HNSW index (``CREATE INDEX USING hnsw``) is created best-effort; creation
  failure is silently ignored so the adapter still works without it.

Requires: ``pip install pgvector 'psycopg[binary]'``
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable, Iterator, Optional

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The pgvector backend needs 'pgvector' and 'psycopg'. "
        "Install with: pip install pgvector 'psycopg[binary]'"
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

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: vd metric -> pgvector distance operator.
_DISTANCE_OP = {
    "cosine": "<=>",
    "l2": "<->",
    "dot": "<#>",
}

#: vd metric -> pgvector HNSW operator class.
_HNSW_OPS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "dot": "vector_ip_ops",
}


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _resolve_dsn(
    dsn: Optional[str],
    url: Optional[str],
) -> str:
    """
    Return the Postgres DSN to use, consulting environment variables as fallback.

    Priority: explicit *dsn* argument → explicit *url* argument →
    ``$DATABASE_URL`` → ``$POSTGRES_DSN``.

    Parameters
    ----------
    dsn : str, optional
        A libpq-style connection string or keyword-value string.
    url : str, optional
        A ``postgresql://…`` URL (alias for *dsn*; accepted for familiarity).

    Raises
    ------
    ValueError
        If no DSN can be determined from arguments or environment.
    """
    resolved = dsn or url or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_DSN")
    if not resolved:
        raise ValueError(
            "No Postgres DSN found. Pass dsn=..., url=..., or set "
            "$DATABASE_URL / $POSTGRES_DSN."
        )
    return resolved


def _open_connection(dsn: str) -> "psycopg.Connection":
    """
    Open a psycopg 3 connection, enable the vector extension, and create the
    ``_vd_collections`` meta table.

    Parameters
    ----------
    dsn : str
        A Postgres connection string or URL.

    Returns
    -------
    psycopg.Connection
        A live, autocommit-off connection with the ``vector`` type registered.
    """
    conn = psycopg.connect(dsn)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _vd_collections "
        "(name text PRIMARY KEY, dimension int, metric text)"
    )
    conn.commit()
    return conn


def _quoted(name: str) -> str:
    """Return a double-quoted, escaped table name safe for SQL interpolation."""
    # Escape any embedded double-quotes by doubling them (SQL standard).
    return '"' + name.replace('"', '""') + '"'


# --------------------------------------------------------------------------- #
# PgvectorCollection
# --------------------------------------------------------------------------- #


class PgvectorCollection(AbstractCollection):
    """
    A ``vd`` collection backed by one Postgres table with a ``vector(N)`` column.

    The table is created lazily on the first write, once the vector dimension is
    known. An HNSW index is added best-effort at that point. Metadata filtering
    is applied client-side.

    Parameters
    ----------
    name : str
        Collection (table) name.
    conn : psycopg.Connection
        The shared psycopg 3 connection owned by the client.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    dimension : int, optional
        Vector dimension; inferred on first write if ``None``.
    metric : str
        Distance metric: ``"cosine"``, ``"l2"``, or ``"dot"``.
    """

    def __init__(
        self,
        name: str,
        conn: "psycopg.Connection",
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
        self._tbl = _quoted(name)
        self._created = self._table_exists()

    @property
    def native(self) -> "psycopg.Connection":
        """The raw psycopg 3 connection (escape hatch)."""
        return self._conn

    # ----- internal helpers ------------------------------------------------- #

    def _table_exists(self) -> bool:
        """Return ``True`` if the backing Postgres table already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            [self.name],
        ).fetchone()
        return row is not None

    def _ensure_table(self, dimension: int) -> None:
        """
        Create the backing table and an HNSW index, once the dimension is known.

        This is a no-op if the table already exists. Index creation is
        best-effort: failure is silently ignored so the adapter still works on
        Postgres builds without HNSW support or without ``maintenance_work_mem``
        large enough.

        Parameters
        ----------
        dimension : int
            The vector dimension to encode in ``vector(dimension)``.
        """
        if self._created:
            return
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._tbl} "
            f"(doc_id text PRIMARY KEY, text text, metadata jsonb, "
            f"embedding vector(%s))",
            [dimension],
        )
        try:
            ops = _HNSW_OPS.get(self.metric, "vector_cosine_ops")
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_quoted(self.name + '_hnsw_idx')} "
                f"ON {self._tbl} USING hnsw (embedding {ops})"
            )
        except Exception:
            pass  # HNSW is an optimisation; its absence is not fatal
        self._conn.commit()
        self._created = True

    # ----- raw primitives --------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._ensure_table(self.dimension)
        self._conn.execute(
            f"INSERT INTO {self._tbl}(doc_id, text, metadata, embedding) "
            f"VALUES (%s, %s, %s, %s) "
            f"ON CONFLICT (doc_id) DO UPDATE SET "
            f"text = EXCLUDED.text, "
            f"metadata = EXCLUDED.metadata, "
            f"embedding = EXCLUDED.embedding",
            [
                doc.id,
                doc.text,
                json.dumps(doc.metadata or {}),
                doc.vector,
            ],
        )
        self._conn.commit()

    def _write_many(self, docs: list[Document]) -> None:
        """Bulk upsert using ``executemany`` for better throughput."""
        if not docs:
            return
        self._ensure_table(self.dimension)
        self._conn.executemany(
            f"INSERT INTO {self._tbl}(doc_id, text, metadata, embedding) "
            f"VALUES (%s, %s, %s, %s) "
            f"ON CONFLICT (doc_id) DO UPDATE SET "
            f"text = EXCLUDED.text, "
            f"metadata = EXCLUDED.metadata, "
            f"embedding = EXCLUDED.embedding",
            [
                [d.id, d.text, json.dumps(d.metadata or {}), d.vector]
                for d in docs
            ],
        )
        self._conn.commit()

    def _read(self, key: str) -> Document:
        if not self._created:
            raise KeyError(key)
        row = self._conn.execute(
            f"SELECT text, metadata, embedding FROM {self._tbl} WHERE doc_id = %s",
            [key],
        ).fetchone()
        if row is None:
            raise KeyError(key)
        text, metadata, embedding = row
        return Document(
            id=key,
            text=text or "",
            vector=list(embedding) if embedding is not None else None,
            metadata=metadata if isinstance(metadata, dict) else json.loads(metadata or "{}"),
        )

    def _drop(self, key: str) -> None:
        if not self._created:
            raise KeyError(key)
        exists = self._conn.execute(
            f"SELECT 1 FROM {self._tbl} WHERE doc_id = %s", [key]
        ).fetchone()
        if exists is None:
            raise KeyError(key)
        self._conn.execute(f"DELETE FROM {self._tbl} WHERE doc_id = %s", [key])
        self._conn.commit()

    def _keys(self) -> Iterator[str]:
        if not self._created:
            return iter(())
        rows = self._conn.execute(f"SELECT doc_id FROM {self._tbl}").fetchall()
        return iter(r[0] for r in rows)

    def _count(self) -> int:
        if not self._created:
            return 0
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._tbl}").fetchone()
        return row[0] if row else 0

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
        op = _DISTANCE_OP.get(self.metric, "<=>")
        fetch = overfetch_limit(limit, filter)
        rows = self._conn.execute(
            f"SELECT doc_id, text, metadata, "
            f"embedding {op} %s AS dist "
            f"FROM {self._tbl} "
            f"ORDER BY dist "
            f"LIMIT %s",
            [vector, fetch],
        ).fetchall()
        results = [
            {
                "id": doc_id,
                "text": text or "",
                "score": score_from_distance(dist, self.metric),
                "metadata": (
                    metadata if isinstance(metadata, dict)
                    else json.loads(metadata or "{}")
                ),
            }
            for doc_id, text, metadata, dist in rows
        ]
        return apply_client_filter(results, filter, limit=limit)


# --------------------------------------------------------------------------- #
# PgvectorClient
# --------------------------------------------------------------------------- #


@register_backend("pgvector")
class PgvectorClient(AbstractClient):
    """
    pgvector client.

    Connects to a running PostgreSQL instance with the pgvector extension
    installed. Each ``vd`` collection becomes one Postgres table; collection
    metadata (name, dimension, metric) is persisted in a ``_vd_collections``
    table in the same database.

    Parameters
    ----------
    dsn : str, optional
        A libpq connection string (e.g.
        ``"postgresql://user:pw@localhost:5432/mydb"``) or keyword-value form
        (e.g. ``"host=localhost dbname=mydb"``).
    url : str, optional
        Alias for *dsn*; both spellings are accepted.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder passed to every
        collection.

    Notes
    -----
    If neither *dsn* nor *url* is given the adapter reads ``$DATABASE_URL``
    first, then ``$POSTGRES_DSN``. If none of the four sources is available,
    a :class:`ValueError` is raised at construction time.

    The psycopg 3 ``autocommit`` flag is left at its default (``False``). Each
    write method commits explicitly. For higher-throughput batch loading, obtain
    the raw connection via ``client.client`` (i.e. ``PgvectorClient.native``)
    and manage transactions manually.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dsn: Optional[str] = None,
        url: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        resolved_dsn = _resolve_dsn(dsn, url)
        self._client = _open_connection(resolved_dsn)

    # ----- helpers ---------------------------------------------------------- #

    def _meta_row(self, name: str):
        """Return the ``(dimension, metric)`` row for *name*, or ``None``."""
        return self._client.execute(
            "SELECT dimension, metric FROM _vd_collections WHERE name = %s",
            [name],
        ).fetchone()

    # ----- AbstractClient primitives --------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> PgvectorCollection:
        """
        Create a new ``vd`` collection backed by a Postgres table.

        Parameters
        ----------
        name : str
            Collection name (becomes the SQL table name).
        dimension : int, optional
            Vector dimension. May be ``None`` — the table is created lazily on
            first write once the dimension is known from the first vector.
        metric : str
            Distance metric: ``"cosine"``, ``"l2"``, or ``"dot"``.
        **index_config
            Ignored for this adapter (HNSW tuning goes through ``maintenance_work_mem``
            and Postgres ``ALTER INDEX`` rather than creation parameters).

        Raises
        ------
        ValueError
            If a collection (and therefore a row in ``_vd_collections``) with
            that name already exists.
        """
        if self._meta_row(name) is not None:
            raise ValueError(f"Collection {name!r} already exists")
        self._client.execute(
            "INSERT INTO _vd_collections(name, dimension, metric) VALUES (%s, %s, %s)",
            [name, dimension, metric],
        )
        self._client.commit()
        return PgvectorCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    def get_collection(self, name: str) -> PgvectorCollection:
        """
        Return an existing collection.

        Parameters
        ----------
        name : str
            Collection name.

        Raises
        ------
        KeyError
            If no collection with that name exists.
        """
        row = self._meta_row(name)
        if row is None:
            raise KeyError(f"Collection {name!r} does not exist")
        dimension, metric = row
        return PgvectorCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric or "cosine",
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection and its backing table.

        Parameters
        ----------
        name : str
            Collection name.

        Raises
        ------
        KeyError
            If no collection with that name exists.
        """
        if self._meta_row(name) is None:
            raise KeyError(f"Collection {name!r} does not exist")
        self._client.execute(f"DROP TABLE IF EXISTS {_quoted(name)}")
        self._client.execute(
            "DELETE FROM _vd_collections WHERE name = %s", [name]
        )
        self._client.commit()

    def list_collections(self) -> Iterator[str]:
        """Iterate collection names registered in ``_vd_collections``."""
        rows = self._client.execute(
            "SELECT name FROM _vd_collections ORDER BY name"
        ).fetchall()
        return iter(r[0] for r in rows)

    def close(self) -> None:
        """Close the underlying psycopg 3 connection."""
        self._client.close()
