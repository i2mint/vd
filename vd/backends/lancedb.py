"""
LanceDB backend.

LanceDB is an embedded, multimodal vector database on the Lance columnar
format — embedded like Chroma, but with cheap schema evolution, versioning,
and a storage layout that is portable (copy the ``.lance`` directory) and
S3-native. Each ``vd`` collection is one Lance table with columns
``(id, text, vector, metadata)``; metadata travels as a JSON string and the
canonical filter is applied client-side (over-fetching candidates first).

Requires: ``pip install lancedb``
"""

from __future__ import annotations

import json
import tempfile
from typing import Callable, Iterable, Iterator, Optional

try:
    import lancedb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The lancedb backend needs the 'lancedb' package. "
        "Install it with: pip install lancedb"
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

#: vd metric -> LanceDB distance-type name.
_METRIC = {"cosine": "cosine", "l2": "l2", "dot": "dot"}


def _sql_str(value: str) -> str:
    """Quote a string for a LanceDB SQL ``where`` predicate."""
    return "'" + value.replace("'", "''") + "'"


def _table_names(db) -> list[str]:
    """Return the table names of a LanceDB connection, across client versions."""
    listed = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    # Newer clients return a response object with a ``.tables`` attribute.
    return list(getattr(listed, "tables", listed))


class LanceDBCollection(AbstractCollection):
    """A collection backed by one Lance table."""

    def __init__(
        self,
        name: str,
        db,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._db = db
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric

    @property
    def _table(self):
        """Open the underlying Lance table, or ``None`` if not created yet."""
        try:
            return self._db.open_table(self.name)
        except Exception:
            return None

    @property
    def native(self):
        """The raw Lance table (escape hatch), or ``None`` before the first write."""
        return self._table

    @staticmethod
    def _row(doc: Document) -> dict:
        return {
            "id": doc.id,
            "text": doc.text,
            "vector": doc.vector,
            "metadata": json.dumps(doc.metadata or {}),
        }

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._write_many([doc])

    def _write_many(self, docs: list[Document]) -> None:
        rows = [self._row(d) for d in docs]
        table = self._table
        if table is None:
            self._db.create_table(self.name, data=rows)
        else:
            (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )

    def _read(self, key: str) -> Document:
        table = self._table
        if table is None:
            raise KeyError(key)
        hits = table.search().where(f"id = {_sql_str(key)}").limit(1).to_list()
        if not hits:
            raise KeyError(key)
        return self._to_document(hits[0])

    def _drop(self, key: str) -> None:
        table = self._table
        if table is None:
            raise KeyError(key)
        if not table.search().where(f"id = {_sql_str(key)}").limit(1).to_list():
            raise KeyError(key)
        table.delete(f"id = {_sql_str(key)}")

    def _keys(self) -> Iterator[str]:
        table = self._table
        if table is None:
            return iter(())
        return iter(table.to_arrow().column("id").to_pylist())

    def _count(self) -> int:
        table = self._table
        return table.count_rows() if table is not None else 0

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        table = self._table
        if table is None:
            return []
        hits = (
            table.search(vector)
            .metric(_METRIC.get(self.metric, "cosine"))
            .limit(overfetch_limit(limit, filter))
            .to_list()
        )
        results = []
        for hit in hits:
            doc = self._to_document(hit)
            results.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": score_from_distance(hit.get("_distance", 0.0), self.metric),
                    "metadata": doc.metadata,
                }
            )
        return apply_client_filter(results, filter, limit=limit)

    @staticmethod
    def _to_document(row: dict) -> Document:
        vector = row.get("vector")
        return Document(
            id=row["id"],
            text=row.get("text") or "",
            vector=list(vector) if vector is not None else None,
            metadata=json.loads(row.get("metadata") or "{}"),
        )


@register_backend("lancedb")
class LanceDBClient(AbstractClient):
    """
    LanceDB client.

    Parameters
    ----------
    path : str, optional
        Directory for the Lance database (or an ``s3://`` URI). If omitted, a
        temporary directory is used and the data is not persisted.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        path: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        self._path = path or tempfile.mkdtemp(prefix="vd_lancedb_")
        self._client = lancedb.connect(self._path)
        self._metrics: dict[str, str] = {}

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> LanceDBCollection:
        if name in set(_table_names(self._client)) or name in self._metrics:
            raise ValueError(f"Collection {name!r} already exists")
        self._metrics[name] = metric
        return LanceDBCollection(
            name, self._client, embedder=self._embedder,
            dimension=dimension, metric=metric,
        )

    def get_collection(self, name: str) -> LanceDBCollection:
        # A collection exists either as a created Lance table or as one that
        # was create_collection'd but not yet written to.
        if name not in set(_table_names(self._client)) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        return LanceDBCollection(
            name, self._client, embedder=self._embedder,
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        if name not in set(_table_names(self._client)) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        if name in set(_table_names(self._client)):
            self._client.drop_table(name)
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        names = set(_table_names(self._client)) | set(self._metrics)
        return iter(sorted(names))
