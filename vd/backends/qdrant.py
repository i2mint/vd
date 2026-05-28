"""
Qdrant backend.

Qdrant is the cleanest mapping of a real vector database onto the ``vd``
facade: one Rust binary that runs embedded (in-memory or a local path), as a
Docker server, or as a managed cloud cluster — all behind the same Python SDK.
It has rich native payload filtering, so this adapter is the one that performs
a *real* filter translation: the canonical ``vd`` filter AST is compiled to a
``qdrant_client.models.Filter`` (no client-side post-filtering).

Document ids are arbitrary strings; Qdrant point ids must be UUIDs or
unsigned ints, so each id is mapped to a deterministic UUID5 and the original
is kept in the point payload.

Requires: ``pip install qdrant-client``
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from qdrant_client import QdrantClient, models
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The qdrant backend needs the 'qdrant-client' package. "
        "Install it with: pip install qdrant-client"
    ) from e

from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.filters import SUPPORTED_FILTER_OPERATORS
from vd.util import register_backend

#: vd metric -> Qdrant distance.
_DISTANCE = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "l2": models.Distance.EUCLID,
}

#: Payload keys vd reserves; user metadata lives nested under ``metadata``.
_ID_KEY = "_vd_id"
_TEXT_KEY = "_vd_text"


def _point_id(doc_id: str) -> str:
    """Map an arbitrary document id to a deterministic Qdrant UUID point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def _to_qdrant_filter(ast: Optional[Filter]) -> Optional["models.Filter"]:
    """
    Compile a canonical ``vd`` filter AST to a ``qdrant_client.models.Filter``.

    User metadata is stored nested under the ``metadata`` payload key, so every
    field reference is translated to a ``metadata.<field>`` path.
    """
    if not ast:
        return None
    must: list = []
    should: list = []
    must_not: list = []

    for key, cond in ast.items():
        if key == "$and":
            must += [_to_qdrant_filter(sub) for sub in cond]
        elif key == "$or":
            should += [_to_qdrant_filter(sub) for sub in cond]
        elif key == "$not":
            must_not.append(_to_qdrant_filter(cond))
        else:
            qkey = f"metadata.{key}"
            if not isinstance(cond, dict):
                must.append(
                    models.FieldCondition(key=qkey, match=models.MatchValue(value=cond))
                )
            else:
                _compile_field(qkey, cond, must, must_not)

    return models.Filter(
        must=must or None, should=should or None, must_not=must_not or None
    )


def _compile_field(qkey: str, cond: dict, must: list, must_not: list) -> None:
    """Translate one ``{field: {op: operand}}`` clause into Qdrant conditions."""
    range_kw: dict[str, Any] = {}
    for op, operand in cond.items():
        if op == "$eq":
            must.append(
                models.FieldCondition(key=qkey, match=models.MatchValue(value=operand))
            )
        elif op == "$ne":
            must_not.append(
                models.FieldCondition(key=qkey, match=models.MatchValue(value=operand))
            )
        elif op == "$in":
            must.append(
                models.FieldCondition(
                    key=qkey, match=models.MatchAny(any=list(operand))
                )
            )
        elif op == "$nin":
            must_not.append(
                models.FieldCondition(
                    key=qkey, match=models.MatchAny(any=list(operand))
                )
            )
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            range_kw[op[1:]] = operand
        elif op == "$exists":
            empty = models.IsEmptyCondition(is_empty=models.PayloadField(key=qkey))
            (must_not if operand else must).append(empty)
    if range_kw:
        must.append(models.FieldCondition(key=qkey, range=models.Range(**range_kw)))


class QdrantCollection(AbstractCollection):
    """A collection backed by one Qdrant collection. Native payload filtering."""

    # Qdrant covers the entire canonical filter language natively.
    supported_filter_operators = SUPPORTED_FILTER_OPERATORS

    def __init__(
        self,
        name: str,
        client: QdrantClient,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._client = client
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric

    @property
    def native(self) -> QdrantClient:
        """The raw ``QdrantClient`` (escape hatch)."""
        return self._client

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection lazily, once the dimension is known."""
        if not self._client.collection_exists(self.name):
            self._client.create_collection(
                collection_name=self.name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=_DISTANCE.get(self.metric, models.Distance.COSINE),
                ),
            )

    @staticmethod
    def _payload(doc: Document) -> dict:
        return {_ID_KEY: doc.id, _TEXT_KEY: doc.text, "metadata": doc.metadata or {}}

    def _point(self, doc: Document) -> "models.PointStruct":
        return models.PointStruct(
            id=_point_id(doc.id), vector=doc.vector, payload=self._payload(doc)
        )

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._ensure_collection()
        self._client.upsert(self.name, points=[self._point(doc)])

    def _write_many(self, docs: list[Document]) -> None:
        self._ensure_collection()
        self._client.upsert(self.name, points=[self._point(d) for d in docs])

    def _read(self, key: str) -> Document:
        if not self._client.collection_exists(self.name):
            raise KeyError(key)
        points = self._client.retrieve(
            self.name, ids=[_point_id(key)], with_payload=True, with_vectors=True
        )
        if not points:
            raise KeyError(key)
        return self._to_document(points[0])

    def _drop(self, key: str) -> None:
        if not self._client.collection_exists(self.name) or not self._client.retrieve(
            self.name, ids=[_point_id(key)]
        ):
            raise KeyError(key)
        self._client.delete(
            self.name, points_selector=models.PointIdsList(points=[_point_id(key)])
        )

    def _keys(self) -> Iterator[str]:
        if not self._client.collection_exists(self.name):
            return iter(())
        ids: list[str] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.name, limit=256, offset=offset, with_payload=[_ID_KEY]
            )
            ids += [p.payload[_ID_KEY] for p in points]
            if offset is None:
                break
        return iter(ids)

    def _count(self) -> int:
        if not self._client.collection_exists(self.name):
            return 0
        return self._client.count(self.name).count

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        if not self._client.collection_exists(self.name):
            return []
        response = self._client.query_points(
            self.name,
            query=vector,
            limit=limit,
            query_filter=_to_qdrant_filter(filter),
            with_payload=True,
            **kwargs,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            score = point.score
            # Qdrant `point.score` per metric (see vd.base "Score semantics"):
            #   - cosine: cosine similarity in [-1, 1]  → matches vd canonical
            #   - dot:    raw inner product              → matches vd canonical
            #   - euclid: a *distance* value (lower-is-better); Qdrant's
            #     own sort orders ascending in that case. The existing
            #     transform 1/(1+d) matches vd's canonical l2 score directly
            #     (no un-negation), so leave it as-is. If a future Qdrant
            #     client version switches Euclid to higher-is-better, this
            #     branch must be revisited.
            results.append(
                {
                    "id": payload.get(_ID_KEY, str(point.id)),
                    "text": payload.get(_TEXT_KEY, ""),
                    "score": 1.0 / (1.0 + score) if self.metric == "l2" else score,
                    "metadata": payload.get("metadata", {}),
                }
            )
        return results

    @staticmethod
    def _to_document(point) -> Document:
        payload = point.payload or {}
        vector = point.vector
        return Document(
            id=payload.get(_ID_KEY, str(point.id)),
            text=payload.get(_TEXT_KEY, ""),
            vector=list(vector) if vector is not None else None,
            metadata=payload.get("metadata", {}),
        )


@register_backend("qdrant")
class QdrantClientAdapter(AbstractClient):
    """
    Qdrant client.

    Parameters
    ----------
    path : str, optional
        A local directory for embedded persistent mode.
    url : str, optional
        URL of a Qdrant server / cloud cluster. With ``api_key`` for cloud.
    api_key : str, optional
        API key for Qdrant Cloud.
    location : str, optional
        Passed straight to ``QdrantClient`` (e.g. ``":memory:"``). The default,
        when neither ``path`` nor ``url`` is given, is ``":memory:"``.
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        location: Optional[str] = None,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        if url is not None:
            self._client = QdrantClient(url=url, api_key=api_key, **config)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(location=location or ":memory:")
        self._metrics: dict[str, str] = {}

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> QdrantCollection:
        if self._client.collection_exists(name) or name in self._metrics:
            raise ValueError(f"Collection {name!r} already exists")
        self._metrics[name] = metric
        collection = QdrantCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )
        if dimension is not None:  # eager create when the dimension is known
            collection._ensure_collection()
        return collection

    def get_collection(self, name: str) -> QdrantCollection:
        if not self._client.collection_exists(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        return QdrantCollection(
            name,
            self._client,
            embedder=self._embedder,
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        if not self._client.collection_exists(name) and name not in self._metrics:
            raise KeyError(f"Collection {name!r} does not exist")
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        names = {c.name for c in self._client.get_collections().collections}
        names |= set(self._metrics)
        return iter(sorted(names))

    def close(self) -> None:
        """Close the underlying Qdrant client."""
        self._client.close()
