"""
ChromaDB backend.

Chroma is the fastest path to a working RAG demo: ``pip install chromadb``,
no server, optional on-disk persistence. This adapter supports the three
Chroma deployment modes — ephemeral in-process, persistent on disk, and a
remote Chroma server — selected by the arguments to ``vd.connect('chroma', ...)``.

Requires: ``pip install chromadb``
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional

try:
    import chromadb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The chroma backend needs the 'chromadb' package. "
        "Install it with: pip install chromadb"
    ) from e

from vd.backends._helpers import score_from_distance
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

#: Operators Chroma's ``where`` clause supports natively. Chroma has no
#: ``$not`` and no ``$exists`` — those raise UnsupportedFilterError so the
#: caller can simplify the filter or drop to ``collection.native``.
CHROMA_FILTER_OPERATORS = SUPPORTED_FILTER_OPERATORS - {"$not", "$exists"}

#: vd metric name -> Chroma HNSW space name.
_SPACE = {"cosine": "cosine", "dot": "ip", "l2": "l2"}


def _to_chroma_where(filter: Optional[Filter]) -> Optional[dict]:
    """
    Translate a canonical ``vd`` filter to a Chroma ``where`` clause.

    Chroma's ``where`` is itself MongoDB-ish, so the translation is mostly
    structural: bare values become explicit ``$eq``, and any dict with more
    than one condition is wrapped in an explicit ``$and`` (Chroma requires it).

    Examples
    --------
    >>> _to_chroma_where({'year': 2024})
    {'year': {'$eq': 2024}}
    >>> _to_chroma_where({'a': 1, 'b': 2})
    {'$and': [{'a': {'$eq': 1}}, {'b': {'$eq': 2}}]}
    """
    if not filter:
        return None
    clauses: list[dict] = []
    for key, cond in filter.items():
        if key in ("$and", "$or"):
            clauses.append({key: [_to_chroma_where(f) for f in cond]})
        elif isinstance(cond, dict):
            # One {field: {op: operand}} clause per operator.
            for op, operand in cond.items():
                clauses.append({key: {op: operand}})
        else:
            clauses.append({key: {"$eq": cond}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaCollection(AbstractCollection):
    """A :class:`~vd.base.Collection` wrapping a ``chromadb`` collection."""

    supported_filter_operators = CHROMA_FILTER_OPERATORS

    def __init__(
        self,
        chroma_collection,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        metric: str = "cosine",
    ):
        self.name = chroma_collection.name
        self._collection = chroma_collection
        self._embedder = embedder
        self.dimension = None  # Chroma infers it from the first vector
        self.metric = metric

    @property
    def native(self):
        """The raw ``chromadb`` collection (escape hatch)."""
        return self._collection

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._collection.upsert(
            ids=[doc.id],
            documents=[doc.text],
            embeddings=[doc.vector],
            metadatas=[doc.metadata or None],
        )

    def _write_many(self, docs: list[Document]) -> None:
        self._collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            embeddings=[d.vector for d in docs],
            metadatas=[d.metadata or None for d in docs],
        )

    def _read(self, key: str) -> Document:
        got = self._collection.get(
            ids=[key], include=["documents", "embeddings", "metadatas"]
        )
        if not got["ids"]:
            raise KeyError(key)
        embeddings = got.get("embeddings")
        return Document(
            id=got["ids"][0],
            text=(got["documents"] or [""])[0] or "",
            vector=list(embeddings[0]) if embeddings is not None and len(embeddings) else None,
            metadata=(got["metadatas"] or [{}])[0] or {},
        )

    def _drop(self, key: str) -> None:
        if not self._collection.get(ids=[key])["ids"]:
            raise KeyError(key)
        self._collection.delete(ids=[key])

    def _keys(self) -> Iterator[str]:
        return iter(self._collection.get(include=[])["ids"])

    def _count(self) -> int:
        return self._collection.count()

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=limit,
            where=_to_chroma_where(filter),
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        if not result["ids"] or not result["ids"][0]:
            return []
        ids, docs = result["ids"][0], result["documents"][0]
        metas, dists = result["metadatas"][0], result["distances"][0]
        return [
            {
                "id": ids[i],
                "text": docs[i] if docs else "",
                "score": score_from_distance(dists[i], self.metric),
                "metadata": metas[i] or {} if metas else {},
            }
            for i in range(len(ids))
        ]


@register_backend("chroma")
class ChromaClient(AbstractClient):
    """
    ChromaDB client.

    Parameters
    ----------
    persist_directory : str, optional
        Persist on disk at this path (``chromadb.PersistentClient``). If
        omitted and no ``host`` is given, an ephemeral in-process client is
        used (data lost on exit).
    host, port : str, int, optional
        Connect to a running Chroma server (``chromadb.HttpClient``).
    embedder : callable, optional
        Optional ``text -> vector`` convenience embedder.

    Examples
    --------
    >>> import vd                                                  # doctest: +SKIP
    >>> client = vd.connect('chroma', persist_directory='./db')    # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        if host is not None:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        # Remember each collection's metric so score conversion is correct.
        self._metrics: dict[str, str] = {}

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> ChromaCollection:
        space = _SPACE.get(metric, "cosine")
        try:
            chroma_collection = self._client.create_collection(
                name=name, configuration={"hnsw": {"space": space}}, **index_config
            )
        except TypeError:
            # Older chromadb: space goes in metadata, not configuration.
            chroma_collection = self._client.create_collection(
                name=name, metadata={"hnsw:space": space}, **index_config
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                raise ValueError(f"Collection {name!r} already exists") from e
            raise
        self._metrics[name] = metric
        return ChromaCollection(
            chroma_collection, embedder=self._embedder, metric=metric
        )

    def get_collection(self, name: str) -> ChromaCollection:
        try:
            chroma_collection = self._client.get_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e
        return ChromaCollection(
            chroma_collection,
            embedder=self._embedder,
            metric=self._metrics.get(name, "cosine"),
        )

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name=name)
        except Exception as e:
            raise KeyError(f"Collection {name!r} does not exist") from e
        self._metrics.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        return (c.name for c in self._client.list_collections())
