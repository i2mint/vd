"""
In-memory backend — the reference adapter.

Stores documents in a plain ``dict`` and answers queries with brute-force
similarity in pure Python. Always available (no third-party dependency), and
the canonical example of how a backend implements the small set of raw
primitives that :class:`~vd.base.AbstractCollection` builds on. Use it for
tests, notebooks, and corpora small enough that an ANN index is overkill.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, Optional

from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.filters import matches_filter
from vd.util import cosine_similarity, euclidean_distance, register_backend


def _similarity(a: Vector, b: Vector, metric: str) -> float:
    """Higher-is-better similarity of two vectors under ``metric``."""
    if metric == "dot":
        return sum(x * y for x, y in zip(a, b))
    if metric == "l2":
        return 1.0 / (1.0 + euclidean_distance(a, b))
    return cosine_similarity(a, b)


class MemoryCollection(AbstractCollection):
    """A collection backed by an in-process ``dict[str, Document]``."""

    def __init__(
        self,
        name: str,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._docs: dict[str, Document] = {}

    @property
    def native(self) -> dict[str, Document]:
        """The raw ``dict[str, Document]`` backing this collection (escape hatch)."""
        return self._docs

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._docs[doc.id] = doc

    def _read(self, key: str) -> Document:
        return self._docs[key]  # KeyError propagates naturally

    def _drop(self, key: str) -> None:
        del self._docs[key]  # KeyError propagates naturally

    def _keys(self) -> Iterator[str]:
        return iter(self._docs)

    def _count(self) -> int:
        return len(self._docs)

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        scored: list[SearchResult] = []
        for doc in self._docs.values():
            if doc.vector is None:
                continue
            if filter and not matches_filter(doc.metadata, filter):
                continue
            scored.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": _similarity(vector, doc.vector, self.metric),
                    "metadata": dict(doc.metadata),
                }
            )
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]


@register_backend("memory")
class MemoryClient(AbstractClient):
    """
    In-memory vector database client.

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')
    >>> col = client.create_collection('docs')
    >>> col['a'] = vd.Document(id='a', text='cat', vector=[1.0, 0.0])
    >>> col['b'] = vd.Document(id='b', text='dog', vector=[0.0, 1.0])
    >>> [r['id'] for r in col.search([0.9, 0.1], limit=1)]
    ['a']
    """

    def __init__(self, *, embedder: Optional[Callable[[str], Vector]] = None, **config):
        super().__init__(embedder=embedder, **config)
        self._collections: dict[str, MemoryCollection] = {}

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> MemoryCollection:
        if name in self._collections:
            raise ValueError(f"Collection {name!r} already exists")
        collection = MemoryCollection(
            name, embedder=self._embedder, dimension=dimension, metric=metric
        )
        self._collections[name] = collection
        return collection

    def get_collection(self, name: str) -> MemoryCollection:
        if name not in self._collections:
            raise KeyError(f"Collection {name!r} does not exist")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        if name not in self._collections:
            raise KeyError(f"Collection {name!r} does not exist")
        del self._collections[name]

    def list_collections(self) -> Iterator[str]:
        return iter(self._collections)
