"""
FAISS backend.

FAISS is a *library*, not a database — the reference implementation of the ANN
algorithms. This adapter wraps a flat (exact) FAISS index in an ``IndexIDMap2``
so it accepts incremental add/remove, and keeps document text + metadata in an
in-process sidecar (FAISS itself stores only vectors). It is the right choice
for an exact recall@k baseline or an in-memory benchmark.

FAISS has no metadata filtering, so ``vd`` applies the canonical filter
client-side (over-fetching candidates first). Persistence is explicit: pass
``path=`` to :func:`vd.connect` and call :meth:`FaissClient.close` (or the
context manager) to flush index + sidecar to disk.

Requires: ``pip install faiss-cpu``  (or ``faiss-gpu`` via conda)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

try:
    import faiss
    import numpy as np
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The faiss backend needs 'faiss-cpu' and 'numpy'. "
        "Install with: pip install faiss-cpu numpy"
    ) from e

from vd.backends._helpers import apply_client_filter, overfetch_limit
from vd.base import (
    AbstractClient,
    AbstractCollection,
    Document,
    Filter,
    SearchResult,
    Vector,
)
from vd.util import register_backend


def _new_index(dimension: int, metric: str):
    """Build a flat (exact) FAISS index wrapped for incremental id-mapped use."""
    if metric == "l2":
        base = faiss.IndexFlatL2(dimension)
    else:  # cosine and dot both use inner product (cosine = IP on normalized)
        base = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap2(base)


class FaissCollection(AbstractCollection):
    """A collection backed by an in-memory FAISS flat index + a sidecar store."""

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
        self._index = _new_index(dimension, metric) if dimension else None
        # Sidecar: FAISS stores only vectors, so text/metadata live here.
        self._docs: dict[str, Document] = {}
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int = 0

    @property
    def native(self):
        """The raw ``faiss`` index (escape hatch)."""
        return self._index

    # ----- internal ------------------------------------------------------- #

    def _prepare(self, vector: Vector) -> "np.ndarray":
        """Return a 2-D float32 array, L2-normalized when the metric is cosine."""
        arr = np.asarray([vector], dtype="float32")
        if self.metric == "cosine":
            faiss.normalize_L2(arr)
        return arr

    def _ensure_index(self) -> None:
        if self._index is None:
            self._index = _new_index(self.dimension, self.metric)

    # ----- raw primitives ------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        self._ensure_index()
        if doc.id in self._id_to_int:  # upsert: drop the stale vector first
            self._remove_int(self._id_to_int[doc.id])
        int_id = self._next_int
        self._next_int += 1
        self._id_to_int[doc.id] = int_id
        self._int_to_id[int_id] = doc.id
        self._docs[doc.id] = doc
        self._index.add_with_ids(
            self._prepare(doc.vector), np.asarray([int_id], dtype="int64")
        )

    def _remove_int(self, int_id: int) -> None:
        self._index.remove_ids(np.asarray([int_id], dtype="int64"))
        doc_id = self._int_to_id.pop(int_id, None)
        if doc_id is not None:
            self._id_to_int.pop(doc_id, None)

    def _read(self, key: str) -> Document:
        if key not in self._docs:
            raise KeyError(key)
        return self._docs[key]

    def _drop(self, key: str) -> None:
        if key not in self._id_to_int:
            raise KeyError(key)
        self._remove_int(self._id_to_int[key])
        self._docs.pop(key, None)

    def _keys(self) -> Iterator[str]:
        return iter(list(self._docs))

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
        if self._index is None or self._index.ntotal == 0:
            return []
        fetch = min(overfetch_limit(limit, filter), self._index.ntotal)
        scores, int_ids = self._index.search(self._prepare(vector), fetch)
        results: list[SearchResult] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:
                continue
            doc_id = self._int_to_id.get(int(int_id))
            if doc_id is None:
                continue
            doc = self._docs[doc_id]
            # IndexFlatIP -> higher is better; IndexFlatL2 -> lower is better.
            value = float(score)
            results.append(
                {
                    "id": doc_id,
                    "text": doc.text,
                    "score": value if self.metric != "l2" else 1.0 / (1.0 + value),
                    "metadata": dict(doc.metadata),
                }
            )
        return apply_client_filter(results, filter, limit=limit)

    # ----- persistence ---------------------------------------------------- #

    def _save(self, directory: Path) -> None:
        """Flush this collection's index and sidecar to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / f"{self.name}.faiss"))
        sidecar = {
            "metric": self.metric,
            "dimension": self.dimension,
            "next_int": self._next_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            "docs": {
                k: {"id": d.id, "text": d.text, "metadata": d.metadata}
                for k, d in self._docs.items()
            },
        }
        (directory / f"{self.name}.meta.json").write_text(json.dumps(sidecar))

    @classmethod
    def _load(
        cls,
        name: str,
        directory: Path,
        *,
        embedder: Optional[Callable[[str], Vector]],
    ) -> "FaissCollection":
        """Reconstruct a collection previously written by :meth:`_save`."""
        sidecar = json.loads((directory / f"{name}.meta.json").read_text())
        col = cls(
            name,
            embedder=embedder,
            dimension=sidecar["dimension"],
            metric=sidecar["metric"],
        )
        col._next_int = sidecar["next_int"]
        col._int_to_id = {int(k): v for k, v in sidecar["int_to_id"].items()}
        col._id_to_int = {v: k for k, v in col._int_to_id.items()}
        index_path = directory / f"{name}.faiss"
        if index_path.exists():
            col._index = faiss.read_index(str(index_path))
        for doc_id, d in sidecar["docs"].items():
            vec = None
            int_id = col._id_to_int.get(doc_id)
            if col._index is not None and int_id is not None:
                vec = list(col._index.reconstruct(int_id))
            col._docs[doc_id] = Document(
                id=d["id"], text=d["text"], vector=vec, metadata=d["metadata"]
            )
        return col


@register_backend("faiss")
class FaissClient(AbstractClient):
    """
    FAISS client.

    Parameters
    ----------
    path : str, optional
        A directory to persist collections in. Without it, everything is
        in-memory and lost on exit. Persistence is flushed on :meth:`close`
        (or context-manager exit), not on every write.
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
        self._dir = Path(path) if path else None
        self._collections: dict[str, FaissCollection] = {}
        if self._dir and self._dir.exists():
            for meta in self._dir.glob("*.meta.json"):
                name = meta.name[: -len(".meta.json")]
                self._collections[name] = FaissCollection._load(
                    name, self._dir, embedder=self._embedder
                )

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> FaissCollection:
        if name in self._collections:
            raise ValueError(f"Collection {name!r} already exists")
        col = FaissCollection(
            name, embedder=self._embedder, dimension=dimension, metric=metric
        )
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> FaissCollection:
        if name not in self._collections:
            raise KeyError(f"Collection {name!r} does not exist")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        if name not in self._collections:
            raise KeyError(f"Collection {name!r} does not exist")
        del self._collections[name]
        if self._dir:
            for suffix in (".faiss", ".meta.json"):
                p = self._dir / f"{name}{suffix}"
                if p.exists():
                    p.unlink()

    def list_collections(self) -> Iterator[str]:
        return iter(self._collections)

    def close(self) -> None:
        """Flush every collection to disk (if a ``path`` was given)."""
        if self._dir is not None:
            for col in self._collections.values():
                col._save(self._dir)
