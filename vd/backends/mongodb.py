"""
MongoDB Atlas Vector Search backend for ``vd``.

**What it is.**  MongoDB Atlas Vector Search exposes a ``$vectorSearch``
aggregation-pipeline stage that performs approximate nearest-neighbour search
directly inside the Atlas managed cluster (or, since 2025, against the
self-managed Community Edition in public preview).  Documents are ordinary
MongoDB documents stored in a collection; one field holds the dense embedding
vector and the rest hold the text payload and user-defined metadata.  The
driver used here is ``pymongo`` 4.x (the official sync Python client).

**When to use it.**  Reach for this adapter when:

- You already run MongoDB Atlas and want vector search without adding a second
  database service.
- Your workload benefits from running aggregation pipelines that mix
  ``$vectorSearch`` with other MongoDB stages (``$match``, ``$lookup``,
  ``$group``, etc.) in a single round-trip.
- The M0 permanent free tier (512 MB, one vector index) is enough for a demo
  or prototype.

**How this adapter maps onto Atlas.**

- A ``vd`` *client* → one Atlas database (default ``"vd"``).
- A ``vd`` *collection* → one MongoDB collection inside that database.
- A ``vd`` *document* → one MongoDB document with the shape
  ``{"_id": id, "text": text, "embedding": vector, "metadata": {...}}``.
- :meth:`MongoDBCollection.search` issues a ``$vectorSearch`` aggregation
  pipeline.  The score returned by ``$meta: "vectorSearchScore"`` is
  *higher-is-better* and is used directly as the ``vd`` result ``score``.

**The Atlas vector search index is created automatically.** On the first
:meth:`~MongoDBCollection.search`, this adapter creates the vector search
index — on the ``"embedding"`` field, with the collection's dimension and
metric — via ``create_search_index``, then blocks until Atlas reports it
queryable. This makes the backend behave like every other ``vd`` adapter
(which all create their own index). Index creation requires an Atlas-capable
deployment: Atlas, or a local ``mongodb-atlas-local`` / AtlasCLI deployment —
it is **not** available on a plain ``mongod``. The index name defaults to
``"vector_index"``; override it with ``vector_index="my_name"`` on
:class:`MongoDBClient`, or pass ``index="my_name"`` to
``collection.search()`` to use a different, externally-managed index.

On the **M0 free tier**, only one vector search index is allowed per cluster.

Requires: ``pip install pymongo``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection as PymongoCollection
    from pymongo.errors import CollectionInvalid, OperationFailure
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The mongodb backend needs the 'pymongo' package. "
        "Install with: pip install pymongo"
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
from vd.filters import SUPPORTED_FILTER_OPERATORS
from vd.util import register_backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default name for the Atlas vector search index (created out-of-band).
_DEFAULT_VECTOR_INDEX = "vector_index"

#: Minimum numCandidates for $vectorSearch; Atlas requires numCandidates >= limit.
_MIN_NUM_CANDIDATES = 100

#: Multiplier applied to ``limit`` when computing ``numCandidates``.
_CANDIDATES_MULTIPLIER = 10

#: vd metric names → Atlas similarity function names (used for documentation;
#: the index similarity is set at index-creation time, not per-query).
_METRIC_TO_ATLAS_SIMILARITY = {
    "cosine": "cosine",
    "dot": "dotProduct",
    "l2": "euclidean",
}


# ---------------------------------------------------------------------------
# Helper: build a $vectorSearch aggregation pipeline
# ---------------------------------------------------------------------------


def _build_vector_search_pipeline(
    vector: Vector,
    *,
    limit: int,
    index: str,
) -> list[dict]:
    """
    Build the MongoDB aggregation pipeline for a ``$vectorSearch`` query.

    The pipeline consists of two stages:

    1. ``$vectorSearch`` — performs the ANN search.
    2. ``$project`` — selects ``text``, ``metadata``, and the
       ``vectorSearchScore`` meta-field (higher-is-better cosine/dot/
       euclidean similarity score assigned by Atlas).

    Metadata filtering is **not** done here. Atlas ``$vectorSearch`` can only
    pre-filter on fields explicitly declared as ``filter`` fields in the index
    definition — but ``vd``'s filter language is open-ended, so filtering is
    applied client-side (over-fetch, then :func:`apply_client_filter`), exactly
    as the pgvector / redis / weaviate / milvus adapters do. This keeps filter
    semantics identical across every backend.

    The ``numCandidates`` parameter controls the candidate pool. Atlas requires
    ``numCandidates >= limit``; we use
    ``max(limit * _CANDIDATES_MULTIPLIER, _MIN_NUM_CANDIDATES)``.

    Parameters
    ----------
    vector : list[float]
        The query embedding vector.
    limit : int
        Number of nearest neighbours to return (already over-fetched by the
        caller when a filter is present).
    index : str
        Name of the Atlas vector search index.

    Returns
    -------
    list[dict]
        A two-stage aggregation pipeline.
    """
    num_candidates = max(limit * _CANDIDATES_MULTIPLIER, _MIN_NUM_CANDIDATES)
    return [
        {
            "$vectorSearch": {
                "index": index,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "metadata": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]


# ---------------------------------------------------------------------------
# Helper: rebuild a vd Document from a raw MongoDB document
# ---------------------------------------------------------------------------


def _raw_to_document(raw: dict) -> Document:
    """
    Rebuild a :class:`~vd.base.Document` from a raw MongoDB document.

    The stored shape is ``{"_id": id, "text": text, "embedding": vector,
    "metadata": {...}}`` (see :meth:`MongoDBCollection._write`); this is the
    inverse mapping.

    Parameters
    ----------
    raw : dict
        A document as returned by ``pymongo``'s ``find_one`` / ``aggregate``.

    Returns
    -------
    Document
    """
    return Document(
        id=raw["_id"],
        text=raw.get("text", ""),
        vector=raw.get("embedding"),
        metadata=raw.get("metadata") or {},
    )


# ---------------------------------------------------------------------------
# MongoDBCollection
# ---------------------------------------------------------------------------


class MongoDBCollection(AbstractCollection):
    """
    A ``vd`` collection backed by a single MongoDB collection.

    Documents are stored as::

        {
            "_id":       <str doc id>,
            "text":      <str text content>,
            "embedding": <list[float] dense vector>,
            "metadata":  <dict user metadata>,
        }

    Nearest-neighbour search is delegated to Atlas ``$vectorSearch``.
    The ``vectorSearchScore`` returned by Atlas is higher-is-better and is
    used directly as the vd result ``score`` regardless of metric.

    The Atlas vector search index named by ``vector_index`` is created
    automatically on the first :meth:`search` (see :meth:`_ensure_search_index`);
    the adapter then blocks until Atlas reports it queryable.

    Parameters
    ----------
    name : str
        Collection name (also the MongoDB collection name).
    pymongo_collection : pymongo.collection.Collection
        The raw pymongo collection handle.
    embedder : callable, optional
        Optional ``text -> vector`` convenience callable.
    dimension : int, optional
        Vector dimension.  Inferred from the first written vector if ``None``.
    metric : str
        Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
    vector_index : str
        Name of the Atlas vector search index to use in ``$vectorSearch``.
    """

    supported_filter_operators = SUPPORTED_FILTER_OPERATORS

    def __init__(
        self,
        name: str,
        pymongo_collection: PymongoCollection,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        vector_index: str = _DEFAULT_VECTOR_INDEX,
    ):
        self.name = name
        self._coll = pymongo_collection
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._vector_index = vector_index
        #: Set once the Atlas vector search index is confirmed queryable.
        self._index_ready = False

    @property
    def native(self) -> PymongoCollection:
        """The raw ``pymongo.collection.Collection`` handle (escape hatch)."""
        return self._coll

    # ----- Atlas vector search index --------------------------------------- #

    def _ensure_search_index(self) -> None:
        """
        Create the Atlas vector search index and wait until it is queryable.

        Lazy and idempotent: invoked on the first :meth:`search`, once the
        vector dimension is known. The index is created on the ``"embedding"``
        field with this collection's ``dimension`` and ``metric``; the adapter
        then blocks until Atlas reports it queryable. A no-op once ready.

        Raises
        ------
        RuntimeError
            If the dimension is still unknown, if the deployment is not
            Atlas-capable (no ``$listSearchIndexes`` support), or if the index
            does not become queryable within the timeout.
        """
        if self._index_ready:
            return
        try:
            existing = {idx["name"] for idx in self._coll.list_search_indexes()}
        except OperationFailure as exc:
            raise RuntimeError(
                "This MongoDB deployment does not support Atlas Vector Search. "
                "Connect to Atlas or a local 'mongodb-atlas-local' / AtlasCLI "
                "deployment — a plain mongod cannot run $vectorSearch."
            ) from exc
        if self._vector_index not in existing:
            if self.dimension is None:
                raise RuntimeError(
                    f"Cannot create the Atlas vector search index for "
                    f"collection {self.name!r}: the vector dimension is unknown. "
                    f"Write a document first, or pass dimension= to "
                    f"create_collection."
                )
            from pymongo.operations import SearchIndexModel

            similarity = _METRIC_TO_ATLAS_SIMILARITY.get(self.metric, "cosine")
            self._coll.create_search_index(
                SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.dimension,
                                "similarity": similarity,
                            }
                        ]
                    },
                    name=self._vector_index,
                    type="vectorSearch",
                )
            )
        self._wait_until_queryable()
        self._index_ready = True

    def _wait_until_queryable(
        self, *, timeout: float = 120.0, poll_interval: float = 1.0
    ) -> None:
        """Block until the vector search index reports ``queryable``."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for idx in self._coll.list_search_indexes(self._vector_index):
                if idx.get("queryable"):
                    return
            time.sleep(poll_interval)
        raise RuntimeError(
            f"Atlas vector search index {self._vector_index!r} for collection "
            f"{self.name!r} did not become queryable within {timeout:.0f}s."
        )

    # ----- raw primitives -------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """
        Upsert one document into the MongoDB collection.

        Uses ``replace_one`` with ``upsert=True`` so the operation is
        idempotent: calling it twice with the same ``doc.id`` overwrites the
        existing document rather than creating a duplicate.
        """
        self._coll.replace_one(
            {"_id": doc.id},
            {
                "_id": doc.id,
                "text": doc.text,
                "embedding": doc.vector,
                "metadata": doc.metadata or {},
            },
            upsert=True,
        )

    def _write_many(self, docs: list[Document]) -> None:
        """
        Bulk-upsert many documents using an unordered ``bulk_write``.

        Uses one ``ReplaceOne`` request per document (with ``upsert=True``),
        sent as a single batch.  ``ordered=False`` allows Atlas to
        parallelise the writes and continue past individual errors.
        """
        from pymongo import ReplaceOne

        operations = [
            ReplaceOne(
                {"_id": doc.id},
                {
                    "_id": doc.id,
                    "text": doc.text,
                    "embedding": doc.vector,
                    "metadata": doc.metadata or {},
                },
                upsert=True,
            )
            for doc in docs
        ]
        self._coll.bulk_write(operations, ordered=False)

    def _read(self, key: str) -> Document:
        """
        Fetch one document by id; raise ``KeyError`` if absent.
        """
        raw = self._coll.find_one({"_id": key})
        if raw is None:
            raise KeyError(key)
        return _raw_to_document(raw)

    def _drop(self, key: str) -> None:
        """
        Delete one document by id; raise ``KeyError`` if absent.
        """
        result = self._coll.delete_one({"_id": key})
        if result.deleted_count == 0:
            raise KeyError(key)

    def _keys(self) -> Iterator[str]:
        """
        Yield document ids by scanning the collection with a projection.

        Uses ``{"_id": 1}`` to minimise network transfer — only the ``_id``
        field is fetched per document.
        """
        cursor = self._coll.find({}, {"_id": 1})
        return (doc["_id"] for doc in cursor)

    def _count(self) -> int:
        """Return the exact count of documents in the collection."""
        return self._coll.count_documents({})

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Run a ``$vectorSearch`` aggregation pipeline and return result dicts.

        Parameters
        ----------
        vector : list[float]
            Query embedding vector.
        limit : int
            Maximum number of results.
        filter : dict or None
            Canonical vd filter AST (translated to MQL internally).
        **kwargs
            Optional overrides:

            - ``index`` (str): name of the Atlas vector search index to use,
              overriding the collection-level ``vector_index`` default.
            - ``num_candidates`` (int): explicit ``numCandidates`` value to
              pass to ``$vectorSearch``; overrides the computed default of
              ``max(limit * 10, 100)``.

        Yields
        ------
        dict
            ``{"id", "text", "score", "metadata"}`` — score is the Atlas
            ``vectorSearchScore``, higher-is-better.
        """
        index = kwargs.get("index", self._vector_index)
        # Auto-create/await our managed index; a caller-supplied index= is
        # assumed to be managed externally and is used as-is.
        if index == self._vector_index:
            self._ensure_search_index()

        # Over-fetch when filtering, then filter client-side — Atlas can only
        # pre-filter on index-declared fields, so the canonical vd evaluator is
        # applied locally (identical to every other server-backed adapter).
        fetch = overfetch_limit(limit, filter)
        pipeline = _build_vector_search_pipeline(vector, limit=fetch, index=index)

        # Allow callers to override numCandidates via kwargs.
        if "num_candidates" in kwargs:
            pipeline[0]["$vectorSearch"]["numCandidates"] = int(
                kwargs["num_candidates"]
            )

        results = [
            {
                "id": raw["_id"],
                "text": raw.get("text", ""),
                "score": raw.get("score", 0.0),
                "metadata": raw.get("metadata", {}),
            }
            for raw in self._coll.aggregate(pipeline)
        ]
        return apply_client_filter(results, filter, limit=limit)


# ---------------------------------------------------------------------------
# MongoDBClient
# ---------------------------------------------------------------------------


@register_backend("mongodb")
class MongoDBClient(AbstractClient):
    """
    MongoDB Atlas Vector Search client.

    Connects to a MongoDB deployment (Atlas or self-managed) via the URI in
    the ``MONGODB_URI`` environment variable or the ``uri`` keyword argument.
    All ``vd`` collections map to MongoDB collections inside the database
    named by ``database`` (default ``"vd"``).

    The Atlas vector search index on each collection's ``"embedding"`` field
    is created automatically on the first search, named by ``vector_index``
    (default ``"vector_index"``). Pass ``index="my_index"`` to
    ``collection.search(...)`` to use a different, externally-managed index.

    Parameters
    ----------
    embedder : callable, optional
        Optional ``text -> vector`` convenience callable passed to each
        collection so text inputs are accepted.
    uri : str, optional
        A MongoDB connection string, e.g.
        ``"mongodb+srv://user:pass@cluster.mongodb.net/"`` for Atlas or
        ``"mongodb://localhost:27017/"`` for local.  Falls back to the
        ``MONGODB_URI`` environment variable when ``None``.  Raises
        ``ValueError`` if neither is provided.
    database : str
        The MongoDB database to use (default ``"vd"``).  One database maps
        to one ``vd`` client; use separate clients for separate databases.
    vector_index : str
        Default Atlas vector search index name (default ``"vector_index"``).
        Passed to every collection; individual ``search()`` calls can
        override it with ``index="..."``.
    **config
        Additional keyword arguments forwarded to
        :class:`vd.base.AbstractClient` and stored as ``self.config``.

    Raises
    ------
    ValueError
        If no URI is provided and the ``MONGODB_URI`` environment variable is
        not set.

    Examples
    --------
    ::

        import os
        import vd

        client = vd.connect("mongodb")          # reads MONGODB_URI from env
        col = client.get_or_create_collection("my_docs", dimension=1536)
        col["doc1"] = vd.Document(id="doc1", text="hello", vector=[0.1]*1536)
        # The Atlas vector search index is created on the first search call;
        # that call blocks until Atlas reports the index queryable.
        for hit in col.search([0.1]*1536, limit=5):
            print(hit["id"], hit["score"])
    """

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        uri: Optional[str] = None,
        database: str = "vd",
        vector_index: str = _DEFAULT_VECTOR_INDEX,
        **config,
    ):
        super().__init__(embedder=embedder, **config)
        resolved_uri = uri or os.environ.get("MONGODB_URI")
        if not resolved_uri:
            raise ValueError(
                "No MongoDB URI supplied. Either pass uri='mongodb+srv://...' "
                "or set the MONGODB_URI environment variable."
            )
        self._client: MongoClient = MongoClient(resolved_uri)
        self._db = self._client[database]
        self._vector_index = vector_index
        self._database_name = database

    # ----- client interface ------------------------------------------------ #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> MongoDBCollection:
        """
        Create a new MongoDB collection and return a ``MongoDBCollection``.

        This creates the *document collection* inside the database.  The Atlas
        vector search index on the ``"embedding"`` field must still be created
        separately (see module docstring).

        Parameters
        ----------
        name : str
            Collection name.
        dimension : int, optional
            Vector dimension stored in the collection object for later
            dimension-mismatch checking.  Not enforced at the MongoDB level.
        metric : str
            Distance metric (``"cosine"``, ``"dot"``, or ``"l2"``).  Stored
            in the collection object; the actual Atlas index similarity is set
            at index-creation time in the Atlas UI/API.
        **index_config
            Additional backend-specific options (currently unused; reserved
            for future Atlas index configuration hints).

        Raises
        ------
        ValueError
            If a collection named ``name`` already exists in the database.
        """
        try:
            self._db.create_collection(name)
        except CollectionInvalid as exc:
            raise ValueError(
                f"Collection {name!r} already exists in database "
                f"{self._database_name!r}."
            ) from exc
        return MongoDBCollection(
            name,
            self._db[name],
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
            vector_index=self._vector_index,
        )

    def get_collection(self, name: str) -> MongoDBCollection:
        """
        Return an existing ``MongoDBCollection``; raise ``KeyError`` if absent.

        Parameters
        ----------
        name : str
            Name of the collection to retrieve.

        Raises
        ------
        KeyError
            If the collection does not exist in the database.
        """
        if name not in self._db.list_collection_names():
            raise KeyError(
                f"Collection {name!r} does not exist in database "
                f"{self._database_name!r}."
            )
        return MongoDBCollection(
            name,
            self._db[name],
            embedder=self._embedder,
            vector_index=self._vector_index,
        )

    def delete_collection(self, name: str) -> None:
        """
        Drop a MongoDB collection; raise ``KeyError`` if absent.

        Parameters
        ----------
        name : str
            Name of the collection to drop.

        Raises
        ------
        KeyError
            If the collection does not exist in the database.
        """
        if name not in self._db.list_collection_names():
            raise KeyError(
                f"Collection {name!r} does not exist in database "
                f"{self._database_name!r}."
            )
        self._db.drop_collection(name)

    def list_collections(self) -> Iterator[str]:
        """
        Iterate over collection names in the current database.

        Returns
        -------
        Iterator[str]
            Names of all MongoDB collections in the database, in the order
            returned by the server.
        """
        return iter(self._db.list_collection_names())

    def close(self) -> None:
        """Close the underlying ``MongoClient`` and release its resources."""
        self._client.close()
