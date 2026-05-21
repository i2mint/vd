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

**IMPORTANT — the Atlas vector search index must be created out-of-band.**
``MongoDBClient.create_collection`` creates the *MongoDB collection* (the
document container) but it **cannot** create the Atlas vector search index.
The vector search index must be created separately, before ``search()`` is
called, via:

1. The Atlas UI: *Atlas → Browse Collections → Search Indexes → Create
   Search Index → Atlas Vector Search*, or
2. The Atlas Data API / Admin API, or
3. The MongoDB Atlas Terraform provider.

When you create the index, set:

- *field*: ``"embedding"``
- *type*: ``"knnVector"``
- *dimensions*: the embedding dimension you are using (e.g. ``1536``)
- *similarity*: ``"cosine"`` (or ``"euclidean"`` / ``"dotProduct"`` to match
  your ``metric`` choice)

The default index name expected by this adapter is ``"vector_index"``; pass
``vector_index="my_name"`` to ``MongoDBClient`` or ``index="my_name"`` to
``collection.search()`` to override.

On the **M0 free tier**, only one vector search index is allowed per cluster.

Requires: ``pip install pymongo``
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection as PymongoCollection
    from pymongo.errors import CollectionInvalid
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The mongodb backend needs the 'pymongo' package. "
        "Install with: pip install pymongo"
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
# Filter translation helpers
# ---------------------------------------------------------------------------


def _prefix_field(key: str) -> str:
    """
    Return the MongoDB document path for a vd filter field key.

    User metadata is stored under the ``metadata`` sub-document, so every
    plain field key is prefixed with ``"metadata."``.  Operator keys that
    start with ``"$"`` are returned unchanged.

    Examples
    --------
    >>> _prefix_field("year")
    'metadata.year'
    >>> _prefix_field("$and")
    '$and'
    """
    return key if key.startswith("$") else f"metadata.{key}"


def _to_mongo_filter(ast: Optional[Filter]) -> Optional[dict]:
    """
    Translate a canonical ``vd`` filter AST to a MongoDB Query Language document.

    The canonical vd filter dialect is already modelled after MQL, so the
    translation is nearly identity.  The one structural change: every
    *field* key (not operator key) must be prefixed with ``"metadata."``
    because user metadata is stored nested under that sub-document in the
    MongoDB document schema.  Operator keys (``$and``, ``$or``, ``$not``,
    ``$eq``, ``$gt``, ...) are left unchanged and MongoDB handles them
    natively.

    Logical operators are recursed through transparently.  ``$not`` wraps
    its sub-document in MQL's ``{"$not": {...}}`` form for field conditions.

    Parameters
    ----------
    ast : dict or None
        A filter in the canonical ``vd`` dialect (see :mod:`vd.filters`).
        ``None`` or empty returns ``None`` (no filter applied).

    Returns
    -------
    dict or None
        A MongoDB Query Language document ready to pass as ``filter`` in a
        ``$vectorSearch`` stage, or ``None`` when no filtering is needed.

    Examples
    --------
    >>> _to_mongo_filter(None)
    >>> _to_mongo_filter({})
    >>> _to_mongo_filter({"year": 2024})
    {'metadata.year': 2024}
    >>> _to_mongo_filter({"year": {"$gte": 2020}})
    {'metadata.year': {'$gte': 2020}}
    >>> _to_mongo_filter({"$and": [{"year": 2024}, {"tag": "ai"}]})
    {'$and': [{'metadata.year': 2024}, {'metadata.tag': 'ai'}]}
    """
    if not ast:
        return None

    result: dict = {}
    for key, value in ast.items():
        if key == "$and":
            # $and takes a list of sub-filter documents
            result["$and"] = [_to_mongo_filter(sub) for sub in value]
        elif key == "$or":
            # $or takes a list of sub-filter documents
            result["$or"] = [_to_mongo_filter(sub) for sub in value]
        elif key == "$not":
            # $not in the vd AST wraps a single sub-filter; translate each
            # field condition inside it with prefixed keys.
            result.update(_to_mongo_not(value))
        else:
            # Plain field condition — prefix the key with "metadata."
            result[_prefix_field(key)] = value

    return result or None


def _to_mongo_not(sub: Filter) -> dict:
    """
    Translate a ``$not`` sub-filter into MQL negation form.

    MQL ``$not`` operates on a single field condition
    (``{"field": {"$not": {...}}}``), whereas the vd ``$not`` wraps an
    entire sub-filter.  We translate by prefixing each field and wrapping
    its condition in ``{"$not": condition}``.  Nested logical operators
    inside ``$not`` are passed through recursively.

    Parameters
    ----------
    sub : dict
        The sub-filter expression to negate.

    Returns
    -------
    dict
        An MQL document expressing the negation.
    """
    result: dict = {}
    for key, value in sub.items():
        if key in ("$and", "$or", "$not"):
            # Logical operators inside $not: recurse via _to_mongo_filter
            translated = _to_mongo_filter({key: value})
            if translated:
                result.update(translated)
        else:
            prefixed = _prefix_field(key)
            if isinstance(value, dict):
                result[prefixed] = {"$not": value}
            else:
                # Bare equality — express as $not: {$eq: value}
                result[prefixed] = {"$not": {"$eq": value}}
    return result


# ---------------------------------------------------------------------------
# Helper: build a $vectorSearch aggregation pipeline
# ---------------------------------------------------------------------------


def _build_vector_search_pipeline(
    vector: Vector,
    *,
    limit: int,
    index: str,
    mongo_filter: Optional[dict],
) -> list[dict]:
    """
    Build the MongoDB aggregation pipeline for a ``$vectorSearch`` query.

    The pipeline consists of two stages:

    1. ``$vectorSearch`` — performs the ANN search.
    2. ``$project`` — selects ``text``, ``metadata``, and the
       ``vectorSearchScore`` meta-field (higher-is-better cosine/dot/
       euclidean similarity score assigned by Atlas).

    The ``numCandidates`` parameter controls the pre-filter candidate pool.
    Atlas requires ``numCandidates >= limit``; we use
    ``max(limit * _CANDIDATES_MULTIPLIER, _MIN_NUM_CANDIDATES)`` as the
    default, matching common practice from the Atlas documentation.

    Parameters
    ----------
    vector : list[float]
        The query embedding vector.
    limit : int
        Number of nearest neighbours to return.
    index : str
        Name of the Atlas vector search index (must exist out-of-band).
    mongo_filter : dict or None
        An MQL filter document (already translated by ``_to_mongo_filter``),
        or ``None`` to skip metadata filtering.

    Returns
    -------
    list[dict]
        A two-stage aggregation pipeline.
    """
    num_candidates = max(limit * _CANDIDATES_MULTIPLIER, _MIN_NUM_CANDIDATES)
    vector_search_stage: dict[str, Any] = {
        "index": index,
        "path": "embedding",
        "queryVector": vector,
        "numCandidates": num_candidates,
        "limit": limit,
    }
    if mongo_filter:
        vector_search_stage["filter"] = mongo_filter

    return [
        {"$vectorSearch": vector_search_stage},
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

    The Atlas vector search index named by ``vector_index`` must exist on
    the field ``"embedding"`` of this collection *before* :meth:`search` is
    called.  This adapter does not create it — see the module docstring for
    how to do so via the Atlas UI or API.

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

    @property
    def native(self) -> PymongoCollection:
        """The raw ``pymongo.collection.Collection`` handle (escape hatch)."""
        return self._coll

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
        mongo_filter = _to_mongo_filter(filter)
        pipeline = _build_vector_search_pipeline(
            vector, limit=limit, index=index, mongo_filter=mongo_filter
        )

        # Allow callers to override numCandidates via kwargs.
        if "num_candidates" in kwargs:
            pipeline[0]["$vectorSearch"]["numCandidates"] = int(
                kwargs["num_candidates"]
            )

        results = []
        for raw in self._coll.aggregate(pipeline):
            results.append(
                {
                    "id": raw["_id"],
                    "text": raw.get("text", ""),
                    "score": raw.get("score", 0.0),
                    "metadata": raw.get("metadata", {}),
                }
            )
        return results


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
    **must be created out-of-band** (Atlas UI, Admin API, or Terraform) before
    :meth:`~MongoDBCollection.search` works.  Use the ``vector_index`` name
    you gave that index when constructing this client (or override per-query
    with ``collection.search(..., index="my_index")``).

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
        # Create the Atlas vector search index on "embedding" before search!
        col["doc1"] = vd.Document(id="doc1", text="hello", vector=[0.1]*1536)
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
