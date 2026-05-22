"""
Weaviate backend.

Weaviate is a BSD-3-licensed, cloud-native vector database with first-class
hybrid (BM25 + vector) search, a modular vectorizer architecture (plug in
OpenAI, Cohere, HuggingFace, Ollama, etc. to embed for you), and a typed
Python v4 client. It runs as a self-hosted Docker container, a managed Weaviate
Cloud cluster, or — for small experiments — as a local server on localhost.

This adapter maps the ``vd`` facade onto Weaviate v4's collection model:

- Each vd *collection* → one Weaviate *class* (Weaviate class names must begin
  with an uppercase letter; a vd name like ``"articles"`` is stored under the
  class name ``"Articles"``). A dict on the client keeps the original vd name
  so ``list_collections`` and ``get_collection`` always round-trip the exact
  name the user supplied.
- The adapter uses **self-provided vectors** (``Configure.Vectors.self_provided``):
  vd supplies vectors; Weaviate just stores and indexes them. Weaviate's
  own vectorizer modules are available via the ``native`` escape hatch.
- Three Weaviate properties are reserved: ``text`` (the document text),
  ``vd_id`` (the original vd document id), and ``metadata_json`` (the
  metadata dict serialized to a JSON string). Metadata is stored as a single
  JSON string because vd's canonical metadata filter is evaluated client-side
  (Weaviate's typed filter builder cannot operate on arbitrary JSON fields
  without a fixed schema per key). The adapter over-fetches and calls
  ``apply_client_filter`` exactly like the FAISS backend.
- Each document is given a deterministic UUID5 from its vd id
  (``uuid.uuid5(uuid.NAMESPACE_OID, doc_id)``), so ``_write`` is always an
  idempotent upsert: insert on first write, replace on subsequent writes.

Connection modes
----------------
- ``weaviate.connect_to_local()`` — local server on localhost:8080/50051.
- ``weaviate.connect_to_local(host=..., port=..., grpc_port=...)`` — local
  server at a custom address.
- ``weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=...)``
  — Weaviate Cloud (formerly WCS) with an API key.

Requires: ``pip install -U weaviate-client``
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Iterable, Iterator, Optional

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The weaviate backend needs the 'weaviate-client' package. "
        "Install with: pip install -U weaviate-client"
    ) from e

from vd.backends._helpers import (
    apply_client_filter,
    overfetch_limit,
    score_from_distance,
)
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

#: Weaviate property that stores the original vd document id.
_VD_ID_PROP = "vd_id"

#: Weaviate property that stores the document text.
_TEXT_PROP = "text"

#: Weaviate property that stores the metadata dict as a JSON string.
_META_PROP = "metadata_json"

#: vd metric -> Weaviate vector distance.  Only used to set the collection's
#: index config at creation time; actual score conversion uses score_from_distance.
_METRIC_TO_DISTANCE = {
    "cosine": "cosine",
    "dot": "dot",
    "l2": "l2-squared",
}


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _weaviate_class_name(vd_name: str) -> str:
    """Map a vd collection name to a Weaviate class name (first char uppercased).

    Weaviate requires class names to start with an uppercase letter.

    Examples
    --------
    >>> _weaviate_class_name("articles")
    'Articles'
    >>> _weaviate_class_name("MyDocs")
    'MyDocs'
    """
    if not vd_name:
        raise ValueError("Collection name must not be empty.")
    return vd_name[0].upper() + vd_name[1:]


def _object_uuid(doc_id: str) -> str:
    """Return a deterministic UUID5 string for a vd document id.

    Using ``uuid.NAMESPACE_OID`` so writes are idempotent upserts: the same
    vd id always maps to the same Weaviate object UUID.

    Examples
    --------
    >>> _object_uuid("doc1") == _object_uuid("doc1")
    True
    >>> _object_uuid("doc1") != _object_uuid("doc2")
    True
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, doc_id))


def _create_collection_schema(
    weaviate_client: Any,
    class_name: str,
    *,
    metric: str = "cosine",
) -> Any:
    """Create a Weaviate collection for vd, using self-provided vectors.

    Falls back to ``Configure.Vectors.none()`` if ``self_provided`` is
    unavailable in older v4.x sub-versions.

    Parameters
    ----------
    weaviate_client :
        A live ``weaviate.WeaviateClient``.
    class_name :
        The Weaviate class name (first char uppercase).
    metric :
        Distance metric — ``"cosine"``, ``"dot"``, or ``"l2"``.

    Returns
    -------
    The newly created Weaviate collection handle.
    """
    try:
        vector_config = Configure.Vectors.self_provided()
    except AttributeError:  # pragma: no cover — older SDK sub-version fallback
        vector_config = Configure.Vectors.none()

    properties = [
        Property(name=_TEXT_PROP, data_type=DataType.TEXT),
        Property(name=_VD_ID_PROP, data_type=DataType.TEXT),
        Property(name=_META_PROP, data_type=DataType.TEXT),
    ]

    return weaviate_client.collections.create(
        name=class_name,
        vector_config=vector_config,
        properties=properties,
    )


def _result_from_object(obj: Any, *, metric: str) -> SearchResult:
    """Convert one Weaviate query result object to a vd :data:`SearchResult` dict.

    Parameters
    ----------
    obj :
        A Weaviate result object with ``.properties`` and ``.metadata``.
    metric :
        The collection distance metric, used to convert distance to score.
    """
    props = obj.properties or {}
    distance = obj.metadata.distance if obj.metadata else 0.0
    score = score_from_distance(distance, metric)
    raw_meta = props.get(_META_PROP) or "{}"
    try:
        metadata = json.loads(raw_meta)
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    return {
        "id": props.get(_VD_ID_PROP, str(obj.uuid)),
        "text": props.get(_TEXT_PROP, ""),
        "score": score,
        "metadata": metadata,
    }


def _doc_from_object(obj: Any) -> Document:
    """Reconstruct a :class:`~vd.base.Document` from a fetched Weaviate object.

    Parameters
    ----------
    obj :
        A Weaviate object returned by ``fetch_object_by_id`` (with vector).
    """
    props = obj.properties or {}
    raw_meta = props.get(_META_PROP) or "{}"
    try:
        metadata = json.loads(raw_meta)
    except (json.JSONDecodeError, TypeError):
        metadata = {}
    vector = list(obj.vector.get("default") or []) if obj.vector else None
    return Document(
        id=props.get(_VD_ID_PROP, str(obj.uuid)),
        text=props.get(_TEXT_PROP, ""),
        vector=vector,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# WeaviateCollection
# --------------------------------------------------------------------------- #


class WeaviateCollection(AbstractCollection):
    """
    A collection backed by one Weaviate class.

    Metadata is stored as a JSON string in a single ``metadata_json`` property
    and filtered client-side via :func:`~vd.backends._helpers.apply_client_filter`,
    exactly like the FAISS backend.  This design keeps the Weaviate schema
    fixed regardless of the user's metadata shape.

    Parameters
    ----------
    name :
        The vd collection name (original, potentially lowercase).
    weaviate_client :
        A live ``weaviate.WeaviateClient`` instance owned by the adapter client.
    embedder :
        Optional ``text -> vector`` callable.
    dimension :
        Vector dimension (``None`` means inferred from the first written vector).
    metric :
        Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
    """

    def __init__(
        self,
        name: str,
        weaviate_client: Any,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ):
        self.name = name
        self._weaviate_client = weaviate_client
        self._embedder = embedder
        self.dimension = dimension
        self.metric = metric
        self._class_name: str = _weaviate_class_name(name)

    @property
    def native(self) -> Any:
        """The raw Weaviate collection handle (escape hatch for advanced use)."""
        return self._weaviate_client.collections.get(self._class_name)

    # ----- internal helpers ------------------------------------------------- #

    def _wcol(self) -> Any:
        """Return the underlying Weaviate collection object."""
        return self._weaviate_client.collections.get(self._class_name)

    def _class_exists(self) -> bool:
        """Return ``True`` if the Weaviate class for this collection exists."""
        try:
            # collections.get() returns a lazy handle without a server check —
            # collections.exists() is the actual existence query.
            return bool(self._weaviate_client.collections.exists(self._class_name))
        except Exception:
            return False

    # ----- raw primitives --------------------------------------------------- #

    def _write(self, doc: Document) -> None:
        """Upsert one document (idempotent via deterministic UUID5)."""
        obj_uuid = _object_uuid(doc.id)
        props = {
            _TEXT_PROP: doc.text,
            _VD_ID_PROP: doc.id,
            _META_PROP: json.dumps(doc.metadata or {}),
        }
        wcol = self._wcol()
        try:
            wcol.data.insert(
                properties=props,
                vector=doc.vector,
                uuid=obj_uuid,
            )
        except Exception:
            # Object already exists — replace it to achieve upsert semantics.
            wcol.data.replace(
                properties=props,
                vector=doc.vector,
                uuid=obj_uuid,
            )

    def _write_many(self, docs: list[Document]) -> None:
        """Batch upsert using Weaviate's context-manager batch API."""
        wcol = self._wcol()
        # Weaviate v4 batch: use the collection-level batch context manager.
        with wcol.batch.dynamic() as batch:
            for doc in docs:
                batch.add_object(
                    properties={
                        _TEXT_PROP: doc.text,
                        _VD_ID_PROP: doc.id,
                        _META_PROP: json.dumps(doc.metadata or {}),
                    },
                    vector=doc.vector,
                    uuid=_object_uuid(doc.id),
                )

    def _read(self, key: str) -> Document:
        """Fetch one document by vd id; raise ``KeyError`` if absent."""
        obj_uuid = _object_uuid(key)
        wcol = self._wcol()
        obj = wcol.query.fetch_object_by_id(obj_uuid, include_vector=True)
        if obj is None:
            raise KeyError(key)
        return _doc_from_object(obj)

    def _drop(self, key: str) -> None:
        """Delete one document; raise ``KeyError`` if absent."""
        obj_uuid = _object_uuid(key)
        wcol = self._wcol()
        existing = wcol.query.fetch_object_by_id(obj_uuid)
        if existing is None:
            raise KeyError(key)
        wcol.data.delete_by_id(obj_uuid)

    def _keys(self) -> Iterator[str]:
        """Iterate vd document ids by scanning the Weaviate collection."""
        wcol = self._wcol()
        for obj in wcol.iterator(include_vector=False):
            props = obj.properties or {}
            yield props.get(_VD_ID_PROP, str(obj.uuid))

    def _count(self) -> int:
        """Return the number of documents via Weaviate's aggregate API."""
        wcol = self._wcol()
        result = wcol.aggregate.over_all(total_count=True)
        return result.total_count or 0

    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """
        Nearest-neighbor search with client-side metadata filtering.

        Weaviate's native filter builder works on typed schema properties;
        because vd stores all metadata in a single JSON string, filtering is
        done client-side: over-fetch ``overfetch_limit`` candidates, then call
        :func:`~vd.backends._helpers.apply_client_filter`.
        """
        from weaviate.classes.query import MetadataQuery

        fetch = overfetch_limit(limit, filter)
        wcol = self._wcol()
        response = wcol.query.near_vector(
            near_vector=vector,
            limit=fetch,
            return_metadata=MetadataQuery(distance=True),
            include_vector=False,
            **kwargs,
        )
        raw_results = [
            _result_from_object(obj, metric=self.metric) for obj in response.objects
        ]
        return apply_client_filter(raw_results, filter, limit=limit)


# --------------------------------------------------------------------------- #
# WeaviateClient
# --------------------------------------------------------------------------- #


@register_backend("weaviate")
class WeaviateClient(AbstractClient):
    """
    Weaviate client.

    Manages a live ``weaviate.WeaviateClient`` connection and a mapping from
    vd collection names to their Weaviate class names.

    Parameters
    ----------
    embedder :
        Optional ``text -> vector`` convenience callable.
    url :
        Weaviate Cloud cluster URL.  When given together with ``api_key``,
        uses ``weaviate.connect_to_weaviate_cloud``.
    api_key :
        API key for Weaviate Cloud authentication.
    host :
        Hostname or IP of a self-hosted Weaviate server.  Triggers
        ``weaviate.connect_to_local(host=host, ...)``.
    port :
        HTTP port of the self-hosted server (default 8080).
    grpc_port :
        gRPC port of the self-hosted server (default 50051).
    **config :
        Extra keyword arguments forwarded to the ``AbstractClient`` base.

    Connection selection logic
    --------------------------
    1. ``url`` + ``api_key`` → ``connect_to_weaviate_cloud``
    2. ``host`` → ``connect_to_local(host=host, port=port, grpc_port=grpc_port)``
    3. (default) → ``connect_to_local()`` (localhost:8080/50051)

    Examples
    --------
    >>> # Local server (Docker running on localhost):
    >>> # client = WeaviateClient()
    >>> # Cloud cluster:
    >>> # client = WeaviateClient(url="https://…weaviate.cloud", api_key="…")
    """

    backend_name: str = "weaviate"

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8080,
        grpc_port: int = 50051,
        **config,
    ):
        super().__init__(embedder=embedder, **config)

        if url is not None and api_key is not None:
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.classes.init.Auth.api_key(api_key),
            )
        elif host is not None:
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
            )
        else:
            self._client = weaviate.connect_to_local()

        # Map vd name → Weaviate class name for all collections in this session.
        # Populated by create_collection; discovered lazily by list_collections.
        self._name_map: dict[str, str] = {}

    # ----- internal helpers ------------------------------------------------- #

    def _class_exists(self, class_name: str) -> bool:
        """Return ``True`` if the Weaviate class ``class_name`` currently exists."""
        try:
            # collections.get() returns a lazy handle without a server check —
            # collections.exists() is the actual existence query.
            return bool(self._client.collections.exists(class_name))
        except Exception:
            return False

    def _vd_name_for_class(self, class_name: str) -> str:
        """Reverse-look up the vd name for a Weaviate class name.

        Falls back to lowercasing the class name when the class was created
        outside this session and is not in ``_name_map``.
        """
        for vd_name, cn in self._name_map.items():
            if cn == class_name:
                return vd_name
        # Fallback: reconstruct by lowercasing the first letter.
        return class_name[0].lower() + class_name[1:] if class_name else class_name

    def _collection_for(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
    ) -> WeaviateCollection:
        """Build a :class:`WeaviateCollection` wrapper for an existing class."""
        return WeaviateCollection(
            name,
            self._client,
            embedder=self._embedder,
            dimension=dimension,
            metric=metric,
        )

    # ----- AbstractClient interface ----------------------------------------- #

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> WeaviateCollection:
        """
        Create a new collection, raising ``ValueError`` if it already exists.

        Parameters
        ----------
        name :
            vd collection name (may be lowercase; mapped to a Weaviate class
            name by uppercasing the first character).
        dimension :
            Vector dimension.  ``None`` is accepted (Weaviate infers from the
            first written vector); required if you want dimension-mismatch
            detection before the first write.
        metric :
            Distance metric: ``"cosine"`` (default), ``"dot"``, or ``"l2"``.
        **index_config :
            Reserved for future per-collection Weaviate index tuning; currently
            unused.

        Raises
        ------
        ValueError
            If a collection named ``name`` already exists (either in this session
            or already present on the Weaviate server).
        """
        class_name = _weaviate_class_name(name)
        if name in self._name_map or self._class_exists(class_name):
            raise ValueError(f"Collection {name!r} already exists")

        _create_collection_schema(self._client, class_name, metric=metric)
        self._name_map[name] = class_name
        return self._collection_for(name, dimension=dimension, metric=metric)

    def get_collection(self, name: str) -> WeaviateCollection:
        """
        Return an existing collection; raise ``KeyError`` if absent.

        Parameters
        ----------
        name :
            vd collection name as originally supplied to ``create_collection``.

        Raises
        ------
        KeyError
            If no collection of that name exists.
        """
        class_name = _weaviate_class_name(name)
        if name not in self._name_map and not self._class_exists(class_name):
            raise KeyError(f"Collection {name!r} does not exist")
        # Re-register in name_map if discovered via class existence check.
        if name not in self._name_map:
            self._name_map[name] = class_name
        return self._collection_for(name)

    def delete_collection(self, name: str) -> None:
        """
        Drop a collection; raise ``KeyError`` if absent.

        Parameters
        ----------
        name :
            vd collection name.

        Raises
        ------
        KeyError
            If no collection of that name exists.
        """
        class_name = _weaviate_class_name(name)
        if name not in self._name_map and not self._class_exists(class_name):
            raise KeyError(f"Collection {name!r} does not exist")
        if self._class_exists(class_name):
            self._client.collections.delete(class_name)
        self._name_map.pop(name, None)

    def list_collections(self) -> Iterator[str]:
        """
        Iterate vd collection names currently present on the server.

        Names that were registered this session are returned as-is.  Classes
        discovered on the server that are not in the session registry are
        reverse-mapped (first char lowercased).
        """
        # Enumerate all classes visible on the server.
        all_classes = self._client.collections.list_all(simple=True)
        seen: set[str] = set()
        # Prefer session-registered names (exact round-trip).
        for vd_name, class_name in self._name_map.items():
            if class_name in all_classes:
                yield vd_name
                seen.add(class_name)
        # Surface any extra classes on the server not in the session map.
        for class_name in all_classes:
            if class_name not in seen:
                vd_name = self._vd_name_for_class(class_name)
                self._name_map.setdefault(vd_name, class_name)
                yield vd_name

    def close(self) -> None:
        """Close the underlying Weaviate client connection (required in v4)."""
        self._client.close()
