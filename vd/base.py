"""
Core contracts, data model, and abstract bases for the ``vd`` vectorDB facade.

``vd`` is a facade over vector databases. This module is the single source of
truth for the contract every backend adapter satisfies and that all of ``vd``'s
higher-level tooling (search, io, migration, analytics, ...) is written against.

The contract, smallest-first:

- :class:`Document` — the unit stored in a collection: ``id``, ``text``,
  ``vector``, ``metadata``.
- :class:`Collection` — a ``MutableMapping[str, Document]`` *plus* a
  :meth:`~Collection.search` method. This is the one retrieval extension.
- :class:`Client` — a ``Mapping[str, Collection]``: a live connection to one
  backend, through which collections are created, fetched, and dropped.
- :class:`AbstractCollection` / :class:`AbstractClient` — adapter-author
  conveniences. A backend implements a handful of *raw primitives*; these bases
  supply everything users actually see (flexible inputs, optional text
  embedding, ``egress`` transforms, batch helpers, dimension checks) uniformly.

Embedding is deliberately **external**. ``vd`` stores and searches *vectors*;
turning text into vectors is another package's job (e.g. ``ef``). A
:class:`Client` may be handed an optional ``embedder`` callable purely as a
convenience, so ``collection["k"] = "some text"`` and
``collection.search("query text")`` work. With no embedder configured those
text forms raise :class:`EmbeddingRequiredError`, and the caller must pass
:class:`Document` objects (or pre-computed vectors) directly.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

Text = str
TextKey = str  # Document ID / URI — the key in a collection
Metadata = dict[str, Any]
Vector = list[float]
VectorMapping = Mapping[TextKey, Vector]
SearchResult = dict[str, Any]  # {"id", "text", "score", "metadata", ...}
Filter = dict[str, Any]  # canonical MongoDB-style filter (see vd.filters)

#: A document may be supplied to batch operations in several flexible shapes.
DocumentInput = Union[
    str,  # just text (id auto-generated)
    tuple,  # (text, id) | (text, metadata) | (text, id, metadata)
    "Document",  # a fully-formed Document
]

#: Distance metrics the facade understands. Adapters map these to their own
#: spellings (e.g. ``"l2"`` -> Qdrant ``Distance.EUCLID``).
METRICS = frozenset({"cosine", "dot", "l2"})

# --------------------------------------------------------------------------- #
# Score semantics — the cross-backend contract for SearchResult["score"]
# --------------------------------------------------------------------------- #
#
# Every :data:`SearchResult` carries a ``score`` field. ``vd``'s contract for
# that number is **higher-is-better, per-metric canonical similarity**:
#
# ============  ===============================  ======================
# metric        canonical score                  range
# ============  ===============================  ======================
# ``cosine``    ``1 - cosine_distance``          ``[-1, 1]``
# ``dot``       raw inner product                ``(-inf, +inf)``
# ``l2``        ``1 / (1 + euclidean_distance)`` ``(0, 1]``
# ============  ===============================  ======================
#
# Rationale:
#
# - **Same backend, different metrics** stay comparable (all three are
#   higher-better).
# - **Same metric, different backends** stay comparable: ``vd``'s own
#   ``reciprocal_rank_fusion`` / ``deduplicate_results`` / ``multi_query_search``
#   helpers and consumers like ``ef.SearchHit`` all assume this scale, so an
#   adapter that returns ``1 / (1 + raw_distance)`` for cosine instead of
#   ``1 - raw_distance`` would mis-rank only across adapters but consistently
#   confuse score-threshold logic.
#
# The reference implementations are :func:`vd.backends.memory._similarity`
# (in-memory adapter) and :func:`vd.backends._helpers.score_from_distance`
# (distance-returning adapters). Adapters whose backend natively returns a
# higher-is-better score on a *different* per-metric scale (e.g. Elasticsearch
# kNN, MongoDB Atlas, Pinecone) **document the deviation in their adapter
# docstring** rather than silently rescaling, because rescaling a backend's
# own combined-ranking score can change ordering for ties. The deviation is
# the cost of using that backend's native scoring.
#
# See issue #9 for the history of this contract.

# Re-exported from vd.filters; imported lazily inside methods to avoid a cycle
# (vd.filters imports UnsupportedFilterError from this module).


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class VdError(Exception):
    """Base class for every error ``vd`` raises on its own behalf."""


class StaticIndexError(VdError):
    """
    Raised on a write to a static (immutable) index.

    Some backends — notably a plain FAISS flat index — build an index that
    cannot accept incremental ``__setitem__`` / ``__delitem__`` after creation.
    Such collections set :attr:`AbstractCollection.supports_incremental_writes`
    to ``False`` and raise this on write. Callers branch on that flag *before*
    triggering the error, and use the adapter's documented ``rebuild()`` path.
    """


class UnsupportedFilterError(VdError, ValueError):
    """
    Raised when a metadata filter uses an operator a backend cannot honor.

    The canonical, backend-agnostic filter language lives in :mod:`vd.filters`
    (a MongoDB-style JSON dialect). When a filter uses an operator outside a
    backend's documented subset — or one that does not exist at all — this is
    raised, so the caller can simplify the filter or drop to the backend's
    native filter via the escape hatch (``collection.native``).
    """


class UnsupportedCapabilityError(VdError, NotImplementedError):
    """
    Raised when an operation needs a capability the backend lacks.

    Prefer feature-discovery — ``isinstance(collection, SupportsHybrid)`` — over
    catching this, but it is the clear, typed fallback when an optional
    operation is called on a backend that does not implement it.
    """


class EmbeddingRequiredError(VdError, RuntimeError):
    """
    Raised when text is given but no embedder is configured.

    ``vd`` operates on vectors. Passing raw text to ``collection[key] = text``
    or ``collection.search(text)`` only works when the :class:`Client` was
    created with an ``embedder``. Otherwise, pass a :class:`Document` with a
    ``vector`` (or a pre-computed query vector) directly.
    """


class BackendNotInstalledError(VdError, ImportError):
    """
    Raised when a known backend's Python package is not installed.

    Distinct from an *unknown* backend name (a plain ``ValueError``): the
    backend exists in ``vd``'s provider registry, but its client library is
    missing. The message carries the ``pip install`` command to fix it.
    """


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


@dataclass
class Document:
    """
    The unit stored in a :class:`Collection`.

    Parameters
    ----------
    id : str
        Unique identifier; the key under which the document lives in a
        collection.
    text : str
        The text content. May be empty for vector-first use cases where no
        text is associated with a vector.
    vector : list[float], optional
        The embedding. If ``None`` when written, the collection embeds
        ``text`` with its client's ``embedder`` — or raises
        :class:`EmbeddingRequiredError` if none is configured.
    metadata : dict
        Arbitrary metadata, used for filtering and carried through search
        results.

    Examples
    --------
    >>> doc = Document(id="doc1", text="Hello world")
    >>> doc.id, doc.text, doc.metadata
    ('doc1', 'Hello world', {})
    >>> Document(id="v1", vector=[0.1, 0.2]).text
    ''
    """

    id: str
    text: str = ""
    vector: Optional[Vector] = None
    metadata: Metadata = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Protocols — the contract clients code against
# --------------------------------------------------------------------------- #


@runtime_checkable
class Collection(Protocol):
    """
    A collection of documents: ``MutableMapping[str, Document]`` + ``search``.

    The mapping half is storage; :meth:`search` is the single retrieval
    extension. This minimal surface is everything ``vd``'s tooling depends on.
    Batch insertion is an *optional* capability — see :class:`SupportsBatch`.
    """

    def __setitem__(self, key: str, value: Union[str, tuple, Document]) -> None:
        """Insert or replace a document (idempotent upsert)."""
        ...

    def __getitem__(self, key: str) -> Document:
        """Return the document for ``key``; raise ``KeyError`` if absent."""
        ...

    def __delitem__(self, key: str) -> None:
        """Delete a document; raise ``KeyError`` if absent."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate document ids."""
        ...

    def __len__(self) -> int:
        """Return the number of documents."""
        ...

    def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> Iterator[SearchResult]:
        """Return the ``limit`` documents most similar to ``query``."""
        ...


@runtime_checkable
class Client(Protocol):
    """
    A live connection to one backend: ``Mapping[str, Collection]``.

    Collections are created explicitly (so create-time parameters such as
    ``dimension`` and ``metric`` can be supplied) and fetched either by
    :meth:`get_collection` or by mapping access ``client[name]``.
    """

    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> Collection:
        """Create a new collection; raise ``ValueError`` if it exists."""
        ...

    def get_collection(self, name: str) -> Collection:
        """Return an existing collection; raise ``KeyError`` if absent."""
        ...

    def delete_collection(self, name: str) -> None:
        """Drop a collection; raise ``KeyError`` if absent."""
        ...

    def list_collections(self) -> Iterator[str]:
        """Iterate collection names."""
        ...


# --------------------------------------------------------------------------- #
# Capability protocols — opt-in, feature-discovered with isinstance
# --------------------------------------------------------------------------- #


@runtime_checkable
class SupportsBatch(Protocol):
    """
    A collection that supports efficient batch insertion.

    ``add_documents`` and ``upsert`` are *not* part of the minimal
    :class:`Collection` contract. Every adapter built on
    :class:`AbstractCollection` happens to provide them, but generic code
    should still feature-discover::

        if isinstance(collection, SupportsBatch):
            collection.add_documents(many_docs, batch_size=256)
    """

    def add_documents(
        self, documents: Iterable[DocumentInput], *, batch_size: int = 100
    ) -> None: ...

    def upsert(self, document: Document) -> None: ...


@runtime_checkable
class SupportsHybrid(Protocol):
    """
    A collection that supports native hybrid (dense + lexical) search.

    Hybrid search has no syntactic convergence across vector databases, so it
    is an opt-in capability, never baseline. Prefer the top-level
    :func:`vd.hybrid_search` — it dispatches to this protocol when the
    collection implements it and falls back to a pure-Python BM25 + RRF
    fusion otherwise. Feature-discover directly only when you specifically
    need to refuse the fallback path::

        if isinstance(collection, SupportsHybrid):
            hits = collection.hybrid_search("query text", limit=20)

    The portable contract is **Reciprocal Rank Fusion** (every native backend
    supports it). Weighted-blend (``alpha``) and other backend-specific
    fusion variants are accepted via ``**kwargs`` and documented per adapter
    — they are not portable across backends.

    Parameters
    ----------
    query : str or list[float]
        Query text (embedded via the collection's embedder if configured)
        or a pre-computed query vector for the dense side.
    query_text : str, optional
        Explicit text for the lexical side. Defaults to ``query`` when
        ``query`` is a string. **Required** when ``query`` is a vector.
    limit : int
        Number of fused results to return.
    filter : dict, optional
        Canonical ``vd`` metadata filter applied to both sub-searches.
    k_dense, k_lexical : int, optional
        How many results to fetch from each sub-search before fusion.
        Both default to ``max(4 * limit, 50)``. Widen for higher recall.
    rrf_k : int
        Reciprocal Rank Fusion constant (typically 60).
    egress : callable, optional
        Transform applied to each fused result before it is yielded.
    **kwargs
        Backend-specific knobs (e.g. ``alpha=0.7`` on weaviate,
        ``ranker="weighted"`` on milvus). Documented per adapter.
    """

    def hybrid_search(
        self,
        query: Union[str, Vector],
        *,
        query_text: Optional[str] = None,
        limit: int = 10,
        filter: Optional[Filter] = None,
        k_dense: Optional[int] = None,
        k_lexical: Optional[int] = None,
        rrf_k: int = 60,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> Iterator[SearchResult]: ...


# --------------------------------------------------------------------------- #
# Async protocols — opt-in, every backend covered by the default wrapper
# --------------------------------------------------------------------------- #


@runtime_checkable
class AsyncCollection(Protocol):
    """
    The async sibling of :class:`Collection`.

    Same conceptual surface — storage + ``search`` — but every method is
    awaitable and iterators are :class:`~typing.AsyncIterator`. The mapping
    interface is exposed as explicit ``get`` / ``set`` / ``delete`` / ``keys``
    / ``count`` methods (the stdlib's ``MutableMapping`` ABC has no async
    counterpart; explicit methods are the Motor / aiopg convention).

    Construct via :func:`vd.connect_async`; the universal
    :class:`AsyncCollectionWrapper` in :mod:`vd.asynchronous` adapts every
    backend to this protocol by dispatching to the sync API through
    :func:`asyncio.to_thread`. Backends with native async SDKs override the
    wrapper and additionally satisfy :class:`SupportsNativeAsync`.
    """

    async def get(self, key: str) -> "Document": ...

    async def set(self, key: str, value: Union[str, tuple, "Document"]) -> None: ...

    async def delete(self, key: str) -> None: ...

    def keys(self) -> AsyncIterator[str]: ...

    async def count(self) -> int: ...

    def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> AsyncIterator[SearchResult]: ...


@runtime_checkable
class AsyncClient(Protocol):
    """
    The async sibling of :class:`Client`.

    Same operations — collection create / fetch / drop / list — exposed as
    awaitables and async iterators. Construct via :func:`vd.connect_async`.
    """

    async def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> AsyncCollection: ...

    async def get_collection(self, name: str) -> AsyncCollection: ...

    async def delete_collection(self, name: str) -> None: ...

    def list_collections(self) -> AsyncIterator[str]: ...


@runtime_checkable
class SupportsNativeAsync(Protocol):
    """
    Marker protocol set on async clients/collections that use a backend's
    native async SDK rather than the universal :func:`asyncio.to_thread`
    wrapper.

    Why care: in high-concurrency event-loop apps (FastAPI, Starlette, etc.),
    a ``to_thread``-wrapped backend still blocks a worker thread per request.
    For real non-blocking I/O, prefer collections that satisfy this protocol.
    The wrapper sets this attribute to ``False``; native adapters set it to
    ``True``. ``isinstance(c, SupportsNativeAsync)`` matches both — check
    ``c.native_async`` for the boolean.
    """

    native_async: bool


# --------------------------------------------------------------------------- #
# Helpers shared by the abstract bases
# --------------------------------------------------------------------------- #


def _to_float_list(vector: Any) -> Vector:
    """
    Coerce any vector-like (list, tuple, numpy array, ...) to ``list[float]``.

    Examples
    --------
    >>> _to_float_list((1, 2, 3))
    [1.0, 2.0, 3.0]
    """
    tolist = getattr(vector, "tolist", None)
    if tolist is not None:  # numpy array / pandas Series
        vector = tolist()
    return [float(x) for x in vector]


def _coerce_document(key: str, value: Union[str, tuple, Document]) -> Document:
    """
    Normalize a ``collection[key] = value`` assignment to a :class:`Document`.

    ``value`` may be raw text, a ``(text, ...)`` tuple, or a ``Document``. The
    resulting document's ``id`` is always ``key``.

    Examples
    --------
    >>> _coerce_document("a", "hello").text
    'hello'
    >>> _coerce_document("a", ("hello", {"k": 1})).metadata
    {'k': 1}
    >>> d = _coerce_document("a", Document(id="ignored", text="t"))
    >>> d.id
    'a'
    """
    if isinstance(value, Document):
        if value.id != key:
            value = Document(
                id=key, text=value.text, vector=value.vector, metadata=value.metadata
            )
        return value
    if isinstance(value, str):
        return Document(id=key, text=value)
    if isinstance(value, tuple):
        # (text, metadata) | (text, id) | (text, id, metadata) — id forced to key.
        if len(value) == 2:
            text, second = value
            metadata = second if isinstance(second, dict) else {}
            return Document(id=key, text=text, metadata=metadata)
        if len(value) == 3:
            text, _id, metadata = value
            return Document(id=key, text=text, metadata=metadata or {})
        if len(value) == 1:
            return Document(id=key, text=value[0])
    raise TypeError(
        f"Cannot interpret {type(value).__name__} as a document. Pass text, a "
        f"(text, metadata) tuple, or a vd.Document."
    )


# --------------------------------------------------------------------------- #
# AbstractCollection — adapter-author base
# --------------------------------------------------------------------------- #


class AbstractCollection(MutableMapping):
    """
    Base class implementing the :class:`Collection` contract for adapters.

    A backend subclasses this and implements the *raw primitives* below;
    everything users see is provided here, once, uniformly:

    - flexible ``__setitem__`` inputs (text / tuple / :class:`Document`),
    - optional text embedding when a ``Document`` arrives without a vector,
    - text-query embedding in :meth:`search`,
    - central filter validation against :attr:`supported_filter_operators`,
    - ``egress`` result transforms,
    - batch helpers (:meth:`add_documents`, :meth:`upsert`),
    - eager dimension-mismatch detection.

    Subclass responsibilities (raw primitives)
    ------------------------------------------
    ``_write(doc)``
        Upsert one document. Its ``vector`` is guaranteed non-``None`` and
        dimension-checked.
    ``_read(key) -> Document``
        Fetch one document; raise ``KeyError`` if absent.
    ``_drop(key)``
        Delete one document; raise ``KeyError`` if absent.
    ``_keys() -> Iterator[str]``
        Iterate document ids.
    ``_count() -> int``
        Number of documents.
    ``_query(vector, *, limit, filter, **kwargs) -> Iterable[SearchResult]``
        Raw nearest-neighbor search. ``filter`` is the canonical AST — the
        adapter translates it. Each result is a dict with at least ``id``,
        ``text``, ``score``, ``metadata``.

    Optional overrides
    ------------------
    ``_write_many(docs)``
        Efficient bulk upsert. Defaults to a loop over ``_write``.
    ``native`` (property)
        The raw backend collection handle (escape hatch).
    """

    #: Filter operators this backend can honor. Default: the full language.
    #: Adapters narrow this; :meth:`search` validates against it.
    supported_filter_operators: frozenset = frozenset()  # set in __init_subclass__

    #: Whether the backend accepts writes after creation. Static-index backends
    #: set this ``False`` and raise :class:`StaticIndexError` on write.
    supports_incremental_writes: bool = True

    # Instance attributes adapters are expected to set in their __init__:
    name: str = ""
    dimension: Optional[int] = None
    metric: str = "cosine"
    _embedder: Optional[Callable[[str], Vector]] = None

    def __init_subclass__(cls, **kwargs):
        # Default an adapter's supported operators to the full language unless
        # it declares its own narrower subset.
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("supported_filter_operators"):
            from vd.filters import SUPPORTED_FILTER_OPERATORS

            cls.supported_filter_operators = SUPPORTED_FILTER_OPERATORS

    # ----- embedding (optional convenience) -------------------------------- #

    @property
    def has_embedder(self) -> bool:
        """Whether a text->vector embedder is configured on this collection."""
        return self._embedder is not None

    def embed(self, text: str) -> Vector:
        """
        Embed ``text`` to a vector, or raise :class:`EmbeddingRequiredError`.

        This is the single place text becomes a vector inside ``vd``.
        """
        if self._embedder is None:
            raise EmbeddingRequiredError(
                f"Collection {self.name!r} has no embedder: cannot turn text "
                f"into a vector. Either create the client with "
                f"`vd.connect(backend, embedder=...)`, or pass a vd.Document "
                f"with a `vector` (and a pre-computed query vector to search)."
            )
        return _to_float_list(self._embedder(text))

    # ----- dimension bookkeeping ------------------------------------------ #

    def _vet_vector(self, vector: Any) -> Vector:
        """Coerce to ``list[float]`` and enforce/learn the collection dimension."""
        vec = _to_float_list(vector)
        if self.dimension is None:
            self.dimension = len(vec)
        elif len(vec) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch in collection {self.name!r}: got "
                f"{len(vec)}, expected {self.dimension}. A collection's "
                f"dimension is fixed once data is written; use a fresh "
                f"collection for a different embedding model."
            )
        return vec

    def _ensure_vector(self, doc: Document) -> Document:
        """Return ``doc`` with a vetted vector, embedding its text if needed."""
        vector = doc.vector if doc.vector is not None else self.embed(doc.text)
        doc.vector = self._vet_vector(vector)
        return doc

    def _resolve_query(self, query: Union[str, Vector]) -> Vector:
        """Turn a text-or-vector query into a vetted query vector."""
        vector = self.embed(query) if isinstance(query, str) else query
        return self._vet_vector(vector)

    def _resolve_hybrid_inputs(
        self,
        query: Union[str, Vector],
        query_text: Optional[str],
    ) -> tuple[Vector, str]:
        """
        Normalize ``(query, query_text)`` for ``hybrid_search`` into ``(vec, text)``.

        - ``query`` may be a string (used for both the dense and lexical sides
          when ``query_text`` is omitted) or a pre-computed vector (then
          ``query_text`` is **required**).
        - The returned ``vec`` is dimension-vetted; the returned ``text`` is
          guaranteed to be a non-empty string for the lexical side.
        """
        if isinstance(query, str):
            text = query_text if query_text is not None else query
            vec = self._vet_vector(self.embed(query))
        else:
            if query_text is None:
                raise ValueError(
                    "hybrid_search needs a `query_text` for the lexical side "
                    "when `query` is a vector. Either pass query_text=..., or "
                    "pass `query` as a string and let the embedder handle both."
                )
            text = query_text
            vec = self._vet_vector(query)
        if not text:
            raise ValueError("hybrid_search needs a non-empty lexical query string.")
        return vec, text

    # ----- MutableMapping interface --------------------------------------- #

    def __setitem__(self, key: str, value: Union[str, tuple, Document]) -> None:
        if not self.supports_incremental_writes:
            raise StaticIndexError(
                f"Collection {self.name!r} uses a static index and cannot "
                f"accept writes after creation. Rebuild it instead."
            )
        doc = self._ensure_vector(_coerce_document(key, value))
        self._write(doc)

    def __getitem__(self, key: str) -> Document:
        return self._read(key)

    def __delitem__(self, key: str) -> None:
        if not self.supports_incremental_writes:
            raise StaticIndexError(
                f"Collection {self.name!r} uses a static index and cannot "
                f"delete documents. Rebuild it instead."
            )
        self._drop(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return self._count()

    # ----- search --------------------------------------------------------- #

    def search(
        self,
        query: Union[str, Vector],
        *,
        limit: int = 10,
        filter: Optional[Filter] = None,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> Iterator[SearchResult]:
        """
        Return the ``limit`` documents most similar to ``query``.

        Parameters
        ----------
        query : str or list[float]
            Query text (embedded via the client's ``embedder``) or a
            pre-computed query vector.
        limit : int
            Maximum number of results.
        filter : dict, optional
            Metadata filter in the canonical ``vd`` dialect (see
            :mod:`vd.filters`). Validated against this backend's
            :attr:`supported_filter_operators` before the query runs, so an
            unsupported operator fails with a clear :class:`UnsupportedFilterError`.
        egress : callable, optional
            Transform applied to each result dict before it is yielded.
        **kwargs
            Backend-specific search options, passed through to ``_query``.

        Yields
        ------
        dict
            ``{"id", "text", "score", "metadata"}`` — or whatever ``egress``
            returns. ``score`` is a higher-is-better, per-metric canonical
            similarity (see the "Score semantics" table at the top of
            :mod:`vd.base`): cosine in ``[-1, 1]``, dot in ``(-inf, +inf)``,
            l2 squashed to ``(0, 1]``. Adapters whose backend returns a
            native combined-ranking score on a different scale (e.g.
            Elasticsearch, Atlas, Pinecone) document the deviation in
            their own docstring.
        """
        from vd.filters import validate_filter

        validate_filter(filter, supported=self.supported_filter_operators)
        query_vector = self._resolve_query(query)
        for result in self._query(query_vector, limit=limit, filter=filter, **kwargs):
            yield egress(result) if egress is not None else result

    # ----- hybrid orchestration (used by SupportsHybrid adapters) --------- #

    def _hybrid_via_rrf(
        self,
        query: Union[str, Vector],
        lexical_query: Callable[..., Iterable[SearchResult]],
        *,
        query_text: Optional[str] = None,
        limit: int = 10,
        filter: Optional[Filter] = None,
        k_dense: Optional[int] = None,
        k_lexical: Optional[int] = None,
        rrf_k: int = 60,
        egress: Optional[Callable[[SearchResult], Any]] = None,
        **kwargs,
    ) -> Iterator[SearchResult]:
        """
        Run dense ``_query`` + a backend-specific ``lexical_query`` and fuse with RRF.

        The pattern every adapter's :meth:`hybrid_search` calls. The adapter
        passes in its lexical primitive (``self._lexical_query``); this method
        handles input resolution, filter validation, over-fetch defaults, and
        Reciprocal Rank Fusion.

        ``lexical_query`` must accept ``(query_text, *, limit, filter, **kwargs)``
        and return an iterable of result dicts shaped like ``_query`` results.
        """
        from vd.filters import validate_filter
        from vd.search import _HYBRID_OVERFETCH_FLOOR, _rrf_fuse

        validate_filter(filter, supported=self.supported_filter_operators)
        vec, text = self._resolve_hybrid_inputs(query, query_text)
        k_dense_eff = (
            k_dense if k_dense is not None else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
        )
        k_lex_eff = (
            k_lexical
            if k_lexical is not None
            else max(4 * limit, _HYBRID_OVERFETCH_FLOOR)
        )
        # NOTE: ``**kwargs`` carry backend-specific native-fusion knobs (e.g.
        # ``alpha=0.7``). This client-RRF orchestration cannot honor them and
        # deliberately drops them rather than risk leaking them into the dense
        # or lexical sub-query calls (which would raise TypeError on most
        # backends). Adapters wanting native fusion should override
        # ``hybrid_search`` and call their backend's fused API directly.
        del kwargs  # unused on this path
        dense = list(self._query(vec, limit=k_dense_eff, filter=filter))
        lex = list(lexical_query(text, limit=k_lex_eff, filter=filter))
        for hit in _rrf_fuse([dense, lex], rrf_k=rrf_k, limit=limit):
            yield egress(hit) if egress is not None else hit

    # ----- batch convenience (also satisfies SupportsBatch) --------------- #

    def add_documents(
        self,
        documents: Iterable[DocumentInput],
        *,
        batch_size: int = 100,
    ) -> None:
        """
        Add many documents, embedding and writing them in batches.

        Each item may be a string, a ``(text, ...)`` tuple, or a
        :class:`Document` (see :data:`DocumentInput`). Items without an ``id``
        get a deterministic auto-generated one.
        """
        from vd.util import normalize_document_input

        batch: list[Document] = []
        for item in documents:
            doc = normalize_document_input(item, auto_id=True)
            self._ensure_vector(doc)
            batch.append(doc)
            if len(batch) >= batch_size:
                self._write_many(batch)
                batch = []
        if batch:
            self._write_many(batch)

    def upsert(self, document: Document) -> None:
        """Insert or replace ``document`` (equivalent to ``self[doc.id] = doc``)."""
        self[document.id] = document

    # ----- escape hatch --------------------------------------------------- #

    @property
    def native(self) -> Any:
        """
        The raw backend collection handle — a supported, documented escape hatch.

        Use it to reach backend-specific features the facade does not expose,
        rather than circumventing ``vd``. Returns ``None`` if the adapter has
        no distinct native object.
        """
        return getattr(self, "_native", None)

    # ----- raw primitives — adapters MUST implement ----------------------- #

    @abstractmethod
    def _write(self, doc: Document) -> None:
        """Upsert one document (its ``vector`` is set and dimension-checked)."""

    @abstractmethod
    def _read(self, key: str) -> Document:
        """Fetch one document; raise ``KeyError`` if absent."""

    @abstractmethod
    def _drop(self, key: str) -> None:
        """Delete one document; raise ``KeyError`` if absent."""

    @abstractmethod
    def _keys(self) -> Iterator[str]:
        """Iterate document ids."""

    @abstractmethod
    def _count(self) -> int:
        """Return the number of documents."""

    @abstractmethod
    def _query(
        self,
        vector: Vector,
        *,
        limit: int,
        filter: Optional[Filter],
        **kwargs,
    ) -> Iterable[SearchResult]:
        """Raw nearest-neighbor search returning result dicts."""

    # ----- raw primitives — adapters MAY override ------------------------- #

    def _write_many(self, docs: list[Document]) -> None:
        """Bulk upsert. Override for a backend with an efficient batch path."""
        for doc in docs:
            self._write(doc)


# --------------------------------------------------------------------------- #
# AbstractClient — adapter-author base
# --------------------------------------------------------------------------- #


class AbstractClient(Mapping):
    """
    Base class implementing the :class:`Client` contract for adapters.

    A :class:`Client` is a ``Mapping[str, Collection]``. A backend subclasses
    this and implements :meth:`create_collection`, :meth:`get_collection`,
    :meth:`delete_collection`, and :meth:`list_collections`; the mapping
    behavior, the :meth:`get_or_create_collection` convenience, the ``client``
    escape hatch, and context-manager support come for free.

    Parameters
    ----------
    embedder : callable, optional
        A ``text -> vector`` function. Passed to every collection so text
        inputs are accepted as a convenience. ``None`` (the default) makes the
        client vector-only.
    **config
        Backend-specific connection configuration.
    """

    #: The registry name of this backend (e.g. ``"chroma"``). Adapters set it.
    backend_name: str = ""

    def __init__(
        self,
        *,
        embedder: Optional[Callable[[str], Vector]] = None,
        **config,
    ):
        self._embedder = embedder
        self.config = config

    # ----- adapters MUST implement ---------------------------------------- #

    @abstractmethod
    def create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> Collection:
        """
        Create a new collection.

        Parameters
        ----------
        name : str
            Collection name.
        dimension : int, optional
            Vector dimension. May be ``None`` for backends that can infer it
            from the first written vector; required up front by backends that
            cannot.
        metric : str
            Distance metric: ``"cosine"``, ``"dot"``, or ``"l2"``.
        **index_config
            Backend-specific index tuning (HNSW ``M``/``ef``, IVF ``nlist``, ...).
            Documented per adapter; never abstracted into a common enum.

        Raises
        ------
        ValueError
            If a collection of that name already exists.
        """

    @abstractmethod
    def get_collection(self, name: str) -> Collection:
        """Return an existing collection; raise ``KeyError`` if absent."""

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Drop a collection; raise ``KeyError`` if absent."""

    @abstractmethod
    def list_collections(self) -> Iterator[str]:
        """Iterate collection names."""

    # ----- provided: Mapping interface ------------------------------------ #

    def __getitem__(self, name: str) -> Collection:
        return self.get_collection(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_collections())

    def __len__(self) -> int:
        return sum(1 for _ in self.list_collections())

    def __contains__(self, name: object) -> bool:
        try:
            self.get_collection(name)  # type: ignore[arg-type]
            return True
        except (KeyError, TypeError):
            return False

    # ----- provided: conveniences ----------------------------------------- #

    def get_or_create_collection(
        self,
        name: str,
        *,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        **index_config,
    ) -> Collection:
        """
        Return the collection ``name``, creating it if it does not exist.

        The common idiom that every consumer otherwise re-implements as a
        ``try get_collection / except KeyError: create_collection``.
        """
        try:
            return self.get_collection(name)
        except KeyError:
            return self.create_collection(
                name, dimension=dimension, metric=metric, **index_config
            )

    @property
    def client(self) -> Any:
        """
        The raw backend client — a supported, documented escape hatch.

        Drop to it for backend-specific operations the facade does not expose.
        Returns ``None`` for backends with no external client object (e.g. the
        in-memory backend).
        """
        return getattr(self, "_client", None)

    def close(self) -> None:
        """Release backend resources. Default no-op; adapters override as needed."""

    def __enter__(self) -> "AbstractClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
