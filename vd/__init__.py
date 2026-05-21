"""
``vd`` — one Pythonic interface to every vector database.

``vd`` is a **facade over vector databases**. Its purpose is to let you operate
on any vectorDB, and switch between them with a one-argument change, while
keeping each backend's particular power one escape hatch away. It does three
things:

1. **Choose** — :func:`recommend_backend`, :func:`print_backends_table` and the
   provider registry help you (or an AI agent) pick the right backend.
2. **Set up** — :func:`check_requirements` and :func:`setup_guide` diagnose and
   walk you through installing and starting a backend.
3. **Operate** — :func:`connect` returns a uniform client; collections behave
   as ``MutableMapping`` of :class:`Document` plus a :meth:`search` method.

Quick start
-----------
>>> import vd
>>> client = vd.connect('memory')          # switch DB = change this one word
>>> col = client.create_collection('docs')
>>> col['a'] = vd.Document(id='a', text='cats', vector=[1.0, 0.0])
>>> col['b'] = vd.Document(id='b', text='dogs', vector=[0.0, 1.0])
>>> [hit['id'] for hit in col.search([0.9, 0.1], limit=1)]
['a']

Embedding is external
---------------------
``vd`` stores and searches *vectors*. Turning text into vectors is another
package's job (e.g. ``ef``). Pass an ``embedder`` to :func:`connect` only for
the *convenience* of writing/searching raw text; otherwise pass
:class:`Document` objects carrying vectors, and pre-computed query vectors.
"""

from __future__ import annotations

from pathlib import Path as _Path

__version__ = "0.2.0"


def skills_dir() -> _Path:
    """Return the path to the bundled AI-agent skills directory."""
    return _Path(__file__).parent / "data" / "skills"


# Importing this package registers every installed backend adapter.
import vd.backends  # noqa: E402,F401

# ----- core contracts & data model ----------------------------------------- #
from vd.base import (  # noqa: E402
    AbstractClient,
    AbstractCollection,
    BackendNotInstalledError,
    Client,
    Collection,
    Document,
    DocumentInput,
    EmbeddingRequiredError,
    Filter,
    Metadata,
    SearchResult,
    StaticIndexError,
    SupportsBatch,
    SupportsHybrid,
    UnsupportedCapabilityError,
    UnsupportedFilterError,
    VdError,
    Vector,
)

# ----- the entry point & registry ------------------------------------------ #
from vd.util import (  # noqa: E402
    connect,
    cosine_similarity,
    euclidean_distance,
    id_and_score,
    id_only,
    id_text_score,
    list_backends,
    normalize_document_input,
    register_backend,
    text_only,
)

# ----- canonical metadata-filter language ---------------------------------- #
from vd.filters import (  # noqa: E402
    SUPPORTED_FILTER_OPERATORS,
    matches_filter,
    validate_filter,
)

# ----- choosing a backend (provider registry) ------------------------------ #
from vd.providers import (  # noqa: E402
    compare_backends,
    get_backend_characteristics,
    get_backend_info,
    get_install_instructions,
    install_command,
    list_all_backends,
    list_available_backends,
    print_backends_table,
    print_comparison,
    print_recommendation,
    provider,
    providers,
    recommend_backend,
)

# ----- setting a backend up ------------------------------------------------ #
from vd.requirements import (  # noqa: E402
    check_requirements,
    install_backend,
    setup_guide,
)

# ----- import / export ----------------------------------------------------- #
from vd.io import (  # noqa: E402
    export_collection,
    export_to_directory,
    export_to_json,
    export_to_jsonl,
    import_collection,
    import_from_directory,
    import_from_json,
    import_from_jsonl,
)

# ----- migration between backends ------------------------------------------ #
from vd.migration import (  # noqa: E402
    copy_collection,
    migrate_client,
    migrate_collection,
)

# ----- analytics ----------------------------------------------------------- #
from vd.analytics import (  # noqa: E402
    collection_stats,
    find_duplicates,
    find_outliers,
    metadata_distribution,
    sample_collection,
    validate_collection,
)

# ----- text preprocessing (a convenience; ef owns real segmentation) ------- #
from vd.text import (  # noqa: E402
    chunk_documents,
    chunk_text,
    clean_text,
    extract_metadata,
    normalize_whitespace,
    truncate_text,
)

# ----- health & benchmarking ----------------------------------------------- #
from vd.health import (  # noqa: E402
    benchmark_insert,
    benchmark_search,
    health_check_backend,
    health_check_collection,
)

# ----- advanced search ----------------------------------------------------- #
from vd.search import (  # noqa: E402
    deduplicate_results,
    multi_query_search,
    reciprocal_rank_fusion,
    search_similar_to_document,
)

# ----- configuration files ------------------------------------------------- #
from vd.config import (  # noqa: E402
    connect_from_config,
    create_example_config,
    load_config,
    save_config,
)

# ----- time-indexed wrapper ------------------------------------------------ #
from vd.time_indexed import (  # noqa: E402
    TimeIndexedCollection,
    TimestampLike,
    WindowSlice,
    count_docs,
    mean_vector,
    parse_window,
    to_datetime,
    to_iso,
)

__all__ = [
    "skills_dir",
    "__version__",
    # entry point
    "connect",
    "register_backend",
    # core contracts & data model
    "Document",
    "Client",
    "Collection",
    "AbstractClient",
    "AbstractCollection",
    "Vector",
    "Filter",
    "Metadata",
    "SearchResult",
    "DocumentInput",
    # exceptions
    "VdError",
    "StaticIndexError",
    "UnsupportedFilterError",
    "UnsupportedCapabilityError",
    "EmbeddingRequiredError",
    "BackendNotInstalledError",
    # capability protocols
    "SupportsBatch",
    "SupportsHybrid",
    # filter language
    "matches_filter",
    "validate_filter",
    "SUPPORTED_FILTER_OPERATORS",
    # choosing a backend
    "list_backends",
    "list_available_backends",
    "list_all_backends",
    "print_backends_table",
    "providers",
    "provider",
    "get_backend_info",
    "get_backend_characteristics",
    "get_install_instructions",
    "install_command",
    "compare_backends",
    "print_comparison",
    "recommend_backend",
    "print_recommendation",
    # setting up a backend
    "check_requirements",
    "setup_guide",
    "install_backend",
    # egress functions
    "text_only",
    "id_only",
    "id_and_score",
    "id_text_score",
    # vector math
    "cosine_similarity",
    "euclidean_distance",
    "normalize_document_input",
    # import / export
    "export_collection",
    "import_collection",
    "export_to_jsonl",
    "import_from_jsonl",
    "export_to_json",
    "import_from_json",
    "export_to_directory",
    "import_from_directory",
    # migration
    "migrate_collection",
    "migrate_client",
    "copy_collection",
    # analytics
    "collection_stats",
    "metadata_distribution",
    "find_duplicates",
    "find_outliers",
    "sample_collection",
    "validate_collection",
    # text preprocessing
    "chunk_text",
    "chunk_documents",
    "clean_text",
    "normalize_whitespace",
    "truncate_text",
    "extract_metadata",
    # health
    "health_check_backend",
    "health_check_collection",
    "benchmark_search",
    "benchmark_insert",
    # advanced search
    "multi_query_search",
    "search_similar_to_document",
    "reciprocal_rank_fusion",
    "deduplicate_results",
    # configuration
    "load_config",
    "save_config",
    "connect_from_config",
    "create_example_config",
    # time-indexed wrapper
    "TimeIndexedCollection",
    "TimestampLike",
    "WindowSlice",
    "count_docs",
    "mean_vector",
    "parse_window",
    "to_datetime",
    "to_iso",
]
