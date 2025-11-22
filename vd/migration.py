"""
Migration utilities for moving data between backends.

Provides functions to migrate collections between different vector database
backends while preserving all data, metadata, and embeddings.
"""

from typing import Any, Callable, Optional, Union

from vd.base import Client, Collection, Document


def migrate_collection(
    source_collection: Collection,
    target_collection: Collection,
    *,
    batch_size: int = 100,
    preserve_vectors: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """
    Migrate a collection from one backend to another.

    Parameters
    ----------
    source_collection : Collection
        Source collection to migrate from
    target_collection : Collection
        Target collection to migrate to
    batch_size : int, default 100
        Number of documents to migrate per batch
    preserve_vectors : bool, default True
        Whether to preserve pre-computed vectors
    progress_callback : callable, optional
        Function called with (current, total) to report progress
    skip_existing : bool, default False
        If True, skip documents that already exist in target

    Returns
    -------
    dict
        Migration statistics with keys:
        - total: Total documents in source
        - migrated: Number of documents migrated
        - skipped: Number of documents skipped
        - failed: Number of failures
        - errors: List of error messages

    Examples
    --------
    >>> import vd
    >>> # Create source and target
    >>> source_client = vd.connect('memory')  # doctest: +SKIP
    >>> target_client = vd.connect('chroma', persist_directory='./data')  # doctest: +SKIP
    >>> source = source_client.get_collection('my_docs')  # doctest: +SKIP
    >>> target = target_client.create_collection('my_docs')  # doctest: +SKIP
    >>>
    >>> # Migrate
    >>> stats = vd.migrate_collection(source, target)  # doctest: +SKIP
    >>> print(f"Migrated {stats['migrated']} documents")  # doctest: +SKIP
    """
    stats = {
        'total': len(source_collection),
        'migrated': 0,
        'skipped': 0,
        'failed': 0,
        'errors': [],
    }

    batch = []
    processed = 0

    for doc_id in source_collection:
        try:
            # Check if should skip
            if skip_existing and doc_id in target_collection:
                stats['skipped'] += 1
                processed += 1
                if progress_callback:
                    progress_callback(processed, stats['total'])
                continue

            # Get source document
            doc = source_collection[doc_id]

            # Optionally clear vectors to force re-embedding
            if not preserve_vectors:
                doc.vector = None

            batch.append(doc)

            # Flush batch
            if len(batch) >= batch_size:
                target_collection.add_documents(batch)
                stats['migrated'] += len(batch)
                processed += len(batch)
                batch = []

                if progress_callback:
                    progress_callback(processed, stats['total'])

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"Error migrating {doc_id}: {str(e)}")
            processed += 1
            if progress_callback:
                progress_callback(processed, stats['total'])

    # Flush remaining
    if batch:
        try:
            target_collection.add_documents(batch)
            stats['migrated'] += len(batch)
        except Exception as e:
            stats['failed'] += len(batch)
            stats['errors'].append(f"Error in final batch: {str(e)}")

    return stats


def migrate_client(
    source_client: Client,
    target_client: Client,
    *,
    collection_names: Optional[list[str]] = None,
    batch_size: int = 100,
    preserve_vectors: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> dict[str, Any]:
    """
    Migrate all (or selected) collections from one client to another.

    Parameters
    ----------
    source_client : Client
        Source database client
    target_client : Client
        Target database client
    collection_names : list of str, optional
        Specific collections to migrate. If None, migrates all.
    batch_size : int, default 100
        Batch size for migration
    preserve_vectors : bool, default True
        Whether to preserve vectors
    progress_callback : callable, optional
        Function called with (collection_name, current, total)

    Returns
    -------
    dict
        Overall migration statistics

    Examples
    --------
    >>> import vd
    >>> source = vd.connect('memory')  # doctest: +SKIP
    >>> target = vd.connect('chroma', persist_directory='./backup')  # doctest: +SKIP
    >>> stats = vd.migrate_client(source, target)  # doctest: +SKIP
    """
    # Get collections to migrate
    if collection_names is None:
        collection_names = list(source_client.list_collections())

    overall_stats = {
        'collections_total': len(collection_names),
        'collections_migrated': 0,
        'collections_failed': 0,
        'total_documents': 0,
        'migrated_documents': 0,
        'failed_documents': 0,
        'errors': [],
    }

    for coll_name in collection_names:
        try:
            # Get source collection
            source_coll = source_client.get_collection(coll_name)

            # Create or get target collection
            try:
                target_coll = target_client.create_collection(coll_name)
            except ValueError:
                # Collection exists, get it
                target_coll = target_client.get_collection(coll_name)

            # Create progress callback for this collection
            coll_callback = None
            if progress_callback:
                coll_callback = lambda c, t: progress_callback(coll_name, c, t)

            # Migrate
            coll_stats = migrate_collection(
                source_coll,
                target_coll,
                batch_size=batch_size,
                preserve_vectors=preserve_vectors,
                progress_callback=coll_callback,
            )

            overall_stats['collections_migrated'] += 1
            overall_stats['total_documents'] += coll_stats['total']
            overall_stats['migrated_documents'] += coll_stats['migrated']
            overall_stats['failed_documents'] += coll_stats['failed']
            overall_stats['errors'].extend(coll_stats['errors'])

        except Exception as e:
            overall_stats['collections_failed'] += 1
            overall_stats['errors'].append(f"Failed to migrate collection {coll_name}: {str(e)}")

    return overall_stats


def copy_collection(
    source: tuple[str, str] | Collection,
    target: tuple[str, str, dict] | Collection,
    *,
    batch_size: int = 100,
    preserve_vectors: bool = True,
) -> dict[str, Any]:
    """
    Copy a collection with flexible source/target specification.

    Parameters
    ----------
    source : tuple or Collection
        Either a Collection object or (backend_name, collection_name) tuple
    target : tuple or Collection
        Either a Collection object or (backend_name, collection_name, config) tuple
    batch_size : int
        Batch size for copying
    preserve_vectors : bool
        Whether to preserve vectors

    Returns
    -------
    dict
        Migration statistics

    Examples
    --------
    >>> import vd
    >>> # Copy between backends
    >>> stats = vd.copy_collection(  # doctest: +SKIP
    ...     source=('memory', 'docs'),
    ...     target=('chroma', 'docs', {'persist_directory': './data'}),
    ... )
    """
    from vd import connect

    # Resolve source
    if isinstance(source, tuple):
        backend_name, coll_name = source
        client = connect(backend_name)
        source_coll = client.get_collection(coll_name)
    else:
        source_coll = source

    # Resolve target
    if isinstance(target, tuple):
        if len(target) == 3:
            backend_name, coll_name, config = target
        else:
            backend_name, coll_name = target
            config = {}
        client = connect(backend_name, **config)
        try:
            target_coll = client.create_collection(coll_name)
        except ValueError:
            target_coll = client.get_collection(coll_name)
    else:
        target_coll = target

    return migrate_collection(
        source_coll,
        target_coll,
        batch_size=batch_size,
        preserve_vectors=preserve_vectors,
    )
