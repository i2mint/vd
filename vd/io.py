"""
Import/export utilities for vd collections.

This module provides functions to export collections to various formats
and import data from different sources.
"""

import json
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from vd.base import Collection, Document


def export_to_jsonl(
    collection: Collection,
    output_path: Union[str, Path],
    *,
    include_vectors: bool = True,
) -> int:
    """
    Export a collection to JSONL (JSON Lines) format.

    Each line is a JSON object representing a document.

    Parameters
    ----------
    collection : Collection
        Collection to export
    output_path : str or Path
        Output file path
    include_vectors : bool, default True
        Whether to include embedding vectors in the export

    Returns
    -------
    int
        Number of documents exported

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> docs['doc1'] = "Hello"  # doctest: +SKIP
    >>> vd.export_to_jsonl(docs, 'backup.jsonl')  # doctest: +SKIP
    1
    """
    output_path = Path(output_path)
    count = 0

    with output_path.open('w', encoding='utf-8') as f:
        for doc_id in collection:
            doc = collection[doc_id]
            data = {
                'id': doc.id,
                'text': doc.text,
                'metadata': doc.metadata,
            }
            if include_vectors and doc.vector:
                data['vector'] = doc.vector

            f.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1

    return count


def import_from_jsonl(
    collection: Collection,
    input_path: Union[str, Path],
    *,
    batch_size: int = 100,
    skip_existing: bool = False,
) -> int:
    """
    Import documents from JSONL format into a collection.

    Parameters
    ----------
    collection : Collection
        Collection to import into
    input_path : str or Path
        Input file path
    batch_size : int, default 100
        Batch size for adding documents
    skip_existing : bool, default False
        If True, skip documents with IDs that already exist

    Returns
    -------
    int
        Number of documents imported

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> vd.import_from_jsonl(docs, 'backup.jsonl')  # doctest: +SKIP
    1
    """
    input_path = Path(input_path)
    count = 0
    batch = []

    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            # Check if we should skip
            if skip_existing and data['id'] in collection:
                continue

            # Create document
            doc = Document(
                id=data['id'],
                text=data['text'],
                vector=data.get('vector'),
                metadata=data.get('metadata', {}),
            )

            batch.append(doc)
            count += 1

            # Flush batch
            if len(batch) >= batch_size:
                collection.add_documents(batch)
                batch = []

    # Flush remaining
    if batch:
        collection.add_documents(batch)

    return count


def export_to_json(
    collection: Collection,
    output_path: Union[str, Path],
    *,
    include_vectors: bool = True,
    indent: Optional[int] = 2,
) -> int:
    """
    Export a collection to JSON format.

    Creates a JSON array of all documents.

    Parameters
    ----------
    collection : Collection
        Collection to export
    output_path : str or Path
        Output file path
    include_vectors : bool, default True
        Whether to include embedding vectors
    indent : int, optional
        JSON indentation (None for compact)

    Returns
    -------
    int
        Number of documents exported
    """
    output_path = Path(output_path)
    documents = []

    for doc_id in collection:
        doc = collection[doc_id]
        data = {
            'id': doc.id,
            'text': doc.text,
            'metadata': doc.metadata,
        }
        if include_vectors and doc.vector:
            data['vector'] = doc.vector

        documents.append(data)

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=indent)

    return len(documents)


def import_from_json(
    collection: Collection,
    input_path: Union[str, Path],
    *,
    batch_size: int = 100,
    skip_existing: bool = False,
) -> int:
    """
    Import documents from JSON format into a collection.

    Parameters
    ----------
    collection : Collection
        Collection to import into
    input_path : str or Path
        Input file path
    batch_size : int, default 100
        Batch size for adding documents
    skip_existing : bool, default False
        If True, skip documents with IDs that already exist

    Returns
    -------
    int
        Number of documents imported
    """
    input_path = Path(input_path)

    with input_path.open('r', encoding='utf-8') as f:
        documents_data = json.load(f)

    count = 0
    batch = []

    for data in documents_data:
        # Check if we should skip
        if skip_existing and data['id'] in collection:
            continue

        # Create document
        doc = Document(
            id=data['id'],
            text=data['text'],
            vector=data.get('vector'),
            metadata=data.get('metadata', {}),
        )

        batch.append(doc)
        count += 1

        # Flush batch
        if len(batch) >= batch_size:
            collection.add_documents(batch)
            batch = []

    # Flush remaining
    if batch:
        collection.add_documents(batch)

    return count


def export_to_directory(
    collection: Collection,
    output_dir: Union[str, Path],
    *,
    include_vectors: bool = True,
) -> int:
    """
    Export collection as a directory with one JSON file per document.

    Useful for version control and easy browsing.

    Parameters
    ----------
    collection : Collection
        Collection to export
    output_dir : str or Path
        Output directory path
    include_vectors : bool, default True
        Whether to include vectors

    Returns
    -------
    int
        Number of documents exported
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export metadata
    metadata = {
        'collection_name': getattr(collection, 'name', 'unknown'),
        'total_documents': len(collection),
    }
    (output_dir / '_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )

    count = 0
    for doc_id in collection:
        doc = collection[doc_id]
        data = {
            'id': doc.id,
            'text': doc.text,
            'metadata': doc.metadata,
        }
        if include_vectors and doc.vector:
            data['vector'] = doc.vector

        # Sanitize filename
        safe_id = doc_id.replace('/', '_').replace('\\', '_')
        file_path = output_dir / f"{safe_id}.json"

        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        count += 1

    return count


def import_from_directory(
    collection: Collection,
    input_dir: Union[str, Path],
    *,
    batch_size: int = 100,
    skip_existing: bool = False,
    pattern: str = '*.json',
) -> int:
    """
    Import documents from a directory of JSON files.

    Parameters
    ----------
    collection : Collection
        Collection to import into
    input_dir : str or Path
        Input directory path
    batch_size : int, default 100
        Batch size for adding documents
    skip_existing : bool, default False
        If True, skip documents with IDs that already exist
    pattern : str, default '*.json'
        File pattern to match

    Returns
    -------
    int
        Number of documents imported
    """
    input_dir = Path(input_dir)
    count = 0
    batch = []

    for file_path in input_dir.glob(pattern):
        # Skip metadata file
        if file_path.name == '_metadata.json':
            continue

        data = json.loads(file_path.read_text(encoding='utf-8'))

        # Check if we should skip
        if skip_existing and data['id'] in collection:
            continue

        # Create document
        doc = Document(
            id=data['id'],
            text=data['text'],
            vector=data.get('vector'),
            metadata=data.get('metadata', {}),
        )

        batch.append(doc)
        count += 1

        # Flush batch
        if len(batch) >= batch_size:
            collection.add_documents(batch)
            batch = []

    # Flush remaining
    if batch:
        collection.add_documents(batch)

    return count


# Convenience functions to add to Collection (can be monkey-patched or used via helper)

def export_collection(
    collection: Collection,
    output_path: Union[str, Path],
    *,
    format: str = 'jsonl',
    **kwargs,
) -> int:
    """
    Export a collection to a file in the specified format.

    Parameters
    ----------
    collection : Collection
        Collection to export
    output_path : str or Path
        Output file/directory path
    format : str
        Export format: 'jsonl', 'json', 'directory'
    **kwargs
        Additional format-specific options

    Returns
    -------
    int
        Number of documents exported

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> vd.export_collection(docs, 'backup.jsonl')  # doctest: +SKIP
    """
    if format == 'jsonl':
        return export_to_jsonl(collection, output_path, **kwargs)
    elif format == 'json':
        return export_to_json(collection, output_path, **kwargs)
    elif format == 'directory':
        return export_to_directory(collection, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


def import_collection(
    collection: Collection,
    input_path: Union[str, Path],
    *,
    format: Optional[str] = None,
    **kwargs,
) -> int:
    """
    Import documents into a collection from a file.

    Parameters
    ----------
    collection : Collection
        Collection to import into
    input_path : str or Path
        Input file/directory path
    format : str, optional
        Import format: 'jsonl', 'json', 'directory'
        If None, inferred from file extension
    **kwargs
        Additional format-specific options

    Returns
    -------
    int
        Number of documents imported

    Examples
    --------
    >>> import vd
    >>> client = vd.connect('memory')  # doctest: +SKIP
    >>> docs = client.create_collection('test')  # doctest: +SKIP
    >>> vd.import_collection(docs, 'backup.jsonl')  # doctest: +SKIP
    """
    input_path = Path(input_path)

    # Infer format if not specified
    if format is None:
        if input_path.is_dir():
            format = 'directory'
        elif input_path.suffix == '.jsonl':
            format = 'jsonl'
        elif input_path.suffix == '.json':
            format = 'json'
        else:
            raise ValueError(f"Cannot infer format from path: {input_path}")

    if format == 'jsonl':
        return import_from_jsonl(collection, input_path, **kwargs)
    elif format == 'json':
        return import_from_json(collection, input_path, **kwargs)
    elif format == 'directory':
        return import_from_directory(collection, input_path, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")
