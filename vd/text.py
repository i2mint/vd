"""
Text preprocessing and chunking utilities for vd.

Provides functions to clean, normalize, and chunk text before adding to
vector databases.
"""

import re
from typing import Any, Iterator, Optional


def clean_text(
    text: str,
    *,
    lowercase: bool = False,
    remove_extra_whitespace: bool = True,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_numbers: bool = False,
    remove_punctuation: bool = False,
) -> str:
    """
    Clean and normalize text.

    Parameters
    ----------
    text : str
        Text to clean
    lowercase : bool, default False
        Convert to lowercase
    remove_extra_whitespace : bool, default True
        Collapse multiple spaces/newlines
    remove_urls : bool, default False
        Remove URLs
    remove_emails : bool, default False
        Remove email addresses
    remove_numbers : bool, default False
        Remove numbers
    remove_punctuation : bool, default False
        Remove punctuation

    Returns
    -------
    str
        Cleaned text

    Examples
    --------
    >>> text = "Hello   World!  Visit https://example.com"
    >>> clean_text(text, remove_urls=True)
    'Hello World! Visit'
    >>> clean_text(text, lowercase=True, remove_punctuation=True)
    'hello world visit https examplecom'
    """
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

    if remove_emails:
        text = re.sub(r'\S+@\S+', '', text)

    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    if lowercase:
        text = text.lower()

    if remove_extra_whitespace:
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        # Collapse multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()

    return text


def chunk_text(
    text: str,
    chunk_size: int = 500,
    *,
    overlap: int = 50,
    strategy: str = 'chars',
    preserve_sentences: bool = True,
) -> list[str]:
    """
    Chunk text into smaller pieces.

    Parameters
    ----------
    text : str
        Text to chunk
    chunk_size : int, default 500
        Target size of each chunk (in characters or tokens depending on strategy)
    overlap : int, default 50
        Number of characters/tokens to overlap between chunks
    strategy : str, default 'chars'
        Chunking strategy:
        - 'chars': Character-based chunking
        - 'words': Word-based chunking
        - 'sentences': Sentence-based chunking
        - 'paragraphs': Paragraph-based chunking
    preserve_sentences : bool, default True
        Try to avoid breaking sentences when using chars/words strategy

    Returns
    -------
    list of str
        List of text chunks

    Examples
    --------
    >>> text = "This is sentence one. This is sentence two. This is sentence three."
    >>> chunks = chunk_text(text, chunk_size=30, strategy='chars')
    >>> len(chunks) >= 2
    True

    >>> chunks = chunk_text(text, strategy='sentences')
    >>> len(chunks)
    3
    """
    if not text or chunk_size <= 0:
        return []

    if strategy == 'paragraphs':
        # Split on double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    elif strategy == 'sentences':
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    elif strategy == 'words':
        words = text.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append(' '.join(chunk_words))
            i += chunk_size - overlap

        return chunks

    elif strategy == 'chars':
        if not preserve_sentences:
            # Simple character-based chunking
            chunks = []
            i = 0

            while i < len(text):
                chunk = text[i : i + chunk_size]
                chunks.append(chunk)
                i += chunk_size - overlap

            return chunks

        # Try to preserve sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size
            if current_length + sentence_len > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Take last part for overlap
                    overlap_text = ' '.join(current_chunk)
                    if len(overlap_text) > overlap:
                        overlap_text = overlap_text[-overlap:]
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_len + 1
                else:
                    current_chunk = [sentence]
                    current_length = sentence_len
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + (1 if current_chunk else 0)

        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def chunk_documents(
    documents: Iterator[tuple[str, str | dict]],
    chunk_size: int = 500,
    *,
    overlap: int = 50,
    strategy: str = 'chars',
    id_template: str = '{doc_id}_chunk_{chunk_num}',
    preserve_metadata: bool = True,
) -> Iterator[tuple[str, str, dict]]:
    """
    Chunk multiple documents while preserving metadata.

    Parameters
    ----------
    documents : iterator of tuples
        Iterator of (doc_id, text) or (doc_id, text, metadata) tuples
    chunk_size : int
        Size of each chunk
    overlap : int
        Overlap between chunks
    strategy : str
        Chunking strategy (see chunk_text)
    id_template : str
        Template for chunk IDs. Can use {doc_id} and {chunk_num}
    preserve_metadata : bool
        Whether to copy metadata to all chunks

    Yields
    ------
    tuple
        (chunk_id, chunk_text, metadata) tuples

    Examples
    --------
    >>> docs = [('doc1', 'Long text...', {'author': 'Alice'})]
    >>> chunks = list(chunk_documents(docs, chunk_size=20))  # doctest: +SKIP
    >>> len(chunks) >= 1  # doctest: +SKIP
    True
    """
    for doc_data in documents:
        if len(doc_data) == 2:
            doc_id, text = doc_data
            metadata = {}
        else:
            doc_id, text, metadata = doc_data

        # Chunk the text
        chunks = chunk_text(text, chunk_size, overlap=overlap, strategy=strategy)

        # Yield chunks with IDs and metadata
        for i, chunk in enumerate(chunks):
            chunk_id = id_template.format(doc_id=doc_id, chunk_num=i)

            chunk_metadata = metadata.copy() if preserve_metadata else {}
            chunk_metadata['chunk_num'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            chunk_metadata['source_doc_id'] = doc_id

            yield (chunk_id, chunk, chunk_metadata)


def extract_metadata(
    text: str,
    *,
    extract_title: bool = True,
    extract_length: bool = True,
    extract_word_count: bool = True,
    extract_language: bool = False,
) -> dict[str, Any]:
    """
    Extract metadata from text.

    Parameters
    ----------
    text : str
        Text to analyze
    extract_title : bool
        Extract first line as title
    extract_length : bool
        Add text length
    extract_word_count : bool
        Add word count
    extract_language : bool
        Detect language (requires langdetect)

    Returns
    -------
    dict
        Extracted metadata

    Examples
    --------
    >>> text = "My Title\\n\\nThis is the content."
    >>> meta = extract_metadata(text)
    >>> meta['title']
    'My Title'
    >>> meta['char_count']
    28
    """
    metadata = {}

    if extract_title:
        lines = text.strip().split('\n')
        if lines:
            metadata['title'] = lines[0].strip()[:200]  # Limit title length

    if extract_length:
        metadata['char_count'] = len(text)

    if extract_word_count:
        metadata['word_count'] = len(text.split())

    if extract_language:
        try:
            from langdetect import detect

            try:
                metadata['language'] = detect(text)
            except:
                metadata['language'] = 'unknown'
        except ImportError:
            pass

    return metadata


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Replaces tabs, multiple spaces, and multiple newlines with single versions.

    Parameters
    ----------
    text : str
        Text to normalize

    Returns
    -------
    str
        Normalized text

    Examples
    --------
    >>> normalize_whitespace("Hello\\t\\tWorld  \\n\\n\\nTest")
    'Hello World \\nTest'
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    # Collapse multiple newlines to single
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def truncate_text(text: str, max_length: int, *, suffix: str = '...') -> str:
    """
    Truncate text to maximum length.

    Parameters
    ----------
    text : str
        Text to truncate
    max_length : int
        Maximum length
    suffix : str
        Suffix to add to truncated text

    Returns
    -------
    str
        Truncated text

    Examples
    --------
    >>> truncate_text("This is a long text", 10)
    'This is...'
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
