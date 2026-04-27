---
name: vd-ingest
description: >-
  Corpus-ingestion tooling for the vd package. Use this skill when the user has
  documents, files, articles, or long text they need to load into a vd
  collection — including cleaning text, splitting it into chunks, attaching
  metadata, and adding the result in batches. Trigger on requests like "load
  these docs into a vector DB", "chunk this text", "preprocess before
  embedding", or "bulk insert".
audience: users
---

# Ingesting a corpus into vd

Most useful work in `vd` starts with: "I have N documents, get them indexed."
This skill covers the ingest pipeline:

```
raw text  →  clean_text  →  chunk_documents  →  add_documents  →  collection
```

You will not always need every step. For a small set of already-clean snippets,
just use `__setitem__` or `add_documents` directly (see `vd-quickstart`). Reach
for this skill when the user's input is messy (HTML, URLs, mixed whitespace),
long (chunking needed for embedding context windows), or large enough that
batch insertion matters.

## Cleaning text

```python
import vd

clean = vd.clean_text(
    raw,
    lowercase=False,                # default — keep case unless asked
    remove_extra_whitespace=True,   # default
    remove_urls=False,
    remove_emails=False,
    remove_numbers=False,
    remove_punctuation=False,
)
```

All cleaning flags are keyword-only and default to off (except
`remove_extra_whitespace=True`). Don't lowercase or strip punctuation by
default — modern embedding models handle them, and dropping them can hurt
search quality (e.g. "AI" vs. "ai", or sentence boundaries).

For a lighter touch, just normalize whitespace:

```python
vd.normalize_whitespace(text)   # collapse repeated spaces / newlines
vd.truncate_text(text, max_length=2000, suffix='...')
```

## Chunking text

Long documents must be split before embedding — most embedding models cap
context around 8k tokens, and similarity is more meaningful on focused chunks.

```python
chunks: list[str] = vd.chunk_text(
    text,
    chunk_size=500,             # positional or kw
    overlap=50,                 # kw — chars overlap between consecutive chunks
    strategy='chars',           # 'chars' | 'words' | 'sentences' | 'paragraphs'
    preserve_sentences=True,    # try to break on sentence boundary
)
```

Picking a strategy:

- `'chars'` — simplest, predictable size, but cuts mid-word. Default.
- `'words'` — word-level boundary, good for prose.
- `'sentences'` — best when retrieval needs to surface coherent statements.
- `'paragraphs'` — each chunk is one paragraph; only useful when paragraphs
  are short enough to fit `chunk_size`.

`overlap` keeps context across cut boundaries. ~10% of `chunk_size` is a
reasonable default. Don't set `overlap >= chunk_size` (you'll loop or get
empty chunks).

## Chunking many documents at once

When you have a stream of `(doc_id, text_or_metadata)` items and want chunked
output with metadata preserved, use `chunk_documents`:

```python
documents = [
    ('article_1', 'Long article text ...'),
    ('article_2', ('Another long body...', {'author': 'Alice', 'year': 2024})),
]

chunked = vd.chunk_documents(
    documents,
    chunk_size=500,
    overlap=50,
    strategy='sentences',
    id_template='{doc_id}_chunk_{chunk_num}',  # how to mint chunk IDs
    preserve_metadata=True,                    # default — copy parent metadata
)
# chunked yields tuples of (chunk_id, chunk_text, metadata_dict)
```

The yielded metadata also gets `chunk_num`, `total_chunks`, and `parent_id`
fields added so downstream queries can reassemble or filter.

## Extracting metadata from text

`extract_metadata` pulls a small set of cheap features:

```python
meta = vd.extract_metadata(
    text,
    extract_title=True,        # first heading or first line
    extract_length=True,       # character count
    extract_word_count=True,
    extract_language=False,    # off by default — can be slow / requires deps
)
# {'title': '...', 'length': 12345, 'word_count': 2102, ...}
```

Use this when the source is a markdown / plaintext document and you want some
metadata for filtering without writing a parser. For richer extraction (dates,
authors, domain tags), the user should plug in their own extractor and pass
the result through.

## End-to-end ingest pattern

```python
import vd

client = vd.connect('memory')
docs = client.create_collection('articles')

raw_documents = load_my_corpus()  # iterable of (doc_id, text, metadata)

# 1. Clean (only if needed for the source)
prepared = (
    (doc_id, vd.clean_text(text, remove_urls=True), meta)
    for doc_id, text, meta in raw_documents
)

# 2. Chunk while preserving metadata
chunked = vd.chunk_documents(
    ((doc_id, (text, meta)) for doc_id, text, meta in prepared),
    chunk_size=800,
    overlap=80,
    strategy='sentences',
    id_template='{doc_id}#{chunk_num}',
)

# 3. Convert to add_documents inputs and batch-insert
docs.add_documents(
    ((text, chunk_id, meta) for chunk_id, text, meta in chunked),
    batch_size=100,
)
```

`add_documents` accepts the four input shapes from `vd-quickstart`
(`str`, `(text, id)`, `(text, metadata)`, `(text, id, metadata)`, or a
`vd.Document`). Batch inserts amortize embedding-API and backend round-trips.

## When to skip parts of the pipeline

- **Skip `clean_text`** if the corpus is already clean or you're paying an
  embedding model that handles formatting fine.
- **Skip `chunk_*`** if individual documents are already short (< few KB).
  Chunking adds dedup work later (see `vd-ops` `find_duplicates`).
- **Skip `add_documents`** in favor of `docs[id] = ...` for tiny manual sets.
  Reach for `add_documents` once you're inserting more than ~10 docs.

## Common gotchas

- **`chunk_documents` input shape is awkward.** It expects an iterable of
  `(doc_id, text)` *or* `(doc_id, (text, metadata))`. The metadata sits inside
  the second element as a tuple, not as a third element. The example above is
  the right shape.
- **Pre-computed vectors short-circuit cleaning.** If you build a `Document`
  with a `vector`, `vd` won't re-embed — so cleaning the text *after* you
  embedded won't change retrieval. Clean before you embed.
- **`chunk_size` is in characters by default**, even if you pick
  `strategy='words'` or `'sentences'` — the strategy controls *boundaries*,
  not the unit of `chunk_size`. So `chunk_size=500` with `strategy='sentences'`
  means "build chunks ≤ 500 chars but break on sentence boundaries".
- **`add_documents` is eager.** It will iterate the whole input at insert time
  (in batches). Don't pass a one-shot generator if you also need to log it.
- **IDs must be unique within a collection.** When chunking, the
  `id_template` default already includes `chunk_num`, so collisions only happen
  if the same parent `doc_id` is processed twice. Keep parent IDs unique.

## See also

- `vd-quickstart` — the basics of `connect`, `create_collection`, `search`
- `vd-search` — querying once data is in
- `vd-ops` — analytics on the ingested collection (find duplicates, validate,
  benchmark)
