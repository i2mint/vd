---
name: vd-ops
description: >-
  Operational tooling for the vd package. Use this skill when the user is
  managing a vd collection beyond reads/writes — exporting or importing data
  (JSONL / JSON / directory), migrating between backends, computing collection
  statistics, finding duplicates / outliers, validating integrity, running
  health checks, benchmarking search or insert latency, or driving any of the
  above from the `vd` CLI.
audience: users
---

# Operational tasks on vd collections

This skill covers the maintenance side of `vd`: backups, migrations, audits,
and benchmarks. For inserting data see `vd-ingest`; for querying, see
`vd-search`.

Most ops are available both as Python functions and as `vd` CLI subcommands.
Both are listed below — pick the surface that matches the user's workflow.

## Export and import

```python
import vd

# Export
vd.export_collection(docs, 'backup.jsonl', format='jsonl')   # default — best for big sets
vd.export_collection(docs, 'backup.json',  format='json')    # single file
vd.export_collection(docs, './backup_dir', format='directory')  # one file per doc

# Import
vd.import_collection(docs, 'backup.jsonl')
vd.import_collection(docs, 'backup.jsonl', skip_existing=True)  # don't overwrite
```

`export_collection`/`import_collection` auto-detect format from extension when
the kwarg is omitted on import. `export_collection` returns the number of docs
written; `import_collection` returns the number imported.

CLI:

```bash
vd export memory my_docs -o backup.jsonl
vd export memory my_docs -o backup.json -f json
vd import chroma my_docs -i backup.jsonl
```

Choose a format:

- **`jsonl`** — one document per line. Streams. Best for >1000 docs. Default.
- **`json`** — single object with all documents. Easy to inspect by hand.
- **`directory`** — one file per document. Useful for diffs and version
  control, but slow on large sets.

The lower-level helpers (`export_to_jsonl`, `import_from_jsonl`, etc.) are
also exported if you need format-specific control; otherwise prefer the
`*_collection` facades.

## Migration between backends

Use this when the user wants to move a collection from one backend to another
(e.g., `memory` prototype → `chroma` for persistence) without going through
disk.

```python
source_client = vd.connect('memory')
target_client = vd.connect('chroma', persist_directory='./vector_db')

source = source_client.get_collection('docs')
target = target_client.create_collection('docs')

stats = vd.migrate_collection(
    source,
    target,
    batch_size=100,
    preserve_vectors=True,   # reuse existing embeddings — no re-embed cost
    progress_callback=lambda cur, tot: print(f"{cur}/{tot}"),
    skip_existing=False,
)
```

For migrating *all* collections of a client at once:

```python
vd.migrate_client(
    source_client,
    target_client,
    collection_names=['docs1', 'docs2'],   # None = all source collections
    batch_size=100,
    preserve_vectors=True,
)
```

`copy_collection` is a flexible wrapper that takes either a `Collection` or a
`(client_name, collection_name)` tuple for source/target — useful for scripts
that build the args dynamically:

```python
vd.copy_collection(
    source=('memory', 'docs'),
    target=('chroma', 'docs', {'persist_directory': './vector_db'}),
    batch_size=100,
)
```

CLI:

```bash
vd migrate memory source_docs chroma target_docs
```

`preserve_vectors=True` is almost always what you want — re-embedding can
cost real money on hosted models.

## Collection analytics

```python
# Comprehensive stats
stats = vd.collection_stats(docs)
# {'total_documents': N, 'avg_text_length': ..., 'metadata_fields': [...], ...}

# Distribution of values for one metadata field
dist = vd.metadata_distribution(docs, 'category', top_n=10)
# {'tech': 412, 'science': 277, ...}

# Validate integrity (missing fields, weird metadata, etc.)
report = vd.validate_collection(docs)
if not report['valid']:
    for issue in report['issues']:
        print(f"Issue: {issue}")

# Find near-duplicate or duplicate docs
dupes = vd.find_duplicates(docs, threshold=0.95, method='cosine')
# list of (id_a, id_b, similarity)

# Find outliers — docs dissimilar to their nearest neighbors
out = vd.find_outliers(docs, n_neighbors=5, threshold=0.3)
# list of (id, mean_similarity_to_neighbors)

# Sample documents
random_ids  = vd.sample_collection(docs, n=10, method='random', seed=42)
diverse_ids = vd.sample_collection(docs, n=10, method='diverse')
```

CLI:

```bash
vd stats memory my_docs       # add -v for verbose
vd validate memory my_docs
```

When to use which:

- **`collection_stats`** — quick "is this collection healthy / what's in it"
  overview. Cheap.
- **`validate_collection`** — flag structural problems (empty texts, missing
  vectors, malformed metadata). Run before exporting / migrating.
- **`find_duplicates`** — surfaces near-dup chunks (especially after
  `chunk_documents` with overlap). `threshold=0.95` is a sensible default;
  raise it if you only want exact dupes.
- **`find_outliers`** — useful for QA; documents very far from their
  neighbors are often noise (broken parses, wrong language, etc.).
- **`sample_collection`** with `method='diverse'` — better than `'random'`
  when you want to *eyeball* the variety of content, since it picks
  representative documents across the embedding space.

## Health checks and benchmarks

```python
# Check that a backend is reachable / configured
health = vd.health_check_backend('chroma', persist_directory='./vector_db')
# {'status': 'healthy' | 'unhealthy', 'available': True, 'error': ...}

# Check that a specific collection is accessible
health = vd.health_check_collection(docs)

# Benchmark search latency
results = vd.benchmark_search(
    docs,
    query="test query",
    n_queries=100,
    limit=10,
)
# {'avg_latency': ..., 'p50': ..., 'p95': ..., 'queries_per_second': ...}

# Benchmark insertion throughput (synthetic random docs)
results = vd.benchmark_insert(docs, n_documents=100, batch_size=10)
```

CLI:

```bash
vd health memory
vd benchmark memory my_docs -q "test query" --queries 100
```

`benchmark_search` and `benchmark_insert` give per-call latencies and
throughput in seconds — multiply by 1000 for ms when reporting.

## CLI cheat sheet

The `vd` CLI is installed as a `[project.scripts]` entry point. All commands
take `<backend>` as the first positional and read backend config from env vars
or `--persist-directory` flags as needed:

```bash
vd backends                          # list registered & available backends
vd backends --planned                # include planned backends in the list
vd install chroma                    # print install instructions for chroma
vd health <backend>                  # health check
vd stats <backend> <collection>      # stats (-v for verbose)
vd validate <backend> <collection>   # integrity check
vd export <backend> <collection> -o <file>  [-f jsonl|json|directory]
vd import <backend> <collection> -i <file>
vd migrate <src_be> <src_coll> <tgt_be> <tgt_coll>
vd benchmark <backend> <collection> -q "query" --queries 100
```

When the user wants automation (cron, CI), the CLI is usually the right
surface. When they want one-off introspection inside a notebook, prefer the
Python functions.

## Common gotchas

- **`migrate_collection` requires the target collection to already exist.**
  Call `target_client.create_collection(name)` first (or use `copy_collection`,
  which takes a `(backend, name, config)` tuple and creates the target for
  you).
- **`preserve_vectors=False` re-embeds.** That can be slow and costly on
  hosted embedding APIs. Only set it when you've changed embedding model.
- **`find_duplicates` and `find_outliers` are O(N²) in the naive case.** They
  walk the collection. Don't run them on millions of docs without sampling
  first via `sample_collection`.
- **`benchmark_search` is sequential.** It does not parallelize; the numbers
  reflect single-client serial latency, not max backend throughput.
- **Directory export creates one file per doc.** On a corpus of 10k+ docs
  you'll hit filesystem limits and slow operations. Use JSONL there.
- **`validate_collection` is best-effort.** It checks the kinds of problems
  the package knows about (empty text, missing vectors). It does not catch
  semantic issues like "all my embeddings are zero vectors" — for that, look
  at `find_outliers` or visual sampling.
- **CLI persistence for `chroma`** typically reads `--persist-directory` or a
  `VD_*` env var (see `vd-backend-choose`). If a CLI command silently uses
  in-memory state when you wanted disk, that's the cause.

## See also

- `vd-quickstart` — connect / create / search basics
- `vd-backend-choose` — picking and configuring backends; YAML config files
- `vd-ingest` — getting data in (cleaning, chunking, batch insert)
- `vd-search` — querying once data is in
