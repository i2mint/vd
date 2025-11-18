#!/usr/bin/env python
"""
Command-line interface for vd.

Provides a CLI tool for common vd operations like listing backends,
exporting/importing collections, health checks, and more.
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_backends(args):
    """List available backends."""
    import vd

    if args.planned:
        vd.print_backends_table(include_planned=True)
    else:
        vd.print_backends_table(include_planned=False)


def cmd_install_info(args):
    """Get installation instructions for a backend."""
    import vd

    try:
        instructions = vd.get_install_instructions(args.backend)
        print(instructions)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_health_check(args):
    """Check backend health."""
    import vd

    status = vd.health_check_backend(args.backend)

    print(f"\nBackend: {status['backend']}")
    print(f"Status: {status['status']}")
    print(f"Available: {status['available']}")
    print(f"Message: {status['message']}")

    if status['details']:
        print("\nDetails:")
        for key, value in status['details'].items():
            print(f"  {key}: {value}")


def cmd_export(args):
    """Export a collection."""
    import vd

    # Connect to backend
    client = vd.connect(args.backend)

    # Get collection
    collection = client.get_collection(args.collection)

    # Export
    count = vd.export_collection(
        collection,
        args.output,
        format=args.format,
        include_vectors=not args.no_vectors,
    )

    print(f"✓ Exported {count} documents to {args.output}")


def cmd_import(args):
    """Import into a collection."""
    import vd

    # Connect to backend
    client = vd.connect(args.backend)

    # Create or get collection
    try:
        collection = client.create_collection(args.collection)
        print(f"Created new collection: {args.collection}")
    except ValueError:
        collection = client.get_collection(args.collection)
        print(f"Using existing collection: {args.collection}")

    # Import
    count = vd.import_collection(
        collection,
        args.input,
        format=args.format,
        skip_existing=args.skip_existing,
    )

    print(f"✓ Imported {count} documents")


def cmd_stats(args):
    """Show collection statistics."""
    import vd

    # Connect and get collection
    client = vd.connect(args.backend)
    collection = client.get_collection(args.collection)

    # Get stats
    stats = vd.collection_stats(collection)

    print(f"\nCollection: {args.collection}")
    print("=" * 60)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Documents with vectors: {stats['has_vectors']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} chars")
    print(f"Min text length: {stats['min_text_length']}")
    print(f"Max text length: {stats['max_text_length']}")
    print(f"Total characters: {stats['total_chars']:,}")

    if stats['embedding_dimension']:
        print(f"Embedding dimension: {stats['embedding_dimension']}")

    if stats['metadata_fields']:
        print(f"\nMetadata fields: {', '.join(sorted(stats['metadata_fields']))}")

        if args.verbose:
            print("\nMetadata field counts:")
            for field, count in sorted(stats['metadata_field_counts'].items()):
                print(f"  {field}: {count}")


def cmd_validate(args):
    """Validate a collection."""
    import vd

    # Connect and get collection
    client = vd.connect(args.backend)
    collection = client.get_collection(args.collection)

    # Validate
    report = vd.validate_collection(collection)

    print(f"\nCollection: {args.collection}")
    print("=" * 60)
    print(f"Valid: {report['valid']}")

    if report['issues']:
        print(f"\nIssues ({len(report['issues'])}):")
        for issue in report['issues']:
            print(f"  ✗ {issue}")

    if report['warnings']:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  ⚠ {warning}")

    print(f"\nStats:")
    for key, value in report['stats'].items():
        print(f"  {key}: {value}")


def cmd_migrate(args):
    """Migrate a collection between backends."""
    import vd

    # Source
    source_client = vd.connect(args.source_backend)
    source_coll = source_client.get_collection(args.source_collection)

    # Target
    target_client = vd.connect(args.target_backend)
    try:
        target_coll = target_client.create_collection(args.target_collection)
        print(f"Created target collection: {args.target_collection}")
    except ValueError:
        target_coll = target_client.get_collection(args.target_collection)
        print(f"Using existing target collection: {args.target_collection}")

    # Progress callback
    def progress(current, total):
        pct = (current / total) * 100 if total > 0 else 0
        print(f"\rProgress: {current}/{total} ({pct:.1f}%)", end='', flush=True)

    print(f"Migrating from {args.source_backend} to {args.target_backend}...")

    # Migrate
    stats = vd.migrate_collection(
        source_coll,
        target_coll,
        batch_size=args.batch_size,
        preserve_vectors=not args.recompute_vectors,
        progress_callback=progress,
    )

    print()  # New line after progress
    print("\nMigration complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Migrated: {stats['migrated']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")

    if stats['errors']:
        print(f"\nErrors:")
        for error in stats['errors'][:10]:  # Show first 10
            print(f"  {error}")


def cmd_benchmark(args):
    """Benchmark search performance."""
    import vd

    # Connect and get collection
    client = vd.connect(args.backend)
    collection = client.get_collection(args.collection)

    print(f"Benchmarking search on {args.collection}...")
    print(f"Running {args.queries} queries...")

    # Benchmark
    results = vd.benchmark_search(
        collection,
        args.query,
        n_queries=args.queries,
        limit=args.limit,
    )

    print("\nResults:")
    print("=" * 60)
    print(f"Total time: {results['total_time']:.3f}s")
    print(f"Average latency: {results['avg_latency']*1000:.2f}ms")
    print(f"Min latency: {results['min_latency']*1000:.2f}ms")
    print(f"Max latency: {results['max_latency']*1000:.2f}ms")
    print(f"P50: {results['p50']*1000:.2f}ms")
    print(f"P95: {results['p95']*1000:.2f}ms")
    print(f"P99: {results['p99']*1000:.2f}ms")
    print(f"Throughput: {results['queries_per_second']:.1f} queries/sec")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='vd - Vector Database Facades CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available backends
  vd backends

  # Get installation instructions
  vd install chroma

  # Export a collection
  vd export memory my_docs -o backup.jsonl

  # Import a collection
  vd import chroma my_docs -i backup.jsonl

  # Get collection statistics
  vd stats memory my_docs

  # Migrate between backends
  vd migrate memory my_docs chroma my_docs

  # Benchmark search
  vd benchmark memory my_docs -q "test query"
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Backends command
    backends_parser = subparsers.add_parser('backends', help='List available backends')
    backends_parser.add_argument(
        '--planned', action='store_true', help='Include planned backends'
    )
    backends_parser.set_defaults(func=cmd_backends)

    # Install info command
    install_parser = subparsers.add_parser(
        'install', help='Get installation instructions'
    )
    install_parser.add_argument('backend', help='Backend name')
    install_parser.set_defaults(func=cmd_install_info)

    # Health check command
    health_parser = subparsers.add_parser('health', help='Check backend health')
    health_parser.add_argument('backend', help='Backend name')
    health_parser.set_defaults(func=cmd_health_check)

    # Export command
    export_parser = subparsers.add_parser('export', help='Export a collection')
    export_parser.add_argument('backend', help='Backend name')
    export_parser.add_argument('collection', help='Collection name')
    export_parser.add_argument('-o', '--output', required=True, help='Output file')
    export_parser.add_argument(
        '-f', '--format', choices=['jsonl', 'json', 'directory'], default='jsonl'
    )
    export_parser.add_argument(
        '--no-vectors', action='store_true', help='Exclude vectors'
    )
    export_parser.set_defaults(func=cmd_export)

    # Import command
    import_parser = subparsers.add_parser('import', help='Import into a collection')
    import_parser.add_argument('backend', help='Backend name')
    import_parser.add_argument('collection', help='Collection name')
    import_parser.add_argument('-i', '--input', required=True, help='Input file')
    import_parser.add_argument(
        '-f', '--format', choices=['jsonl', 'json', 'directory']
    )
    import_parser.add_argument(
        '--skip-existing', action='store_true', help='Skip existing documents'
    )
    import_parser.set_defaults(func=cmd_import)

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show collection statistics')
    stats_parser.add_argument('backend', help='Backend name')
    stats_parser.add_argument('collection', help='Collection name')
    stats_parser.add_argument('-v', '--verbose', action='store_true')
    stats_parser.set_defaults(func=cmd_stats)

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a collection')
    validate_parser.add_argument('backend', help='Backend name')
    validate_parser.add_argument('collection', help='Collection name')
    validate_parser.set_defaults(func=cmd_validate)

    # Migrate command
    migrate_parser = subparsers.add_parser(
        'migrate', help='Migrate collection between backends'
    )
    migrate_parser.add_argument('source_backend', help='Source backend')
    migrate_parser.add_argument('source_collection', help='Source collection')
    migrate_parser.add_argument('target_backend', help='Target backend')
    migrate_parser.add_argument('target_collection', help='Target collection')
    migrate_parser.add_argument('--batch-size', type=int, default=100)
    migrate_parser.add_argument(
        '--recompute-vectors', action='store_true', help='Recompute embeddings'
    )
    migrate_parser.set_defaults(func=cmd_migrate)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark search')
    benchmark_parser.add_argument('backend', help='Backend name')
    benchmark_parser.add_argument('collection', help='Collection name')
    benchmark_parser.add_argument('-q', '--query', default='test query')
    benchmark_parser.add_argument('--queries', type=int, default=100)
    benchmark_parser.add_argument('--limit', type=int, default=10)
    benchmark_parser.set_defaults(func=cmd_benchmark)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
