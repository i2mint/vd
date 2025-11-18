"""
Demo script showing the utility features of vd.

This demonstrates:
- Import/Export collections
- Migration between backends
- Collection analytics and statistics
- Text preprocessing and chunking
- Health checks and benchmarking
- Advanced search features
"""

import hashlib
import tempfile
from pathlib import Path

import vd


def mock_embedding_function(text: str) -> list[float]:
    """Simple mock embedding for demo."""
    text_hash = hashlib.md5(text.encode()).digest()
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]


def main():
    print("=" * 80)
    print("VD Utilities Demo")
    print("=" * 80)
    print()

    # Set up client and collection
    client = vd.connect('memory', embedding_model=mock_embedding_function)
    docs = client.create_collection('demo_docs')

    # Add some sample documents
    print("Setting up sample collection...")
    docs['doc1'] = (
        "Machine learning is a subset of artificial intelligence",
        {'category': 'AI', 'year': 2023},
    )
    docs['doc2'] = (
        "Deep learning uses neural networks with multiple layers",
        {'category': 'AI', 'year': 2023},
    )
    docs['doc3'] = (
        "Python is a popular programming language for data science",
        {'category': 'Programming', 'year': 2024},
    )
    docs['doc4'] = (
        "Natural language processing helps computers understand human language",
        {'category': 'AI', 'year': 2024},
    )
    print(f"✓ Created collection with {len(docs)} documents")
    print()

    # 1. Collection Statistics
    print("1. Collection Statistics:")
    print("-" * 80)
    stats = vd.collection_stats(docs)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Documents with vectors: {stats['has_vectors']}")
    print(f"Average text length: {stats['avg_text_length']:.1f} chars")
    print(f"Metadata fields: {', '.join(stats['metadata_fields'])}")
    print()

    # 2. Metadata Distribution
    print("2. Metadata Distribution:")
    print("-" * 80)
    category_dist = vd.metadata_distribution(docs, 'category')
    print("Category distribution:")
    for category, count in category_dist.items():
        print(f"  {category}: {count}")
    print()

    year_dist = vd.metadata_distribution(docs, 'year')
    print("Year distribution:")
    for year, count in year_dist.items():
        print(f"  {year}: {count}")
    print()

    # 3. Collection Validation
    print("3. Collection Validation:")
    print("-" * 80)
    report = vd.validate_collection(docs)
    print(f"Valid: {report['valid']}")
    print(f"Total documents: {report['stats']['total_documents']}")
    print(f"Issues: {len(report['issues'])}")
    print(f"Warnings: {len(report['warnings'])}")
    print()

    # 4. Text Preprocessing
    print("4. Text Preprocessing:")
    print("-" * 80)
    sample_text = "  This is   some TEXT with  extra    spaces and URLs like https://example.com  "
    cleaned = vd.clean_text(
        sample_text, lowercase=True, remove_extra_whitespace=True, remove_urls=True
    )
    print(f"Original: '{sample_text}'")
    print(f"Cleaned:  '{cleaned}'")
    print()

    # 5. Text Chunking
    print("5. Text Chunking:")
    print("-" * 80)
    long_text = "Machine learning is amazing. It helps solve complex problems. Neural networks are powerful. They can learn from data. Deep learning is a subset. It uses multiple layers."
    chunks = vd.chunk_text(long_text, chunk_size=50, strategy='sentences')
    print(f"Original text ({len(long_text)} chars):")
    print(f"  {long_text}")
    print(f"\nChunked into {len(chunks)} pieces:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {chunk}")
    print()

    # 6. Advanced Search - Multi-query
    print("6. Advanced Search - Multi-query:")
    print("-" * 80)
    queries = ["artificial intelligence", "programming language"]
    results = list(vd.multi_query_search(docs, queries, limit=3, combine='best'))
    print(f"Searching for: {queries}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['id']}: {result['text'][:50]}... (score: {result['score']:.3f})")
    print()

    # 7. Search Similar Documents
    print("7. Find Similar Documents:")
    print("-" * 80)
    similar = list(vd.search_similar_to_document(docs, 'doc1', limit=2))
    print(f"Documents similar to 'doc1':")
    for i, result in enumerate(similar, 1):
        print(f"  {i}. {result['id']}: {result['text'][:50]}... (score: {result['score']:.3f})")
    print()

    # 8. Import/Export
    print("8. Import/Export:")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / 'export.jsonl'

        # Export
        count = vd.export_collection(docs, export_path, format='jsonl')
        print(f"✓ Exported {count} documents to {export_path.name}")

        # Create new collection and import
        client2 = vd.connect('memory', embedding_model=mock_embedding_function)
        docs2 = client2.create_collection('imported_docs')
        count = vd.import_collection(docs2, export_path)
        print(f"✓ Imported {count} documents into new collection")
        print(f"  New collection size: {len(docs2)}")
    print()

    # 9. Migration
    print("9. Collection Migration:")
    print("-" * 80)
    target_client = vd.connect('memory', embedding_model=mock_embedding_function)
    target_docs = target_client.create_collection('migrated_docs')

    migration_stats = vd.migrate_collection(docs, target_docs, batch_size=2)
    print(f"✓ Migration completed:")
    print(f"  Total: {migration_stats['total']}")
    print(f"  Migrated: {migration_stats['migrated']}")
    print(f"  Failed: {migration_stats['failed']}")
    print()

    # 10. Health Check
    print("10. Backend Health Check:")
    print("-" * 80)
    health = vd.health_check_backend('memory', embedding_model=mock_embedding_function)
    print(f"Backend: {health['backend']}")
    print(f"Status: {health['status']}")
    print(f"Available: {health['available']}")
    print()

    # 11. Performance Benchmark
    print("11. Search Performance Benchmark:")
    print("-" * 80)
    results = vd.benchmark_search(docs, "machine learning", n_queries=10, limit=5)
    print(f"Average latency: {results['avg_latency']*1000:.2f}ms")
    print(f"P95 latency: {results['p95']*1000:.2f}ms")
    print(f"Throughput: {results['queries_per_second']:.1f} queries/sec")
    print()

    # 12. Sample Collection
    print("12. Sample Collection:")
    print("-" * 80)
    sample_ids = vd.sample_collection(docs, n=2, method='random', seed=42)
    print(f"Random sample of 2 documents: {sample_ids}")

    diverse_ids = vd.sample_collection(docs, n=2, method='diverse')
    print(f"Diverse sample of 2 documents: {diverse_ids}")
    print()

    print("=" * 80)
    print("Utilities Summary:")
    print("  ✓ Collection analytics and statistics")
    print("  ✓ Import/Export (JSONL, JSON, directory)")
    print("  ✓ Migration between backends")
    print("  ✓ Text preprocessing and chunking")
    print("  ✓ Advanced search (multi-query, similarity)")
    print("  ✓ Health checks and benchmarking")
    print("  ✓ Collection validation and sampling")
    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
