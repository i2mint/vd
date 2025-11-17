"""
Example usage of the vd package.

This script demonstrates the main features of the vd package, including:
- Connecting to a backend
- Creating collections
- Adding documents
- Searching with filters
- Using egress functions
"""


def mock_embedding_function(text: str) -> list[float]:
    """Simple mock embedding function for demo purposes."""
    import hashlib

    text_hash = hashlib.md5(text.encode()).digest()
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]


def main():
    import vd

    print("=" * 70)
    print("VD Package - Vector Database Facades")
    print("=" * 70)
    print()

    # List available backends
    print("Available backends:", vd.list_backends())
    print()

    # Connect to memory backend
    print("Connecting to memory backend...")
    client = vd.connect('memory', embedding_model=mock_embedding_function)
    print("✓ Connected successfully")
    print()

    # Create a collection
    print("Creating a collection...")
    docs = client.create_collection('tech_articles')
    print("✓ Collection 'tech_articles' created")
    print()

    # Add documents in various formats
    print("Adding documents...")

    # Simple string format
    docs['doc1'] = "Machine learning is a subset of artificial intelligence"

    # Tuple with metadata
    docs['doc2'] = (
        "Deep learning uses neural networks with multiple layers",
        {'category': 'AI', 'year': 2024, 'views': 1500}
    )

    # Document object
    doc3 = vd.Document(
        id='doc3',
        text='Python is a popular programming language for data science',
        metadata={'category': 'Programming', 'year': 2023, 'views': 2000}
    )
    docs.upsert(doc3)

    # Batch add
    docs.add_documents([
        ("Natural language processing helps computers understand human language", {'category': 'AI', 'year': 2024}),
        ("JavaScript is widely used for web development", {'category': 'Programming', 'year': 2023}),
        ("Cloud computing enables scalable infrastructure", {'category': 'Cloud', 'year': 2024}),
    ])

    print(f"✓ Added {len(docs)} documents")
    print()

    # Basic search
    print("Basic search for 'artificial intelligence':")
    print("-" * 70)
    results = list(docs.search("artificial intelligence", limit=3))
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']}: {result['text'][:50]}...")
        print(f"   Score: {result['score']:.4f}")
        if result['metadata']:
            print(f"   Metadata: {result['metadata']}")
    print()

    # Search with filter
    print("Search for 'programming' with filter (category='AI'):")
    print("-" * 70)
    results = list(docs.search(
        "programming language",
        filter={'category': 'AI'},
        limit=3
    ))
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']}: {result['text'][:50]}...")
        print(f"   Category: {result['metadata'].get('category')}")
    print()

    # Search with egress function
    print("Search using egress function (text only):")
    print("-" * 70)
    texts = list(docs.search(
        "computer science",
        limit=3,
        egress=vd.text_only
    ))
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    print()

    # Advanced filter
    print("Advanced filter (year >= 2024 AND views >= 1000):")
    print("-" * 70)
    results = list(docs.search(
        "technology",
        filter={'$and': [
            {'year': {'$gte': 2024}},
            {'views': {'$gte': 1000}}
        ]},
        limit=5
    ))
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']}: {result['text'][:40]}...")
        print(f"   Year: {result['metadata'].get('year')}, Views: {result['metadata'].get('views')}")
    print()

    # Demonstrate collection operations
    print("Collection operations:")
    print("-" * 70)
    print(f"Total documents: {len(docs)}")
    print(f"Document IDs: {list(docs)}")
    print()

    # Retrieve a specific document
    doc = docs['doc1']
    print(f"Retrieved doc1: {doc.text}")
    print()

    # Delete a document
    del docs['doc1']
    print(f"Deleted doc1. Remaining documents: {len(docs)}")
    print()

    # List all collections
    print("All collections:")
    print(list(client.list_collections()))
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
