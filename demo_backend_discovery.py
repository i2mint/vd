"""
Demo script showing the backend discovery features of vd.

This demonstrates how users can:
- List all available backends
- See what needs to be installed
- Get installation instructions
- Use the helpful error messages
"""

import hashlib

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
    print("VD Backend Discovery Demo")
    print("=" * 80)
    print()

    # Show the formatted table of all backends
    print("1. View all backends with print_backends_table():")
    print("-" * 80)
    vd.print_backends_table()

    # Show currently available backends
    print("\n2. List currently available backends:")
    print("-" * 80)
    available = vd.list_available_backends()
    print(f"Available: {available}")
    print()

    # Show all backends (just registered ones)
    print("3. List registered backends (vd.list_backends()):")
    print("-" * 80)
    registered = vd.list_backends()
    print(f"Registered: {registered}")
    print()

    # Get info about a specific backend
    print("4. Get information about ChromaDB:")
    print("-" * 80)
    chroma_info = vd.get_backend_info('chroma')
    print(f"Name: {chroma_info['name']}")
    print(f"Description: {chroma_info['description']}")
    print(f"Available: {chroma_info['available']}")
    print(f"Features: {', '.join(chroma_info['features'])}")
    print()

    # Get installation instructions
    print("5. Get installation instructions for a backend:")
    print("-" * 80)
    # Try Pinecone (which is planned but not yet implemented)
    try:
        instructions = vd.get_install_instructions('pinecone')
        print(instructions)
    except ValueError as e:
        print(f"Error: {e}")
    print()

    # Show what happens when you try to connect to unavailable backend
    print("6. Trying to connect to an unavailable backend:")
    print("-" * 80)
    try:
        client = vd.connect('pinecone')
    except ValueError as e:
        print(f"Got helpful error message:")
        print(e)
    print()

    # Show what happens with unknown backend
    print("7. Trying to connect to an unknown backend:")
    print("-" * 80)
    try:
        client = vd.connect('nonexistent')
    except ValueError as e:
        print(f"Got helpful error message:")
        print(e)
    print()

    # Successfully connect to memory backend
    print("8. Successfully connecting to available backend:")
    print("-" * 80)
    client = vd.connect('memory', embedding_model=mock_embedding_function)
    print(f"✓ Connected to memory backend successfully!")
    print()

    # Show all backends including planned ones
    print("9. View ALL backends including planned ones:")
    print("-" * 80)
    vd.print_backends_table(include_planned=True)

    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
