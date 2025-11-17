"""
Pytest configuration and fixtures for vd tests.
"""

import hashlib


def mock_embedding_function(text: str) -> list[float]:
    """
    Simple mock embedding function for testing.

    Creates a deterministic "embedding" based on the text hash.
    Not semantically meaningful, but consistent and testable.
    """
    # Create a deterministic hash-based embedding
    text_hash = hashlib.md5(text.encode()).digest()
    # Convert to floats between -1 and 1
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    # Pad to typical embedding size (e.g., 16 dimensions for testing)
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]
