"""
Backend implementations for various vector databases.

This module contains concrete implementations of the BaseBackend protocol
for different vector database systems.

Available backends are registered here and can be accessed via vd.connect().
"""

# Import backends to trigger registration
from vd.backends.memory import MemoryBackend  # noqa: F401

# Optional backends - import only if dependencies are available
try:
    from vd.backends.chroma import ChromaBackend  # noqa: F401
except ImportError:
    pass  # ChromaDB not installed

__all__ = ['MemoryBackend']
