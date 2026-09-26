"""
Server-free unit tests for the pgvector adapter's value conversions.

The live contract tests (``test_core.py`` etc.) only run when a Postgres +
pgvector server is reachable; these run wherever the ``pgvector`` client is
installed.
"""

import pytest

pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

from vd.backends.pgvector import _embedding_to_list  # noqa: E402


def test_embedding_to_list_accepts_pgvector_vector():
    """pgvector-python >= 0.5 returns its own (non-iterable) Vector type (#26)."""
    from pgvector import Vector

    assert _embedding_to_list(Vector([1.0, 2.5])) == [1.0, 2.5]


def test_embedding_to_list_accepts_numpy_and_sequences():
    np = pytest.importorskip("numpy")
    assert _embedding_to_list(np.array([1.0, 2.0], dtype="float32")) == [1.0, 2.0]
    assert _embedding_to_list((1, 2)) == [1.0, 2.0]
    assert all(isinstance(x, float) for x in _embedding_to_list([1, 2]))
