"""
Core functionality tests for the vd package.

Tests the main interface functions and basic operations that should work
across all backends.
"""

import hashlib

import pytest

import vd
from vd import Document, connect, list_backends


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


def test_import():
    """Test that vd can be imported and has expected attributes."""
    assert hasattr(vd, 'connect')
    assert hasattr(vd, 'Document')
    assert hasattr(vd, 'list_backends')


def test_list_backends():
    """Test listing available backends."""
    backends = list_backends()
    assert isinstance(backends, list)
    assert 'memory' in backends


def test_document_creation():
    """Test Document dataclass creation."""
    doc = Document(id='test1', text='Hello world')
    assert doc.id == 'test1'
    assert doc.text == 'Hello world'
    assert doc.vector is None
    assert doc.metadata == {}

    doc_with_meta = Document(
        id='test2',
        text='Test',
        metadata={'category': 'test', 'year': 2024}
    )
    assert doc_with_meta.metadata['category'] == 'test'
    assert doc_with_meta.metadata['year'] == 2024


class TestMemoryBackend:
    """Test suite for the memory backend."""

    @pytest.fixture
    def client(self):
        """Create a memory client for testing."""
        return connect('memory', embedding_model=mock_embedding_function)

    @pytest.fixture
    def collection(self, client):
        """Create a test collection."""
        return client.create_collection('test_collection')

    def test_connect(self):
        """Test connecting to memory backend."""
        client = connect('memory', embedding_model=mock_embedding_function)
        assert client is not None

    def test_create_collection(self, client):
        """Test creating a collection."""
        collection = client.create_collection('test1')
        assert collection is not None
        assert collection.name == 'test1'

    def test_create_duplicate_collection(self, client):
        """Test that creating duplicate collection raises error."""
        client.create_collection('dup_test')
        with pytest.raises(ValueError, match="already exists"):
            client.create_collection('dup_test')

    def test_get_collection(self, client):
        """Test getting an existing collection."""
        client.create_collection('get_test')
        collection = client.get_collection('get_test')
        assert collection is not None
        assert collection.name == 'get_test'

    def test_get_nonexistent_collection(self, client):
        """Test getting a non-existent collection raises error."""
        with pytest.raises(KeyError, match="does not exist"):
            client.get_collection('nonexistent')

    def test_list_collections(self, client):
        """Test listing collections."""
        client.create_collection('list1')
        client.create_collection('list2')
        collections = list(client.list_collections())
        assert 'list1' in collections
        assert 'list2' in collections

    def test_delete_collection(self, client):
        """Test deleting a collection."""
        client.create_collection('del_test')
        client.delete_collection('del_test')
        with pytest.raises(KeyError):
            client.get_collection('del_test')

    def test_add_document_string(self, collection):
        """Test adding a document as a string."""
        collection['doc1'] = "This is a test document"
        doc = collection['doc1']
        assert doc.text == "This is a test document"
        assert doc.id == 'doc1'
        assert doc.vector is not None  # Should be auto-generated

    def test_add_document_object(self, collection):
        """Test adding a Document object."""
        doc = Document(id='doc2', text='Another test', metadata={'type': 'test'})
        collection['doc2'] = doc
        retrieved = collection['doc2']
        assert retrieved.text == 'Another test'
        assert retrieved.metadata['type'] == 'test'

    def test_get_nonexistent_document(self, collection):
        """Test getting a non-existent document raises KeyError."""
        with pytest.raises(KeyError):
            _ = collection['nonexistent']

    def test_delete_document(self, collection):
        """Test deleting a document."""
        collection['doc1'] = "Test"
        del collection['doc1']
        with pytest.raises(KeyError):
            _ = collection['doc1']

    def test_iterate_documents(self, collection):
        """Test iterating over document IDs."""
        collection['doc1'] = "Test 1"
        collection['doc2'] = "Test 2"
        collection['doc3'] = "Test 3"

        doc_ids = list(collection)
        assert len(doc_ids) == 3
        assert 'doc1' in doc_ids
        assert 'doc2' in doc_ids
        assert 'doc3' in doc_ids

    def test_len(self, collection):
        """Test getting the number of documents."""
        assert len(collection) == 0
        collection['doc1'] = "Test"
        assert len(collection) == 1
        collection['doc2'] = "Test 2"
        assert len(collection) == 2

    def test_search_basic(self, collection):
        """Test basic search functionality."""
        collection['doc1'] = "Machine learning is a subset of AI"
        collection['doc2'] = "Deep learning uses neural networks"
        collection['doc3'] = "Python is a programming language"

        results = list(collection.search("artificial intelligence", limit=2))
        assert len(results) <= 2
        assert all('id' in r for r in results)
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)

        # First result should be most relevant
        assert results[0]['score'] >= results[1]['score']

    def test_search_with_filter(self, collection):
        """Test search with metadata filter."""
        collection['doc1'] = ("AI article", {'category': 'tech', 'year': 2024})
        collection['doc2'] = ("ML article", {'category': 'tech', 'year': 2023})
        collection['doc3'] = ("News article", {'category': 'news', 'year': 2024})

        # Filter by category
        results = list(collection.search("article", filter={'category': 'tech'}))
        assert len(results) == 2
        assert all(r['metadata']['category'] == 'tech' for r in results)

        # Filter by year
        results = list(collection.search("article", filter={'year': 2024}))
        assert len(results) == 2
        assert all(r['metadata']['year'] == 2024 for r in results)

    def test_search_with_egress(self, collection):
        """Test search with egress function."""
        collection['doc1'] = "First document"
        collection['doc2'] = "Second document"

        # Get only text
        texts = list(collection.search("document", egress=lambda r: r['text']))
        assert all(isinstance(t, str) for t in texts)
        assert 'document' in texts[0].lower()

    def test_add_documents_batch(self, collection):
        """Test batch adding documents."""
        docs = [
            "First article about AI",
            ("Second article", "doc2"),
            ("Third article", {'category': 'tech'}),
        ]

        collection.add_documents(docs)
        assert len(collection) == 3

    def test_upsert(self, collection):
        """Test upsert operation."""
        doc = Document(id='upsert1', text='Original text')
        collection.upsert(doc)
        assert collection['upsert1'].text == 'Original text'

        # Update
        doc_updated = Document(id='upsert1', text='Updated text')
        collection.upsert(doc_updated)
        assert collection['upsert1'].text == 'Updated text'


class TestUtilityFunctions:
    """Test utility functions."""

    def test_text_only_egress(self):
        """Test text_only egress function."""
        result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
        assert vd.text_only(result) == 'Hello'

    def test_id_only_egress(self):
        """Test id_only egress function."""
        result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
        assert vd.id_only(result) == 'doc1'

    def test_id_and_score_egress(self):
        """Test id_and_score egress function."""
        result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
        assert vd.id_and_score(result) == ('doc1', 0.9)

    def test_id_text_score_egress(self):
        """Test id_text_score egress function."""
        result = {'id': 'doc1', 'text': 'Hello', 'score': 0.9}
        assert vd.id_text_score(result) == ('doc1', 'Hello', 0.9)


class TestFilterOperators:
    """Test filter operators in memory backend."""

    @pytest.fixture
    def collection(self):
        """Create a collection with test data."""
        client = connect('memory', embedding_model=mock_embedding_function)
        coll = client.create_collection('filter_test')

        coll['doc1'] = ("Article 1", {'year': 2020, 'views': 100, 'tags': ['python']})
        coll['doc2'] = ("Article 2", {'year': 2021, 'views': 500, 'tags': ['ai']})
        coll['doc3'] = ("Article 3", {'year': 2022, 'views': 1000, 'tags': ['ml']})
        coll['doc4'] = ("Article 4", {'year': 2023, 'views': 1500, 'tags': ['ai', 'python']})

        return coll

    def test_filter_gte(self, collection):
        """Test $gte (greater than or equal) filter."""
        results = list(collection.search("article", filter={'year': {'$gte': 2022}}))
        assert len(results) == 2
        assert all(r['metadata']['year'] >= 2022 for r in results)

    def test_filter_lte(self, collection):
        """Test $lte (less than or equal) filter."""
        results = list(collection.search("article", filter={'views': {'$lte': 500}}))
        assert len(results) == 2
        assert all(r['metadata']['views'] <= 500 for r in results)

    def test_filter_gt(self, collection):
        """Test $gt (greater than) filter."""
        results = list(collection.search("article", filter={'views': {'$gt': 500}}))
        assert len(results) == 2
        assert all(r['metadata']['views'] > 500 for r in results)

    def test_filter_lt(self, collection):
        """Test $lt (less than) filter."""
        results = list(collection.search("article", filter={'year': {'$lt': 2022}}))
        assert len(results) == 2
        assert all(r['metadata']['year'] < 2022 for r in results)

    def test_filter_in(self, collection):
        """Test $in filter."""
        results = list(collection.search("article", filter={'tags': {'$in': ['python']}}))
        # Should find docs with 'python' tag
        assert len(results) >= 1

    def test_filter_and(self, collection):
        """Test $and logical operator."""
        results = list(collection.search(
            "article",
            filter={'$and': [
                {'year': {'$gte': 2021}},
                {'views': {'$gte': 1000}}
            ]}
        ))
        assert len(results) == 2
        assert all(r['metadata']['year'] >= 2021 and r['metadata']['views'] >= 1000 for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
