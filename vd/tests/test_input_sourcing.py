"""
Tests for the input_sourcing module.

These tests demonstrate the functionality of the source_variables decorator,
which provides flexible variable resolution for API endpoints.
"""

import pytest
import asyncio
from functools import partial

from vd.wip.input_sourcing import (
    resolve_data,
    _get_function_from_store,
    source_variables,
    mock_mall,
)


# Helper for async testing
async def async_call(func, *args, **kwargs):
    """Helper to call both sync and async functions."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_resolve_data():
    """Test the resolve_data function with different input types."""
    mall = {"segments": {"greeting": "hello", "farewell": "goodbye"}}

    # Test resolving a string key that exists in the store
    result = resolve_data("greeting", "segments", mall)
    assert result == "hello"

    # Test resolving a string key that doesn't exist in the store
    result = resolve_data("nonexistent", "segments", mall)
    assert result == "nonexistent"  # Should return the original value

    # Test resolving a non-string value
    result = resolve_data(42, "segments", mall)
    assert result == 42  # Should return the original value

    # Test with store that doesn't exist
    result = resolve_data("greeting", "nonexistent_store", mall)
    assert result == "greeting"  # Should return the original value


@pytest.mark.asyncio
async def test_get_function_from_store():
    """Test the _get_function_from_store function."""
    mall = {
        "functions": {
            "simple": lambda x: x * 2,
            "complex": lambda x, multiplier=1: x * multiplier,
        }
    }

    # Test getting a simple function
    func = await _get_function_from_store("simple", "functions", mall)
    assert func(5) == 10

    # Test getting a parameterized function
    func = await _get_function_from_store(
        {"complex": {"multiplier": 3}}, "functions", mall
    )
    assert func(4) == 12

    # Test with non-existent function
    with pytest.raises(KeyError):
        await _get_function_from_store("nonexistent", "functions", mall)

    # Test with non-existent store
    with pytest.raises(KeyError):
        await _get_function_from_store("simple", "nonexistent_store", mall)


@pytest.mark.asyncio
async def test_basic_resolution():
    """Test basic variable resolution with the source_variables decorator."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
    )
    async def embed_text(segments, embedder):
        return embedder(segments)

    # Case 1: Pass direct values (no resolution needed)
    text = "test"
    embedder = lambda t: [ord(c) for c in t]
    result = await embed_text(segments=text, embedder=embedder)
    assert result == [116, 101, 115, 116]  # ASCII values of "test"

    # Case 2: Resolve segment from store
    result = await embed_text(segments="greeting", embedder=embedder)
    assert result == [104, 101, 108, 108, 111]  # ASCII values of "hello"


@pytest.mark.asyncio
async def test_function_resolution():
    """Test resolving a function from a store."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
    )
    async def embed_with_named_embedder(segments, embedder):
        return embedder(segments)

    # Case 1: Resolve embedder by name
    result = await embed_with_named_embedder(segments="hello", embedder="default")
    assert result == [104, 101, 108, 108, 111]  # ASCII values of "hello"

    # Case 2: Resolve both segments and embedder from stores
    result = await embed_with_named_embedder(segments="greeting", embedder="default")
    assert result == [
        104,
        101,
        108,
        108,
        111,
    ]  # Same result as "greeting" resolves to "hello"


@pytest.mark.asyncio
async def test_parameterized_function():
    """Test parameterizing a function retrieved from a store."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={
            "resolver": _get_function_from_store,
            "store_key": "embedders",
        },
    )
    async def embed_with_params(segments, embedder):
        return embedder(segments)

    # Use parameterized "advanced" embedder with multiplier=2
    result = await embed_with_params(
        segments="hello", embedder={"advanced": {"multiplier": 2}}
    )
    # Each ASCII value should be doubled
    assert result == [208, 202, 216, 216, 222]  # 2 * ASCII values of "hello"


@pytest.mark.asyncio
async def test_conditional_resolution():
    """Test conditional resolution based on input type."""
    # Define a condition that only resolves short strings (presumed to be keys)
    is_likely_key = lambda x: isinstance(x, str) and len(x) < 10

    @source_variables(
        segments={
            "resolver": resolve_data,
            "store_key": "segments",
            "condition": is_likely_key,  # Only resolve short strings
        },
    )
    async def conditional_embed(segments, embedder):
        return embedder(segments)

    # Case 1: Short string is treated as a key and resolved
    result = await conditional_embed(
        segments="greeting",  # Short string, will be resolved
        embedder=lambda t: [ord(c) for c in t],
    )
    assert result == [104, 101, 108, 108, 111]  # ASCII for "hello"

    # Case 2: Long string is treated as literal value
    long_text = "this is a very long string that should not be treated as a key"
    result = await conditional_embed(
        segments=long_text,  # Long string, will NOT be resolved
        embedder=lambda t: len(t),  # Just return the length
    )
    assert result == len(long_text)  # The length of the long string


@pytest.mark.asyncio
async def test_output_transformation():
    """Test transforming the output with an egress function."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
        egress=lambda x: {"embeddings": x},  # Wrap result in an object
    )
    async def embed_with_formatted_output(segments, embedder):
        return embedder(segments)

    # Test with resolved embedder and segments
    result = await embed_with_formatted_output(segments="greeting", embedder="default")

    # The result should be wrapped in an object with "embeddings" key
    assert "embeddings" in result
    assert result["embeddings"] == [104, 101, 108, 108, 111]  # ASCII for "hello"


@pytest.mark.asyncio
async def test_full_pipeline():
    """Test a complete processing pipeline with multiple resolution steps."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
        clusterer={"resolver": _get_function_from_store, "store_key": "clusterers"},
        egress=lambda x: {"clusters": x},
    )
    async def embed_and_cluster(segments, embedder, clusterer):
        # Core pipeline logic
        embeddings = embedder(segments)
        # Ensure embeddings is a list of embeddings for clusterer
        embeddings_list = (
            [embeddings] if not isinstance(embeddings[0], list) else embeddings
        )
        return clusterer(embeddings_list)

    # Test the complete pipeline
    result = await embed_and_cluster(
        segments="greeting", embedder="default", clusterer="default"
    )

    # ASCII sum of "hello" is 532, mod 3 = 1
    assert result == {"clusters": [1]}

    # Test with parameterized embedder
    result = await embed_and_cluster(
        segments="greeting",
        embedder={"advanced": {"multiplier": 2}},  # Double all values
        clusterer="default",
    )

    # Double ASCII sum is 1064, mod 3 = 2
    assert result == {"clusters": [2]}


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with the decorator."""

    @source_variables(
        segments={
            "resolver": resolve_data,
            "store_key": "segments",
            "mode": "store_only",
        },
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
    )
    async def embed_with_error_handling(segments, embedder):
        return embedder(segments)

    # Case 1: Invalid segment key (should raise KeyError in store_only mode)
    with pytest.raises(KeyError):
        await embed_with_error_handling(segments="nonexistent", embedder="default")

    # Case 2: Invalid embedder key
    with pytest.raises(KeyError) as excinfo:
        await embed_with_error_handling(segments="greeting", embedder="nonexistent")
    assert "not found in store" in str(excinfo.value)


@pytest.mark.asyncio
async def test_custom_mall_provider():
    """Test using a custom mall provider function."""

    # Define a function that provides a user-specific mall
    def get_user_mall(user_id=None):
        if user_id == "user1":
            return {
                "segments": {
                    "greeting": "hola",  # Spanish greeting
                },
                "embedders": mock_mall["embedders"],  # Reuse mock embedders
                "clusterers": mock_mall["clusterers"],  # Reuse mock clusterers
            }
        # Fall back to default mall
        return mock_mall

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
        mall=lambda: get_user_mall("user1"),  # Get user1's mall
    )
    async def embed_with_user_mall(segments, embedder):
        return embedder(segments)

    # Test with user-specific segment
    result = await embed_with_user_mall(segments="greeting", embedder="default")

    # Should get ASCII for "hola" (user1's greeting)
    assert result == [104, 111, 108, 97]


@pytest.mark.asyncio
async def test_sync_function_with_source_variables():
    """Test that the decorator works with synchronous functions too."""

    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
    )
    def sync_embed_text(segments, embedder):
        return embedder(segments)

    # Test with direct value
    result = await sync_embed_text(segments="test", embedder=lambda t: len(t))
    assert result == 4

    # Test with store resolution
    result = await sync_embed_text(segments="greeting", embedder=lambda t: len(t))
    assert result == 5  # Length of "hello"


def test_sync_resolve_data():
    """Test resolve_data with synchronous usage."""
    mall = {"test_store": {"key1": "value1", "key2": 42}}

    # Test successful resolution
    assert resolve_data("key1", "test_store", mall) == "value1"
    assert resolve_data("key2", "test_store", mall) == 42

    # Test fallback to original value
    assert resolve_data("nonexistent", "test_store", mall) == "nonexistent"
    assert resolve_data(123, "test_store", mall) == 123


@pytest.mark.asyncio
async def test_integration_with_fastapi():
    """Test how the decorator integrates with FastAPI routes."""

    # Mock request object
    class MockRequest:
        def __init__(self, json_data):
            self.json_data = json_data

        async def json(self):
            return self.json_data

    # Mock endpoint function (simulating FastAPI handler)
    async def fastapi_endpoint(request):
        data = await request.json()

        # Extract parameters from request
        segments = data.get("segments", "hello")
        embedder = data.get("embedder", "default")

        # Call our decorated function
        return await embed_with_api_format(segments=segments, embedder=embedder)

    # Define the core function with our decorator
    @source_variables(
        segments={'resolver': resolve_data, 'store_key': 'segments'},
        embedder={'resolver': _get_function_from_store, 'store_key': 'embedders'},
        egress=lambda x: {"embeddings": x, "status": "success"},
    )
    async def embed_with_api_format(segments, embedder):
        return embedder(segments)

    # Test 1: Simple request with direct values
    request = MockRequest({"segments": "test"})
    response = await fastapi_endpoint(request)
    assert response["status"] == "success"
    assert response["embeddings"] == [116, 101, 115, 116]  # ASCII for "test"

    # Test 2: Request with store keys
    request = MockRequest({"segments": "greeting", "embedder": "default"})
    response = await fastapi_endpoint(request)
    assert response["status"] == "success"
    assert response["embeddings"] == [104, 101, 108, 108, 111]  # ASCII for "hello"


# Add this to make the tests runnable from command line
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
