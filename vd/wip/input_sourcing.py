"""
Tests for the source_variables decorator pattern.

These tests demonstrate the functionality of the source_variables decorator,
which provides flexible variable resolution for API endpoints.
"""

from functools import partial, wraps
from typing import Dict, Any, Callable, List, Union, Tuple


# Mock implementation of resolve_data
def resolve_data(data, store_key, mall):
    """
    Resolve data from a store if it's a string key, otherwise return as is.

    This is a hybrid resolver that works with both direct values and store keys.
    """
    if isinstance(data, str) and store_key in mall and data in mall[store_key]:
        return mall[store_key][data]
    return data


# Mock implementation of _get_function_from_store
async def _get_function_from_store(key, store_key, mall):
    """
    Get a processing function from the mall.

    Handles both string keys and {key: params} dictionaries for parameterization.
    """
    if store_key not in mall:
        raise KeyError(f"Store '{store_key}' not found in mall")

    store = mall[store_key]

    # Handle dict case for parameterized functions
    if isinstance(key, dict):
        if len(key) != 1:
            raise KeyError(f"Dict key must contain exactly one item, got {len(key)}")

        func_key = next(iter(key))
        func_kwargs = key[func_key]

        if func_key not in store:
            raise KeyError(f"Key '{func_key}' not found in store '{store_key}'")

        # Get base function and create partial with kwargs
        base_func = store[func_key]
        return partial(base_func, **func_kwargs)

    # Handle string key case
    if key not in store:
        raise KeyError(f"Key '{key}' not found in store '{store_key}'")

    return store[key]


# Implementation of source_variables decorator
def source_variables(__var_store_suffix="s", **config):
    """
    Decorator to handle variable sourcing and transformation.

    Config options per variable:
    - resolver: Function to resolve the variable
    - store_key: Store to use for resolution
    - mode: 'hybrid' (default) or 'store_only'
    - condition: Function to determine if resolution should be applied
    - ingress: Function to transform store-retrieved value

    Special config keys:
    - mall: Function or value to use as the mall
    - egress: Function to transform the final result
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Handle special configs
            mall_provider = config.get("mall", lambda: mock_mall)
            egress_func = config.get("egress")

            # Get mall
            mall = mall_provider() if callable(mall_provider) else mall_provider

            # Process each configured variable
            for var_name, var_config in config.items():
                if var_name in ("mall", "egress"):  # Skip special configs
                    continue

                if var_name not in kwargs:  # Skip if variable not in kwargs
                    continue

                # Extract config or use defaults
                if callable(var_config):  # Simple resolver function
                    resolver = var_config
                    store_key = var_name + __var_store_suffix
                    mode = "hybrid"
                    condition = lambda x: True
                    ingress = lambda obj, v: obj
                else:  # Detailed config
                    resolver = var_config.get("resolver", resolve_data)
                    store_key = var_config.get(
                        "store_key", var_name + __var_store_suffix
                    )
                    mode = var_config.get("mode", "hybrid")
                    condition = var_config.get("condition", lambda x: True)
                    ingress = var_config.get("ingress", lambda obj, v: obj)

                # Apply resolution if condition is met
                value = kwargs[var_name]
                if condition(value):
                    if asyncio.iscoroutinefunction(resolver):
                        resolved_value = await resolver(value, store_key, mall)
                    else:
                        resolved_value = resolver(value, store_key, mall)

                    # New check: for store_only mode, raise error if resolution did not change the value.
                    if (
                        mode == "store_only"
                        and isinstance(value, str)
                        and resolved_value == value
                    ):
                        raise KeyError(
                            f"Key '{value}' not found in store '{store_key}'"
                        )

                    # Apply ingress transformation for dict-based inputs
                    if isinstance(value, dict) and len(value) == 1:
                        obj_key = next(iter(value))
                        obj_value = value[obj_key]
                        if (
                            isinstance(resolved_value, dict)
                            and obj_key in resolved_value
                        ):
                            base_obj = resolved_value[obj_key]
                            kwargs[var_name] = ingress(base_obj, obj_value)
                        else:
                            kwargs[var_name] = ingress(resolved_value, obj_value)
                    else:
                        kwargs[var_name] = resolved_value

            # Call the original function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Apply egress transformation if configured
            if egress_func:
                return egress_func(result)
            return result

        return wrapper

    return decorator


# -------------------------------------------------------------------------------------
# Test and demonstrate the functionality of the source_variables decorator


import pytest
import asyncio

# Configure pytest to use asyncio
pytest_plugins = ["pytest_asyncio"]


# Helper for async testing
async def async_call(func, *args, **kwargs):
    """Helper to call both sync and async functions."""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


#
# TESTS
#


# Mock mall and stores for testing
mock_mall = {
    "embedders": {
        "default": lambda text: (
            [ord(c) for c in text] if isinstance(text, str) else text
        ),
        "advanced": lambda text, multiplier=1: (
            [ord(c) * multiplier for c in text]
            if isinstance(text, str)
            else [v * multiplier for v in text]
        ),
    },
    "segments": {
        "greeting": "hello",
        "farewell": "goodbye",
    },
    "clusterers": {
        # Ensure clusterer always receives a list of embeddings (list of lists)
        "default": lambda embeddings: [
            sum(e) % 3
            for e in (embeddings if isinstance(embeddings[0], list) else [embeddings])
        ],
        "binary": lambda embeddings: [
            1 if sum(e) > 500 else 0
            for e in (embeddings if isinstance(embeddings[0], list) else [embeddings])
        ],
    },
}


@pytest.mark.asyncio
async def test_basic_resolution():
    """
    Demonstrates basic variable resolution with the source_variables decorator.

    This test shows how to resolve a segment key to its actual text value
    and then pass it to a simple embedding function.
    """

    # Define a simple function that just calls an embedder on segments
    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
    )
    async def embed_text(segments, embedder):
        # Core logic is simple: just call embedder on segments
        return embedder(segments)

    # Case 1: Pass direct values (no resolution needed)
    text = "test"
    embedder = lambda t: [ord(c) for c in t]
    result = await embed_text(segments=text, embedder=embedder)

    # The result should be ASCII values of "test"
    assert result == [116, 101, 115, 116]

    # Case 2: Resolve segment from store
    result = await embed_text(segments="greeting", embedder=embedder)

    # The result should be ASCII values of "hello" (from store)
    assert result == [104, 101, 108, 108, 111]


@pytest.mark.asyncio
async def test_function_resolution():
    """
    Demonstrates resolving a function from a store.

    This test shows how to resolve an embedder function by name and
    use it to process segments.
    """

    # Define a function that embeds text using a named embedder
    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
    )
    async def embed_with_named_embedder(segments, embedder):
        # Core logic remains simple despite the complex resolution
        return embedder(segments)

    # Case 1: Resolve embedder by name
    result = await embed_with_named_embedder(segments="hello", embedder="default")

    # The result should be ASCII values of "hello" using the default embedder
    assert result == [104, 101, 108, 108, 111]

    # Case 2: Resolve both segments and embedder from stores
    result = await embed_with_named_embedder(segments="greeting", embedder="default")

    # Same result as it resolves "greeting" to "hello"
    assert result == [104, 101, 108, 108, 111]


@pytest.mark.asyncio
async def test_parameterized_function():
    """
    Demonstrates parameterizing a function retrieved from a store.

    This test shows how to resolve and parameterize an embedder using
    the {name: params} dictionary pattern.
    """

    # Define a function that supports parameterized embedders
    @source_variables(
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={
            "resolver": _get_function_from_store,
            "store_key": "embedders",
            # No need for explicit ingress as the resolver already handles partials
        },
    )
    async def embed_with_params(segments, embedder):
        return embedder(segments)

    # Case: Use parameterized "advanced" embedder with multiplier=2
    result = await embed_with_params(
        segments="hello", embedder={"advanced": {"multiplier": 2}}
    )

    # Each ASCII value should be doubled
    assert result == [208, 202, 216, 216, 222]  # 2 * ASCII values of "hello"


@pytest.mark.asyncio
async def test_conditional_resolution():
    """
    Demonstrates conditional resolution based on input type.

    This test shows how to apply resolution only to certain inputs
    using a condition function.
    """
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
    """
    Demonstrates transforming the output with an egress function.

    This test shows how to format the output of the function according
    to API requirements.
    """

    # Define a function with output transformation
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
    """
    Demonstrates a complete processing pipeline with multiple resolution steps.

    This test shows how to create a pipeline that:
    1. Resolves segments from a store
    2. Resolves an embedder function from a store
    3. Embeds the segments
    4. Passes the embeddings to a clusterer
    5. Formats the output
    """

    # Define a complete pipeline function
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
async def test_integration_with_fastapi():
    """
    Demonstrates how the decorator integrates with FastAPI routes.

    This test shows a mock FastAPI endpoint that uses the decorator
    to handle parameter resolution.
    """

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
        segments={"resolver": resolve_data, "store_key": "segments"},
        embedder={"resolver": _get_function_from_store, "store_key": "embedders"},
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


@pytest.mark.asyncio
async def test_error_handling():
    """
    Demonstrates error handling with the decorator.

    This test shows how errors are propagated when store keys
    or stores don't exist.
    """

    # Define a function to test error handling
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

    # Case 1: Invalid segment key
    with pytest.raises(KeyError) as excinfo:
        await embed_with_error_handling(segments="nonexistent", embedder="default")

    # Case 2: Invalid embedder key
    with pytest.raises(KeyError) as excinfo:
        await embed_with_error_handling(segments="greeting", embedder="nonexistent")
    assert "not found in store" in str(excinfo.value)


@pytest.mark.asyncio
async def test_custom_mall_provider():
    """
    Demonstrates using a custom mall provider function.

    This test shows how to use a function that provides a user-specific mall,
    enabling support for multi-tenant systems.
    """

    # Define a function that provides a user-specific mall
    def get_user_mall(user_id=None):
        # In a real system, this would look up user-specific stores
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

    # Define a function that uses the user-specific mall
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


# Add this to make the tests runnable from command line
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
