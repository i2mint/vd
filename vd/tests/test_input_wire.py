import pytest
import asyncio

from vd.wip.input_wire import input_wiring

# Language: python


class MockStore:
    """Mock store that behaves like a dictionary"""

    def __init__(self, initial_data=None):
        self._data = initial_data or {}

    def __call__(self):
        """Make the store callable to match dependency_injector behavior"""
        return self._data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data


class MockMall:
    """Mock mall container that provides access to stores"""

    def __init__(self):
        self.store1 = MockStore({"x": 10})
        self.store2 = MockStore({"y": 5})
        self.store3 = MockStore()


@pytest.fixture
def mall():
    return MockMall()


def test_input_wiring_sync_no_save(mall):
    def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={"a": "store1", "b": "store2"},
        mall=mall,
    )
    # Calling without the extra keyword `save_as`
    result = wrapped("x", "y", 2)
    # Expected: 10 + (5*2) = 20
    assert result == 20
    # Since save_as not provided, output should not be stored.
    assert "result" not in mall.store3()


def test_missing_key_sync(mall):
    def multiply(a, b):
        return a * b

    wrapped = input_wiring(
        multiply,
        global_param_to_store={"a": "store1", "b": "store2"},
        mall=mall,
    )
    with pytest.raises(KeyError):
        wrapped("invalid", "y")


def test_input_wiring_sync_with_save_raises_type_error(mall):
    def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={"a": "store1", "b": "store2"},
        mall=mall,
    )
    # Expect TypeError due to unexpected keyword argument "save_as"
    with pytest.raises(TypeError):
        wrapped("x", "y", 2, save_as="result")


@pytest.mark.asyncio
async def test_input_wiring_async_no_save(mall):
    async def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={"a": "store1", "b": "store2"},
        mall=mall,
    )
    result = await wrapped("x", "y", 2)
    assert result == 20
    assert "result" not in mall.store3()


@pytest.mark.asyncio
async def test_input_wiring_async_with_save_raises_type_error(mall):
    async def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={"a": "store1", "b": "store2"},
        mall=mall,
    )
    with pytest.raises(TypeError):
        await wrapped("x", "y", 2, save_as="result")
