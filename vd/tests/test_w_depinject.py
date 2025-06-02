import pytest
import asyncio
from dependency_injector import containers, providers
from vd.wip.w_depinject import input_wiring

# Language: python

class TestContainer(containers.DeclarativeContainer):
    store1 = providers.Singleton(dict)
    store2 = providers.Singleton(dict)
    store3 = providers.Singleton(dict)

@pytest.fixture
def mall():
    cont = TestContainer()
    cont.store1()['x'] = 10
    cont.store2()['y'] = 5
    return cont

def test_input_wiring_sync_no_save(mall):
    def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        mall=mall,
    )
    # Calling without the extra keyword `save_as`
    result = wrapped('x', 'y', 2)
    # Expected: 10 + (5*2) = 20
    assert result == 20
    # Since save_as not provided, output should not be stored.
    assert 'result' not in mall.store3()

def test_missing_key_sync(mall):
    def multiply(a, b):
        return a * b

    wrapped = input_wiring(
        multiply,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        mall=mall,
    )
    with pytest.raises(KeyError):
        wrapped('invalid', 'y')

def test_input_wiring_sync_with_save_raises_type_error(mall):
    def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        mall=mall,
    )
    # Expect TypeError due to unexpected keyword argument "save_as"
    with pytest.raises(TypeError):
        wrapped('x', 'y', 2, save_as='result')

@pytest.mark.asyncio
async def test_input_wiring_async_no_save(mall):
    async def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        mall=mall,
    )
    result = await wrapped('x', 'y', 2)
    assert result == 20
    assert 'result' not in mall.store3()

@pytest.mark.asyncio
async def test_input_wiring_async_with_save_raises_type_error(mall):
    async def add(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        add,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        mall=mall,
    )
    with pytest.raises(TypeError):
        await wrapped('x', 'y', 2, save_as='result')