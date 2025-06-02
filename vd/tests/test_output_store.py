import pytest
from vd.wip.crude import prepare_for_crude_dispatch


# A simple function to add two numbers.
def add(a, b):
    return a + b


# Test that when a non-empty save_name is provided the output is stored.
def test_simple_output_store():
    output_store = {}
    # Wrap the function so that it takes a save_name parameter and stores output.
    wrapped_add = prepare_for_crude_dispatch(add, mall={}, output_store=output_store)
    result = wrapped_add(2, 3, save_name="addition_test")
    assert result == 5
    assert "addition_test" in output_store
    assert output_store["addition_test"] == 5


# A function that squares its input.
def square(x):
    return x * x


# Test that when no save_name is provided an auto_namer is used to compute the key.
def test_auto_namer_output_store():
    output_store = {}
    auto_namer = lambda *, arguments, output: f"square_{arguments['x']}"
    wrapped_square = prepare_for_crude_dispatch(
        square, mall={}, output_store=output_store, auto_namer=auto_namer
    )
    # Call without providing save_name; auto_namer should supply one.
    result = wrapped_square(4)
    assert result == 16
    # The auto-generated key should be "square_4"
    assert "square_4" in output_store
    assert output_store["square_4"] == 16


# A function that returns an iterable.
def range_list(n):
    return list(range(n))


# Test output storing with multiple values.
def test_multi_value_output_store():
    output_store = {}
    # For multi-value storing, save_name_param is None and store_multi_values is True.
    auto_namer = lambda *, arguments, output: f"r_{arguments['n']}_{output}"
    wrapped_range = prepare_for_crude_dispatch(
        range_list,
        mall={},
        output_store=output_store,
        store_multi_values=True,
        save_name_param=None,
        auto_namer=auto_namer,
    )
    result = wrapped_range(3)
    assert result == [0, 1, 2]
    # Each output item should be stored individually.
    for item in result:
        key = f"r_3_{item}"
        assert key in output_store
        assert output_store[key] == item
