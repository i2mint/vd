"""Value Dispatch using Dependency Injectior Package"""

from functools import wraps
import inspect


# TODO: Make output_wiring too


def input_wiring(func, global_param_to_store, mall=None):
    """
    Wrap a function to source its inputs from stores.

    Args:
        func: The function to wrap (sync or async).
        global_param_to_store: Dict mapping parameter names to store names (e.g., {'a': 'store1'}).
        mall: The mapping container to access the stores that are referenced in global_param_to_store

    Returns:
        A wrapped function that resolves inputs from stores and optionally stores outputs.
    """
    if mall is None:
        raise ValueError("A mall must be provided to access storage systems.")

    sig = inspect.signature(func)
    # Determine which parameters to resolve based on global_param_to_store
    sig_names = sig.parameters.keys()
    param_to_store = {
        param: global_param_to_store[param]
        for param in sig_names
        if param in global_param_to_store
    }

    # New inner helper function to resolve inputs
    def resolve_bound_args(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        for param in param_to_store:
            key = bound_args.arguments[param]
            store_name = param_to_store[param]
            store = getattr(mall, store_name)()
            if key not in store:
                raise KeyError(f"Key '{key}' not found in store '{store_name}'")
            bound_args.arguments[param] = store[key]
        return bound_args

    # Define the wrapper based on whether the function is async
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            bound_args = resolve_bound_args(*args, **kwargs)
            # Call the function with resolved arguments
            result = await func(*bound_args.args, **bound_args.kwargs)

            return result

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = resolve_bound_args(*args, **kwargs)
            result = func(*bound_args.args, **bound_args.kwargs)

            return result

    return wrapper


def example_with_dependency_injector():
    # Example container (for reference, typically defined by the user)
    from dependency_injector import containers, providers

    class StorageContainer(containers.DeclarativeContainer):
        store1 = providers.Singleton(dict)
        store2 = providers.Singleton(dict)
        store3 = providers.Singleton(dict)

    container = StorageContainer()
    container.store1()["one"] = 1
    container.store2()["two"] = 2

    def my_function(a, b, c):
        return a + b * c

    wrapped = input_wiring(
        my_function,
        global_param_to_store={"a": "store1", "b": "store2"},
        output_store="store3",
        container=container,
    )

    result = wrapped("one", "two", 3)
    print(result)  # 7
