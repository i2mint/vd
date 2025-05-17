"""Value Dispatch using Dependency Injectior Package"""

from dependency_injector import containers, providers
from functools import wraps
import inspect
from i2 import Sig  # Requires i2 package; alternatively, use inspect.signature
from typing import Optional


def io_wiring(
    func, global_param_to_store, output_store=None, container=None, save_param='save_as'
):
    """
    Wrap a function to wire its inputs and outputs to storage systems using Dependency Injector.

    Args:
        func: The function to wrap (sync or async).
        global_param_to_store: Dict mapping parameter names to store names (e.g., {'a': 'store1'}).
        output_store: Name of the store to save outputs (optional).
        container: Dependency Injector container holding the stores.
        save_param: Name of the parameter for specifying the output key (default: 'save_as').

    Returns:
        A wrapped function that resolves inputs from stores and optionally stores outputs.
    """
    if container is None:
        raise ValueError("A container must be provided to access storage systems.")

    sig = Sig(func)
    # Determine which parameters to resolve based on global_param_to_store
    param_to_store = {
        param: global_param_to_store[param]
        for param in sig.names
        if param in global_param_to_store
    }

    # Define the wrapper based on whether the function is async
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            save_key = kwargs.pop(save_param, None)  # pop save_as before binding
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Resolve inputs for parameters in param_to_store
            for param in param_to_store:
                key = bound_args.arguments[param]
                store_name = param_to_store[param]
                store = getattr(container, store_name)()
                if key not in store:
                    raise KeyError(f"Key '{key}' not found in store '{store_name}'")
                bound_args.arguments[param] = store[key]

            # Call the function with resolved arguments
            result = await func(*bound_args.args, **bound_args.kwargs)

            # Store output if specified
            if output_store and save_key:  # use save_key instead of kwargs
                output_store_instance = getattr(container, output_store)()
                output_store_instance[save_key] = result

            return result

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            save_key = kwargs.pop(save_param, None)  # pop save_as before binding
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param in param_to_store:
                key = bound_args.arguments[param]
                store_name = param_to_store[param]
                store = getattr(container, store_name)()
                if key not in store:
                    raise KeyError(f"Key '{key}' not found in store '{store_name}'")
                bound_args.arguments[param] = store[key]

            result = func(*bound_args.args, **bound_args.kwargs)

            if output_store and save_key:  # use save_key instead of kwargs
                output_store_instance = getattr(container, output_store)()
                output_store_instance[save_key] = result

            return result

    # Adjust the signature for better documentation
    new_params = [
        param.replace(annotation=str) if param.name in param_to_store else param
        for param in sig.params
    ]
    save_param_obj = inspect.Parameter(
        save_param,
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=Optional[str],
    )
    new_params.append(save_param_obj)
    new_sig = inspect.Signature(
        parameters=new_params, return_annotation=sig.return_annotation
    )
    wrapper.__signature__ = new_sig

    return wrapper


# Example container (for reference, typically defined by the user)
if __name__ == "__main__":

    class StorageContainer(containers.DeclarativeContainer):
        store1 = providers.Singleton(dict)
        store2 = providers.Singleton(dict)
        store3 = providers.Singleton(dict)

    container = StorageContainer()
    container.store1()['one'] = 1
    container.store2()['two'] = 2

    def my_function(a, b, c):
        return a + b * c

    wrapped = io_wiring(
        my_function,
        global_param_to_store={'a': 'store1', 'b': 'store2'},
        output_store='store3',
        container=container,
    )

    result = wrapped('one', 'two', 3, save_as='result_key')
    print(result)  # 7
    print(container.store3()['result_key'])  # 7
