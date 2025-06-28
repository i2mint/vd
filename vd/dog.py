import pytest
from typing import Callable, Any, Dict, List, Tuple, get_args
from functools import partial
from collections.abc import MutableMapping
from dol import Pipe

# --- Import au for async operations ---
from au.base import async_compute, FileSystemStore, SerializationFormat
import os
import tempfile


# --- Core Data Operation Graph (DOG) Implementation ---
class _DOG:
    def __init__(
        self,
        operation_signatures: Dict[str, Any],
        data_stores: Dict[str, Any],
        operation_implementations: Dict[str, Any],
        sourced_argnames: Any = None,  # can be None, True, or Dict[str, str]
    ):
        """
        Initializes the Data Operation Graph (DOG) with abstract operation definitions,
        data store configurations, and concrete function implementations.

        Args:
            operation_signatures: A dictionary mapping operation type names (str)
                                  to Callable type hints defining input/output types.
                                  Example: {'embedder': Callable[[Segments], Embeddings]}
            data_stores: A dictionary configuring data repositories. Keys are store names (str),
                         values are dicts with 'type' (a Python class/NewType for the data type)
                         and 'store' (a MutableMapping instance).
                         Example: {'segments': {'type': Segments, 'store': {}}}
            operation_implementations: A dictionary mapping operation type names (str)
                                       to dictionaries of concrete function implementations.
                                       Example: {'embedder': {'constant': lambda s: ..., ...}}
            sourced_argnames: A dictionary mapping argument names to data store names.
                              This is used to automatically source argument values from
                              the appropriate data stores based on the operation signatures.
                              Example: {'arg1': 'store1', 'arg2': 'store2'}
        """
        self.operation_signatures = operation_signatures
        self.operation_implementations = operation_implementations
        self._return_type_to_store_name_map = {}
        for store_name, store_config in data_stores.items():
            data_type_class = store_config['type']
            self._return_type_to_store_name_map[data_type_class] = store_name
        self.data_stores = {
            name: config['store'] for name, config in data_stores.items()
        }

        if sourced_argnames is True:
            # All argnames matching data_stores keys are sourced from stores of same name
            sourced_argnames = {k: k for k in self.data_stores.keys()}
        elif not sourced_argnames:
            sourced_argnames = {}
        else:
            sourced_argnames = dict(sourced_argnames)

        self.sourced_argnames = self._validate_sourced_argnames(sourced_argnames)
        self._output_counter = 0

    def _validate_sourced_argnames(self, sourced_argnames):
        """
        Normalize and validate sourced_argnames.
        If True, use all data_store keys as argnames.
        If dict, validate all stores exist.
        If None or empty, return {}.
        """
        for arg, store in sourced_argnames.items():
            if store not in self.data_stores:
                raise ValueError(
                    f"sourced_argnames: store '{store}' for argument '{arg}' is not a valid data store name"
                )
        return sourced_argnames

    # Note: This is a local crudifier (input sourcer)
    # TODO: Consider using crude.py or other general tool
    def _source_args(self, func_impl, args, kwargs):
        """Replace sourced arguments in kwargs with values from the appropriate data stores."""
        if not self.sourced_argnames:
            return args, kwargs
        import inspect

        sig = inspect.signature(func_impl)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        for argname, storename in self.sourced_argnames.items():
            if argname in bound.arguments:
                val = bound.arguments[argname]
                store = self.data_stores[storename]
                if isinstance(val, str) and val in store:
                    bound.arguments[argname] = store[val]
        return bound.args, bound.kwargs

    def _get_output_store_name_and_type(self, func_impl: Callable):
        output_store_name = None
        return_type_class = None
        # Step 1: Find the abstract operation signature that `func_impl` belongs to.
        # This involves iterating through `operation_implementations` to find `func_impl`.
        for op_type_name, impl_dict in self.operation_implementations.items():
            if func_impl in impl_dict.values():
                # Once the operation type name is found (e.g., 'embedder'),
                # retrieve its corresponding Callable signature.
                signature_callable = self.operation_signatures.get(op_type_name)

                if signature_callable:
                    # Step 2: Extract the return type class from the Callable signature.
                    # For Callable[[A, B], R], get_args returns (A, B, R). The last element is R.
                    signature_args = get_args(signature_callable)
                    if signature_args:  # Ensure there are arguments/return type defined
                        return_type_class = signature_args[-1]
                    break  # Found the signature, no need to search further

        # Step 3: Use the inferred return type class to determine the target data store name.
        if return_type_class:
            output_store_name = self._return_type_to_store_name_map.get(
                return_type_class
            )

        return output_store_name, return_type_class

    def _get_next_output_key(self, output_store_name):
        self._output_counter += 1
        # Use .json extension for stores that require it
        return f"output_{output_store_name}_{self._output_counter}.json"


class DOG(_DOG):
    def call(self, func_impl: Callable, *args, **kwargs) -> Tuple[str, str]:
        """
        Executes a given concrete function implementation with provided args.
        The function's output is then automatically stored in the appropriate
        data store, and a reference to its location is returned.

        Args:
            func_impl: The concrete function to execute (must be one from self.operation_implementations).
            *args: Variable positional arguments to be passed to func_impl.
            **kwargs: Variable keyword arguments to be passed to func_impl.

        Returns:
            A tuple (output_store_name, output_value_key) where:
                output_store_name (str): The name of the data store where the output was placed.
                output_value_key (str): The unique key under which the output is stored within that data store.

        Raises:
            ValueError: If the output store cannot be determined or is invalid.
        """
        args, kwargs = self._source_args(func_impl, args, kwargs)
        output_store_name, _ = self._get_output_store_name_and_type(func_impl)
        # Step 4: Validate that a target data store was successfully identified and exists.
        if not output_store_name or output_store_name not in self.data_stores:
            raise ValueError(
                f"Could not determine a valid output data store for function implementation: {func_impl}. "
                f"Mapped store name: '{output_store_name}'. "
                f"Available data stores: {list(self.data_stores.keys())}"
            )

        # Step 5: Execute the concrete function with the provided args & kwargs.
        output_data = func_impl(*args, **kwargs)

        # Step 6: Generate a unique key for the output and store it in the determined data store.
        output_key = self._get_next_output_key(output_store_name)
        self.data_stores[output_store_name][output_key] = output_data

        # Step 7: Return the reference (store name and key) to the stored output.
        return output_store_name, output_key


class ADOG(_DOG):
    def __init__(
        self,
        operation_signatures: Dict[str, Any],
        data_stores: Dict[str, Any],
        operation_implementations: Dict[str, Any],
        *,
        base_path: str = None,
        ttl_seconds: int = 3600,
        serialization: SerializationFormat = SerializationFormat.JSON,
        middleware: List[Any] = None,
        sourced_argnames: Dict[str, str] = None,
    ):
        super().__init__(
            operation_signatures,
            data_stores,
            operation_implementations,
            sourced_argnames=sourced_argnames,
        )
        if base_path is None:
            base_path = tempfile.mkdtemp(prefix="adog_store_")
        self._adog_base_path = base_path
        self._adog_ttl_seconds = ttl_seconds
        self._adog_serialization = serialization
        self._adog_middleware = middleware or []
        self._async_wrappers = {}
        # Only replace output stores (those that are used as outputs of operations)
        # Get the actual return types from operation signatures
        output_types = set()
        for op_signature in self.operation_signatures.values():
            signature_args = get_args(op_signature)
            if signature_args:  # Ensure there are arguments/return type defined
                return_type = signature_args[-1]  # Last element is the return type
                output_types.add(return_type)

        output_store_names = set()
        for output_type in output_types:
            store_name = self._return_type_to_store_name_map.get(output_type)
            if store_name:
                output_store_names.add(store_name)

        for store_name in output_store_names:
            store = self.data_stores[store_name]
            if not isinstance(store, FileSystemStore):
                store_path = os.path.join(self._adog_base_path, store_name)
                os.makedirs(store_path, exist_ok=True)
                self.data_stores[store_name] = FileSystemStore(
                    store_path,
                    ttl_seconds=self._adog_ttl_seconds,
                    serialization=self._adog_serialization,
                )

    def call(self, func_impl: Callable, *args, **kwargs) -> Tuple[str, str]:
        args, kwargs = self._source_args(func_impl, args, kwargs)
        output_store_name, _ = self._get_output_store_name_and_type(func_impl)
        if not output_store_name or output_store_name not in self.data_stores:
            raise ValueError(
                f"Could not determine a valid output data store for function implementation: {func_impl}. "
                f"Mapped store name: '{output_store_name}'. "
                f"Available data stores: {list(self.data_stores.keys())}"
            )
        output_store = self.data_stores[output_store_name]
        if func_impl not in self._async_wrappers:
            async_func = async_compute(
                store=output_store,
                ttl_seconds=self._adog_ttl_seconds,
                serialization=self._adog_serialization,
                middleware=self._adog_middleware,
            )(func_impl)
            self._async_wrappers[func_impl] = async_func
        else:
            async_func = self._async_wrappers[func_impl]
        handle = async_func(*args, **kwargs)
        return output_store_name, handle.key
