## __init__.py

```python
"""Core tools to build simple interfaces to complex data sources and bend the interface to your will (and need)"""

import os

# from contextlib import suppress

file_sep = os.path.sep


def kvhead(store, n=1):
    """Get the first item of a kv store, or a list of the first n items"""
    if n == 1:
        for k in store:
            return k, store[k]
    else:
        return [(k, store[k]) for i, k in enumerate(store) if i < n]


def ihead(store, n=1):
    """Get the first item of an iterable, or a list of the first n items"""
    if n == 1:
        for item in iter(store):
            return item
    else:
        return [item for i, item in enumerate(store) if i < n]


# from dol.base import (
#     Collection,  # base class for collections (adds to collections.abc.Collection)
#     MappingViewMixin,
#     KvReader,  # base class for kv readers (adds to collections.abc.Mapping)
#     KvPersister,  # base for kv persisters (adds to collections.abc.MutableMapping)
#     Reader,  # TODO: deprecate? (now KvReader)
#     Persister,  # TODO: deprecate? (now KvPersister)
#     kv_walk,  # walk a kv store
#     Store,  # base class for stores (adds hooks for key and value transforms)
#     BaseKeysView,  # base class for keys views
#     BaseValuesView,  # base class for values views
#     BaseItemsView,  # base class for items views
#     KT,  # Key type,
#     VT,  # Value type
# )


from dol.tools import (
    store_aggregate,  # aggregate stores keys and values into an aggregate object (e.g. string concatenation)
)

from dol.kv_codecs import ValueCodecs, KeyCodecs

from dol.base import (
    Collection,  # base class for collections (adds to collections.abc.Collection)
    MappingViewMixin,
    KvReader,  # base class for kv readers (adds to collections.abc.Mapping)
    KvPersister,  # base for kv persisters (adds to collections.abc.MutableMapping)
    Reader,  # TODO: deprecate? (now KvReader)
    Persister,  # TODO: deprecate? (now KvPersister)
    kv_walk,  # walk a kv store
    Store,  # base class for stores (adds hooks for key and value transforms)
)


from dol.base import KT, VT, BaseKeysView, BaseValuesView, BaseItemsView


from dol.zipfiledol import (
    zip_compress,
    zip_decompress,
    to_zip_file,
    ZipReader,
    ZipInfoReader,
    FilesOfZip,  # read-only access to files in a zip archive
    FileStreamsOfZip,
    FlatZipFilesReader,
    ZipFiles,  # read-write-delete access to files in a zip archive
    ZipStore,  # back-compatible alias of ZipFiles
    ZipFileStreamsReader,
    remove_mac_junk_from_zip,
    tar_compress,
    tar_decompress,
)

from dol.filesys import (
    Files,  # read-write-delete access to files; relative paths, bytes values
    FilesReader,  # read-only version of LocalFiles,
    TextFiles,  # read-write-delete access to text files; relative paths, str values
    ensure_dir,  # function to create a directory, if missing
    mk_dirs_if_missing,  # store deco to create directories on write, when missing
    MakeMissingDirsStoreMixin,  # Mixin to enable auto-dir-making on write
    resolve_path,  # to get a full path (resolve ~ and .),
    resolve_dir,  # to get a full path (resolve ~ and .) and ensure it is a directory
    DirReader,  # recursive read-only access to directories,
    temp_dir,  # make a temporary directory,
    PickleFiles,  # CRUD access to pickled files
    JsonFiles,  # CRUD access to jsob files,
    Jsons,  # Same as JsonFiles, but with added .json extension handling
    create_directories,
    process_path,
    subfolder_stores,  # a store of stores, each store corresponding to a subfolder
)

from dol.util import (
    non_colliding_key,  # make a key that does not collide with existing keys
    get_app_folder,  # get the path to a folder for application data
    get_app_config_folder,  # get the path to a folder for application config
    AttributeMapping,  # a mapping that provides attribute-access to the keys that are valid attribute names
    AttributeMutableMapping,  # a mutable mapping version of AttributeMapping
    Pipe,  # chain functions
    lazyprop,  # lazy evaluation of properties
    partialclass,  # partial class instantiation
    groupby,  # group items according to group keys
    regroupby,  # recursive version of groupby
    igroupby,
    not_a_mac_junk_path,  # filter function to filter out mac junk paths
    instance_checker,  # make filter function that checks the type of an object
    chain_get,  # a function to perform chained get operations (i.e. path keys get)
    written_bytes,  # transform a file-writing function into a bytes-writing function
    written_key,  # writes an object to a key and returns the key.
    read_from_bytes,  # transform a file-reading function into a bytes-reading function
)

from dol.trans import (
    wrap_kvs,  # transform store key and/or value
    filt_iter,  # filter store keys (and contains ready to use filters as attributes)
    cached_keys,  # cache store keys
    add_decoder,  # add a decoder (i.e. outcomming value transformer) to a store
    add_ipython_key_completions,  # add ipython key completions
    insert_hash_method,  # add a hash method to store
    add_path_get,  # add a path_get method to store
    add_path_access,  # add path_get and path_set methods to store
    flatten,  # flatten a nested store
    kv_wrap,  # different interface to wrap_kvs
    disable_delitem,  # disable ability to delete
    disable_setitem,  # disable ability to write to a store
    mk_read_only,  # disable ability to write to a store or delete its keys
    add_aliases,  # delegation-wrap any object and add aliases for its methods
    insert_aliases,  # insert aliases for special (dunder) store methods,
    add_missing_key_handling,  # add a missing key handler to a store
    cache_iter,  # being deprecated
    store_decorator,  # Helper to make store decorators
    redirect_getattr_to_getitem,  # redirect attribute access to __getitem__
)

from dol.caching import (
    cache_this,  # cache the result of "property" methods in a store
    add_extension,  # a helper (for cache_this) to make key functions
    lru_cache_method,  # A decorator to cache the result of a method, ignoring the first argument
    WriteBackChainMap,  # write-back cache
    mk_cached_store,  # (old alias of cache_vals) wrap a store so it uses a cache
    cache_vals,  # wrap a store so it uses a cache
    store_cached,  # func memorizer using a specific store as its "memory"
    store_cached_with_single_key,
    ensure_clear_to_kv_store,  # add a clear method to a store (removed by default)
    flush_on_exit,  # make a store become a context manager that flushes cache on exit
    mk_write_cached_store,
)

from dol.appendable import mk_item2kv_for, appendable

from dol.naming import (
    StrTupleDict,  # convert from and to strings, tuples, and dicts.
    mk_store_from_path_format_store_cls,
)

from dol.paths import (
    flatten_dict,  # flatten a nested Mapping, getting a dict
    leaf_paths,  # get the paths to the leaves of a Mapping
    KeyTemplate,  # express strings, tuples, and dict keys from a string template
    mk_relative_path_store,  # transform path store into relative path store
    KeyPath,  # a class to represent a path to a key
    paths_getter,  # to make mapping extractors that use path_get
    path_get,  # get a value from a path
    path_set,  # set a value from a path
    path_filter,  # search through paths of a Mapping
    add_prefix_filtering,  # add a prefix filtering method to a store
    # PathMappedData,  # A mapping that extracts data from a mapping according to paths
)

from dol.dig import trace_getitem  # trace getitem calls, stepping through the layers

from dol.explicit import ExplicitKeyMap, invertible_maps, KeysReader


from dol.sources import (
    FlatReader,  # A flat view of a store of stores (a sort of union of stores)
    SequenceKvReader,
    FuncReader,
    Attrs,
    ObjReader,
    FanoutReader,
    FanoutPersister,
    CascadedStores,  # multi-store writes to all stores and reads from first store.
)


def __getattr__(name):
    """Handle deprecated imports at module level."""
    if name == "get_app_data_folder":
        import warnings

        warnings.warn(
            "`get_app_data_folder` is deprecated. Use `get_app_config_folder` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_app_config_folder
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

## appendable.py

```python
"""
Tools to add append-functionality to key-val stores. The main function is
    `appendable_store_cls = add_append_functionality_to_store_cls(store_cls, item2kv, ...)`
You give it the `store_cls` you want to sub class, and a item -> (key, val) function, and you get a store (subclass) that
has a `store.append(item)` method. Also includes an extend method (that just called appends in a loop.

See add_append_functionality_to_store_cls docs for examples.
"""

import time
import types
from typing import Optional
from collections.abc import Callable

from dol.trans import store_decorator
from dol.util import exhaust

utc_now = time.time


def define_extend_as_seq_of_appends(obj):
    """Inject an extend method in obj that will used append method.

    Args:
        obj: Class (type) or instance of an object that has an "append" method.

    Returns: The obj, but with that extend method.

    >>> class A:
    ...     def __init__(self):
    ...         self.t = list()
    ...     def append(self, item):
    ...         self.t.append(item)
    ...
    >>> AA = define_extend_as_seq_of_appends(A)
    >>> a = AA()
    >>> a.extend([1,2,3])
    >>> a.t
    [1, 2, 3]
    >>> a.extend([10, 20])
    >>> a.t
    [1, 2, 3, 10, 20]
    >>> a = A()
    >>> a = define_extend_as_seq_of_appends(a)
    >>> a.extend([1,2,3])
    >>> a.t
    [1, 2, 3]
    >>> a.extend([10, 20])
    >>> a.t
    [1, 2, 3, 10, 20]

    """
    assert hasattr(
        obj, "append"
    ), f"Your object needs to have an append method! Object was: {obj}"

    def extend(self, items):
        for item in items:
            self.append(item)

    if isinstance(obj, type):
        obj = type(obj.__name__, (obj,), {})
        obj.extend = extend
    else:
        obj.extend = types.MethodType(extend, obj)
    return obj


########################################################################################################################


class NotSpecified:
    pass


class mk_item2kv_for:
    """A bunch of functions to make item2kv functions

    A few examples (see individual methods' docs for more examples)

    >>> # item_to_key
    >>> item2kv = mk_item2kv_for.item_to_key(item2key=lambda item: item['L'] )
    >>> item2kv({'L': 'let', 'I': 'it', 'G': 'go'})
    ('let', {'L': 'let', 'I': 'it', 'G': 'go'})
    >>>
    >>> # utc_key
    >>> import time
    >>> item2key = mk_item2kv_for.utc_key()
    >>> k, v = item2key('some data')
    >>> assert abs(time.time() - k) < 0.01  # which asserts that k is indeed a (current) utc timestamp
    >>> assert v == 'some data'  # just the item itself
    >>>
    >>> # item_to_key_params_and_val
    >>> item_to_kv = mk_item2kv_for.item_to_key_params_and_val(lambda x: ((x['L'], x['I']), x['G']), '{}/{}')
    >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
    ('let/it', 'go')
    >>>
    >>> # fields
    >>> item_to_kv = mk_item2kv_for.fields(['L', 'I'])
    >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
    ({'L': 'let', 'I': 'it'}, {'G': 'go'})
    >>> item_to_kv = mk_item2kv_for.fields(('G', 'L'), keep_field_in_value=True)
    >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})  # note the order of the key is not ('G', 'L')...
    ({'L': 'let', 'G': 'go'}, {'L': 'let', 'I': 'it', 'G': 'go'})
    >>> item_to_kv = mk_item2kv_for.fields(('G', 'L'), key_as_tuple=True)  # but ('G', 'L') order is respected here
    >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
    (('go', 'let'), {'I': 'it'})
    """

    @staticmethod
    def attr(attr_name):
        """Make item2kv from an attribute name (the value will be the item itself).

        Args:
            attr_name: The attribute name to use as the key
        Returns: an item -> (key, val) function

        >>> ref_getter =mk_item2kv_for.attr("ref")
        >>> from collections import namedtuple
        >>> A = namedtuple('A', ['ref'])
        >>> a = A(ref='some_ref')
        >>> ref_getter(a)
        ('some_ref', A(ref='some_ref'))

        """

        def item2kv(item):
            return getattr(item, attr_name), item

        return item2kv

    @staticmethod
    def kv_pairs():
        """
        Essentially, the identity. Is used when the items are already (key, val) pairs.
        """

        def item2kv(item):
            return item

        return item2kv

    @staticmethod
    def item_to_key(item2key):
        """Make item2kv from a item2key function (the value will be the item itself).

        Args:
            item2key: an item -> key function

        Returns: an item -> (key, val) function

        >>> item2key = lambda item: item['G']  # use value of 'L' as the key
        >>> item2key({'L': 'let', 'I': 'it', 'G': 'go'})
        'go'
        >>> item2kv = mk_item2kv_for.item_to_key(item2key)
        >>> item2kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ('go', {'L': 'let', 'I': 'it', 'G': 'go'})
        """

        def item2kv(item):
            return item2key(item), item

        return item2kv

    @staticmethod
    def field(field, keep_field_in_value=True, dflt_if_missing=NotSpecified):
        """item2kv that uses a specific key of a (mapping) item as the key

        Note: If keep_field_in_value=False, the field will be popped OUT of the item.
         If that's not the desired effect, one should feed copies of the items (e.g. map(dict.copy, items))

        :param field: The field (value) to use as the returned key
        :param keep_field_in_value: Whether to leave the field in the item. If False, will pop it out
        :param dflt_if_missing: If specified (even None) will use the specified key as the key, if the field is missig
        :return: A item2kv function

        >>> item2kv = mk_item2kv_for.field('G')
        >>> item2kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ('go', {'L': 'let', 'I': 'it', 'G': 'go'})
        >>> item2kv = mk_item2kv_for.field('G', keep_field_in_value=False)
        >>> item2kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ('go', {'L': 'let', 'I': 'it'})
        >>> item2kv = mk_item2kv_for.field('G', dflt_if_missing=None)
        >>> item2kv({'L': 'let', 'I': 'it', 'DIE': 'go'})
        (None, {'L': 'let', 'I': 'it', 'DIE': 'go'})

        """
        if dflt_if_missing is NotSpecified:
            if keep_field_in_value:

                def item2kv(item):
                    return item[field], item

            else:

                def item2kv(item):
                    return item.pop(field), item

        else:
            if keep_field_in_value:

                def item2kv(item):
                    return item.get(field, dflt_if_missing), item

            else:

                def item2kv(item):
                    return item.pop(field, dflt_if_missing), item

        return item2kv

    @staticmethod
    def utc_key(offset_s=0, factor=1, *, time_postproc: Callable | None = None):
        """Make an item2kv function that uses the current time as the key, and the unchanged item as a value.
        The offset_s, which is added to the output key, can be used, for example, to align to another system's clock,
        or to get a more accurate timestamp of an event.

        Use case for offset_s:
            * Align to another system's clock
            * Get more accurate timestamping of an event. For example, in situations where the item is a chunk of live
            streaming data and we want the key (timestamp) to represent the timestamp of the beginning of the chunk.
            Without an offset_s, the timestamp would be the timestamp after the last byte of the chunk was produced,
            plus the time it took to reach the present function. If we know the data production rate (e.g. sample rate)
            and the average lag to get to the present function, we can get a more accurate timestamp for the beginning
            of the chunk

        Args:
            offset_s: An offset (in seconds, possibly negative) to add to the current time.

        Returns: an item -> (current_utc_s, item) function

        >>> import time
        >>> item2key = mk_item2kv_for.utc_key()
        >>> k, v = item2key('some data')
        >>> assert abs(time.time() - k) < 0.01  # which asserts that k is indeed a (current) utc timestamp
        >>> assert v == 'some data'  # just the item itself

        """
        if time_postproc is None:

            def item2kv(item):
                # Note: If real time accuracy is needed, you should use your own optimized
                # item2kv function.
                return factor * utc_now() + offset_s, item

        else:

            def item2kv(item):
                # Note: If real time accuracy is needed, you should use your own optimized
                # item2kv function.
                return time_postproc(factor * utc_now() + offset_s), item

        return item2kv

    @staticmethod
    def uuid_key(hex=True):
        """Make an item2kv function that uses a uuid hex as the key.
        This uses uuid.uuid1() to generate the uuid, so it is not cryptographically
        secure, and it may not be unique if you generate more than 10k uuids per second.
        One advantage though, is that the uuid is time-based, so it can be used to sort
        the keys in the order they were IDed.

        Returns: an item -> (uuid, item) function

        >>> import uuid
        >>> item2key = mk_item2kv_for.uuid_key()
        >>> k, v = item2key('some data')
        >>> v
        'some data'
        >>> isinstance(k, str)
        True
        >>> k  # doctest: +SKIP
        '0d8a9930aaf411ee9f605e03a02258c9'
        >>> item2key = mk_item2kv_for.uuid_key(hex=False)
        >>> k, v = item2key('some data')
        >>> isinstance(k, uuid.UUID)
        True

        """
        import uuid

        if hex:

            def item2kv(item):
                return uuid.uuid1().hex, item

        else:

            def item2kv(item):
                return uuid.uuid1(), item

        return item2kv

    @staticmethod
    def item_to_key_params_and_val(item_to_key_params_and_val, key_str_format):
        """Make item2kv from a function that produces key_params and val,
        and a key_template that will produce a string key from the key_params

        Args:
            item_to_key_params_and_val: an item -> (key_params, val) function
            key_str_format: A string format such that
                    key_str_format.format(*key_params) or
                    key_str_format.format(**key_params)
                will produce the desired key string

        Returns: an item -> (key, val) function

        >>> # Using tuple key params with unnamed string format fields
        >>> item_to_kv = mk_item2kv_for.item_to_key_params_and_val(lambda x: ((x['L'], x['I']), x['G']), '{}/{}')
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ('let/it', 'go')
        >>>
        >>> # Using dict key params with named string format fields
        >>> item_to_kv = mk_item2kv_for.item_to_key_params_and_val(
        ...                             lambda x: ({'second': x['L'], 'first': x['G']}, x['I']), '{first}_{second}')
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ('go_let', 'it')
        """

        def item2kv(item):
            key_params, val = item_to_key_params_and_val(item)
            if isinstance(key_params, dict):
                return key_str_format.format(**key_params), val
            else:
                return key_str_format.format(*key_params), val

        return item2kv

    @staticmethod
    def fields(fields, keep_field_in_value=False, key_as_tuple=False):
        """Make item2kv from specific fields of a Mapping (i.e. dict-like object) item.

        Note: item2kv will not mutate item (even if keep_field_in_value=False).

        Args:
            fields: The sequence (list, tuple, etc.) of item fields that should be used to create the key.
            keep_field_in_value: Set to True to return the item as is, as the value
            key_as_tuple: Set to True if you want keys to be tuples (note that the fields order is important here!)

        Returns: an item -> (item[fields], item[not in fields]) function

        >>> item_to_kv = mk_item2kv_for.fields('L')
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ({'L': 'let'}, {'I': 'it', 'G': 'go'})
        >>> item_to_kv = mk_item2kv_for.fields(['L', 'I'])
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
        ({'L': 'let', 'I': 'it'}, {'G': 'go'})
        >>> item_to_kv = mk_item2kv_for.fields(('G', 'L'), keep_field_in_value=True)
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})  # note the order of the key is not ('G', 'L')...
        ({'L': 'let', 'G': 'go'}, {'L': 'let', 'I': 'it', 'G': 'go'})
        >>> item_to_kv = mk_item2kv_for.fields(('G', 'L'), key_as_tuple=True)  # but ('G', 'L') order is respected here
        >>> item_to_kv({'L': 'let', 'I': 'it', 'G': 'go'})
        (('go', 'let'), {'I': 'it'})

        """
        if isinstance(fields, str):
            fields_set = {fields}
            fields = (fields,)
        else:
            fields_set = set(fields)

        def item2kv(item):
            if keep_field_in_value:
                key = dict()
                for k, v in item.items():
                    if k in fields_set:
                        key[k] = v
                val = item
            else:
                key = dict()
                val = dict()
                for k, v in item.items():
                    if k in fields_set:
                        key[k] = v
                    elif not keep_field_in_value:
                        val[k] = v

            if key_as_tuple:
                return tuple(key[f] for f in fields), val
            else:
                return key, val

        return item2kv


@store_decorator
def appendable(store_cls=None, *, item2kv, return_keys=False):
    """Makes a new class with append (and consequential extend) methods

    Args:
        store_cls: The store class to subclass
        item2kv: The function that produces a (key, val) pair from an item
        new_store_name: The name to give the new class (default will be 'Appendable' + store_cls.__name__)

    Returns: A subclass of store_cls with two additional methods: append, and extend.


    >>> item_to_kv = lambda item: (item['L'], item)  # use value of 'L' as the key, and value is the item itself
    >>> MyStore = appendable(dict, item2kv=item_to_kv)
    >>> s = MyStore(); s.append({'L': 'let', 'I': 'it', 'G': 'go'}); list(s.items())
    [('let', {'L': 'let', 'I': 'it', 'G': 'go'})]

    Use mk_item2kv.from_item_to_key_params_and_val with tuple key params

    >>> item_to_kv = appendable.mk_item2kv_for.item_to_key_params_and_val(lambda x: ((x['L'], x['I']), x['G']), '{}/{}')
    >>> MyStore = appendable(item2kv=item_to_kv)(dict)  # showing the append(...)(store) form
    >>> s = MyStore(); s.append({'L': 'let', 'I': 'it', 'G': 'go'}); list(s.items())
    [('let/it', 'go')]

    Use mk_item2kv.from_item_to_key_params_and_val with dict key params

    >>> item_to_kv = appendable.mk_item2kv_for.item_to_key_params_and_val(
    ...     lambda x: ({'L': x['L'], 'G': x['G']}, x['I']), '{G}_{L}')
    >>> @appendable(item2kv=item_to_kv)  # showing the @ form
    ... class MyStore(dict):
    ...     pass
    >>> s = MyStore(); s.append({'L': 'let', 'I': 'it', 'G': 'go'}); list(s.items())
    [('go_let', 'it')]

    Use mk_item2kv.fields to get a tuple key from item fields,
    defining the sub-dict of the remaining fields to be the value.
    Also showing here how you can decorate the instance itself.

    >>> item_to_kv = appendable.mk_item2kv_for.fields(['G', 'L'], key_as_tuple=True)
    >>> d = {}
    >>> s = appendable(d, item2kv=item_to_kv)
    >>> s.append({'L': 'let', 'I': 'it', 'G': 'go'}); list(s.items())
    [(('go', 'let'), {'I': 'it'})]

    You can make the "append" and "extend" methods to return the new generated keys by
    using the "return_keys" flag.

    >>> d = {}
    >>> s = appendable(d, item2kv=item_to_kv, return_keys=True)
    >>> s.append({'L': 'let', 'I': 'it', 'G': 'go'})
    ('go', 'let')
    """

    def append(self, item):
        k, v = item2kv(item)
        self[k] = v
        if return_keys:
            return k

    def extend(self, items):
        def gen_keys():
            for item in items:
                yield self.append(item)

        gen = gen_keys()
        if return_keys:
            return list(gen)
        exhaust(gen)

    return type(
        "Appendable" + store_cls.__name__,
        (store_cls,),
        {"append": append, "extend": extend},
    )


add_append_functionality_to_store_cls = appendable  # for back compatibility

appendable.mk_item2kv_for = mk_item2kv_for  # adding as attribute for convenient access

from collections.abc import Sequence
from typing import Optional
from collections.abc import Iterable

NotAVal = type("NotAVal", (), {})()  # singleton instance to distinguish from None


from collections.abc import MutableMapping
from functools import partial
from operator import add


def read_add_write(store, key, iterable, add_iterables=add):
    """Retrieves"""
    if key in store:
        store[key] = add_iterables(store[key], iterable)
    else:
        store[key] = iterable


class Extender:
    """Extends a value in a store.

    The value in the store (if it exists) must be an iterable.
    The value to extend must also be an iterable.

    Unless a different ``extend_store_value`` function is given,
    the sum of the two iterables must be an iterable.

    The default ``extend_store_value`` is such that if the key is not in the store,
    the value is simply written in the store.

    The default ``append_method`` is ``None``, which means that the ``append`` method
    is not defined. If you want to define it, you can pass a function that takes
    the ``Extender`` instance as first argument, and the object to append as second
    argument. The ``append`` method will then be defined as a partial of this function
    with the ``Extender`` instance as first argument.

    >>> store = {'a': 'pple'}
    >>> # test normal extend
    >>> a_extender = Extender(store, 'a')
    >>> a_extender.extend('sauce')
    >>> store
    {'a': 'pplesauce'}
    >>> # test creation (when key is not in store)
    >>> b_extender = Extender(store, 'b')
    >>> b_extender.extend('anana')
    >>> store
    {'a': 'pplesauce', 'b': 'anana'}
    >>> # you can use the += operator too
    >>> b_extender += ' split'
    >>> store
    {'a': 'pplesauce', 'b': 'anana split'}

    """

    def __init__(
        self,
        store: MutableMapping,
        key,
        *,
        extend_store_value=read_add_write,
        append_method=None,
    ):
        self.store = store
        self.key = key
        self.extend_store_value = extend_store_value

        # Note: Not sure this is a good idea.
        # Note:   I'm not documenting it or testing it until I let class mature.
        # Note: Yes, I tried making this a method of the class, but it became ugly.
        if append_method is not None:
            self.append = partial(append_method, self)

    def extend(self, iterable):
        """Extend the iterable stored in"""
        return self.extend_store_value(self.store, self.key, iterable)

    __iadd__ = extend  # Note: Better to forward dunders to non-dunder-methods

    # TODO: Should we even have this? Is it violating the purity of the class?
    @property
    def value(self):
        return self.store[self.key]


#
# class FixedSizeStack(Sequence):
#     """A finite Sequence that can have no more than one element.
#
#     >>> t = FixedSizeStack(maxsize=1)
#     >>> assert len(t) == 0
#     >>>
#     >>> t.append('something')
#     >>> assert len(t) == 1
#     >>> assert t[0] == 'something'
#     >>>
#     >>> t.append('something else')
#     >>> assert len(t) == 1  # still only one item
#     >>> assert t[0] == 'something'  # still the same item
#
#     Not that we'd ever these methods of FirstAppendOnly,
#     but know that FirstAppendOnly is a collection.abc.Sequence, so...
#
#     >>> t[:1] == t[:10] == t[::-1] == t[::-10] == t[0:2:10] == list(reversed(t)) == ['something']
#     True
#     >>>
#     >>> assert t.count('something') == 1
#     >>> assert t.index('something') == 0
#
#     """
#
#     def __init__(self, iterable: Optional[Iterable] = None, *, maxsize: int):
#         self.maxsize = maxsize
#         self.data = [NotAVal] * maxsize
#         # self.data = (isinstance(iterable, Iterable) and list(iterable)) or []
#         # if iterable is not None:
#         #     pass
#         self.cursor = 0
#
#     def append(self, v):
#         if self.cursor < self.maxsize:
#             self.data[self.cursor] = v
#             self.cursor += 1
#
#     def __len__(self):
#         return self.cursor
#
#     def __getitem__(self, k):
#         if isinstance(k, int):
#             if k < self.cursor:
#                 return self.data[k]
#             else:
#                 raise IndexError(
#                     f"There are only {len(self)} items: You asked for self[{k}]."
#                 )
#         elif isinstance(k, slice):
#             return self.data[: self.cursor][k]
#         else:
#             raise IndexError(
#                 f"A {self.__class__} instance can only have one value, or none at all."
#             )
#


class FirstAppendOnly(Sequence):
    """A finite Sequence that can have no more than one element.

    >>> t = FirstAppendOnly()
    >>> assert len(t) == 0
    >>>
    >>> t.append('something')
    >>> assert len(t) == 1
    >>> assert t[0] == 'something'
    >>>
    >>> t.append('something else')
    >>> assert len(t) == 1  # still only one item
    >>> assert t[0] == 'something'  # still the same item
    >>>
    >>> # Not that we'd ever these methods of FirstAppendOnly, but know that FirstAppendOnly is a collection.abc.Sequence, so...
    >>> t[:1] == t[:10] == t[::-1] == t[::-10] == t[0:2:10] == list(reversed(t)) == ['something']
    True
    >>>
    >>> t.count('something') == 1
    True
    >>> t.index('something') == 0
    True
    """

    def __init__(self):
        self.val = NotAVal

    def append(self, v):
        if self.val == NotAVal:
            self.val = v

    def __len__(self):
        return int(self.val != NotAVal)

    def __getitem__(self, k):
        if len(self) == 0:
            raise IndexError(f"There are no items in this {self.__class__} instance")
        elif k == 0:
            return self.val
        elif isinstance(k, slice):
            return [self.val][k]
        else:
            raise IndexError(
                f"A {self.__class__} instance can only have one value, or none at all."
            )

    # @staticmethod
    # def from


# def add_append_functionality_to_str_key_store(store_cls,
#                                               item_to_key_params_and_val,
#                                               key_template=None,
#                                               new_store_name=None):
#     def item_to_kv(item):
#         nonlocal key_template
#         if key_template is None:
#             key_params, _ = item_to_key_params_and_val(item)
#             key_template = path_sep.join('{{{}}}'.format(p) for p in key_params)
#         key_params, val = item_to_key_params_and_val(item)
#         return key_template.format(**key_params), val
#
#     return add_append_functionality_to_store_cls(store_cls, item_to_kv, new_store_name)
```

## base.py

```python
"""
Base classes for making stores.
In the language of the collections.abc module, a store is a MutableMapping that is configured to work with a specific
representation of keys, serialization of objects (python values), and persistence of the serialized data.

That is, stores offer the same interface as a dict, but where the actual implementation of writes, reads, and listing
are configurable.

Consider the following example. You're store is meant to store waveforms as wav files on a remote server.
Say waveforms are represented in python as a tuple (wf, sr), where wf is a list of numbers and sr is the sample
rate, an int). The __setitem__ method will specify how to store bytes on a remote server, but you'll need to specify
how to SERIALIZE (wf, sr) to the bytes that constitute that wav file: _data_of_obj specifies that.
You might also want to read those wav files back into a python (wf, sr) tuple. The __getitem__ method will get
you those bytes from the server, but the store will need to know how to DESERIALIZE those bytes back into a python
object: _obj_of_data specifies that

Further, say you're storing these .wav files in /some/folder/on/the/server/, but you don't want the store to use
these as the keys. For one, it's annoying to type and harder to read. But more importantly, it's an irrelevant
implementation detail that shouldn't be exposed. THe _id_of_key and _key_of_id pair are what allow you to
add this key interface layer.

These key converters object serialization methods default to the identity (i.e. they return the input as is).
This means that you don't have to implement these as all, and can choose to implement these concerns within
the storage methods themselves.
"""

from functools import partial, update_wrapper
import copyreg
from collections.abc import Collection as CollectionABC
from collections.abc import Mapping, MutableMapping
from collections.abc import (
    KeysView as BaseKeysView,
    ValuesView as BaseValuesView,
    ItemsView as BaseItemsView,
    Set,
)
from typing import Any, Tuple, Union, Optional
from collections.abc import Iterable, Callable

from dol.util import (
    wraps,
    _disabled_clear_method,
    identity_func,
    static_identity_method,
    Key,
    Val,
    Id,
    Data,
    Item,
    KeyIter,
    ValIter,
    ItemIter,
    is_unbound_method,
    is_classmethod,
)
from dol.signatures import Sig


class AttrNames:
    CollectionABC = {"__len__", "__iter__", "__contains__"}
    Mapping = CollectionABC | {
        "keys",
        "get",
        "items",
        "__reversed__",
        "values",
        "__getitem__",
    }
    MutableMapping = Mapping | {
        "setdefault",
        "pop",
        "popitem",
        "clear",
        "update",
        "__delitem__",
        "__setitem__",
    }

    Collection = CollectionABC | {"head"}
    KvReader = (Mapping | {"head"}) - {"__reversed__"}
    KvPersister = (MutableMapping | {"head"}) - {"__reversed__"} - {"clear"}


# TODO: Consider using ContainmentChecker and Sizer attributes which dunders would
#  point to.
class Collection(CollectionABC):
    """The same as collections.abc.Collection, with some modifications:
    - Addition of a ``head``
    """

    def __contains__(self, x) -> bool:
        """
        Check if collection of keys contains k.
        Note: This method loops through all contents of collection to see if query element exists.
        Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
        :return: True if k is in the collection, and False if not
        """
        for existing_x in iter(self):
            if existing_x == x:
                return True
        return False

    def __len__(self) -> int:
        """
        Number of elements in collection of keys.
        Note: This method iterates over all elements of the collection and counts them.
        Therefore it is not efficient, and in most cases should be overridden with a more efficient version.
        :return: The number (int) of elements in the collection of keys.
        """
        # Note: Found that sum(1 for _ in self.__iter__()) was slower for small, slightly faster for big inputs.
        count = 0
        for _ in iter(self):
            count += 1
        return count

    def head(self):
        if hasattr(self, "items"):
            return next(iter(self.items()))
        else:
            return next(iter(self))


# KvCollection = Collection  # alias meant for back-compatibility. Would like to deprecated


# def getitem_based_contains(self, x) -> bool:
#     """
#     Check if collection of keys contains k.
#     Note: This method actually fetches the contents for k, returning False if there's a key error trying to do so
#     Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
#     :return: True if k is in the collection, and False if not
#     """
#
#     try:
#         self.__getitem__(k)
#         return True
#     except KeyError:
#         return False


class MappingViewMixin:
    KeysView: type = BaseKeysView
    ValuesView: type = BaseValuesView
    ItemsView: type = BaseItemsView

    def keys(self) -> KeysView:
        return self.KeysView(self)

    def values(self) -> ValuesView:
        return self.ValuesView(self)

    def items(self) -> ItemsView:
        return self.ItemsView(self)


class KvReader(MappingViewMixin, Collection, Mapping):
    """Acts as a Mapping abc, but with default __len__ (implemented by counting keys)
    and head method to get the first (k, v) item of the store"""

    def head(self):
        """Get the first (key, value) pair"""
        for k, v in self.items():
            return k, v

    def __reversed__(self):
        """The __reversed__ is disabled at the base, but can be re-defined in subclasses.
        Rationale: KvReader is meant to wrap a variety of storage backends or key-value
        perspectives thereof.
        Not all of these would have a natural or intuitive order nor do we want to
        incur the cost of maintaining one systematically.

        If you need a reversed list, here's one way to do it, but note that it
        depends on how self iterates, which is not even assured to be consistent at
        every call:

        .. code-block:: python

            reversed = list(self)[::-1]


        If the keys are comparable, therefore sortable, another natural option would be:

        .. code-block:: python

            reversed = sorted(self)[::-1]

        """
        raise NotImplementedError(__doc__)


Reader = KvReader  # alias for back-compatibility


# TODO: Should we really be using MutableMapping if we're disabling so many of it's methods?
# TODO: Wishful thinking: Define store type so the type is defined by it's methods, not by subclassing.
class KvPersister(KvReader, MutableMapping):
    """Acts as a MutableMapping abc, but disabling the clear and __reversed__ method,
    and computing __len__ by iterating over all keys, and counting them.

    Note that KvPersister is a MutableMapping, and as such, is dict-like.
    But that doesn't mean it's a dict.

    For instance, consider the following code:

    .. code-block:: python

        s = SomeKvPersister()
        s['a']['b'] = 3

    If `s` is a dict, this would have the effect of adding a ('b', 3) item under 'a'.
    But in the general case, this might
    - fail, because the `s['a']` doesn't support sub-scripting (doesn't have a `__getitem__`)
    - or, worse, will pass silently but not actually persist the write as expected (e.g. LocalFileStore)

    Another example: `s.popitem()` will pop a `(k, v)` pair off of the `s` store.
    That is, retrieve the `v` for `k`, delete the entry for `k`, and return a `(k, v)`.
    Note that unlike modern dicts which will return the last item that was stored
     -- that is, LIFO (last-in, first-out) order -- for KvPersisters,
     there's no assurance as to what item will be, since it will depend on the backend storage system
     and/or how the persister was implemented.

    """

    clear = _disabled_clear_method

    # # TODO: Tests and documentation demos needed.
    # def popitem(self):
    #     """pop a (k, v) pair off of the store.
    #     That is, retrieve the v for k, delete the entry for k, and return a (k, v)
    #     Note that unlike modern dicts which will return the last item that was stored
    #      -- that is, LIFO (last-in, first-out) order -- for KvPersisters,
    #      there's no assurance as to what item will be, since it will depend on the backend storage system
    #      and/or how the persister was implemented.
    #     :return:
    #     """
    #     return super(KvPersister, self).popitem()


Persister = KvPersister  # alias for back-compatibility


class NoSuchItem:
    pass


no_such_item = NoSuchItem()

from collections.abc import Set


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            # return getattr(getattr(owner, self.delegate_name), self.attr_name)
            # return getattr(owner, self.attr_name, None)

            # TODO: Would just return self or self.__wrapped__ here, but
            #   self.__wrapped__ would make it hard to debug and
            #   self would fail with unbound methods (why?)
            #   So doing a check here, but would like to find a better solution.
            wrapped_self = getattr(self, "__wrapped__", None)
            if is_classmethod(wrapped_self) or is_unbound_method(wrapped_self):
                return wrapped_self
            else:
                return self

            # wrapped_self = getattr(self, '__wrapped__', None)
            # if not is_classmethod(wrapped_self):
            #     return self
            # else:
            #     return wrapped_self
        else:
            # i.e. return instance.delegate.attr
            return getattr(getattr(instance, self.delegate_name), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(getattr(instance, self.delegate_name), self.attr_name, value)

    def __delete__(self, instance):
        delattr(getattr(instance, self.delegate_name), self.attr_name)


Decorator = Callable[[Callable], Any]  # TODO: Look up typing protocols


def delegate_to(
    wrapped: type,
    class_trans: Callable | None = None,
    delegation_attr: str = "store",
    include=frozenset(),
    ignore=frozenset(),
) -> Decorator:
    # turn include and ignore into sets, if they aren't already
    if not isinstance(include, Set):
        include = set(include)
    if not isinstance(ignore, Set):
        ignore = set(ignore)
    # delegate_attrs = set(delegate_cls.__dict__)
    delegate_attrs = set(dir(wrapped))
    attributes_of_wrapped = (
        include | delegate_attrs - ignore
    )  # TODO: Look at precedence

    def delegation_decorator(wrapper_cls: type):
        @wraps(wrapper_cls, updated=())
        class Wrap(wrapper_cls):
            # _type_of_wrapped = wrapped
            # _delegation_attr = delegation_attr
            _class_trans = class_trans

            @wraps(wrapper_cls.__init__)
            def __init__(self, *args, **kwargs):
                delegate = wrapped(*args, **kwargs)
                super().__init__(delegate)
                assert isinstance(
                    getattr(self, delegation_attr, None), wrapped
                ), f"The wrapper instance has no (expected) {delegation_attr!r} attribute"

            def __reduce__(self):
                return (
                    # reconstructor
                    wrapped_delegator_reconstruct,
                    # args of reconstructor
                    (wrapper_cls, wrapped, class_trans, delegation_attr),
                    # instance state
                    self.__getstate__(),
                )

        attrs = attributes_of_wrapped - set(
            dir(wrapper_cls)
        )  # don't bother adding attributes that the class already has
        # set all the attributes
        for attr in attrs:
            if attr == "__provides__":  # TODO: Hack. Find better solution.
                # This is because __provides__ happened to be in wrapper_cls but not
                # in wrapped.
                # Happened at some point with `from sqldol import SqlRowsReader``
                continue
            wrapped_attr = getattr(wrapped, attr)
            delegated_attribute = update_wrapper(
                wrapper=DelegatedAttribute(delegation_attr, attr),
                wrapped=wrapped_attr,
            )
            setattr(Wrap, attr, delegated_attribute)

        if class_trans:
            Wrap = class_trans(Wrap)
        return Wrap

    return delegation_decorator


def wrapped_delegator_reconstruct(wrapped_cls, wrapped, class_trans, delegation_attr):
    """"""
    type_ = delegator_wrap(wrapped_cls, wrapped, class_trans, delegation_attr)
    # produce an empty object for pickle to pour the
    # __getstate__ values into, via __setstate__
    return copyreg._reconstructor(type_, object, None)


def delegator_wrap(
    delegator: Callable,
    obj: type | Any,
    class_trans=None,
    delegation_attr: str = "store",
):
    """Wrap a ``obj`` (type or instance) with ``delegator``.

    If obj is not a type, trivially returns ``delegator(obj)``.

    The interesting case of ``delegator_wrap`` is when ``obj`` is a type (a class).
    In this case, ``delegator_wrap`` returns a callable (class or function) that has the
    same signature as obj, but that produces instances that are wrapped by ``delegator``

    :param delegator: An instance wrapper. A Callable (type or function -- with only
        one required input) that will return a wrapped version of it's input instance.
    :param obj: The object (class or instance) to be wrapped.
    :return: A wrapped object

    Let's demo this on a simple Delegator class.

    >>> class Delegator:
    ...     i_think = 'therefore I am delegated'  # this is just to verify that we're in a Delegator
    ...     def __init__(self, wrapped_obj):
    ...         self.wrapped_obj = wrapped_obj
    ...     def __getattr__(self, attr):  # delegation: just forward attributes to wrapped_obj
    ...         return getattr(self.wrapped_obj, attr)
    ...     wrap = classmethod(delegator_wrap)  # this is a useful recipe to have the Delegator carry it's own wrapping method

    The only difference between a wrapped object ``Delegator(obj)`` and the original ``obj`` is
    that the wrapped one has a ``i_think`` attribute.
    The wrapped object should otherwise behave the same (on all but special (dunder) methods).
    So let's test this on dictionaries, using the following test function:

    >>> def test_wrapped_d(wrapped_d, original_d):
    ...     '''A function to test a wrapped dict'''
    ...     assert not hasattr(original_d, 'i_think')  # verify that the unwrapped_d doesn't have an i_think attribute
    ...     assert list(wrapped_d.items()) == list(original_d.items())  # verify that wrapped_d has an items that gives us the same thing as origina_d
    ...     assert hasattr(wrapped_d, 'i_think')  # ... but wrapped_d has a i_think attribute
    ...     assert wrapped_d.i_think == 'therefore I am delegated'  # ... and its what we set it to be

    Let's try delegating a dict INSTANCE first:

    >>> d = {'a': 1, 'b': 2}
    >>> wrapped_d = delegator_wrap(Delegator, d)
    >>> test_wrapped_d(wrapped_d, d)

    If we ask ``delegator_wrap`` to wrap a ``dict`` type, we get a subclass of Delegator
    (NOT dict!) whose instances will have the behavior exhibited above:

    >>> WrappedDict = delegator_wrap(Delegator, dict, delegation_attr='wrapped_obj')
    >>> assert issubclass(WrappedDict, Delegator)
    >>> wrapped_d = WrappedDict(a=1, b=2)

    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)

    Now we'll demo/test the ``wrap = classmethod(delegator_wrap)`` trick
    ... with instances

    >>> wrapped_d = Delegator.wrap(d)
    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)

    ... with classes

    >>> WrappedDict = Delegator.wrap(dict, delegation_attr='wrapped_obj')
    >>> wrapped_d = WrappedDict(a=1, b=2)

    >>> test_wrapped_d(wrapped_d, wrapped_d.wrapped_obj)
    >>> class A(dict):
    ...     def foo(self, x):
    ...         pass
    >>> hasattr(A, 'foo')
    True
    >>> WrappedA = Delegator.wrap(A)
    >>> hasattr(WrappedA, 'foo')
    True

    """
    if isinstance(obj, type):
        if isinstance(delegator, type):
            type_decorator = delegate_to(
                obj, class_trans=class_trans, delegation_attr=delegation_attr
            )
            wrap = type_decorator(delegator)
            try:  # try to give the wrap the signature of obj (if it has one)
                wrap.__signature__ = Sig(obj)
            except ValueError:
                pass
            return wrap

        else:
            assert isinstance(delegator, Callable)

            @wraps(obj.__init__)
            def wrap(*args, **kwargs):
                wrapped = obj(*args, **kwargs)
                return delegator(wrapped)

            return wrap
    else:
        return delegator(obj)


class Store(KvPersister):
    """
    By store we mean key-value store. This could be files in a filesystem, objects in s3, or a database. Where and
    how the content is stored should be specified, but StoreInterface offers a dict-like interface to this.
    ::
        __getitem__ calls: _id_of_key			                    _obj_of_data
        __setitem__ calls: _id_of_key		        _data_of_obj
        __delitem__ calls: _id_of_key
        __iter__    calls:	            _key_of_id


    >>> # Default store: no key or value conversion #####################################
    >>> from dol import Store
    >>> s = Store()
    >>> s['foo'] = 33
    >>> s['bar'] = 65
    >>> assert list(s.items()) == [('foo', 33), ('bar', 65)]
    >>> assert list(s.store.items()) == [('foo', 33), ('bar', 65)]  # see that the store contains the same thing
    >>>
    >>> #################################################################################
    >>> # Now let's make stores that have a key and value conversion layer ##############
    >>> # input keys will be upper cased, and output keys lower cased ###################
    >>> # input values (assumed int) will be converted to ascii string, and visa versa ##
    >>> #################################################################################
    >>>
    >>> def test_store(s):
    ...     s['foo'] = 33  # write 33 to 'foo'
    ...     assert 'foo' in s  # __contains__ works
    ...     assert 'no_such_key' not in s  # __nin__ works
    ...     s['bar'] = 65  # write 65 to 'bar'
    ...     assert len(s) == 2  # there are indeed two elements
    ...     assert list(s) == ['foo', 'bar']  # these are the keys
    ...     assert list(s.keys()) == ['foo', 'bar']  # the keys() method works!
    ...     assert list(s.values()) == [33, 65]  # the values() method works!
    ...     assert list(s.items()) == [('foo', 33), ('bar', 65)]  # these are the items
    ...     assert list(s.store.items()) == [('FOO', '!'), ('BAR', 'A')]  # but note the internal representation
    ...     assert s.get('foo') == 33  # the get method works
    ...     assert s.get('no_such_key', 'something') == 'something'  # return a default value
    ...     del(s['foo'])  # you can delete an item given its key
    ...     assert len(s) == 1  # see, only one item left!
    ...     assert list(s.items()) == [('bar', 65)]  # here it is
    >>>
    >>> # We can introduce this conversion layer in several ways. Here's a few... ######################
    >>> # by subclassing ###############################################################################
    >>> class MyStore(Store):
    ...     def _id_of_key(self, k):
    ...         return k.upper()
    ...     def _key_of_id(self, _id):
    ...         return _id.lower()
    ...     def _data_of_obj(self, obj):
    ...         return chr(obj)
    ...     def _obj_of_data(self, data):
    ...         return ord(data)
    >>> s = MyStore(store=dict())  # note that you don't need to specify dict(), since it's the default
    >>> test_store(s)
    >>>
    >>> # by assigning functions to converters ##########################################################
    >>> class MyStore(Store):
    ...     def __init__(self, store, _id_of_key, _key_of_id, _data_of_obj, _obj_of_data):
    ...         super().__init__(store)
    ...         self._id_of_key = _id_of_key
    ...         self._key_of_id = _key_of_id
    ...         self._data_of_obj = _data_of_obj
    ...         self._obj_of_data = _obj_of_data
    ...
    >>> s = MyStore(dict(),
    ...             _id_of_key=lambda k: k.upper(),
    ...             _key_of_id=lambda _id: _id.lower(),
    ...             _data_of_obj=lambda obj: chr(obj),
    ...             _obj_of_data=lambda data: ord(data))
    >>> test_store(s)
    >>>
    >>> # using a Mixin class #############################################################################
    >>> class Mixin:
    ...     def _id_of_key(self, k):
    ...         return k.upper()
    ...     def _key_of_id(self, _id):
    ...         return _id.lower()
    ...     def _data_of_obj(self, obj):
    ...         return chr(obj)
    ...     def _obj_of_data(self, data):
    ...         return ord(data)
    ...
    >>> class MyStore(Mixin, Store):  # note that the Mixin must come before Store in the mro
    ...     pass
    ...
    >>> s = MyStore()  # no dict()? No, because default anyway
    >>> test_store(s)
    >>>
    >>> # adding wrapper methods to an already made Store instance #########################################
    >>> s = Store(dict())
    >>> s._id_of_key=lambda k: k.upper()
    >>> s._key_of_id=lambda _id: _id.lower()
    >>> s._data_of_obj=lambda obj: chr(obj)
    >>> s._obj_of_data=lambda data: ord(data)
    >>> test_store(s)

    Note on defining your own "Mapping Views".

    When you do a `.keys()`, a `.values()` or `.items()` you're getting a `MappingView`
    instance; an iterable and sized container that provides some methods to access
    particular aspects of the wrapped mapping.

    If you need to customize the behavior of these instances, you should avoid
    overriding the `keys`, `values` or `items` methods directly, but instead
    override the `KeysView`, `ValuesView` or `ItemsView` classes that they use.

    For more, see: https://github.com/i2mint/dol/wiki/Mapping-Views

    """

    _state_attrs = ["store", "_class_wrapper"]
    # __slots__ = ('_id_of_key', '_key_of_id', '_data_of_obj', '_obj_of_data')

    def __init__(self, store=dict):
        # self._wrapped_methods = set(dir(Store))

        if isinstance(store, type):
            store = store()

        self.store = store

        if hasattr(self.store, "KeysView"):
            self.KeysView = self.store.KeysView

        if hasattr(self.store, "ValuesView"):
            self.ValuesView = self.store.ValuesView

        if hasattr(self.store, "ItemsView"):
            self.ItemsView = self.store.ItemsView

    _id_of_key = static_identity_method
    _key_of_id = static_identity_method
    _data_of_obj = static_identity_method
    _obj_of_data = static_identity_method

    _max_repr_size = None

    _errors_that_trigger_missing = (
        KeyError,
    )  # another option: (KeyError, FileNotFoundError)

    wrap = classmethod(partial(delegator_wrap, delegation_attr="store"))

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        # Instead of return getattr(self.store, attr), doing the following
        # because self.store had problems with pickling
        return getattr(object.__getattribute__(self, "store"), attr)

    def __dir__(self):
        return list(
            set(dir(self.__class__)).union(self.store.__dir__())
        )  # to forward dir to delegated stream as well

    def __hash__(self):
        return hash(self.store)
        # changed from the following (store.__hash__ was None sometimes (so not callable)
        # return self.store.__hash__()

    # Read ####################################################################

    def __getitem__(self, k: Key) -> Val:
        # essentially: self._obj_of_data(self.store[self._id_of_key(k)])
        _id = self._id_of_key(k)
        try:
            data = self.store[_id]
        except self._errors_that_trigger_missing as error:
            if hasattr(self, "__missing__"):
                data = self.__missing__(k)
            else:
                raise error
        return self._obj_of_data(data)

    def get(self, k: Key, default=None) -> Val:
        try:
            return self[k]
        except KeyError:
            return default

    # Explore ####################################################################
    def __iter__(self) -> KeyIter:
        yield from (self._key_of_id(k) for k in self.store)
        # return map(self._key_of_id, self.store.__iter__())

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, k) -> bool:
        return self._id_of_key(k) in self.store

    def head(self) -> Item:
        k = None
        try:
            for k in self:
                return k, self[k]
        except Exception as e:
            from warnings import warn

            if k is None:
                raise
            else:
                msg = f"Couldn't get data for the key {k}. This could be be...\n"
                msg += "... because it's not a store (just a collection, that doesn't have a __getitem__)\n"
                msg += (
                    "... because there's a layer transforming outcoming keys that are not the ones the store actually "
                    "uses? If you didn't wrap the store with the inverse ingoing keys transformation, "
                    "that would happen.\n"
                )
                msg += (
                    "I'll ask the inner-layer what it's head is, but IT MAY NOT REFLECT the reality of your store "
                    "if you have some filtering, caching etc."
                )
                msg += f"The error messages was: \n{e}"
                warn(msg)

            for _id in self.store:
                return self._key_of_id(_id), self._obj_of_data(self.store[_id])
        # NOTE: Old version didn't work when key mapping was asymmetrical
        # for k, v in self.items():
        #     return k, v

    # Write ####################################################################
    def __setitem__(self, k: Key, v: Val):
        return self.store.__setitem__(self._id_of_key(k), self._data_of_obj(v))

    # def update(self, *args, **kwargs):
    #     return self.store.update(*args, **kwargs)

    # Delete ####################################################################
    def __delitem__(self, k: Key):
        return self.store.__delitem__(self._id_of_key(k))

    # def clear(self):
    #     raise NotImplementedError('''
    #     The clear method was overridden to make dangerous difficult.
    #     If you really want to delete all your data, you can do so by doing:
    #         try:
    #             while True:
    #                 self.popitem()
    #         except KeyError:
    #             pass''')

    # Misc ####################################################################
    # TODO: Review this -- must be a better overall solution!
    def __repr__(self):
        x = repr(self.store)
        if isinstance(self._max_repr_size, int):
            half = int(self._max_repr_size)
            if len(x) > self._max_repr_size:
                x = x[:half] + "  ...  " + x[-half:]
        return x
        # return self.store.__repr__()

    def __getstate__(self) -> dict:
        state = {}
        for attr in Store._state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    def __setstate__(self, state: dict):
        for attr in Store._state_attrs:
            if attr in state:
                setattr(self, attr, state[attr])


# Store.register(dict)  # TODO: Would this be a good idea? To make isinstance({}, Store) be True (though missing head())
KvStore = Store  # alias with explict name

########################################################################################################################
# walking in trees

from typing import KT, VT, Any, TypeVar
from collections.abc import Callable, Iterator
from collections import deque

PT = TypeVar("PT")  # Path Type
inf = float("infinity")


def val_is_mapping(p: PT, k: KT, v: VT) -> bool:
    return isinstance(v, Mapping)


def asis(p: PT, k: KT, v: VT) -> Any:
    return p, k, v


def tuple_keypath_and_val(p: PT, k: KT, v: VT) -> tuple[PT, VT]:
    if p == ():  # we're just begining (the root),
        p = (k,)  # so begin the path with the first key.
    else:
        p = (*p, k)  # extend the path (append the new key)
    return p, v


# TODO: More docs and doctests.
#  This one even merits an extensive usage and example tutorial!
def kv_walk(
    v: Mapping,
    leaf_yield: Callable[[PT, KT, VT], Any] = asis,
    walk_filt: Callable[[PT, KT, VT], bool] = val_is_mapping,
    pkv_to_pv: Callable[[PT, KT, VT], tuple[PT, VT]] = tuple_keypath_and_val,
    *,
    branch_yield: Callable[[PT, KT, VT], Any] = None,
    breadth_first: bool = False,
    p: PT = (),
) -> Iterator[Any]:
    """
    Walks a nested structure of mappings, yielding stuff on the way.

    :param v: A nested structure of mappings
    :param leaf_yield: (pp, k, vv) -> Any, what you want to yield when you encounter
        a leaf node (as define by walk_filt resolving to False)
    :param walk_filt: (p, k, vv) -> (bool) whether to explore the nested structure v further
    :param pkv_to_pv:  (p, k, v) -> (pp, vv)
        where pp is a form of p + k (update of the path with the new node k)
        and vv is the value that will be used by both walk_filt and leaf_yield
    :param p: The path to v (used internally, mainly, to keep track of the path)
    :param breadth_first: Whether to perform breadth-first traversal
        (instead of the default depth-first traversal).
    :param branch_yield: (pp, k, vv) -> Any, optional yield function to yield before
        the recursive walk of a branch. This is useful if you want to yield something
        for every branch, not just the leaves.

    >>> d = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> list(kv_walk(d))
    [(('a',), 'a', 1), (('b', 'c'), 'c', 2), (('b', 'd'), 'd', 3)]
    >>> list(kv_walk(d, lambda p, k, v: '.'.join(p)))
    ['a', 'b.c', 'b.d']

    The `walk_filt` argument allows you to control what values the walk encountered
    should be walked through. This also means that this function is what controls
    when to stop the recursive traversal of the tree, and yield an actual "leaf".

    Say we want to get (path, values) items from a nested mapping/store based on
    a ``levels`` argument that determines what the desired values are.
    This can be done as follows:

    >>> def mk_level_walk_filt(levels):
    ...     return lambda p, k, v: len(p) < levels - 1
    ...
    >>> def leveled_map_walk(m, levels):
    ...     yield from kv_walk(
    ...         m,
    ...         leaf_yield=lambda p, k, v: (p, v),
    ...         walk_filt=mk_level_walk_filt(levels)
    ...     )
    >>> m = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'dragon_con'}}
    ... }
    >>>
    >>> assert (
    ...         list(leveled_map_walk(m, 3))
    ...         == [
    ...             (('a', 'b', 'c'), 42),
    ...             (('aa', 'bb', 'cc'), 'dragon_con')
    ...         ]
    ... )
    >>> assert (
    ...         list(leveled_map_walk(m, 2))
    ...         == [
    ...             (('a', 'b'), {'c': 42}),
    ...             (('aa', 'bb'), {'cc': 'dragon_con'})
    ...         ]
    ... )
    >>>
    >>> assert (
    ...         list(leveled_map_walk(m, 1))
    ...         == [
    ...             (('a',), {'b': {'c': 42}}),
    ...             (('aa',), {'bb': {'cc': 'dragon_con'}})
    ...         ]
    ... )

    Tip: If you want to use ``kv_filt`` to search and extract stuff from a nested
    mapping, you can have your ``leaf_yield`` return a sentinel (say, ``None``) to
    indicate that the value should be skipped, and then filter out the ``None``s from
    your results.

    >>> mm = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'meaning_of_life'}},
    ...     'aaa': {'bbb': 314},
    ... }
    >>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
    >>> list(filter(None, kv_walk(mm, leaf_yield=return_path_if_int_leaf)))
    [(('a', 'b', 'c'), 42), (('aaa', 'bbb'), 314)]

    This "path search" functionality is available as a function in the ``recipes``
    module, as ``search_paths``.

    One last thing. Let's demonstrate the use of `branch_yield` and `breadth_first`.
    Consider the following dictionary:

    >>> d = {'big': {'apple': 1}, 'deal': 3, 'apple': {'pie': 1, 'crumble': 2}}

    Say you wanted to find all the paths that end with 'apple'. You could do:

    >>> from functools import partial
    >>> yield_path_if_ends_with_apple = lambda p, k, v: p if k == 'apple' else None
    >>> walker1 = partial(kv_walk, leaf_yield=yield_path_if_ends_with_apple)
    >>> list(filter(None, walker1(d)))
    [('big', 'apple')]

    It only got `('big', 'apple')` because the `leaf_yield` is only triggered
    for leaf nodes (as defined by the `walk_filt` argument, which defaults to
    `val_is_mapping`). So let's try again, but this time, we'll use `branch_yield`
    to yield the path for every branch (not just the leaves):

    >>> walker2 = partial(walker1, branch_yield=yield_path_if_ends_with_apple)
    >>> list(filter(None, walker2(d)))
    [('big', 'apple'), ('apple',)]

    But this isn't convenient if you'd like your search to finish as soon as you
    find a path ending with `'apple'`. The order here comes from the fact that
    `kv_walk` does a depth-first traversal. If you want to do a breadth-first
    traversal, just say it:

    >>> walker3 = partial(walker2, breadth_first=True)
    >>> list(filter(None, walker3(d)))
    [('apple',), ('big', 'apple')]

    So now, you can get the first apple path by doing:
    >>> next(filter(None, walker3(d)))
    ('apple',)

    """
    if not breadth_first:
        # print(f"1: entered with: v={v}, p={p}")
        for k, vv in v.items():
            # print(f"2: item: k={k}, vv={vv}")
            pp, vv = pkv_to_pv(
                p, k, vv
            )  # update the path with k (and preprocess v if necessary)
            if walk_filt(
                p, k, vv
            ):  # should we recurse? (based on some function of p, k, v)
                # print(f"3: recurse with: pp={pp}, vv={vv}\n")
                if branch_yield:
                    yield branch_yield(pp, k, vv)
                yield from kv_walk(
                    vv,
                    leaf_yield,
                    walk_filt,
                    pkv_to_pv,
                    breadth_first=breadth_first,
                    branch_yield=branch_yield,
                    p=pp,
                )  # recurse
            else:
                # print(f"4: leaf_yield(pp={pp}, k={k}, vv={vv})\n --> {leaf_yield(pp, k, vv)}")
                yield leaf_yield(pp, k, vv)  # yield something computed from p, k, vv
    else:
        queue = deque([(p, v)])

        while queue:
            p, v = queue.popleft()
            for k, vv in v.items():
                pp, vv = pkv_to_pv(p, k, vv)
                if walk_filt(p, k, vv):
                    if branch_yield:
                        yield branch_yield(pp, k, vv)
                    queue.append((pp, vv))
                else:
                    yield leaf_yield(pp, k, vv)


def has_kv_store_interface(o):
    """Check if object has the KvStore interface (that is, has the kv wrapper methods

    Args:
        o: object (class or instance)

    Returns: True if kv has the four key (in/out) and value (in/out) transformation methods

    """
    return (
        hasattr(o, "_id_of_key")
        and hasattr(o, "_key_of_id")
        and hasattr(o, "_data_of_obj")
        and hasattr(o, "_obj_of_data")
    )


from abc import ABCMeta, abstractmethod
from dol.errors import KeyValidationError


def _check_methods(C, *methods):
    """
    Check that all methods listed are in the __dict__ of C, or in the classes of it's mro.
    One trick pony borrowed from collections.abc.
    """
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


# Note: Not sure I want to do key validation this way. Perhaps better injected in _id_of_key?
class KeyValidationABC(metaclass=ABCMeta):
    """
    An ABC for an object writer.
    Single purpose: store an object under a given key.
    How the object is serialized and or physically stored should be defined in a concrete subclass.
    """

    __slots__ = ()

    @abstractmethod
    def is_valid_key(self, k):
        pass

    def check_key_is_valid(self, k):
        if not self.is_valid_key(k):
            raise KeyValidationError(f"key is not valid: {k}")

    @classmethod
    def __subclasshook__(cls, C):
        if cls is KeyValidationABC:
            return _check_methods(C, "is_valid_key", "check_key_is_valid")
        return NotImplemented


########################################################################################################################
# Streams


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class Stream:
    """A layer-able version of the stream interface

        __iter__    calls: _obj_of_data(map)

    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ... '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... '''
    ... )
    >>>
    >>> from dol.base import Stream
    >>>
    >>> class MyStream(Stream):
    ...     def _obj_of_data(self, line):
    ...         return [x.strip() for x in line.strip().split(',')]
    ...
    >>> stream = MyStream(src)
    >>>
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    >>> stream.seek(0)  # oh!... but we consumed the stream already, so let's go back to the beginning
    0
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]
    >>> stream.seek(0)  # reverse again
    0
    >>> next(stream)
    ['a', 'b', 'c']
    >>> next(stream)
    ['1', '2', '3']

    Let's add a filter! There's two kinds you can use.
    One that is applied to the line before the data is transformed by _obj_of_data,
    and the other that is applied after (to the obj).


    >>> from dol.base import Stream
    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ...     '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... ''')
    >>> class MyFilteredStream(MyStream):
    ...     def _post_filt(self, obj):
    ...         return str.isnumeric(obj[0])
    >>>
    >>> s = MyFilteredStream(src)
    >>>
    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> next(s)
    ['1', '2', '3']

    Recipes:

    .. hlist::
        * _pre_iter: involving itertools.islice to skip header lines
        * _pre_iter: involving enumerate to get line indices in stream iterator
        * _pre_iter = functools.partial(map, line_pre_proc_func) to preprocess all lines with line_pre_proc_func
        * _pre_iter: include filter before obj
    """

    def __init__(self, stream):
        self.stream = stream

    wrap = classmethod(partial(delegator_wrap, delegation_attr="stream"))

    # _data_of_obj = static_identity_method  # for write methods
    _pre_iter = static_identity_method
    _obj_of_data = static_identity_method
    _post_filt = stream_util.always_true

    def __iter__(self):
        for line in self._pre_iter(self.stream):
            obj = self._obj_of_data(line)
            if self._post_filt(obj):
                yield obj

        # TODO: See pros and cons of above vs below:
        # yield from filter(self._post_filt,
        #                   map(self._obj_of_data,
        #                       self._pre_iter(self.stream)))

    # _wrapped_methods = {'__iter__'}

    def __next__(self):  # TODO: Pros and cons of having a __next__?
        return next(iter(self))

    def __getattr__(self, attr):
        """Delegate method to wrapped store if not part of wrapper store methods"""
        return getattr(self.stream, attr)
        # if attr in self._wrapped_methods:
        #     return getattr(self, attr)
        # else:
        #     return getattr(self.stream, attr)

    def __enter__(self):
        self.stream.__enter__()
        return self
        # return self._pre_proc(self.stream) # moved to iter to

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stream.__exit__(
            exc_type, exc_val, exc_tb
        )  # TODO: Should we have a _post_proc? Uses?


########################################################################################################################
```

## caching.py

```python
"""
Tools to add caching layers to stores and methods.

This module provides comprehensive caching functionality for Python applications,
offering flexible and powerful caching solutions for both data stores and method calls.

Main Use Cases:
- Property caching: Cache expensive computations that only need to be run once
- Method caching: Cache method results based on arguments, with smart key generation
- Store caching: Add caching layers to data stores for improved performance
- Custom caching strategies: Flexible key generation and cache storage options

Key Tools:

cache_this:
    The main decorator for caching properties and methods. Automatically detects
    whether to use property or method caching based on function signature.
    Supports custom cache storage, key functions, parameter ignoring, and
    serialization hooks.

CachedProperty:
    A descriptor for caching property values with flexible cache storage and
    key generation strategies.

CachedMethod:
    A descriptor for caching method results based on arguments, with support
    for parameter filtering and custom key functions.

KeyStrategy Protocol:
    Extensible system for defining how cache keys are generated, including
    strategies for explicit keys, instance properties, method arguments, and
    composite keys.

Store Decorators:
    Tools like cache_vals, mk_sourced_store, and store_cached for adding
    caching layers to data stores.

Examples:

    Basic property caching:

    >>> class MyClass:
    ...     @cache_this
    ...     def expensive_computation(self):
    ...         return sum(range(1000000))

    Method caching with argument-based keys:

    >>> class Calculator:
    ...     @cache_this(cache={})
    ...     def multiply(self, x, y):
    ...         return x * y

    Custom cache storage and key functions:

    >>> class DataProcessor:
    ...     def __init__(self):
    ...         self.cache = {}
    ...     @cache_this(cache='cache', ignore={'verbose'})
    ...     def process(self, data, mode='fast', verbose=False):
    ...         return len(data) if mode == 'fast' else sum(data)

"""

# -------------------------------------------------------------------------------------

import os
import types
from typing import Optional, KT, VT, Any, Union, T
from collections.abc import Callable
from collections.abc import Mapping

from dol.base import Store
from dol.trans import store_decorator

from functools import RLock, cached_property
from types import GenericAlias
from collections.abc import MutableMapping

Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]
KeyType = Union[KT, Callable[[MethodName], KT]]


def identity(x: T) -> T:
    """Identity function that returns its input unchanged.

    >>> identity(42)
    42
    >>> identity("hello")
    'hello'
    >>> identity([1, 2, 3])
    [1, 2, 3]
    """
    return x


from functools import RLock, partial, wraps
from types import GenericAlias
from collections.abc import MutableMapping
from typing import Optional, TypeVar, Union, Any, Protocol
from collections.abc import Callable

# Type variables
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type
T = TypeVar("T")  # Generic type

# Constants
_NOT_FOUND = object()

# Type definitions
Instance = Any
PropertyFunc = Callable[[Instance], VT]
MethodName = str
Cache = Union[MethodName, MutableMapping[KT, VT]]


def identity(x: T) -> T:
    """Identity function that returns its input unchanged."""
    return x


class KeyStrategy(Protocol):
    """Protocol defining how a key strategy should behave."""

    registered_key_strategies = set()

    def resolve_at_definition(self, method_name: str) -> Any | None:
        """
        Attempt to resolve the key at class definition time.

        Args:
            method_name: The name of the method being decorated.

        Returns:
            The resolved key or None if it can't be resolved at definition time.
        """
        ...

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """
        Resolve the key at runtime.
        By default, this will call resolve_at_definition on method_name.

        Args:
            instance: The instance the property is being accessed on.
            method_name: The name of the method being decorated.

        Returns:
            The resolved key.
        """
        return self.resolve_at_definition(method_name)


def register_key_strategy(cls):
    """Register a class as a KeyStrategy."""
    KeyStrategy.registered_key_strategies.add(cls)
    return cls


@register_key_strategy
class ExplicitKey:
    """Use an explicitly provided key value.

    >>> strategy = ExplicitKey("my_key")
    >>> strategy.resolve_at_definition("method_name")
    'my_key'
    """

    def __init__(self, key: Any):
        """
        Initialize with an explicit key value.

        Args:
            key: The explicit key to use.
        """
        self.key = key

    def resolve_at_definition(self, method_name: str) -> Any:
        """Return the explicit key value at definition time."""
        return self.key


@register_key_strategy
class ApplyToMethodName:
    """Apply a function to the method name to generate the key.

    >>> strategy = ApplyToMethodName(lambda name: f"{name}.cache")
    >>> strategy.resolve_at_definition("my_method")
    'my_method.cache'
    """

    def __init__(self, func: Callable[[str], Any]):
        """
        Initialize with a function to apply to the method name.

        Args:
            func: A function that takes a method name and returns a key.
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> Any:
        """Apply the function to the method name at definition time."""
        return self.func(method_name)


@register_key_strategy
class InstanceProp:
    """Get a key from an instance property."""

    def __init__(self, prop_name: str):
        """
        Initialize with the name of the instance property to use as a key.

        Args:
            prop_name: The name of the property to get from the instance.
        """
        self.prop_name = prop_name

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the instance."""
        return None

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """Get the property value from the instance at runtime."""
        return getattr(instance, self.prop_name)


@register_key_strategy
class ApplyToInstance:
    """Apply a function to the instance to generate the key."""

    def __init__(self, func: Callable[[Any], Any]):
        """
        Initialize with a function to apply to the instance.

        Args:
            func: A function that takes an instance and returns a key.
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the instance."""
        return None

    def resolve_at_runtime(self, instance: Any, method_name: str) -> Any:
        """Apply the function to the instance at runtime."""
        return self.func(instance)


@register_key_strategy
class FromMethodArgs:
    """
    Apply a function to method arguments to generate the key.

    The function receives (self, *args, **kwargs) and should return a cache key.
    """

    def __init__(self, func: Callable):
        """
        Initialize with a function to apply to method arguments.

        Args:
            func: A function that takes (self, *args, **kwargs) and returns a key.
                  Example: lambda self, x, y: f'{x}_{y}'
        """
        self.func = func

    def resolve_at_definition(self, method_name: str) -> None:
        """Cannot resolve at definition time, need the arguments."""
        return None

    def resolve_at_runtime(
        self, instance: Any, method_name: str, *args, **kwargs
    ) -> Any:
        """Apply the function to the instance and method arguments at runtime."""
        return self.func(instance, *args, **kwargs)


@register_key_strategy
class CompositeKey:
    """
    Combine multiple key strategies into a single composite key.

    Useful for creating keys that depend on both instance properties and method arguments.
    """

    def __init__(self, *strategies, separator: str = "_"):
        """
        Initialize with multiple key strategies to combine.

        Args:
            *strategies: KeyStrategy instances to combine
            separator: String to use when joining key parts (default: '_')
        """
        self.strategies = strategies
        self.separator = separator

    def resolve_at_definition(self, method_name: str) -> Any | None:
        """Try to resolve all strategies at definition time."""
        parts = []
        any_unresolved = False

        for strategy in self.strategies:
            part = strategy.resolve_at_definition(method_name)
            if part is None:
                any_unresolved = True
            else:
                parts.append(str(part))

        # If any strategy can't be resolved at definition time, return None
        if any_unresolved:
            return None

        return self.separator.join(parts) if parts else None

    def resolve_at_runtime(
        self, instance: Any, method_name: str, *args, **kwargs
    ) -> Any:
        """Resolve all strategies at runtime and combine them."""
        import inspect

        parts = []

        for strategy in self.strategies:
            # Try runtime resolution first (works for both property and method strategies)
            if hasattr(strategy, "resolve_at_runtime"):
                try:
                    # Check if it's a method-aware strategy by inspecting signature
                    sig = inspect.signature(strategy.resolve_at_runtime)
                    if len(sig.parameters) > 2:
                        # Method-aware strategy - pass args and kwargs
                        part = strategy.resolve_at_runtime(
                            instance, method_name, *args, **kwargs
                        )
                    else:
                        # Property strategy - only pass instance and method_name
                        part = strategy.resolve_at_runtime(instance, method_name)
                except TypeError:
                    # Fallback to definition-time resolution
                    part = strategy.resolve_at_definition(method_name)
            else:
                # Fallback to definition-time resolution
                part = strategy.resolve_at_definition(method_name)

            if part is not None:
                parts.append(str(part))

        return self.separator.join(parts) if parts else None


def _resolve_key_for_cached_prop(key: Any) -> KeyStrategy:
    """
    Convert a key specification to a KeyStrategy instance.

    Args:
        key: The key specification, can be a string, function, or KeyStrategy.

    Returns:
        A KeyStrategy instance.
    """
    if key is None:
        # Default to using the method name as the key
        return ApplyToMethodName(lambda x: x)

    if isinstance(key, tuple(KeyStrategy.registered_key_strategies)):
        # Already a KeyStrategy instance
        return key

    if isinstance(key, str):
        # Explicit string key
        return ExplicitKey(key)

    if callable(key):
        # Check the signature to determine the right strategy
        if hasattr(key, "__code__"):
            co_varnames = key.__code__.co_varnames

            if (
                key.__code__.co_argcount > 0
                and co_varnames
                and co_varnames[0] in ("instance", "self")
            ):
                # Function that takes an instance as first arg
                return ApplyToInstance(key)
            else:
                # Function that operates on something else (like method name)
                return ApplyToMethodName(key)
        else:
            # Callable without a __code__ attribute (like partial)
            return ApplyToMethodName(key)

    # For any other type, treat as an explicit key
    return ExplicitKey(key)


def _resolve_key_for_cached_method(
    key: Any, ignore: set = None, func: Callable = None
) -> KeyStrategy:
    """
    Convert a key specification to a KeyStrategy instance for methods.

    This function is the heart of the flexible key generation system. It takes
    various types of key specifications and converts them into a standardized
    KeyStrategy object that knows how to generate cache keys.

    Args:
        key: The key specification, can be a string, function, or KeyStrategy.
        ignore: Set of parameter names to ignore when using default key function.
        func: The function being cached (needed for default key function).

    Returns:
        A KeyStrategy instance suitable for methods with arguments.
    """
    ignore = ignore or set()

    if key is None:
        # Default case: Auto-generate keys from method arguments
        # This creates keys like "x=1;y=2;mode=fast" from method arguments
        def default_key_func(inst, *args, **kwargs):
            return _default_method_key(func, inst, *args, ignore=ignore, **kwargs)

        return FromMethodArgs(default_key_func)

    if isinstance(key, tuple(KeyStrategy.registered_key_strategies)):
        # User provided a pre-built KeyStrategy instance - use it directly
        return key

    if isinstance(key, str):
        # User provided a fixed string - all method calls use the same cache key
        # Useful when you want to cache only the latest call result
        return ExplicitKey(key)

    if callable(key):
        # User provided a custom function to generate keys from method arguments
        # We need to inspect the function to understand how to use it
        import inspect

        if hasattr(key, "__code__"):
            sig = inspect.signature(key)
            params = list(sig.parameters.keys())

            # Check what the callable expects
            if len(params) == 0:
                # No parameters - use as explicit key
                return ExplicitKey(key())
            elif len(params) == 1 and params[0] in ("self", "instance"):
                # Only takes instance - wrap in ApplyToInstance
                return ApplyToInstance(key)
            else:
                # Takes arguments - use FromMethodArgs
                return FromMethodArgs(key)
        else:
            # Callable without __code__ (like partial) - assume it takes full args
            return FromMethodArgs(key)

    # For any other type, treat as an explicit key
    return ExplicitKey(key)


class CachedProperty:
    """
    Descriptor that caches the result of the first call to a method.

    It generalizes the builtin functools.cached_property class, enabling the user to
    specify a cache object and a key to store the cache value.
    """

    def __init__(
        self,
        func: PropertyFunc,
        cache: Cache | None = None,
        key: str | Callable | KeyStrategy | None = None,
        *,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: bool | MutableMapping = False,
        serialize: Callable[[Any], Any] | None = None,
        deserialize: Callable[[Any], Any] | None = None,
    ):
        """
        Initialize the cached property.

        Args:
            func: The function whose result needs to be cached.
            cache: The cache storage. Can be:
                - A MutableMapping instance (shared across all instances)
                - A string naming an instance attribute that is a MutableMapping
                - A callable that takes the instance and returns a MutableMapping
                  (allows per-instance cache customization)
                Example: cache=lambda self: Files(f'/tmp/cache_{self.user_id}/')
            key: The key to store the cache value. Can be:
                - A string (treated as an explicit key)
                - A function (interpreted based on its signature)
                - A KeyStrategy instance
            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: (value) -> serialized_value
                       Example: serialize=pickle.dumps
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: (serialized_value) -> value
                         Example: deserialize=pickle.loads
        """
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock_factory()
        self.cache = cache
        self.key_strategy = _resolve_key_for_cached_prop(key)
        self.allow_none_keys = allow_none_keys
        self.cache_key = (
            None  # Will be set in __set_name__ if resolvable at definition time
        )
        self.serialize = serialize if serialize is not None else identity
        self.deserialize = deserialize if deserialize is not None else identity

        if pre_cache is not False:
            if pre_cache is True:
                pre_cache = dict()
            else:
                assert isinstance(pre_cache, MutableMapping), (
                    f"`pre_cache` must be a bool or a MutableMapping, "
                    f"Was a {type(pre_cache)}: {pre_cache}"
                )
            self.wrap_cache = partial(cache_vals, cache=pre_cache)
        else:
            self.wrap_cache = identity

    def __set_name__(self, owner, name):
        """
        Set the name of the property.

        Args:
            owner: The class owning the property.
            name: The name of the property.
        """
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same CachedProperty to two different names "
                f"({self.attrname!r} and {name!r})."
            )

        # If func is a descriptor (stacked cache_this), propagate __set_name__ to it
        if isinstance(self.func, (CachedProperty, CachedMethod)):
            self.func.__set_name__(owner, name)

        # Try to resolve the key at definition time
        key = self.key_strategy.resolve_at_definition(self.attrname)

        if key is not None:
            if key is None and not self.allow_none_keys:
                raise TypeError("The key resolved at definition time cannot be None.")
            self.cache_key = key

    def _get_cache_key(self, instance):
        """
        Get the cache key for the instance.

        Args:
            instance: The instance of the class.

        Returns:
            The cache key to use.
        """
        # If we already have a cache_key from definition time, use it
        if self.cache_key is not None:
            return self.cache_key

        # Otherwise, resolve at runtime
        key = self.key_strategy.resolve_at_runtime(instance, self.attrname)

        if key is None and not self.allow_none_keys:
            raise TypeError(
                f"The key resolved at runtime for {self.attrname!r} cannot be None."
            )

        return key

    def __get_cache(self, instance):
        """
        Get the cache for the instance.

        This method handles the three main cache specification patterns:
        1. Cache factories (functions that create cache instances)
        2. Attribute names (strings referring to instance attributes)
        3. Direct cache objects (MutableMapping instances)

        Args:
            instance: The instance of the class.

        Returns:
            The cache storage.
        """
        # Pattern 1: Callable cache factories
        # These enable dynamic, instance-specific cache creation
        # Example: cache=lambda self: Files(f'/cache/{self.user_id}/')
        if callable(self.cache) and not isinstance(self.cache, type):
            # It's a factory function - call it with the instance
            import inspect

            sig = inspect.signature(self.cache)

            if len(sig.parameters) >= 1:
                # Factory expects the instance
                cache = self.cache(instance)
            else:
                # No-argument factory
                cache = self.cache()

            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Cache factory must return a MutableMapping, got {type(cache)}"
                )
            __cache = cache

        elif isinstance(self.cache, str):
            cache = getattr(instance, self.cache, None)
            if cache is None:
                raise TypeError(
                    f"No attribute named '{self.cache}' found on {type(instance).__name__!r} instance."
                )
            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Attribute '{self.cache}' on {type(instance).__name__!r} instance is not a MutableMapping."
                )
            __cache = cache
        elif isinstance(self.cache, MutableMapping):
            __cache = self.cache
        else:
            __cache = instance.__dict__

        return self.wrap_cache(__cache)

    def __get__(self, instance, owner=None):
        """
        Get the value of the cached property.

        Args:
            instance: The instance of the class.
            owner: The owner class.

        Returns:
            The cached value or computed value if not cached.
        """
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use CachedProperty instance without calling __set_name__ on it."
            )
        if self.cache is False:
            # If cache is False, always compute the value
            return self.func(instance)

        # Main caching flow: get cache storage, compute key, then get or compute value
        cache = self._get_cache(instance)
        cache_key = self._get_cache_key(instance)

        return self._get_or_compute(instance, cache, cache_key)

    def _get_cache(self, instance):
        """Get the cache for the instance, handling potential errors."""
        try:
            cache = self.__get_cache(instance)
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        return cache

    def _get_or_compute(self, instance, cache, cache_key):
        """Get cached value or compute it if not found."""
        val = cache.get(cache_key, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # Double-check pattern: verify cache is still empty after acquiring lock
                # This prevents multiple threads from computing the same value simultaneously
                val = cache.get(cache_key, _NOT_FOUND)
                if val is _NOT_FOUND:
                    # Check if func is actually a descriptor (e.g., another CachedProperty)
                    # This enables stacking of cache_this decorators
                    if isinstance(self.func, (CachedProperty, CachedMethod)):
                        # Use descriptor protocol to get the value
                        val = self.func.__get__(instance, type(instance))
                    else:
                        # Normal function call
                        val = self.func(instance)
                    try:
                        # Serialize before storing
                        cache[cache_key] = self.serialize(val)
                    except TypeError as e:
                        msg = (
                            f"The cache on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {cache_key!r} property.\n"
                            f"Error: {e}"
                        )
                        raise TypeError(msg) from None
                    return val  # Return the unserialized value
                else:
                    # Deserialize when retrieving from cache
                    return self.deserialize(val)
        else:
            # Deserialize when retrieving from cache
            return self.deserialize(val)

    __class_getitem__ = classmethod(GenericAlias)

    # TODO: Time-boxed attempt to get a __call__ method to work with the class
    #    (so that you can chain two cache_this decorators, (problem is that the outer
    #    expects the inner to be a function, not an instance of CachedProperty, so
    #    tried to make CachedProperty callable).
    # def __call__(self, instance):
    #     """
    #     Call the cached property.

    #     :param func: The function to be called.
    #     :return: The cached property.
    #     """
    #     cache = self._get_cache(instance)

    #     return self._get_or_compute(instance, cache)


def _default_method_key(func, self, *args, ignore=None, **kwargs):
    """
    Default key function for CachedMethod.

    Uses inspect.signature to bind all arguments to their parameter names,
    then creates a semicolon-separated string of param_name=value pairs.

    Args:
        func: The function being cached
        self: The instance (first argument of the method)
        *args: Positional arguments passed to the method
        **kwargs: Keyword arguments passed to the method
        ignore: Set of parameter names to exclude from the key

    Returns:
        A string like "x=1;y=2;mode=fast" representing all arguments

    Examples:
        >>> def sample_method(self, x, y, mode='fast'): pass
        >>> _default_method_key(sample_method, None, 1, 2)
        'x=1;y=2;mode=fast'
        >>> _default_method_key(sample_method, None, 1, y=3, mode='slow')
        'x=1;y=3;mode=slow'
        >>> _default_method_key(sample_method, None, 1, 2, ignore={'mode'})
        'x=1;y=2'
    """
    import inspect

    ignore = ignore or set()

    # Get the signature of the wrapped function
    sig = inspect.signature(func)

    # Bind the arguments (excluding 'self')
    # We pass args and kwargs as if calling the function without self
    try:
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        # Fallback if binding fails - shouldn't happen but be defensive
        key_parts = [str(arg) for arg in args]
        key_parts.extend(
            f"{k}={v}" for k, v in sorted(kwargs.items()) if k not in ignore
        )
        return ";".join(key_parts) if key_parts else ""

    # Build key from bound arguments
    key_parts = []
    for param_name, param_value in bound.arguments.items():
        # Skip 'self' and any ignored parameters
        if param_name == "self" or param_name in ignore:
            continue

        # Handle VAR_POSITIONAL (*args)
        if sig.parameters[param_name].kind == inspect.Parameter.VAR_POSITIONAL:
            if param_value:  # Only include if non-empty
                key_parts.append(f"{param_name}={param_value}")

        # Handle VAR_KEYWORD (**kwargs)
        elif sig.parameters[param_name].kind == inspect.Parameter.VAR_KEYWORD:
            # Add each kwarg as a separate key=value pair
            for k, v in sorted(param_value.items()):
                if k not in ignore:
                    key_parts.append(f"{k}={v}")

        # Handle regular parameters
        else:
            key_parts.append(f"{param_name}={param_value}")

    return ";".join(key_parts) if key_parts else ""


class CachedMethod:
    """
    Descriptor that caches the result of method calls based on their arguments.

    Similar to CachedProperty but handles methods with arguments, caching results
    based on unique combinations of arguments (excluding self).
    """

    def __init__(
        self,
        func: Callable,
        cache: Cache | None = None,
        key: Callable | None = None,
        *,
        ignore: str | list[str] | None = None,
        allow_none_keys: bool = False,
        lock_factory: Callable = RLock,
        pre_cache: bool | MutableMapping = False,
        serialize: Callable[[Any], Any] | None = None,
        deserialize: Callable[[Any], Any] | None = None,
    ):
        """
        Initialize the cached method.

        Args:
            func: The function whose results need to be cached.
            cache: The cache storage. Can be:
                - A MutableMapping instance (shared across instances)
                - A string naming an instance attribute containing a MutableMapping
                - A callable taking (instance) and returning a MutableMapping
                  This enables instance-specific caching, e.g.:
                  cache=lambda self: Files(f'/cache/{self.user_id}/')
            key: Callable that takes (self, *args, **kwargs) and returns a cache key.
                 Defaults to a function that converts args/kwargs to a string.
            ignore: Parameter name(s) to exclude from cache key computation.
                Can be a string (single parameter) or list of strings (multiple parameters).
                Commonly used to ignore 'self' or parameters like 'verbose' that don't
                affect the result.
            allow_none_keys: Whether to allow None as a valid key.
            lock_factory: Factory function to create a lock.
            pre_cache: If True or a MutableMapping, adds in-memory caching.
            serialize: Optional function to serialize values before storing in cache.
                       Signature: (value) -> serialized_value
                       Example: serialize=pickle.dumps
            deserialize: Optional function to deserialize values retrieved from cache.
                         Signature: (serialized_value) -> value
                         Example: deserialize=pickle.loads
        """
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock_factory()
        self.cache = cache

        # Handle ignore parameter
        if isinstance(ignore, str):
            ignore = [ignore]
        self.ignore = set(ignore or [])

        # Set up key strategy
        self.key_strategy = _resolve_key_for_cached_method(
            key, ignore=self.ignore, func=self.func
        )
        self.allow_none_keys = allow_none_keys
        self.serialize = serialize if serialize is not None else identity
        self.deserialize = deserialize if deserialize is not None else identity

        if pre_cache is not False:
            if pre_cache is True:
                pre_cache = dict()
            else:
                assert isinstance(pre_cache, MutableMapping), (
                    f"`pre_cache` must be a bool or a MutableMapping, "
                    f"Was a {type(pre_cache)}: {pre_cache}"
                )
            self.wrap_cache = partial(cache_vals, cache=pre_cache)
        else:
            self.wrap_cache = identity

    def __set_name__(self, owner, name):
        """
        Set the name of the method.

        Args:
            owner: The class owning the method.
            name: The name of the method.
        """
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same CachedMethod to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def _get_cache_key(self, instance, *args, **kwargs):
        """
        Get the cache key for the method call.

        Args:
            instance: The instance of the class.
            *args: Arguments passed to the method.
            **kwargs: Keyword arguments passed to the method.

        Returns:
            The cache key to use.
        """
        # Resolve key using the strategy
        if hasattr(self.key_strategy, "resolve_at_runtime"):
            # Check if it's a method-aware strategy
            try:
                key = self.key_strategy.resolve_at_runtime(
                    instance, self.attrname, *args, **kwargs
                )
            except TypeError:
                # Fallback for property-style strategies
                key = self.key_strategy.resolve_at_runtime(instance, self.attrname)
        else:
            key = self.key_strategy.resolve_at_definition(self.attrname)

        if key is None and not self.allow_none_keys:
            raise TypeError(f"The key resolved for {self.attrname!r} cannot be None.")

        return key

    def __get_cache(self, instance):
        """
        Get the cache for the instance.

        Args:
            instance: The instance of the class.

        Returns:
            The cache storage.
        """
        # Handle callable cache factories
        if callable(self.cache) and not isinstance(self.cache, type):
            # It's a factory function - call it with the instance
            import inspect

            sig = inspect.signature(self.cache)

            if len(sig.parameters) >= 1:
                # Factory expects the instance
                cache = self.cache(instance)
            else:
                # No-argument factory
                cache = self.cache()

            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Cache factory must return a MutableMapping, got {type(cache)}"
                )
            __cache = cache

        elif isinstance(self.cache, str):
            cache = getattr(instance, self.cache, None)
            if cache is None:
                raise TypeError(
                    f"No attribute named '{self.cache}' found on {type(instance).__name__!r} instance."
                )
            if not isinstance(cache, MutableMapping):
                raise TypeError(
                    f"Attribute '{self.cache}' on {type(instance).__name__!r} instance is not a MutableMapping."
                )
            __cache = cache
        elif isinstance(self.cache, MutableMapping):
            __cache = self.cache
        else:
            __cache = instance.__dict__

        return self.wrap_cache(__cache)

    def __get__(self, instance, owner=None):
        """
        Get the bound method wrapper that provides caching.

        Args:
            instance: The instance of the class.
            owner: The owner class.

        Returns:
            A bound wrapper function that caches method results.
        """
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use CachedMethod instance without calling __set_name__ on it."
            )

        if self.cache is False:
            # If cache is False, always compute the value
            return partial(self.func, instance)

        cache = self._get_cache(instance)

        @wraps(self.func)
        def cached_method_wrapper(*args, **kwargs):
            cache_key = self._get_cache_key(instance, *args, **kwargs)
            return self._get_or_compute(instance, cache, cache_key, args, kwargs)

        return cached_method_wrapper

    def _get_cache(self, instance):
        """Get the cache for the instance, handling potential errors."""
        try:
            cache = self.__get_cache(instance)
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} method."
            )
            raise TypeError(msg) from None
        return cache

    def _get_or_compute(self, instance, cache, cache_key, args, kwargs):
        """Get cached value or compute it if not found."""
        val = cache.get(cache_key, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(cache_key, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance, *args, **kwargs)
                    try:
                        # Serialize before storing
                        cache[cache_key] = self.serialize(val)
                    except TypeError as e:
                        msg = (
                            f"The cache on {type(instance).__name__!r} instance "
                            f"does not support item assignment for caching {cache_key!r} method.\n"
                            f"Error: {e}"
                        )
                        raise TypeError(msg) from None
                    return val  # Return the unserialized value
                else:
                    # Deserialize when retrieving from cache
                    return self.deserialize(val)
        else:
            # Deserialize when retrieving from cache
            return self.deserialize(val)

    __class_getitem__ = classmethod(GenericAlias)


def cache_this(
    func: PropertyFunc = None,
    *,
    cache: Cache | None = None,
    key: KeyType | None = None,
    pre_cache: bool | MutableMapping = False,
    as_property: bool | None = None,
    ignore: str | list[str] | None = None,
    serialize: Callable[[Any], Any] | None = None,
    deserialize: Callable[[Any], Any] | None = None,
):
    r"""
    Unified caching decorator for properties and methods with persistent storage support.

    `cache_this` extends the capabilities of Python's built-in `functools.cached_property`
    and `functools.lru_cache` by providing:

    - **Persistent caching**: Store cached values in files, databases, or any MutableMapping
    - **Flexible cache backends**: Use instance attributes, external stores, or cache factories
    - **Smart key generation**: Automatic argument-based keys for methods with parameter filtering
    - **Serialization support**: Custom serialize/deserialize functions for complex data
    - **Auto-detection**: Automatically chooses property vs method caching based on signature
    - **No LRU eviction**: Unlike lru_cache, values persist until explicitly removed

    Unlike functools.cached_property (properties only) and lru_cache (memory-only with eviction),
    cache_this provides a unified interface for both use cases with persistent storage options.

    :param func: The function to be decorated (usually left empty).
    :param cache: The cache storage. Can be:
        - A MutableMapping instance (shared across instances)
        - A string naming an instance attribute containing a MutableMapping
        - A callable taking (instance) and returning a MutableMapping
          This enables instance-specific caching, e.g.:
          cache=lambda self: Files(f'/cache/{self.user_id}/')
    :param key: For properties: the key to store the cache value, can be a callable
        that will be applied to the method name to make a key, or an explicit string.
        For methods: a callable that takes (self, *args, **kwargs) and returns a cache key.
    :param pre_cache: Default is False. If True, adds an in-memory cache to the method
        to (also) cache the results in memory. If a MutableMapping is given, it will be
        used as the pre-cache.
        This is useful when you want a persistent cache but also want to speed up
        access to the method in the same session.
    :param as_property: If True, force use of CachedProperty. If False, force use of
        CachedMethod. If None (default), auto-detect based on function signature.
    :param ignore: Parameter name(s) to exclude from cache key computation.
        Can be a string (single parameter) or list of strings (multiple parameters).
        Commonly used to ignore 'self' or parameters like 'verbose' that don't
        affect the result.
    :param serialize: Optional function to serialize values before caching.
        Example: serialize=pickle.dumps for binary file storage
    :param deserialize: Optional function to deserialize cached values.
        Example: deserialize=pickle.loads
    :return: The decorated function.

    ## Comprehensive Example

    Here's a complete example showcasing all major features of cache_this:

    >>> import tempfile
    >>> import os
    >>> from pathlib import Path
    >>>
    >>> class DataProcessor:
    ...     def __init__(self, user_id="user123"):
    ...         self.user_id = user_id
    ...         self.memory_cache = {}  # In-memory cache
    ...         self.call_counts = {}   # Track function calls for demo
    ...
    ...     # 1. Basic property caching (like functools.cached_property)
    ...     @cache_this
    ...     def basic_property(self):
    ...         '''Cached in instance.__dict__ by default'''
    ...         self.call_counts['basic_property'] = self.call_counts.get('basic_property', 0) + 1
    ...         return f"computed_value_{self.call_counts['basic_property']}"
    ...
    ...     # 2. Property with custom cache and key
    ...     @cache_this(cache='memory_cache', key='custom_prop_key')
    ...     def custom_cached_property(self):
    ...         '''Cached in instance.memory_cache with custom key'''
    ...         self.call_counts['custom_cached_property'] = self.call_counts.get('custom_cached_property', 0) + 1
    ...         return f"custom_value_{self.call_counts['custom_cached_property']}"
    ...
    ...     # 3. Method caching with argument-based keys
    ...     @cache_this(cache='memory_cache')
    ...     def compute_result(self, x, y, mode='fast'):
    ...         '''Cached based on arguments (x, y, mode)'''
    ...         key = ('compute_result', x, y, mode)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         return x * y * (2 if mode == 'fast' else 3)
    ...
    ...     # 4. Method caching with ignored parameters
    ...     @cache_this(cache='memory_cache', ignore={'verbose', 'debug'})
    ...     def process_data(self, data, algorithm='default', verbose=False, debug=False):
    ...         '''Cache ignores verbose and debug parameters'''
    ...         key = ('process_data', tuple(data), algorithm)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         if verbose: print(f"Processing {data} with {algorithm}")
    ...         return sum(data) * (2 if algorithm == 'default' else 3)
    ...
    ...     # 5. Instance-specific cache factory
    ...     @cache_this(cache=lambda self: {f'{self.user_id}_cache': {}}.get(f'{self.user_id}_cache'))
    ...     def user_specific_computation(self, value):
    ...         '''Each instance gets its own cache based on user_id'''
    ...         key = ('user_specific_computation', value)
    ...         self.call_counts[key] = self.call_counts.get(key, 0) + 1
    ...         return value ** 2

    Now let's test all the features:

    >>> processor = DataProcessor("alice")
    >>>
    >>> # Test basic property caching
    >>> result1 = processor.basic_property
    >>> result2 = processor.basic_property  # Should use cache
    >>> assert result1 == result2 == "computed_value_1"
    >>> assert 'basic_property' in processor.__dict__  # Cached in instance dict
    >>>
    >>> # Test custom cache and key
    >>> result1 = processor.custom_cached_property
    >>> result2 = processor.custom_cached_property  # Should use cache
    >>> assert result1 == result2 == "custom_value_1"
    >>> assert 'custom_prop_key' in processor.memory_cache
    >>>
    >>> # Test method caching with arguments
    >>> result1 = processor.compute_result(3, 4, 'fast')
    >>> result2 = processor.compute_result(3, 4, 'fast')  # Should use cache
    >>> result3 = processor.compute_result(3, 4, 'slow')  # Different args, new computation
    >>> assert result1 == result2 == 24  # 3 * 4 * 2
    >>> assert result3 == 36  # 3 * 4 * 3
    >>>
    >>> # Test parameter ignoring
    >>> result1 = processor.process_data([1, 2, 3], verbose=True)
    Processing [1, 2, 3] with default
    >>> result2 = processor.process_data([1, 2, 3], verbose=False)  # Should use same cache
    >>> result3 = processor.process_data([1, 2, 3], debug=True)     # Should use same cache
    >>> assert result1 == result2 == result3 == 12  # sum([1,2,3]) * 2
    >>>
    >>> # Test instance-specific caching
    >>> result1 = processor.user_specific_computation(5)
    >>> result2 = processor.user_specific_computation(5)  # Should use cache
    >>> assert result1 == result2 == 25  # 5 ** 2
    >>>
    >>> # Different instance should have separate cache
    >>> processor2 = DataProcessor("bob")
    >>> result3 = processor2.user_specific_computation(5)  # Fresh computation
    >>> assert result3 == 25

    Used with no arguments, `cache_this` will cache just as the builtin
    `cached_property` does -- in the instance's `__dict__` attribute.

    >>> class SameAsCachedProperty:
    ...     @cache_this
    ...     def foo(self):
    ...         print("In SameAsCachedProperty.foo...")
    ...         return 42
    ...
    >>> obj = SameAsCachedProperty()
    >>> obj.__dict__  # the cache is empty
    {}
    >>> obj.foo  # when we access foo, it's computed and returned...
    In SameAsCachedProperty.foo...
    42
    >>> obj.__dict__  # ... but also cached
    {'foo': 42}
    >>> obj.foo  # so that the next time we access foo, it's returned from the cache.
    42

    Not that if you specify `cache=False`, you get a property that is computed
    every time it's accessed:

    >>> class NoCache:
    ...     @cache_this(cache=False)
    ...     def foo(self):
    ...         print("In NoCache.foo...")
    ...         return 42
    ...
    >>> obj = NoCache()
    >>> obj.foo
    In NoCache.foo...
    42
    >>> obj.foo
    In NoCache.foo...
    42

    Specify the cache as a dictionary that lives outside the instance:

    >>> external_cache = {}
    >>>
    >>> class CacheWithExternalMapping:
    ...     @cache_this(cache=external_cache)
    ...     def foo(self):
    ...         print("In CacheWithExternalMapping.foo...")
    ...         return 42
    ...
    >>> obj = CacheWithExternalMapping()
    >>> external_cache
    {}
    >>> obj.foo
    In CacheWithExternalMapping.foo...
    42
    >>> external_cache
    {'foo': 42}
    >>> obj.foo
    42

    Specify the cache as an attribute of the instance, and an explicit key:

    >>> class WithCacheInInstanceAttribute:
    ...
    ...     def __init__(self):
    ...         self.my_cache = {}
    ...
    ...     @cache_this(cache='my_cache', key='key_for_foo')
    ...     def foo(self):
    ...         print("In WithCacheInInstanceAttribute.foo...")
    ...         return 42
    ...
    >>> obj = WithCacheInInstanceAttribute()
    >>> obj.my_cache
    {}
    >>> obj.foo
    In WithCacheInInstanceAttribute.foo...
    42
    >>> obj.my_cache
    {'key_for_foo': 42}
    >>> obj.foo
    42

    Now let's see a more involved example that exhibits how `cache_this` would be used
    in real life. Note two things in the example below.

    First, that we use `functools.partial` to fix the parameters of our `cache_this`.
    This enables us to reuse the same `cache_this` in multiple places without all
    the verbosity. We fix that the cache is the attribute `cache` of the instance,
    and that the key is a function that will be computed from the name of the method
    adding a `'.pkl'` extension to it.

    Secondly, we use the `ValueCodecs` from `dol` to provide a pickle codec for storying
    values. The backend store used here is a dictionary, so we don't really need a
    codec to store values, but in real life you would use a persistent storage that
    would require a codec, such as files or a database.

    Thirdly, we'll use a `pre_cache` to store the values in a different cache "before"
    (setting and getting) them in the main cache.
    This is useful, for instance, when you want to persist the values (in the main
    cache), but keep them in memory for faster access in the same session
    (the pre-cache, a dict() instance usually). It can also be used to store and
    use things locally (pre-cache) while sharing them with others by storing them in
    a remote store (main cache).

    Finally, we'll use a dict that logs any setting and getting of values to show
    how the caches are being used.

    >>> from dol import cache_this
    >>>
    >>> from functools import partial
    >>> from dol import ValueCodecs
    >>> from collections import UserDict
    >>>
    >>>
    >>> class LoggedCache(UserDict):
    ...     name = 'cache'
    ...
    ...     def __setitem__(self, key, value):
    ...         print(f"In {self.name}: setting {key} to {value}")
    ...         return super().__setitem__(key, value)
    ...
    ...     def __getitem__(self, key):
    ...         print(f"In {self.name}: getting value of {key}")
    ...         return super().__getitem__(key)
    ...
    >>>
    >>> class CacheA(LoggedCache):
    ...     name = 'CacheA'
    ...
    >>>
    >>> class CacheB(LoggedCache):
    ...     name = 'CacheB'
    ...
    >>>
    >>> cache_with_pickle = partial(
    ...     cache_this,
    ...     cache='cache',  # the cache can be found on the instance attribute `cache`
    ...     key=lambda x: f"{x}.pkl",  # the key is the method name with a '.pkl' extension
    ...     pre_cache=CacheB(),
    ... )
    >>>
    >>>
    >>> class PickleCached:
    ...     def __init__(self, backend_store_factory=CacheA):
    ...         # usually this would be a mapping interface to persistent storage:
    ...         self._backend_store = backend_store_factory()
    ...         self.cache = ValueCodecs.default.pickle(self._backend_store)
    ...
    ...     @cache_with_pickle
    ...     def foo(self):
    ...         print("In PickleCached.foo...")
    ...         return 42
    ...

    >>> obj = PickleCached()
    >>> list(obj.cache)
    []

    >>> obj.foo
    In CacheA: getting value of foo.pkl
    In CacheA: getting value of foo.pkl
    In PickleCached.foo...
    In CacheA: setting foo.pkl to b'\x80\x04K*.'
    42
    >>> obj.foo
    In CacheA: getting value of foo.pkl
    In CacheB: setting foo.pkl to 42
    42

    As usual, it's because the cache now holds something that has to do with `foo`:

    >>> list(obj.cache)
    ['foo.pkl']
    >>> # == ['foo.pkl']

    The value of `'foo.pkl'` is indeed `42`:

    >>> obj.cache['foo.pkl']
    In CacheA: getting value of foo.pkl
    42


    But note that the actual way it's stored in the `_backend_store` is as pickle bytes:

    >>> obj._backend_store['foo.pkl']
    In CacheA: getting value of foo.pkl
    b'\x80\x04K*.'
    >>> # == b'\x80\x04K*.'


    """

    import inspect

    def _should_use_property(func, as_property):
        """
        Determine whether to use CachedProperty or CachedMethod based on function signature.

        This is the core auto-detection logic that makes cache_this work seamlessly
        for both properties (no arguments) and methods (with arguments).
        """
        # Explicit override takes precedence
        if as_property is not None:
            return as_property

        # If func is already a CachedProperty or CachedMethod, treat it as a property
        # (since these are descriptors that work like properties)
        if isinstance(func, (CachedProperty, CachedMethod)):
            return True

        # Auto-detect based on function signature
        try:
            sig = inspect.signature(func)
            # Count all parameters except 'self', including variadic ones (*args, **kwargs)
            non_self_params = [
                p for name, p in sig.parameters.items() if name != "self"
            ]

            # Key insight: If there are any parameters beyond 'self', it's a method
            # that takes arguments and needs argument-based cache keys.
            # If only 'self' parameter exists, it's a property-like function.
            return len(non_self_params) == 0
        except (ValueError, TypeError):
            # If we can't get signature, default to property behavior for backward compatibility
            return True

    # Special case: cache=False means disable caching entirely (compute every time)
    if cache is False:
        if func is None:
            # Decorator factory: @cache_this(cache=False)
            def wrapper(f):
                return property(f)  # Just a plain property, no caching

            return wrapper
        else:
            # Direct decoration: cache_this(some_func, cache=False)
            return property(func)

    # The main case: cache is enabled (default or explicitly set)
    else:
        if func is None:
            # Decorator factory case: @cache_this() or @cache_this(cache=..., key=...)
            def wrapper(f):
                # This is where the magic happens: auto-detect property vs method
                if _should_use_property(f, as_property):
                    # Function has no arguments beyond 'self' -> use property caching
                    return CachedProperty(
                        f,
                        cache=cache,
                        key=key,
                        pre_cache=pre_cache,
                        serialize=serialize,
                        deserialize=deserialize,
                    )
                else:
                    # Function has arguments beyond 'self' -> use method caching
                    return CachedMethod(
                        f,
                        cache=cache,
                        key=key,
                        pre_cache=pre_cache,
                        ignore=ignore,
                        serialize=serialize,
                        deserialize=deserialize,
                    )

            return wrapper

        else:  #   If func is given, we want to return the appropriate cached instance
            if _should_use_property(func, as_property):
                return CachedProperty(
                    func,
                    cache=cache,
                    key=key,
                    pre_cache=pre_cache,
                    serialize=serialize,
                    deserialize=deserialize,
                )
            else:
                return CachedMethod(
                    func,
                    cache=cache,
                    key=key,
                    pre_cache=pre_cache,
                    ignore=ignore,
                    serialize=serialize,
                    deserialize=deserialize,
                )


# add the key strategies as attributes of cache_this to have them easily accessible
for _key_strategy in KeyStrategy.registered_key_strategies:
    setattr(cache_this, _key_strategy.__name__, _key_strategy)


extsep = os.path.extsep


def add_extension(ext=None, name=None):
    """
    Add an extension to a name.

    If name is None, return a partial function that will add the extension to a
    name when called.

    add_extension is a useful helper for making key functions, namely for cache_this.

    >>> add_extension('txt', 'file')
    'file.txt'
    >>> add_txt_ext = add_extension('txt')
    >>> add_txt_ext('file')
    'file.txt'

    Note: If you want to add an extension to a name that already has an extension,
    you can do that, but it will add the extension to the end of the name,
    not replace the existing extension.

    >>> add_txt_ext('file.txt')
    'file.txt.txt'

    Also, bare in mind that if ext starts with the system's extension separator,
    (os.path.extsep), it will be removed.

    >>> add_extension('.txt', 'file') == add_extension('txt', 'file') == 'file.txt'
    True

    """
    if ext.startswith(extsep):
        ext = ext[1:]
    if name is None:
        return partial(add_extension, ext)
    if ext:
        return f"{name}{extsep}{ext}"
    else:
        return name


from functools import lru_cache, partial, wraps


def cached_method(func=None, *, maxsize=128, typed=False):
    """
    A decorator to cache the result of a method, ignoring the first argument (usually `self`).

    This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
    to the method, excluding the first argument (typically `self`). This allows methods of a class to
    be cached while ignoring the instance (`self`) in the cache key.

    Parameters:
    - func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
      will be returned for later application.
    - maxsize (int, optional): The maximum size of the cache. Defaults to 128.
    - typed (bool, optional): If True, cache entries will be different based on argument types, such as
      distinguishing between `1` and `1.0`. Defaults to False.

    Returns:
    - callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    Example:
    >>> class MyClass:
    ...     @cached_method(maxsize=2, typed=True)
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    ...
    >>> obj = MyClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3
    >>> obj.add(1.0, 2.0)  # Different types, recomputation occurs
    Computing 1.0 + 2.0
    3.0
    """
    if func is None:
        # Parametrize cached_method and return a decorator to be applied to a function directly
        return partial(cached_method, maxsize=maxsize, typed=typed)

    # Create a cache, ignoring the first argument (`self`)
    cache = lru_cache(maxsize=maxsize, typed=typed)(
        lambda _, *args, **kwargs: func(_, *args, **kwargs)
    )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the cache but don't include `self` in the arguments for caching
        return cache(None, *args, **kwargs)

    return wrapper


from functools import lru_cache, partial, wraps


def lru_cache_method(func=None, *, maxsize=128, typed=False):
    """
    A decorator to cache the result of a method, ignoring the first argument
    (usually `self`).

    This decorator uses `functools.lru_cache` to cache the method result based on the arguments passed
    to the method, excluding the first argument (typically `self`). This allows methods of a class to
    be cached while ignoring the instance (`self`) in the cache key.

    Parameters:
    - func (callable, optional): The method to be decorated. If not provided, a partially applied decorator
      will be returned for later application.
    - maxsize (int, optional): The maximum size of the cache. Defaults to 128.
    - typed (bool, optional): If True, cache entries will be different based on argument types, such as
      distinguishing between `1` and `1.0`. Defaults to False.

    Returns:
    - callable: A wrapped function with LRU caching applied, ignoring the first argument (`self`).

    Example:

    >>> class MyClass:
    ...     @lru_cache_method
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    >>> obj = MyClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3

    Like `lru_cache`, you can specify the `maxsize` and `typed` parameters:

    >>> class MyOtherClass:
    ...     @lru_cache_method(maxsize=2, typed=True)
    ...     def add(self, x, y):
    ...         print(f"Computing {x} + {y}")
    ...         return x + y
    ...
    >>> obj = MyOtherClass()
    >>> obj.add(1, 2)
    Computing 1 + 2
    3
    >>> obj.add(1, 2)  # Cached result, no recomputation
    3
    >>> obj.add(1.0, 2.0)  # Different types, recomputation occurs
    Computing 1.0 + 2.0
    3.0
    """
    if func is None:
        # Parametrize lru_cache_method and return a decorator to be applied to a function directly
        return partial(lru_cache_method, maxsize=maxsize, typed=typed)

    # Create a cache, ignoring the first argument (`self`)
    cache = lru_cache(maxsize=maxsize, typed=typed)(
        lambda _, *args, **kwargs: func(_, *args, **kwargs)
    )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Call the cache but don't include `self` in the arguments for caching
        return cache(None, *args, **kwargs)

    return wrapper


def cache_property_method(
    cls=None, method_name: MethodName = None, *, cache_decorator: Callable = cache_this
):
    """
    Converts a method of a class into a CachedProperty.

    Essentially, it does what `A.method = cache_this(A.method)` would do, taking care of
    the `__set_name__` problem that you'd run into doing it that way.
    Note that here, you need to say `cache_property_method(A, 'method')`.

    Args:
        cls (type): The class containing the method.
        method_name (str): The name of the method to convert to a cached property.
        cache_decorator (Callable): The decorator to use to cache the method. Defaults to
            `cache_this`. One frequent use case would be to use `functools.partial` to
            fix the cache and key parameters of `cache_this` and inject that.

    Example:

    >>> @cache_property_method(['normal_method', 'property_method'])
    ... class TestClass:
    ...     def normal_method(self):
    ...         print('normal_method called')
    ...         return 1
    ...
    ...     @property
    ...     def property_method(self):
    ...         print('property_method called')
    ...         return 2
    >>>
    >>> c = TestClass()
    >>> c.normal_method
    normal_method called
    1
    >>> c.normal_method
    1
    >>> c.property_method
    property_method called
    2
    >>> c.property_method
    2


    You can also use it like this:

    >>> class TestClass:
    ...     def normal_method(self):
    ...         print('normal_method called')
    ...         return 1
    ...
    ...     @property
    ...     def property_method(self):
    ...         print('property_method called')
    ...         return 2
    >>>
    >>> cache_property_method(
    ...     TestClass,
    ...     [
    ...         'normal_method',
    ...         'property_method',
    ...     ],
    ... )  # doctest: +ELLIPSIS
    <class ...TestClass'>
    >>> c = TestClass()
    >>> c.normal_method
    normal_method called
    1
    >>> c.normal_method
    1
    >>> c.property_method
    property_method called
    2
    >>> c.property_method
    2


    """
    if method_name is None:
        assert cls is not None, (
            "If method_name is None, cls (which will play the role of method_name in "
            "a decorator factory) must not be None."
        )
        method_name = cls
        return partial(
            cache_property_method,
            method_name=method_name,
            cache_decorator=cache_decorator,
        )
    if not isinstance(method_name, str) and isinstance(method_name, Iterable):
        for name in method_name:
            cache_property_method(cls, name, cache_decorator=cache_decorator)
        return cls

    method = getattr(cls, method_name)

    if isinstance(method, property):
        method = method.fget  # Get the original method from the property
    elif isinstance(method, (cached_property, CachedProperty)):
        method = method.func
    # not sure we want to handle (staticmethod, classmethod, but in case:
    # elif isinstance(method, (staticmethod, classmethod)):
    #     method = method.__func__

    cached_method = cache_decorator(method)
    cached_method.__set_name__(cls, method_name)
    setattr(cls, method_name, cached_method)
    return cls


# -------------------------------------------------------------------------------------
import os
from functools import wraps, partial
from collections.abc import Iterable, Callable
from inspect import signature

from dol.trans import store_decorator


def is_a_cache(obj):
    """Check if an object implements the cache interface.

    A cache object must have __contains__, __getitem__, and __setitem__ methods.

    >>> is_a_cache({})  # dict is a valid cache
    True
    >>> is_a_cache([])  # list has these methods but for indexed access
    True
    >>> is_a_cache("string")  # string is not (immutable)
    False
    """
    return all(
        map(
            partial(hasattr, obj),
            ("__contains__", "__getitem__", "__setitem__"),
        )
    )


def get_cache(cache):
    """Convenience function to get a cache (whether it's already an instance, or needs to be validated).

    >>> get_cache({'a': 1})  # Return existing cache instance
    {'a': 1}
    >>> get_cache(dict)()  # Return result of calling cache factory
    {}
    """
    if is_a_cache(cache):
        return cache
    elif callable(cache) and len(signature(cache).parameters) == 0:
        return cache()  # consider it to be a cache factory, and call to make factory


# -------------------------------------------------------------------------------------
# Read caching


# The following is a "Cache-Aside" read-cache with NO builtin cache update or refresh mechanism.
def mk_memoizer(cache):
    """
    Make a memoizer that caches the output of a getter function in a cache.

    Note: This is a specialized memoizer for getter functions/methods, i.e.
    functions/methods that have the signature (instance, key) and return a value.

    :param cache: The cache to use. Must have __getitem__ and __setitem__ methods.
    :return: A memoizer that caches the output of the function in the cache.

    >>> cache = dict()
    >>> @mk_memoizer(cache)
    ... def getter(self, k):
    ...     print(f"getting value for {k}...")
    ...     return k * 10
    ...
    >>> getter(None, 2)
    getting value for 2...
    20
    >>> getter(None, 2)
    20

    """

    def memoize(method):
        @wraps(method)
        def memoizer(self, k):
            if k not in cache:
                val = method(self, k)
                cache[k] = val  # cache it
                return val
            else:
                return cache[k]

        return memoizer

    return memoize


def _mk_cache_instance(cache=None, assert_attrs=()):
    """Make a cache store (if it's not already) from a type or a callable, or just return dict.
    Also assert the presence of given attributes

    >>> _mk_cache_instance(dict(a=1, b=2))
    {'a': 1, 'b': 2}
    >>> _mk_cache_instance(None)
    {}
    >>> _mk_cache_instance(dict)
    {}
    >>> _mk_cache_instance(list, ('__getitem__', '__setitem__'))
    []
    >>> _mk_cache_instance(tuple, ('__getitem__', '__setitem__'))
    Traceback (most recent call last):
        ...
    AssertionError: cache should have the __setitem__ method, but does not: ()

    """
    if isinstance(assert_attrs, str):
        assert_attrs = (assert_attrs,)
    if cache is None:
        cache = {}  # use a dict (memory caching) by default
    elif isinstance(cache, type) or (  # if caching_store is a type...
        not hasattr(cache, "__getitem__")  # ... or is a callable without a __getitem__
        and callable(cache)
    ):
        cache = (
            cache()
        )  # ... assume it's a no-argument callable that makes the instance
    for method in assert_attrs or ():
        assert hasattr(
            cache, method
        ), f"cache should have the {method} method, but does not: {cache}"
    return cache


# TODO: Make it so that the resulting store gets arguments to construct it's own cache
#   right now, only cache instances or no-argument cache types can be used.
#


@store_decorator
def cache_vals(store=None, *, cache=dict):
    """

    Args:
        store: The class of the store you want to cache
        cache: The store you want to use to cache. Anything with a __setitem__(k, v) and a __getitem__(k).
            By default, it will use a dict

    Returns: A subclass of the input store, but with caching (to the cache store)

    >>> from dol.caching import cache_vals
    >>> import time
    >>> class SlowDict(dict):
    ...     sleep_s = 0.2
    ...     def __getitem__(self, k):
    ...         time.sleep(self.sleep_s)
    ...         return super().__getitem__(k)
    ...
    ...
    >>> d = SlowDict({'a': 1, 'b': 2, 'c': 3})
    >>>
    >>> d['a']  # Wow! Takes a long time to get 'a'
    1
    >>> cache = dict()
    >>> CachedSlowDict = cache_vals(store=SlowDict, cache=cache)
    >>>
    >>> s = CachedSlowDict({'a': 1, 'b': 2, 'c': 3})
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: []
    >>> # This will take a LONG time because it's the first time we ask for 'a'
    >>> v = s['a']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a']
    >>> # This will take very little time because we have 'a' in the cache
    >>> v = s['a']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a']
    >>> # But we don't have 'b'
    >>> v = s['b']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a', 'b']
    >>> # But now we have 'b'
    >>> v = s['b']
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c']
    cache: ['a', 'b']
    >>> s['d'] = 4  # and we can do things normally (like put stuff in the store)
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c', 'd']
    cache: ['a', 'b']
    >>> s['d']  # if we ask for it again though, it will take time (the first time)
    4
    >>> print(f"store: {list(s)}\\ncache: {list(cache)}")
    store: ['a', 'b', 'c', 'd']
    cache: ['a', 'b', 'd']
    >>> # Of course, we could write 'd' in the cache as well, to get it quicker,
    >>> # but that's another story: The story of write caches!
    >>>
    >>> # And by the way, your "cache wrapped" store hold a pointer to the cache it's using,
    >>> # so you can take a peep there if needed:
    >>> s._cache
    {'a': 1, 'b': 2, 'd': 4}
    """

    # cache = _mk_cache_instance(cache, assert_attrs=('__getitem__', '__setitem__'))
    assert isinstance(
        store, type
    ), f"store should be a type, was a {type(store)}: {store}"

    class CachedStore(store):
        @wraps(store.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache = _mk_cache_instance(
                cache,
                assert_attrs=("__getitem__", "__contains__", "__setitem__"),
            )
            # self.__getitem__ = mk_memoizer(self._cache)(self.__getitem__)

        def __getitem__(self, k):
            if k not in self._cache:
                val = super(type(self), self).__getitem__(k)
                self._cache[k] = val  # cache it
                return val
            else:
                return self._cache[k]

    return CachedStore


mk_cached_store = cache_vals  # backwards compatibility alias


@store_decorator
def mk_sourced_store(store=None, *, source=None, return_source_data=True):
    """

    Args:
        store: The class of the store you want to cache
        cache: The store you want to use to cache. Anything with a __setitem__(k, v) and a __getitem__(k).
            By default, it will use a dict
        return_source_data:
    Returns: A subclass of the input store, but with caching (to the cache store)


    :param store: The class of the store you're talking to. This store acts as the cache
    :param source: The store that is used to populate the store (cache) when a key is missing there.
    :param return_source_data:
        If True, will return ``source[k]`` as is. This should be used only if ``store[k]`` would return the same.
        If False, will first write to cache (``store[k] = source[k]``) then return ``store[k]``.
        The latter introduces a performance hit (we write and then read again from the cache),
        but ensures consistency (and is useful if the writing or the reading to/from store
        transforms the data in some way.
    :return: A decorated store

    Here are two stores pretending to be local and remote data stores respectively.

    >>> from dol.caching import mk_sourced_store
    >>>
    >>> class Local(dict):
    ...     def __getitem__(self, k):
    ...         print(f"looking for {k} in Local")
    ...         return super().__getitem__(k)
    >>>
    >>> class Remote(dict):
    ...     def __getitem__(self, k):
    ...         print(f"looking for {k} in Remote")
    ...         return super().__getitem__(k)


    Let's make a remote store with two elements in it, and a local store class that asks the remote store for stuff
    if it can't find it locally.

    >>> remote = Remote({'foo': 'bar', 'hello': 'world'})
    >>> SourcedLocal = mk_sourced_store(Local, source=remote)
    >>> s = SourcedLocal({'some': 'local stuff'})
    >>> list(s)  # the local store has one key
    ['some']

    # but if we ask for a key that is in the remote store, it provides it

    >>> assert s['foo'] == 'bar'
    looking for foo in Local
    looking for foo in Remote

    >>> list(s)
    ['some', 'foo']

    See that next time we ask for the 'foo' key, the local store provides it:

    >>> assert s['foo'] == 'bar'
    looking for foo in Local

    >>> assert s['hello'] == 'world'
    looking for hello in Local
    looking for hello in Remote
    >>> list(s)
    ['some', 'foo', 'hello']

    We can still add stuff (locally)...

    >>> s['something'] = 'else'
    >>> list(s)
    ['some', 'foo', 'hello', 'something']
    """
    assert source is not None, "You need to specify a source"

    source = _mk_cache_instance(source, assert_attrs=("__getitem__",))

    assert isinstance(
        store, type
    ), f"store should be a type, was a {type(store)}: {store}"

    if return_source_data:

        class SourcedStore(store):
            _src = source

            def __missing__(self, k):
                # if you didn't have it "locally", ask src for it
                v = self._src[k]  # ... get it from _src,
                self[k] = v  # ... store it in self
                return v  # ... and return it.

    else:

        class SourcedStore(store):
            _src = source

            def __missing__(self, k):
                # if you didn't have it "locally", ask src for it
                v = self._src[k]  # ... get it from _src,
                self[k] = v  # ... store it in self
                return self[k]  # retrieve it again and return

    return SourcedStore


# cache = _mk_cache_instance(cache, assert_attrs=('__getitem__',))
# assert isinstance(store, type), f"store should be a type, was a {type(store)}: {store}"
#
# class CachedStore(store):
#     _cache = cache
#
#     @mk_memoizer(cache)
#     def __getitem__(self, k):
#         return super().__getitem__(k)
#
# return CachedStore


# TODO: Didn't finish this. Finish, doctest, and remove underscore
def _pre_condition_containment(store=None, *, bool_key_func):
    """Adds a custom boolean key function `bool_key_func` before the store_cls.__contains__ check is performed.

    It is meant to be used to create smart read caches.

    This can be used, for example, to perform TTL caching by having `bool_key_func` check on how long
    ago a cache item has been created, and returning False if the item is past it's expiry time.
    """

    class PreContaimentStore(store):
        def __contains__(self, k):
            return bool_key_func(k) and super().__contains__(k)

    return PreContaimentStore


def _slow_but_somewhat_general_hash(*args, **kwargs):
    """
    Attempts to create a hash of the inputs, recursively resolving the most common hurdles (dicts, sets, lists)
    Returns: A hash value for the input

    >>> _slow_but_somewhat_general_hash(1, [1, 2], a_set={1,2}, a_dict={'a': 1, 'b': [1,2]})
    ((1, (1, 2)), (('a_set', (1, 2)), ('a_dict', (('a', 1), ('b', (1, 2))))))
    """
    if len(kwargs) == 0 and len(args) == 1:
        single_val = args[0]
        if hasattr(single_val, "items"):
            return tuple(
                (k, _slow_but_somewhat_general_hash(v)) for k, v in single_val.items()
            )
        elif isinstance(single_val, (set, list)):
            return tuple(single_val)
        else:
            return single_val
    else:
        return (
            tuple(_slow_but_somewhat_general_hash(x) for x in args),
            tuple((k, _slow_but_somewhat_general_hash(v)) for k, v in kwargs.items()),
        )


# TODO: Could add an empty_cache function attribute.
#  Wrap the store cache to track new keys, and delete those (and only those!!) when emptying the store.
def store_cached(store, key_func: Callable):
    """
    Function output memorizer but using a specific (usually persisting) store as it's
    memory and a key_func to compute the key under which to store the output.

    The key can be
    - a single value under which the output should be stored, regardless of the input.
    - a key function that is called on the inputs to create a hash under which the function's output should be stored.

    Args:
        store: The key-value store to use for caching. Must support __getitem__ and __setitem__.
        key_func: The key function that is called on the input of the function to create the key value.

    Note: Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.
    Note: No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.
    Note: No, they shouldn't.

    See Also: store_cached_with_single_key (for a version where the cache store key doesn't depend on function's args)

    >>> # Note: Our doc test will use dict as the store, but to make the functionality useful beyond existing
    >>> # RAM-memorizer, you should use actual "persisting" stores that store in local files, or DBs, etc.
    >>> store = dict()
    >>> @store_cached(store, lambda *args: args)
    ... def my_data(x, y):
    ...     print("Pretend this is a long computation")
    ...     return x + y
    >>> t = my_data(1, 2)  # note the print below (because the function is called
    Pretend this is a long computation
    >>> tt = my_data(1, 2)  # note there's no print (because the function is NOT called)
    >>> assert t == tt
    >>> tt
    3
    >>> my_data(3, 4)  # but different inputs will trigger the actual function again
    Pretend this is a long computation
    7
    >>> my_data._cache
    {(1, 2): 3, (3, 4): 7}
    """
    assert callable(key_func), (
        "key_func should be a callable: "
        "It's called on the wrapped function's input to make a key for the caching store."
    )

    def func_wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            key = key_func(*args, **kwargs)
            if key in store:  # if the store has that key...
                return store[key]  # ... just return the data cached under this key
            else:  # if the store doesn't have it...
                output = func(
                    *args, **kwargs
                )  # ... call the function and get the output
                store[key] = output  # store the output under the key
                return output

        wrapped_func._cache = store
        return wrapped_func

    return func_wrapper


def store_cached_with_single_key(store, key):
    """
    Function output memorizer but using a specific store and key as its memory.

    Use in situations where you have a argument-less function or bound method that computes some data whose dependencies
    are static enough that there's enough advantage to make the data refresh explicit (by deleting the cache entry)
    instead of making it implicit (recomputing/refetching the data every time).

    The key should be a single value under which the output should be stored, regardless of the input.

    Note: The wrapped function comes with a empty_cache attribute, which when called, empties the cache (i.e. removes
    the key from the store)

    Note: The wrapped function has a hidden `_cache` attribute pointing to the store in case you need to peep into it.

    Args:
        store: The cache. The key-value store to use for caching. Must support __getitem__ and __setitem__.
        key: The store key under which to store the output of the function.

    Note: Union[Callable, Any] is equivalent to just Any, but reveals the two cases of a key more clearly.
    Note: No, Union[Callable, Hashable] is not better. For one, general store keys are not restricted to hashable keys.
    Note: No, they shouldn't.

    See Also: store_cached (for a version whose keys are computed from the wrapped function's input.

    >>> # Note: Our doc test will use dict as the store, but to make the functionality useful beyond existing
    >>> # RAM-memorizer, you should use actual "persisting" stores that store in local files, or DBs, etc.
    >>> store = dict()
    >>> @store_cached_with_single_key(store, 'whatevs')
    ... def my_data():
    ...     print("Pretend this is a long computation")
    ...     return [1, 2, 3]
    >>> t = my_data()  # note the print below (because the function is called
    Pretend this is a long computation
    >>> tt = my_data()  # note there's no print (because the function is NOT called)
    >>> assert t == tt
    >>> tt
    [1, 2, 3]
    >>> my_data._cache  # peep in the cache
    {'whatevs': [1, 2, 3]}
    >>> # let's empty the cache
    >>> my_data.empty_cache_entry()
    >>> assert 'whatevs' not in my_data._cache  # see that the cache entry is gone.
    >>> t = my_data()  # so when you call the function again, it prints again!d
    Pretend this is a long computation
    """

    def func_wrapper(func):
        # TODO: Enforce that the func is argument-less or a bound method here?

        # TODO: WhyTF doesn't this work: (unresolved reference)
        # if key is None:
        #     key = '.'.join([func.__module__, func.__qualname___])

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if key in store:  # if the store has that key...
                return store[key]  # ... just return the data cached under this key
            else:
                output = func(*args, **kwargs)
                store[key] = output
                return output

        wrapped_func._cache = store
        wrapped_func.empty_cache_entry = lambda: wrapped_func._cache.__delitem__(key)
        return wrapped_func

    return func_wrapper


def ensure_clear_to_kv_store(store):
    """
    Ensures the store has a working clear method.

    If the store doesn't have a clear method or has the disabled version,
    adds a proper implementation that safely removes all items.

    Args:
        store: A Store class or instance

    Returns:
        The same store with guaranteed clear functionality

    >>> class NoClearing(dict):
    ...     clear = None
    >>> d = NoClearing({'a': 1, 'b': 2})
    >>> d = ensure_clear_to_kv_store(d)
    >>> len(d)
    2
    >>> d.clear()
    >>> len(d)
    0
    """

    def _needs_clear_method(obj):
        """Check if the object needs a clear method added."""
        has_clear = hasattr(obj, "clear")
        if not has_clear:
            return True

        clear_attr = getattr(obj, "clear")
        if clear_attr is None:
            return True

        if (
            hasattr(clear_attr, "__name__")
            and clear_attr.__name__ == "_disabled_clear_method"
        ):
            return True

        return False

    if not _needs_clear_method(store):
        return store

    def _clear_method(self):
        """Remove all items from the store."""
        # Create a separate list to avoid modification during iteration
        keys = list(self.keys())
        for k in keys:
            del self[k]

    # Apply the appropriate clear method
    if isinstance(store, type):
        store.clear = _clear_method
    else:
        store.clear = types.MethodType(_clear_method, store)

    return store


# TODO: Normalize using store_decorator and add control over flush_cache method name
def flush_on_exit(cls):
    new_cls = type(cls.__name__, (cls,), {})

    if not hasattr(new_cls, "__enter__"):

        def __enter__(self):
            return self

        new_cls.__enter__ = __enter__

    if not hasattr(new_cls, "__exit__"):

        def __exit__(self, *args, **kwargs):
            return self.flush_cache()

    else:  # TODO: Untested case where the class already has an __exit__, which we want to call after flush

        @wraps(new_cls.__exit__)
        def __exit__(self, *args, **kwargs):
            self.flush_cache()
            return super(new_cls, self).__exit__(*args, **kwargs)

    new_cls.__exit__ = __exit__

    return new_cls


from dol.util import has_enabled_clear_method


@store_decorator
def mk_write_cached_store(store=None, *, w_cache=dict, flush_cache_condition=None):
    """Wrap a write cache around a store.

    Args:
        w_cache: The store to (write) cache to
        flush_cache_condition: The condition to apply to the cache
            to decide whether it's contents should be flushed or not

    A ``w_cache`` must have a clear method (that clears the cache's contents).
    If you know what you're doing and want to add one to your input kv store,
    you can do so by calling ``ensure_clear_to_kv_store(store)``
    -- this will add a ``clear`` method inplace AND return the resulting store as well.

    We didn't add this automatically because the first thing ``mk_write_cached_store`` will do is call clear,
    to remove all the contents of the store.
    You don't want to do this unwittingly and delete a bunch of precious data!!

    >>> from dol.caching import mk_write_cached_store, ensure_clear_to_kv_store
    >>> from dol.base import Store
    >>>
    >>> def print_state(store):
    ...     print(f"store: {store} ----- store._w_cache: {store._w_cache}")
    ...
    >>> class MyStore(dict): ...
    >>> MyCachedStore = mk_write_cached_store(MyStore, w_cache={})  # wrap MyStore with a (dict) write cache
    >>> s = MyCachedStore()  # make a MyCachedStore instance
    >>> print_state(s)  # print the contents (both store and cache), see that it's empty
    store: {} ----- store._w_cache: {}
    >>> s['hello'] = 'world'  # write 'world' in 'hello'
    >>> print_state(s)  # see that it hasn't been written
    store: {} ----- store._w_cache: {'hello': 'world'}
    >>> s['ding'] = 'dong'
    >>> print_state(s)
    store: {} ----- store._w_cache: {'hello': 'world', 'ding': 'dong'}
    >>> s.flush_cache()  # manually flush the cache
    >>> print_state(s)  # note that store._w_cache is empty, but store has the data now
    store: {'hello': 'world', 'ding': 'dong'} ----- store._w_cache: {}
    >>>
    >>> # But you usually want to use the store as a context manager
    >>> MyCachedStore = mk_write_cached_store(
    ...     MyStore, w_cache={},
    ...     flush_cache_condition=None)
    >>>
    >>> the_persistent_dict = dict()
    >>>
    >>> s = MyCachedStore(the_persistent_dict)
    >>> with s:
    ...     print("===> Before writing data:")
    ...     print_state(s)
    ...     s['hello'] = 'world'
    ...     print("===> Before exiting the with block:")
    ...     print_state(s)
    ...
    ===> Before writing data:
    store: {} ----- store._w_cache: {}
    ===> Before exiting the with block:
    store: {} ----- store._w_cache: {'hello': 'world'}
    >>>
    >>> print("===> After exiting the with block:"); print_state(s)  # Note that the cache store flushed!
    ===> After exiting the with block:
    store: {'hello': 'world'} ----- store._w_cache: {}
    >>>
    >>> # Example of auto-flushing when there's at least two elements
    >>> class MyStore(dict): ...
    ...
    >>> MyCachedStore = mk_write_cached_store(
    ...     MyStore, w_cache={},
    ...     flush_cache_condition=lambda w_cache: len(w_cache) >= 3)
    >>>
    >>> s = MyCachedStore()
    >>> with s:
    ...     for i in range(7):
    ...         s[i] = i * 10
    ...         print_state(s)
    ...
    store: {} ----- store._w_cache: {0: 0}
    store: {} ----- store._w_cache: {0: 0, 1: 10}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {3: 30}
    store: {0: 0, 1: 10, 2: 20} ----- store._w_cache: {3: 30, 4: 40}
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50} ----- store._w_cache: {}
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50} ----- store._w_cache: {6: 60}
    >>> # There was still something left in the cache before exiting the with block. But now...
    >>> print_state(s)
    store: {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60} ----- store._w_cache: {}
    """

    w_cache = _mk_cache_instance(w_cache, ("clear", "__setitem__", "items"))

    if not has_enabled_clear_method(w_cache):
        raise TypeError(
            """w_cache needs to have an enabled clear method to be able to act as a write cache.
        You can wrap w_cache in dol.trans.ensure_clear_method to inject a clear method, 
        but BE WARNED: mk_write_cached_store will immediately delete all contents of `w_cache`!
        So don't give it your filesystem or important DB to delete!
        """
        )
    w_cache.clear()  # assure the cache is empty, by emptying it.

    @flush_on_exit
    class WriteCachedStore(store):
        _w_cache = w_cache
        _flush_cache_condition = staticmethod(flush_cache_condition)

        if flush_cache_condition is None:

            def __setitem__(self, k, v):
                return self._w_cache.__setitem__(k, v)

        else:
            assert callable(flush_cache_condition), (
                "flush_cache_condition must be None or a callable ",
                "taking the (write) cache store as an input and returning"
                "True if and only if the cache should be flushed.",
            )

            def __setitem__(self, k, v):
                r = self._w_cache.__setitem__(k, v)
                if self._flush_cache_condition(self._w_cache):
                    self.flush_cache()
                return r

        if not hasattr(store, "flush"):

            def flush(self, items: Iterable = tuple()):
                for k, v in items:
                    super().__setitem__(k, v)

        def flush_cache(self):
            self.flush(self._w_cache.items())
            return self._w_cache.clear()

    return WriteCachedStore


from collections import ChainMap, deque


# Note: A (big) generalization of this is a set of graphs that determines how to
# operate with multiple (mutuable) mappings: The order in which to search, the stores
# that should be "written back" to according to where the key was found, the stores that
# should be synced with other stores (possibly even when searched), etc.
class WriteBackChainMap(ChainMap):
    """A collections.ChainMap that also 'writes back' when a key is found.

    >>> from dol.caching import WriteBackChainMap
    >>>
    >>> d = WriteBackChainMap({'a': 1, 'b': 2}, {'b': 22, 'c': 33}, {'d': 444})

    In a ``ChainMap``, when you ask for the value for a key, each mapping in the
    sequence is checked for, and the first mapping found that contains it will be
    the one determining the value.

    So here if you look for `b`, though the first mapping will give you the value,
    though the second mapping also contains a `b` with a different value:

    >>> d['b']
    2

    if you ask for `c`, it's the second mapping that will give you the value:

    >>> d['c']
    33

    But unlike with the builtin ``ChainMap``, something else is going to happen here:

    >>> d
    WriteBackChainMap({'a': 1, 'b': 2, 'c': 33}, {'b': 22, 'c': 33}, {'d': 444})

    See that now the first mapping also has the ``('c', 33)`` key-value pair:

    That is what we call "write back".

    When a key is found in a mapping, all previous mappings (which by definition of
    ``ChainMap`` did not have a value for that key) will be revisited and that key-value
    pair will be written in it.

    As in with ``ChainMap``, all writes will be carried out in the first mapping,
    and only the first mapping:

    >>> d['e'] = 5
    >>> d
    WriteBackChainMap({'a': 1, 'b': 2, 'c': 33, 'e': 5}, {'b': 22, 'c': 33}, {'d': 444})

    Example use cases:

    - You're working with a local and a remote source of data. You'd like to list the
    keys available in both, and use the local item if it's available, and if it's not,
    you want it to be sourced from remote, but written in local for quicker access
    next time.

    - You have several sources to look for configuration values: a sequence of
    configuration files/folders to look through (like a unix search path for command
    resolution) and environment variables.
    """

    max_key_search_depth = 1

    def __getitem__(self, key):
        q = deque([])  # a queue to remember the "failed" mappings
        for mapping in self.maps:  # for each mapping
            try:  # try getting  a value for that key
                v = mapping[key]  # Note: can't use 'key in mapping' with defaultdict
                # if that mapping had that key
                for d in q:  # make sure all other previous mappings
                    d[key] = v  # get that value too (* this is the "write back")
                return v  # and then return the value
            except KeyError:  # if you get a key error for that mapping
                q.append(mapping)  # remember that mapping, so you can write back (*)
        # if no such key was found in any of the self.maps...
        return self.__missing__(key)  # ... call __missing__

    def __len__(self):
        return len(
            set().union(*self.maps[: self.max_key_search_depth])
        )  # reuses stored hash values if possible

    def __iter__(self):
        d = {}
        for mapping in reversed(self.maps[: self.max_key_search_depth]):
            d.update(dict.fromkeys(mapping))  # reuses stored hash values if possible
        return iter(d)

    def __contains__(self, key):
        return any(key in m for m in self.maps[: self.max_key_search_depth])


# Experimental #########################################################################################################


def _mk_cache_method_local_path_key(
    method, args, kwargs, ext=".p", path_sep=os.path.sep
):
    """"""
    return (
        method.__module__
        + path_sep
        + method.__qualname__
        + path_sep
        + (
            ",".join(map(str, args))
            + ",".join(f"{k}={v}" for k, v in kwargs.items())
            + ext
        )
    )


class HashableMixin:
    def __hash__(self):
        return id(self)


class HashableDict(HashableMixin, dict):
    """Just a dict, but hashable"""


# NOTE: cache uses (func, args, kwargs). Don't want to make more complex with a bind cast to (func, kwargs) only
def cache_func_outputs(cache=HashableDict):
    cache = get_cache(cache)

    def cache_method_decorator(func):
        @wraps(func)
        def _func(*args, **kwargs):
            k = (func, args, HashableDict(kwargs))
            if k not in cache:
                val = func(*args, **kwargs)
                cache[k] = val  # cache it
                return val
            else:
                return cache[k]

        return _func

    return cache_method_decorator


# from dol import StrTupleDict
#
# def process_fak(module, qualname, args, kwargs):
# #     func, args, kwargs = map(fak_dict.get, ('func', 'args', 'kwargs'))
#     return {
#         'module': module,
#         'qualname': qualname,
#         'args': ",".join(map(str, args)),
#         'kwargs': ",".join(f"{k}={v}" for k, v in kwargs.items())
#     }
#
# t = StrTupleDict(os.path.join("{module}", "{qualname}", "{args},{kwargs}.p"), process_kwargs=process_fak)
#
# t.tuple_to_str((StrTupleDict.__module__, StrTupleDict.__qualname__, (1, 'one'), {'mode': 'lydian'}))
```

## dig.py

```python
"""Layers introspection"""

from functools import partial

# TODO: Make a generator and a index getter for keys and vals (and both).
#  Point is to be able to get views from any level.

not_found = type("NotFound", (), {})()
no_default = type("NoDefault", (), {})()


def get_first_attr_found(store, attrs, default=no_default):
    for attr in attrs:
        a = getattr(store, attr, not_found)
        if a != not_found:
            return a
    if default == no_default:
        raise AttributeError(f"None of the attributes were found: {attrs}")
    else:
        return default


def recursive_get_attr(store, attr, default=None):
    a = getattr(store, attr, None)
    if a is not None:
        return a
    elif hasattr(store, "store"):
        return recursive_get_attr(store.store, attr, default)
    else:
        return default


re_get_attr = recursive_get_attr
dig_up = recursive_get_attr


def store_trans_path(store, arg, method):
    f = getattr(store, method, None)
    if f is not None:
        trans_arg = f(arg)
        yield trans_arg
        if hasattr(store, "store"):
            yield from unravel_key(store.store, trans_arg)


def print_trans_path(store, arg, method, with_type=False):
    gen = (arg, *store_trans_path(store, arg, method))
    if with_type:
        gen = map(lambda x: f"{type(x)}: {x}", gen)
    else:
        gen = map(str, gen)
    print("\n".join(gen))


def last_element(gen):
    x = None
    for x in gen:
        pass
    return x


def inner_most(store, arg, method):
    return last_element(store_trans_path(store, arg, method))


# TODO: Better change the signature to reflect context (k (key) or v (val) instead of arg)
unravel_key = partial(store_trans_path, method="_id_of_key")
print_key_trans_path = partial(print_trans_path, method="_id_of_key")
inner_most_key = partial(inner_most, method="_id_of_key")

# TODO: inner_most_val doesn't really do what one expects. It just does what inner_most_key does
unravel_val = partial(store_trans_path, method="_data_of_obj")
print_val_trans_path = partial(print_trans_path, method="_data_of_obj")
inner_most_val = partial(inner_most, method="_data_of_obj")

from functools import partial


def next_layer(store, layer_attrs=("store",)):
    for attr in layer_attrs:
        attr_val = getattr(store, attr, not_found)
        if attr_val is not not_found:
            return attr_val
    return not_found


def recursive_calls(func, x, sentinel=not_found):
    while True:
        if x is sentinel:
            break
        else:
            yield x
            x = func(x)


def layers(store, layer_attrs=("store",)):
    _next_layer = partial(next_layer, layer_attrs=layer_attrs)
    return list(recursive_calls(_next_layer, store))


def trace_getitem(store, k, layer_attrs=("store",)):
    """A generator of layered steps to inspect a store.

    :param store: An instance that has the base.Store interface
    :param k: A key
    :param layer_attrs: The attribute names that should be checked to get the next layer.
    :return: A generator of (layer, method, value)

    We start with a small dict:

    >>> d = {'a.num': '1000', 'b.num': '2000'}

    Now let's add layers to it. For example, with wrap_kvs:

    >>> from dol.trans import wrap_kvs

    Say that we want the interface to not see the ``'.num'`` strings, and deal with numerical values, not strings.

    >>> s = wrap_kvs(d,
    ...              key_of_id=lambda x: x[:-len('.num')],
    ...              id_of_key=lambda x: x + '.num',
    ...              obj_of_data=lambda x: int(x),
    ...              data_of_obj=lambda x: str(x)
    ...             )
    >>>

    Oh, and we want the interface to display upper case keys.

    >>> ss = wrap_kvs(s,
    ...              key_of_id=lambda x: x.upper(),
    ...              id_of_key=lambda x: x.lower(),
    ...             )

    And we want the numerical unit to be the kilo (that's 1000):

    >>> sss = wrap_kvs(ss,
    ...                obj_of_data=lambda x: x / 1000,
    ...                data_of_obj=lambda x: x * 1000
    ...               )
    >>>
    >>> dict(sss.items())
    {'A': 1.0, 'B': 2.0}

    Well, if we had bugs, we'd like to inspect the various layers, and how they transform the data.

    .. code-block:: python

        # Here's how to do that:

        # >>> for layer, method, value in trace_getitem(sss, 'A'):
        # ...     print(layer, method, value)
        # ...
        # Traceback (most recent call last):
        #   File "<stdin>", line 1, in <module>
        # NameError: name 'trace_getitem' is not defined

    >>> from dol.dig import trace_getitem
    >>>
    >>> for layer, method, value in trace_getitem(sss, 'A'):
    ...     print(layer, method, value)
    ...
    {'a.num': '1000', 'b.num': '2000'} _id_of_key A
    {'a.num': '1000', 'b.num': '2000'} _id_of_key A
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
    {'a.num': '1000', 'b.num': '2000'} _id_of_key a.num
    {'a.num': '1000', 'b.num': '2000'} __getitem__ 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1000
    {'a.num': '1000', 'b.num': '2000'} _obj_of_data 1.0
    """
    _layers = layers(store, layer_attrs)

    layer = None
    for i, layer in enumerate(_layers):
        if hasattr(layer, "_id_of_key"):
            k = layer._id_of_key(k)
            yield (layer, "_id_of_key", k)

    if layer is not None:
        v = layer[k]
        yield (layer, "__getitem__", v)

        for layer in _layers[:i][::-1]:
            if hasattr(layer, "_obj_of_data"):
                v = layer._obj_of_data(v)
                yield (layer, "_obj_of_data", v)


def trace_info(trace, item_func=print):
    for item in trace:
        item_func(item)


def _trace_item_info(item):
    layer, method, value = item
    return f"{layer.__class__.__name__}.{method}: {type(value).__name__}"


def print_trace_info(trace, item_info=_trace_item_info):
    for item in trace:
        print(item_info(item))
```

## errors.py

```python
"""Error objects and utils"""

from collections.abc import Mapping
from inspect import signature


# TODO: More on wrapped_callback: Handle *args too. Make it work with builtins (no signature!)
# TODO: What about traceback?
# TODO: Make it a more general and useful store decorator. Trans store into an getitem exception catching store.
def items_with_caught_exceptions(
    d: Mapping,
    callback=None,
    catch_exceptions=(Exception,),
    yield_callback_output=False,
):
    """
    Do what Mapping.items() does, but catching exceptions when getting the values for a key.


    Some time your `store.items()` is annoying because of some exceptions that happen
    when you're retrieving some value for some of the keys.

    Yes, if that happens, it's that something is wrong with your store, and yes,
    if it's a store that's going to be used a lot, you really should build the right store
    that doesn't have that problem.

    But now that we appeased the paranoid naysayers with that warning, let's get to business:
    Sometimes you just want to get through the hurdle to get the job done. Sometimes your store is good enough,
    except for a few exceptions. Sometimes your store gets it's keys from a large pool of possible keys
    (e.g. github stores or kaggle stores, or any store created by a free-form search seed),
    so you can't really depend on the fact that all the keys given by your key iterator
    will give you a value without exception
    -- especially if you slapped on a bunch of post-processing on the out-coming values.

    So you can right a for loop to iterate over your keys, catch the exceptions, do something with it...

    Or, in many cases, you can just use `items_with_caught_exceptions`.

    :param d: Any Mapping
    :param catch_exceptions: A tuple of exceptions that should be caught
    :param callback: A function that will be called every time an exception is caught.
        The signature of the callback function is required to be:
            k (key), e (error obj), d (mapping), i (index)
        but
    :return: An (key, val) generator with exceptions caught

    >>> from collections.abc import Mapping
    >>> class Test(Mapping):  # a Mapping class that has keys 0..9, but raises of KeyError if the key is not even
    ...     n = 10
    ...     def __iter__(self):
    ...         yield from range(2, self.n)
    ...     def __len__(self):
    ...         return self.n
    ...     def __getitem__(self, k):
    ...         if k % 2 == 0:
    ...             return k
    ...         else:
    ...             raise KeyError('Not even')
    >>>
    >>> list(items_with_caught_exceptions(Test()))
    [(2, 2), (4, 4), (6, 6), (8, 8)]
    >>>
    >>> def my_log(k, e):
    ...     print(k, e)
    >>> list(items_with_caught_exceptions(Test(), callback=my_log))
    3 'Not even'
    5 'Not even'
    7 'Not even'
    9 'Not even'
    [(2, 2), (4, 4), (6, 6), (8, 8)]
    >>> def my_other_log(i):
    ...     print(i)
    >>> list(items_with_caught_exceptions(Test(), callback=my_other_log))
    1
    3
    5
    7
    [(2, 2), (4, 4), (6, 6), (8, 8)]
    """

    # wrap the input callback to make the callback definition less constrained for the user.
    if callback is not None:
        try:
            params = signature(callback).parameters

            def wrapped_callback(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k in params}
                return callback(**kwargs)

        except ValueError:

            def wrapped_callback(k, e, d, i):
                return callback(k, e, d, i)

    else:

        def wrapped_callback(k, e, d, i):
            pass  # do nothing

    for i, k in enumerate(d):  # iterate over keys of the mapping
        try:
            v = d[k]  # try getting the value...
            yield k, v  # and if you do, yield the (k, v) pair
        except (
            catch_exceptions
        ) as e:  # catch the specific exceptions you requested to catch
            t = wrapped_callback(k=k, e=e, d=d, i=i)  # call it
            if yield_callback_output:  # if the user wants the output of the callback
                yield t  # yield it


def _assert_condition(condition, err_msg="", err_cls=AssertionError):
    if not condition:
        raise err_cls(err_msg)


class NotValid(ValueError, TypeError):
    """To use to indicate when an object doesn't fit expected properties"""


class KeyValidationError(NotValid):
    """Error to raise when a key is not valid"""


class NoSuchKeyError(KeyError):
    """When a requested key doesn't exist"""


class NotAllowed(Exception):
    """To use to indicate that something is not allowed"""


class OperationNotAllowed(NotAllowed, NotImplementedError):
    """When a given operation is not allowed (through being disabled, conditioned, or just implemented)"""


class ReadsNotAllowed(OperationNotAllowed):
    """Read OperationNotAllowed"""


class WritesNotAllowed(OperationNotAllowed):
    """Write OperationNotAllowed"""


class DeletionsNotAllowed(OperationNotAllowed):
    """Delete OperationNotAllowed"""


class IterationNotAllowed(OperationNotAllowed):
    """Iteration OperationNotAllowed"""


class OverWritesNotAllowedError(OperationNotAllowed):
    """Error to raise when a writes to existing keys are not allowed"""


class AlreadyExists(ValueError):
    """To use if an object already exists (and shouldn't; for example, to protect overwrites)"""


class MethodNameAlreadyExists(AlreadyExists):
    """To use when a method name already exists (and shouldn't)"""


class MethodFuncNotValid(NotValid):
    """Use when method function is not valid"""


class SetattrNotAllowed(NotAllowed):
    """An attribute was requested to be set, but some conditions didn't apply"""
```

## explicit.py

```python
"""
utils to make stores based on a the input data itself
"""

from collections.abc import Mapping
from typing import KT, VT, TypeVar
from collections.abc import Callable, Collection as CollectionType, Iterator

from dol.base import Collection, KvReader, Store
from dol.trans import kv_wrap
from dol.util import max_common_prefix
from dol.sources import ObjReader  # because it used to be here


Source = TypeVar("Source")  # the source of some values
Getter = Callable[
    [Source, KT], VT
]  # a function that gets a value from a source and a key
# TODO: Might want to make the Getter by generic, so that we can do things like
#   Getter[Mapping] or Getter[Mapping, KeyType] or Getter[Any, KeyType]


class KeysReader(Mapping):
    """
    Mapping defined by keys with a getter function that gets values from keys.

    `KeysReader` is particularly useful in cases where you want to have a mapping
    that lazy-load values for keys from an explicit collection.

    Keywords: Lazy-evaluation, Mapping

    Args:
        src: The source where values will be extracted from.
        key_collection: A collection of keys that will be used to extract values from `src`.
        getter: A function that takes a source and a key, and returns the value for that key.
        key_error_msg: A function that takes a source and a key, and returns an error message.


    Example::

    >>> src = {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}
    >>> key_collection = ['carrot', 'apple']
    >>> getter = lambda src, key: src[key]
    >>> key_reader = KeysReader(src, key_collection, getter)

    Note that the only the keys mentioned by `key_collection` will be iterated through,
    and in the order they are mentioned in `key_collection`.

    >>> list(key_reader)
    ['carrot', 'apple']

    >>> key_reader['apple']
    'pie'
    >>> key_reader['banana']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key 'banana' was not found....key_collection attribute)"

    Let's take the same `src` and `key_collection`, but with a different getter and
    key_error_msg:

    Note that a key_error_msg must be a function that takes a `src` and a `key`,
    in that order and with those argument names. Say you wanted to not use the `src`
    in your message. You would still have to write a function that takes `src` as the
    first argument.

    >>> key_error_msg = lambda src, key: f"Key {key} was not found"  # no source information

    >>> getter = lambda src, key: f"Value for {key} in {src}: {src[key]}"
    >>> key_reader = KeysReader(src, key_collection, getter, key_error_msg=key_error_msg)
    >>> list(key_reader)
    ['carrot', 'apple']
    >>> key_reader['apple']
    "Value for apple in {'apple': 'pie', 'banana': 'split', 'carrot': 'cake'}: pie"
    >>> key_reader['banana']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key banana was not found"

    """

    def __init__(
        self,
        src: Source,
        key_collection: CollectionType[KT],
        getter: Callable[[Source, KT], VT],
        *,
        key_error_msg: Callable[
            [Source, KT], str
        ] = "Key {key} was not found in {src} should be in .key_collection attribute)".format,
    ) -> None:
        self.src = src
        self.key_collection = key_collection
        self.getter = getter
        self.key_error_msg = key_error_msg

    def __getitem__(self, key: KT) -> VT:
        if key in self:
            return self.getter(self.src, key)
        else:
            raise KeyError(self.key_error_msg(src=self.src, key=key))

    def __iter__(self) -> Iterator[KT]:
        yield from self.key_collection

    def __len__(self) -> int:
        return len(self.key_collection)

    def __contains__(self, key: KT) -> bool:
        return key in self.key_collection


# --------------------------------------------------------------------------------------
# Older stuff:


# TODO: Revisit ExplicitKeys and ExplicitKeysWithPrefixRelativization. Not extendible to full store!
class ExplicitKeys(Collection):
    """
    dol.base.Keys implementation that gets it's keys explicitly from a collection given
    at initialization time.
    The key_collection must be a collections.abc.Collection
    (such as list, tuple, set, etc.)

    >>> keys = ExplicitKeys(key_collection=['foo', 'bar', 'alice'])
    >>> 'foo' in keys
    True
    >>> 'not there' in keys
    False
    >>> list(keys)
    ['foo', 'bar', 'alice']
    """

    __slots__ = ("_keys_cache",)

    def __init__(
        self, key_collection: CollectionType
    ):  # don't remove this init: Don't. Need for _keys_cache init
        assert isinstance(key_collection, CollectionType), (
            "key_collection must be a collections.abc.Collection, i.e. have a __len__, __contains__, and __len__."
            "The key_collection you gave me was a {}".format(type(key_collection))
        )
        # self._key_collection = key_collection
        self._keys_cache = key_collection

    def __iter__(self):
        yield from self._keys_cache

    def __len__(self):
        return len(self._keys_cache)

    def __contains__(self, k):
        return k in self._keys_cache


# TODO: Should we deprecate or replace with recipe?
class ExplicitKeysSource(ExplicitKeys, ObjReader, KvReader):
    """
    An object source that uses an explicit keys collection and a specified function to
    read contents for a key.

    >>> s = ExplicitKeysSource([1, 2, 3], str)
    >>> list(s)
    [1, 2, 3]
    >>> list(s.values())
    ['1', '2', '3']

    Main functionality equivalent to recipe:

    >>> def explicit_keys_source(key_collection, _obj_of_key):
    ...     from dol.trans import wrap_kvs
    ...     return wrap_kvs({k: k for k in key_collection}, obj_of_data=_obj_of_key)

    >>> s = explicit_keys_source([1, 2, 3], str)
    >>> list(s)
    [1, 2, 3]
    >>> list(s.values())
    ['1', '2', '3']

    """

    def __init__(self, key_collection: CollectionType, _obj_of_key: Callable):
        """

        :param key_collection: The collection of keys that this source handles
        :param _obj_of_key: The function that returns the contents for a key
        """
        ObjReader.__init__(self, _obj_of_key)
        self._keys_cache = key_collection


class ExplicitKeysStore(ExplicitKeys, Store):
    """Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.

    >>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> list(s)
    ['a', 'b', 'c', 'd']
    >>> ss = ExplicitKeysStore(s, ['d', 'a'])
    >>> len(ss)
    2
    >>> list(ss)
    ['d', 'a']
    >>> list(ss.values())
    [4, 1]
    >>> ss.head()
    ('d', 4)
    """

    def __init__(self, store, key_collection):
        Store.__init__(self, store)
        self._keys_cache = key_collection


from dol.util import invertible_maps


# TODO: Put on the path of deprecation, since KeyCodecs.mapped_keys is a better way to do this.
class ExplicitKeyMap:
    def __init__(self, *, key_of_id: Mapping = None, id_of_key: Mapping = None):
        """

        :param key_of_id:
        :param id_of_key:

        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2})
        >>> km.id_of_key = {1: 'a', 2: 'b'}
        >>> km._key_of_id('b')
        2
        >>> km._id_of_key(1)
        'a'
        >>> # You can specify id_of_key instead
        >>> km = ExplicitKeyMap(id_of_key={1: 'a', 2: 'b'})
        >>> assert km.key_of_id_map == {'a': 1, 'b': 2}
        >>> # You can specify both key_of_id and id_of_key
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2}, id_of_key={1: 'a', 2: 'b'})
        >>> assert km._key_of_id(km._id_of_key(2)) == 2
        >>> assert km._id_of_key(km._key_of_id('b')) == 'b'
        >>> # But they better be inverse of each other!
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2, 'c': 2})
        Traceback (most recent call last):
          ...
        AssertionError: The values of inv_mapping are not unique, so the mapping is not invertible
        >>> km = ExplicitKeyMap(key_of_id={'a': 1, 'b': 2}, id_of_key={1: 'a', 2: 'oh no!!!!'})
        Traceback (most recent call last):
          ...
        AssertionError: mapping and inv_mapping are not inverse of each other!
        """
        id_of_key, key_of_id = invertible_maps(id_of_key, key_of_id)
        self.key_of_id_map = key_of_id
        self.id_of_key_map = id_of_key

    def _key_of_id(self, _id):
        return self.key_of_id_map[_id]

    def _id_of_key(self, k):
        return self.id_of_key_map[k]


class ExplicitKeymapReader(ExplicitKeys, Store):
    """Wrap a store (instance) so that it gets it's keys from an explicit iterable of keys.

    >>> s = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> id_of_key = {'A': 'a', 'C': 'c'}
    >>> ss = ExplicitKeymapReader(s, id_of_key=id_of_key)
    >>> list(ss)
    ['A', 'C']
    >>> ss['C']  # will look up 'C', find 'c', and call the store on that.
    3
    """

    def __init__(self, store, key_of_id=None, id_of_key=None):
        key_trans = ExplicitKeyMap(key_of_id=key_of_id, id_of_key=id_of_key)
        Store.__init__(self, kv_wrap(key_trans)(store))
        ExplicitKeys.__init__(self, key_trans.id_of_key_map.keys())


# ExplicitKeysWithPrefixRelativization: Moved to dol.paths


class ObjDumper:
    def __init__(self, save_data_to_key, data_of_obj=None):
        self.save_data_to_key = save_data_to_key
        if data_of_obj is not None or not callable(data_of_obj):
            raise TypeError("serializer must be None or a callable")
        self.data_of_obj = data_of_obj

    def __call__(self, k, v):
        if self.data_of_obj is not None:
            return self.save_data_to_key(k, self.data_of_obj(v))
        else:
            return self.save_data_to_key(k, v)
```

## filesys.py

```python
"""File system access"""

import os
from os import stat as os_stat
from functools import wraps, partial
from typing import Union, Optional
from collections.abc import Callable, Iterable

from dol.base import Collection, KvReader, KvPersister
from dol.trans import wrap_kvs, store_decorator, filt_iter
from dol.naming import mk_pattern_from_template_and_format_dict
from dol.paths import mk_relative_path_store

file_sep = os.path.sep
inf = float("infinity")


def ensure_slash_suffix(path: str):
    r"""Add a file separation (/ or \) at the end of path str, if not already present."""
    if not path.endswith(file_sep):
        return path + file_sep
    else:
        return path


def paths_in_dir(rootdir, include_hidden=False):
    try:
        for name in os.listdir(rootdir):
            if include_hidden or not name.startswith(
                "."
            ):  # TODO: is dot a platform independent marker for hidden file?
                filepath = os.path.join(rootdir, name)
                if os.path.isdir(filepath):
                    yield ensure_slash_suffix(filepath)
                else:
                    yield filepath
    except FileNotFoundError:
        pass


def iter_filepaths_in_folder_recursively(
    root_folder, max_levels=None, _current_level=0, include_hidden=False
):
    """Recursively generates filepaths of folder (and subfolders, etc.) up to a given level"""
    if max_levels is None:
        max_levels = inf
    for full_path in paths_in_dir(root_folder, include_hidden):
        if os.path.isdir(full_path):
            if _current_level < max_levels:
                yield from iter_filepaths_in_folder_recursively(
                    full_path, max_levels, _current_level + 1, include_hidden
                )
        else:
            if os.path.isfile(full_path):
                yield full_path


def iter_dirpaths_in_folder_recursively(
    root_folder, max_levels=None, _current_level=0, include_hidden=False
):
    """Recursively generates dirpaths of folder (and subfolders, etc.) up to a given level"""
    if max_levels is None:
        max_levels = inf
    for full_path in paths_in_dir(root_folder, include_hidden):
        if os.path.isdir(full_path):
            yield full_path
            if _current_level < max_levels:
                yield from iter_dirpaths_in_folder_recursively(
                    full_path, max_levels, _current_level + 1, include_hidden
                )


def create_directories(dirpath, max_dirs_to_make: int | None = None):
    """
    Create directories up to a specified limit.

    Parameters:
    dirpath (str): The directory path to create.
    max_dirs_to_make (int, optional): The maximum number of directories to create. If None, there's no limit.

    Returns:
    bool: True if the directory was created successfully, False otherwise.

    Raises:
    ValueError: If max_dirs_to_make is negative.

    Examples:
    >>> import tempfile, shutil
    >>> temp_dir = tempfile.mkdtemp()
    >>> target_dir = os.path.join(temp_dir, 'a', 'b', 'c')
    >>> create_directories(target_dir, max_dirs_to_make=2)
    False
    >>> create_directories(target_dir, max_dirs_to_make=3)
    True
    >>> os.path.isdir(target_dir)
    True
    >>> shutil.rmtree(temp_dir)  # Cleanup

    >>> temp_dir = tempfile.mkdtemp()
    >>> target_dir = os.path.join(temp_dir, 'a', 'b', 'c', 'd')
    >>> create_directories(target_dir)
    True
    >>> os.path.isdir(target_dir)
    True
    >>> shutil.rmtree(temp_dir)  # Cleanup
    """
    if max_dirs_to_make is not None and max_dirs_to_make < 0:
        raise ValueError("max_dirs_to_make must be non-negative or None")

    if os.path.exists(dirpath):
        return True

    if max_dirs_to_make is None:
        os.makedirs(dirpath, exist_ok=True)
        return True

    # Calculate the number of directories to create
    dirs_to_make = []
    current_path = dirpath

    while not os.path.exists(current_path):
        dirs_to_make.append(current_path)
        current_path, _ = os.path.split(current_path)

    if len(dirs_to_make) > max_dirs_to_make:
        return False

    # Create directories from the top level down
    for dir_to_make in reversed(dirs_to_make):
        os.mkdir(dir_to_make)

    return True


def process_path(
    *path: Iterable[str],
    ensure_dir_exists: int | bool = False,
    assert_exists: bool = False,
    ensure_endswith_slash: bool = False,
    ensure_does_not_end_with_slash: bool = False,
    expanduser: bool = True,
    expandvars: bool = True,
    abspath: bool = True,
    rootdir: str = "",
) -> str:
    """
    Process a path string, ensuring it exists, and optionally expanding user.

    Args:
        path (Iterable[str]): The path to process. Can be multiple components of a path.
        ensure_dir_exists (bool): Whether to ensure the path exists.
        assert_exists (bool): Whether to assert that the path exists.
        ensure_endswith_slash (bool): Whether to ensure the path ends with a slash.
        ensure_does_not_end_with_slash (bool): Whether to ensure the path does not end with a slash.
        expanduser (bool): Whether to expand the user in the path.
        expandvars (bool): Whether to expand environment variables in the path.
        abspath (bool): Whether to convert the path to an absolute path.
        rootdir (str): The root directory to prepend to the path.

    Returns:
        str: The processed path.

    >>> process_path('a', 'b', 'c')  # doctest: +ELLIPSIS
    '...a/b/c'
    >>> from functools import partial
    >>> process_path('a', 'b', 'c', rootdir='/root/dir/', ensure_endswith_slash=True)
    '/root/dir/a/b/c/'

    """
    path = os.path.join(*path)
    if ensure_endswith_slash and ensure_does_not_end_with_slash:
        raise ValueError(
            "Cannot ensure both ends with slash and does not end with slash."
        )
    if rootdir:
        path = os.path.join(rootdir, path)
    if expanduser:
        path = os.path.expanduser(path)
    if expandvars:
        path = os.path.expandvars(path)
    if abspath:
        path = os.path.abspath(path)
    if ensure_endswith_slash:
        if not path.endswith(os.path.sep):
            path = path + os.path.sep
    if ensure_does_not_end_with_slash:
        if path.endswith(os.path.sep):
            path = path[:-1]
    if ensure_dir_exists:
        if ensure_dir_exists is True:
            ensure_dir_exists = None  # max_dirs_to_make
        create_directories(path, max_dirs_to_make=ensure_dir_exists)
    if assert_exists:
        assert os.path.exists(path), f"Path does not exist: {path}"
    return path


def ensure_dir(
    dirpath,
    *,
    max_dirs_to_make: int | None = None,
    verbose: bool | str | Callable = False,
):
    """Ensure that a directory exists, creating it if necessary.

    :param dirpath: path to the directory to create
    :param max_dirs_to_make: the maximum number of directories to create.
        If None, there's no limit.
    :param verbose: controls verbosity (the noise ensure_dir makes if it make folder)
    :return: the path to the directory

    When the path does not exist, if ``verbose`` is:

    - a ``bool``' a standard message will be printed

    - a ``callable``; will be called on dirpath before directory is created -- you
    can use this to ask the user for confirmation for example

    - a ''string``; this string will be printed


    Usage note: If you want to string or the (argument-less) callable to be dependent
    on ``dirpath``, you need make them so when calling ensure_dir.

    """
    if not os.path.exists(dirpath):
        if verbose:
            if isinstance(verbose, bool):
                print(f"Making the directory: {dirpath}")
            elif isinstance(verbose, Callable):
                callaback = verbose
                callaback(dirpath)
            else:
                string_to_print = verbose
                print(string_to_print)
        create_directories(dirpath, max_dirs_to_make=max_dirs_to_make)
    return dirpath


def temp_dir(dirname="", make_it_if_necessary=True, verbose=False):
    """
    Create and return a path to a temporary directory that's guaranteed to be
    accessible to the user.

    Parameters:
    ----------
    dirname : str
        Optional subdirectory name to append to the temporary directory path
    make_it_if_necessary : bool
        Whether to create the directory if it doesn't exist
    verbose : bool, str, or callable
        Controls verbosity when creating directories

    Returns:
    -------
    str
        Path to a temporary directory that the user has access to

    Notes:
    -----
    This function creates a user-specific temporary directory to avoid permission
    issues with system-wide temporary directories. The directory is guaranteed to
    be accessible to the current user.
    """
    from tempfile import mkdtemp, gettempdir
    import uuid

    # Create a unique user-specific directory under the system temp dir
    user_temp_base = os.path.join(gettempdir(), f"user_{os.getuid()}")

    if dirname:
        # If a specific dirname is provided, use it
        tmpdir = os.path.join(user_temp_base, dirname)
    else:
        # Otherwise create a unique directory with uuid
        unique_id = str(uuid.uuid4())[:8]
        tmpdir = os.path.join(user_temp_base, f"dol_temp_{unique_id}")

    if make_it_if_necessary:
        try:
            ensure_dir(tmpdir, verbose=verbose)
            # Verify we have write access
            test_file = os.path.join(tmpdir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except (PermissionError, OSError):
            # Fallback: create a truly temporary directory with mkdtemp
            if verbose:
                print(f"Could not access {tmpdir}, falling back to mkdtemp()")
            tmpdir = mkdtemp(prefix=f"dol_temp_")

    return tmpdir


mk_tmp_dol_dir = temp_dir  # for backward compatibility


def mk_absolute_path(path_format):
    if path_format.startswith("~"):
        path_format = os.path.expanduser(path_format)
    elif path_format.startswith("."):
        path_format = os.path.abspath(path_format)
    return path_format


# TODO: subpath: Need to be able to allow named and unnamed file format markers (i.e {} and {named})

_dflt_not_valid_error_msg = (
    "Key not valid (usually because does not exist or access not permitted): {}"
)
_dflt_not_found_error_msg = "Key not found: {}"


class KeyValidationError(KeyError):
    pass


# TODO: The validate and try/except is a frequent pattern. Make it a decorator.
def validate_key_and_raise_key_error_on_exception(func):
    @wraps(func)
    def wrapped_method(self, k, *args, **kwargs):
        self.validate_key(k)
        try:
            return func(self, k, *args, **kwargs)
        except Exception as e:
            raise KeyError(str(e))

    return wrapped_method


def resolve_path(path, assert_existence: bool = False):
    """Resolve a path to a full, real, (file or folder) path (opt assert existence).
    That is, resolve situations where ~ and . prefix the paths.
    """
    if path.startswith("."):
        path = os.path.abspath(path)
    elif path.startswith("~"):
        path = os.path.expanduser(path)
    if assert_existence:
        assert os.path.exists(path), f"This path (file or folder) wasn't found: {path}"
    return path


def resolve_dir(
    dirpath: str, assert_existence: bool = False, ensure_existence: bool = False
):
    """Resolve a path to a full, real, path to a directory"""
    dirpath = resolve_path(dirpath)
    if ensure_existence and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    if assert_existence:
        assert os.path.isdir(dirpath), f"This directory wasn't found: {dirpath}"
    return dirpath


def _for_repr(obj, quote="'"):
    """
    >>> _for_repr('a string')
    "'a string'"
    >>> _for_repr(10)
    10
    >>> _for_repr(None)
    'None'
    """
    if isinstance(obj, str):
        obj = f"{quote}{obj}{quote}"
    elif obj is None:
        obj = "None"
    return obj


class FileSysCollection(Collection):
    # rootdir = None  # mentioning here so that the attribute is seen as an attribute before instantiation.

    def __init__(
        self,
        rootdir,
        subpath="",
        pattern_for_field=None,
        max_levels=None,
        *,
        include_hidden=False,
        assert_rootdir_existence=False,
    ):
        self._init_kwargs = {k: v for k, v in locals().items() if k != "self"}
        rootdir = resolve_dir(rootdir, assert_existence=assert_rootdir_existence)
        if max_levels is None:
            max_levels = inf
        subpath_implied_min_levels = len(subpath.split(os.path.sep)) - 1
        assert (
            max_levels >= subpath_implied_min_levels
        ), f"max_levels is {max_levels}, but subpath {subpath} would imply at least {subpath_implied_min_levels}"
        pattern_for_field = pattern_for_field or {}
        self.rootdir = ensure_slash_suffix(rootdir)
        self.subpath = subpath
        self._key_pattern = mk_pattern_from_template_and_format_dict(
            os.path.join(rootdir, subpath), pattern_for_field
        )
        self._max_levels = max_levels
        self.include_hidden = include_hidden

    def is_valid_key(self, k):
        return bool(self._key_pattern.match(k))

    def validate_key(
        self,
        k,
        err_msg_format=_dflt_not_valid_error_msg,
        err_type=KeyValidationError,
    ):
        if not self.is_valid_key(k):
            raise err_type(err_msg_format.format(k))

    def __repr__(self):
        input_str = ", ".join(
            f"{k}={_for_repr(v)}" for k, v in self._init_kwargs.items()
        )
        return f"{type(self).__name__}({input_str})"

    def with_relative_paths(self):
        """Return a copy of self with relative paths"""
        return with_relative_paths(self)


class DirCollection(FileSysCollection):
    def __iter__(self):
        yield from filter(
            self.is_valid_key,
            iter_dirpaths_in_folder_recursively(
                self.rootdir,
                max_levels=self._max_levels,
                include_hidden=self.include_hidden,
            ),
        )

    def __contains__(self, k):
        return self.is_valid_key(k) and os.path.isdir(k)


class FileCollection(FileSysCollection):
    def __iter__(self):
        """
        Iterator of valid filepaths.

        >>> import os
        >>> filepath = __file__  # path to this module
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileCollection(dirpath, max_levels=0)
        >>>
        >>> files_in_this_dir = list(s)
        >>> filepath in files_in_this_dir
        True
        """
        yield from filter(
            self.is_valid_key,
            iter_filepaths_in_folder_recursively(
                self.rootdir,
                max_levels=self._max_levels,
                include_hidden=self.include_hidden,
            ),
        )

    def __contains__(self, k):
        """
        Checks if k is valid and contained in the store

        >>> import os
        >>> filepath = __file__  # path to this module
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileCollection(dirpath, max_levels=0)
        >>>
        >>> filepath in s
        True
        >>> '_this_filepath_will_never_be_valid_' in s
        False
        """
        return self.is_valid_key(k) and os.path.isfile(k)


class FileInfoReader(FileCollection, KvReader):
    def __getitem__(self, k):
        self.validate_key(k)
        return os_stat(k)


class FileBytesReader(FileCollection, KvReader):
    _read_open_kwargs = dict(
        mode="rb",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    )

    @validate_key_and_raise_key_error_on_exception
    def __getitem__(self, k):
        '''
        Gets the bytes contents of the file k.

        >>> import os
        >>> filepath = __file__
        >>> dirpath = os.path.dirname(__file__)  # path of the directory where I (the module file) am
        >>> s = FileBytesReader(dirpath, max_levels=0)
        >>>
        >>> ####### Get the first 9 characters (as bytes) of this module #####################
        >>> s[filepath][:9]
        b'"""File s'
        >>>
        >>> ####### Test key validation #####################
        >>> # this key is not valid since not under the dirpath folder, so should give an exception
        >>> # Skipped because filesys.KeyValidationError vs dol.filesys.KeyValidationError on different systems
        >>> s['not_a_valid_key']  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        filesys.KeyValidationError: 'Key not valid (usually because does not exist or access not permitted): not_a_valid_key'
        >>>
        >>> ####### Test further exceptions (that should be wrapped in KeyError) #####################
        >>> # this key is valid, since under dirpath, but the file itself doesn't exist (hopefully for this test)
        >>> non_existing_file = os.path.join(dirpath, 'non_existing_file')
        >>> try:
        ...     s[non_existing_file]
        ... except KeyError:
        ...     print("KeyError (not FileNotFoundError) was raised.")
        KeyError (not FileNotFoundError) was raised.
        '''
        with open(k, **self._read_open_kwargs) as fp:
            return fp.read()


class LocalFileDeleteMixin:
    @validate_key_and_raise_key_error_on_exception
    def __delitem__(self, k):
        os.remove(k)


class FileBytesPersister(FileBytesReader, KvPersister):
    _write_open_kwargs = dict(
        mode="wb",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    )
    # _make_dirs_if_missing = False

    @validate_key_and_raise_key_error_on_exception
    def __setitem__(self, k, v):
        # TODO: Make this work with validate_key_and_raise_key_error_on_exception
        # if self._make_dirs_if_missing:
        #     dirname = os.path.dirname(k)
        #     os.makedirs(dirname, exist_ok=True)
        with open(k, **self._write_open_kwargs) as fp:
            return fp.write(v)

    @validate_key_and_raise_key_error_on_exception
    def __delitem__(self, k):
        os.remove(k)


# ---------------------------------------------------------------------------------------
# TODO: Once test coverage sufficient, apply this pattern to all other convenience stores

with_relative_paths = partial(mk_relative_path_store, prefix_attr="rootdir")


@with_relative_paths
class FilesReader(FileBytesReader):
    """FileBytesReader with relative paths"""


@with_relative_paths
class Files(FileBytesPersister):
    """FileBytesPersister with relative paths"""


RelPathFileBytesReader = FilesReader
RelPathFileBytesPersister = Files  # back-compatibility alias

# ---------------------------------------------------------------------------------------


class FileStringReader(FileBytesReader):
    _read_open_kwargs = dict(FileBytesReader._read_open_kwargs, mode="rt")


class FileStringPersister(FileBytesPersister):
    _read_open_kwargs = dict(FileBytesReader._read_open_kwargs, mode="rt")
    _write_open_kwargs = dict(FileBytesPersister._write_open_kwargs, mode="wt")


@with_relative_paths(prefix_attr="rootdir")
class TextFilesReader(FileStringReader):
    """FileStringReader with relative paths"""


@with_relative_paths(prefix_attr="rootdir")
class TextFiles(FileStringPersister):
    """FileStringPersister with relative paths"""


RelPathFileStringReader = TextFilesReader
RelPathFileStringPersister = TextFiles


# ------------------------------------ misc --------------------------------------------
import pickle
import json

# TODO: Want to replace with use of ValueCodecs but need to resolve circular imports
pickle_bytes_wrap = wrap_kvs(value_decoder=pickle.loads, value_encoder=pickle.dumps)
json_bytes_wrap = wrap_kvs(
    value_decoder=json.loads, value_encoder=partial(json.dumps, indent=4)
)


# And two factories to make the above more configurable:
def mk_pickle_bytes_wrap(
    *, loads_kwargs: dict | None = None, dumps_kwargs: dict | None = None
) -> Callable:
    """"""
    return wrap_kvs(
        value_decoder=partial(pickle.loads, **(loads_kwargs or {})),
        value_encoder=partial(pickle.dumps, **(dumps_kwargs or {})),
    )


def mk_json_bytes_wrap(
    *, loads_kwargs: dict | None = None, dumps_kwargs: dict | None = None
) -> Callable:
    return wrap_kvs(
        value_decoder=partial(json.loads, **(loads_kwargs or {})),
        value_encoder=partial(json.dumps, **(dumps_kwargs or {})),
    )


class ReprMixin:
    def __repr__(self):
        input_str = ", ".join(
            f"{k}={_for_repr(v)}" for k, v in getattr(self, "_init_kwargs", {}).items()
        )
        return f"{type(self).__name__}({input_str})"


@pickle_bytes_wrap
class PickleFiles(ReprMixin, Files):
    """A store of pickles"""


PickleStore = PickleFiles  # back-compatibility alias


@json_bytes_wrap
class JsonFiles(ReprMixin, TextFiles):
    """A store of json files"""


from dol.trans import affix_key_codec


@affix_key_codec(suffix=".json")
@filt_iter.suffixes(".json")
class Jsons(ReprMixin, JsonFiles):
    """Like JsonFiles, but with added .json extension handling
    Namely: filtering for `.json` extensions but not showing the extension in keys"""


# @wrap_kvs(key_of_id=lambda x: x[:-1], id_of_key=lambda x: x + path_sep)
@mk_relative_path_store(prefix_attr="rootdir")
class PickleStores(DirCollection):
    def __getitem__(self, k):
        return PickleFiles(k)

    def __repr__(self):
        return f"{type(self).__name__}('{self.rootdir}', ...)"


class DirReader(DirCollection, KvReader):
    def __getitem__(self, k):
        return DirReader(k)


# TODO: This, with mk_dirs_if_missing, should replace uses of AutoMkDirsOnSetitemMixin and MakeMissingDirsStoreMixin
def mk_dirs_if_missing_preset(
    self, k, v, *, max_dirs_to_make: int | None = None, verbose=False
):
    """
    Preset function that will make the store create directories on write as needed.
    """
    # TODO: I'm not thrilled in the way I'm doing this; find alternatives
    try:
        super(type(self), self).__setitem__(k, v)
    except Exception:  # general on purpose...
        # TODO: ... But perhaps a more precise (but sufficient) exception list better?
        from dol.dig import inner_most_key

        # get the inner most key, which should be a full path
        _id = inner_most_key(self, k)
        # get the full path of directory needed for this file
        dirname = os.path.dirname(_id)
        # make all the directories needed
        ensure_dir(dirname, max_dirs_to_make=max_dirs_to_make, verbose=verbose)
        # os.makedirs(dirname, exist_ok=True)  # TODO: ensure_dir does this already, no?
        # try writing again
        super(type(self), self).__setitem__(k, v)
        # TODO: Undesirable here: If the setitem still fails, we created dirs
        #  already, for nothing, and are not cleaning up (if clean up need to make
        #  sure to not delete dirs that already existed!)
    finally:
        return v


# TODO: Add more control over mk dir condition (e.g. number of levels, or any key cond)
#   Also, add a verbose option to print the dirs that are being made
#   (see dol.filesys.ensure_dir)
@store_decorator
def mk_dirs_if_missing(
    store_cls=None,
    *,
    max_dirs_to_make: int | None = None,
    verbose: bool | str | Callable = False,
    key_condition=None,  # TODO: not used! Should use! Add to ensure_dir
):
    """Store decorator that will make the store create directories on write as
    needed.

    Note that it'll only effect paths relative to the rootdir, which needs to be
    ensured to exist separatedly.
    """
    _mk_dirs_if_missing_preset = partial(
        mk_dirs_if_missing_preset, max_dirs_to_make=max_dirs_to_make, verbose=verbose
    )
    return wrap_kvs(store_cls, preset=_mk_dirs_if_missing_preset)


# DEPRECATED!!
# This one really smells. Replace uses with mk_dirs_if_missing
class MakeMissingDirsStoreMixin:
    """Will make a local file store automatically create the directories needed to create a file.
    Should be placed before the concrete perisister in the mro but in such a manner so that it receives full paths.
    """

    _verbose: bool | str | Callable = False  # eek! Can't set in init.

    def __setitem__(self, k, v):
        print(
            f"Deprecating message: Consider using the mk_dirs_if_missing decorator instead."
        )
        # TODO: I'm not thrilled in the way I'm doing this; find alternatives
        try:
            super().__setitem__(k, v)
        except Exception:  # general on purpose...
            # TODO: ... But perhaps a more precise (but sufficient) exception list better?
            from dol.dig import inner_most_key

            # get the inner most key, which should be a full path
            _id = inner_most_key(self, k)
            # get the full path of directory needed for this file
            dirname = os.path.dirname(_id)
            # make all the directories needed
            ensure_dir(dirname, self._verbose)
            os.makedirs(dirname, exist_ok=True)
            # try writing again
            super().__setitem__(k, v)
            # TODO: Undesirable here: If the setitem still fails, we created dirs
            #  already, for nothing, and are not cleaning up (if clean up need to make
            #  sure to not delete dirs that already existed!)


# -------------------------------------------------------------------------------------

from dol.kv_codecs import KeyCodecs


def subfolder_stores(
    root_folder,
    *,
    max_levels: int | None = None,
    include_hidden: bool = False,
    relative_paths: bool = True,
    slash_suffix: bool = False,
    folder_to_store=Files,
):
    """
    Create a store of subfolders of a given folder, where the keys are the subfolder
    paths (by default, relative and slash-less) and the values are stores of these
    subfolders.

    By default, all subfolders will be taken, recursively, but this can be controlled by
    the `max_levels` parameter.
    """
    root_folder = ensure_slash_suffix(root_folder)
    wrap = KeyCodecs.affixed(
        prefix=root_folder if relative_paths else "",
        suffix="/" if not slash_suffix else "",
    )
    folders = iter_dirpaths_in_folder_recursively(
        root_folder, max_levels=max_levels, include_hidden=include_hidden
    )
    return wrap({path: folder_to_store(path) for path in folders})
```

## kv_codecs.py

```python
"""
Tools to make Key-Value Codecs (encoder-decoder pairs) from standard library tools.
"""

# ------------------------------------ Codecs ------------------------------------------

from functools import partial
from typing import Any, Optional, KT, VT, Union, Dict
from collections.abc import Callable, Iterable, Mapping
from operator import itemgetter

from dol.trans import (
    Codec,
    ValueCodec,
    KeyCodec,
    KeyValueCodec,
    affix_key_codec,
    store_decorator,
)
from dol.paths import KeyTemplate
from dol.signatures import Sig
from dol.util import named_partial, identity_func, single_nest_in_dict, nest_in_dict

# For the codecs:
import csv
import io


@Sig
def _string(string: str): ...


@Sig
def _csv_rw_sig(
    dialect: str = "excel",
    *,
    delimiter: str = ",",
    quotechar: str | None = '"',
    escapechar: str | None = None,
    doublequote: bool = True,
    skipinitialspace: bool = False,
    lineterminator: str = "\r\n",
    quoting=0,
    strict: bool = False,
): ...


@Sig
def _csv_dict_extra_sig(
    fieldnames, restkey=None, restval="", extrasaction="raise", fieldcasts=None
): ...


__csv_rw_sig = _string + _csv_rw_sig
__csv_dict_sig = _string + _csv_rw_sig + _csv_dict_extra_sig


# Note: @(_string + _csv_rw_sig) made (ax)black choke
@__csv_rw_sig
def csv_encode(string, *args, **kwargs):
    with io.StringIO() as buffer:
        writer = csv.writer(buffer, *args, **kwargs)
        writer.writerows(string)
        return buffer.getvalue()


@__csv_rw_sig
def csv_decode(string, *args, **kwargs):
    with io.StringIO(string) as buffer:
        reader = csv.reader(buffer, *args, **kwargs)
        return list(reader)


@__csv_dict_sig
def csv_dict_encode(string, *args, **kwargs):
    r"""Encode a list of dicts into a csv string.

    >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
    >>> encoded
    'a,b\r\n1,2\r\n3,4\r\n'

    """
    _ = kwargs.pop("fieldcasts", None)  # this one is for decoder only
    with io.StringIO() as buffer:
        writer = csv.DictWriter(buffer, *args, **kwargs)
        writer.writeheader()
        writer.writerows(string)
        return buffer.getvalue()


@__csv_dict_sig
def csv_dict_decode(string, *args, **kwargs):
    r"""Decode a csv string into a list of dicts.

    :param string: The csv string to decode
    :param fieldcasts: A function that takes a row and returns a row with the same keys
        but with values cast to the desired type. If a dict, it should be a mapping
        from fieldnames to cast functions. If an iterable, it should be an iterable of
        cast functions, in which case each cast function will be applied to each element
        of the row, element wise.

    >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> encoded = csv_dict_encode(data, fieldnames=['a', 'b'])
    >>> encoded
    'a,b\r\n1,2\r\n3,4\r\n'
    >>> csv_dict_decode(encoded)
    [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]


    See that you don't get back when you started with. The ints aren't ints anymore!
    You can resolve this by using the fieldcasts argument
    (that's our argument -- not present in builtin csv module).
    I should be a function (that transforms a dict to the one you want) or
    list or tuple of the same size as the row (that specifies the cast function for
    each field)


    >>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts=[int] * 2)
    [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> csv_dict_decode(encoded, fieldnames=['a', 'b'], fieldcasts={'b': float})
    [{'a': '1', 'b': 2.0}, {'a': '3', 'b': 4.0}]

    """
    fieldcasts = kwargs.pop("fieldcasts", lambda row: row)
    if isinstance(fieldcasts, Iterable):
        if isinstance(fieldcasts, dict):
            cast_dict = dict(fieldcasts)
            cast = lambda k: cast_dict.get(k, lambda x: x)
            fieldcasts = lambda row: {k: cast(k)(v) for k, v in row.items()}
        else:
            _casts = list(fieldcasts)
            # apply each cast function to each element of the row, element wise
            fieldcasts = lambda row: {
                k: cast(v) for cast, (k, v) in zip(_casts, row.items())
            }
    with io.StringIO(string) as buffer:
        reader = csv.DictReader(buffer, *args, **kwargs)
        rows = [row for row in reader]

        def remove_first_row_if_only_header(rows):
            first_row = next(iter(rows), None)
            if first_row is not None and all(k == v for k, v in first_row.items()):
                rows.pop(0)

        remove_first_row_if_only_header(rows)
        return list(map(fieldcasts, rows))


def _xml_tree_encode(element, parser=None):
    # Needed to replace original "text" argument with "element" to be consistent with
    # ET.tostring
    import xml.etree.ElementTree as ET

    return ET.fromstring(text=element, parser=parser)


def _xml_tree_decode(
    element,
    encoding=None,
    method=None,
    *,
    xml_declaration=None,
    default_namespace=None,
    short_empty_elements=True,
):
    import xml.etree.ElementTree as ET

    return ET.tostring(
        element,
        encoding,
        method,
        xml_declaration=xml_declaration,
        default_namespace=default_namespace,
        short_empty_elements=short_empty_elements,
    )


def extract_arguments(func, args, kwargs):
    return Sig(func).map_arguments(
        args, kwargs, allow_partial=True, allow_excess=True, ignore_kind=True
    )


def _var_kinds_less_signature(func):
    sig = Sig(func)
    var_kinds = (
        sig.names_of_kind[Sig.VAR_POSITIONAL] + sig.names_of_kind[Sig.VAR_KEYWORD]
    )
    return sig - var_kinds


def _merge_signatures(encoder, decoder, *, exclude=()):
    return (_var_kinds_less_signature(encoder) - exclude) + (
        _var_kinds_less_signature(decoder) - exclude
    )


def _codec_wrap(cls, encoder: Callable, decoder: Callable, **kwargs):
    return cls(
        encoder=partial(encoder, **extract_arguments(encoder, (), kwargs)),
        decoder=partial(decoder, **extract_arguments(decoder, (), kwargs)),
    )


def codec_wrap(cls, encoder: Callable, decoder: Callable, *, exclude=()):
    _cls_codec_wrap = partial(_codec_wrap, cls)
    factory = partial(_cls_codec_wrap, encoder, decoder)
    # TODO: Review this signature here. Should be keyword-only to match what
    #  _codec_wrap implementation imposses, or _codec_wrap should be made to accpt
    #  positional arguments (when encoder/decoder function are not class methods)
    # See: https://github.com/i2mint/dol/discussions/41#discussioncomment-8015800
    sig = _merge_signatures(encoder, decoder, exclude=exclude)
    # Change all arguments to keyword-only
    # sig = sig.ch_kinds(**{k: Sig.KEYWORD_ONLY for k in sig.names})
    return sig(factory)


# wrappers to manage encoder and decoder arguments and signature
value_wrap = named_partial(codec_wrap, ValueCodec, __name__="value_wrap")
key_wrap = named_partial(codec_wrap, KeyCodec, __name__="key_wrap")
key_value_wrap = named_partial(codec_wrap, KeyValueCodec, __name__="key_value_wrap")


class CodecCollection:
    """The base class for collections of codecs.
    Makes sure that the class cannot be instantiated, but only used as a collection.
    Also provides an _iter_codecs method that iterates over the codec names.
    """

    def __init__(self, *args, **kwargs):
        name = getattr(type(self), "__name__", "")
        raise ValueError(
            f"The {name} class is not meant to be instantiated, "
            "but only act as a collection of codec factories"
        )

    @classmethod
    def _iter_codecs(cls):
        def is_value_codec(attr_val):
            func = getattr(attr_val, "func", None)
            name = getattr(func, "__name__", "")
            return name == "_codec_wrap"

        for attr in dir(cls):
            if not attr.startswith("_"):
                attr_val = getattr(cls, attr, None)
                if is_value_codec(attr_val):
                    yield attr


def _add_default_codecs(cls):
    for codec_name in cls._iter_codecs():
        codec_factory = getattr(cls, codec_name)
        dflt_codec = codec_factory()
        setattr(cls.default, codec_name, dflt_codec)
    return cls


@_add_default_codecs
class ValueCodecs(CodecCollection):
    r"""
    A collection of value codec factories using standard lib tools.

    >>> json_codec = ValueCodecs.json()  # call the json codec factory
    >>> encoder, decoder = json_codec
    >>> encoder({'b': 2})
    '{"b": 2}'
    >>> decoder('{"b": 2}')
    {'b': 2}

    The `json_codec` object is also a `Mapping` value wrapper:

    >>> backend = dict()
    >>> interface = json_codec(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> assert backend == {'a': '{"b": 2}'}  # json was written in backend
    >>> interface['a']  # but this json is decoded to a dict when read from interface
    {'b': 2}

    In order not to have to call the codec factory when you just want the default,
    we've made a `default` attribute that contains all the default codecs:

    >>> backend = dict()
    >>> interface = ValueCodecs.default.json(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> assert backend == {'a': '{"b": 2}'}  # json was written in backend

    For times when you want to parametrize your code though, know that you can also
    pass arguments to the encoder and decoder when you make your codec.
    For example, to make a json codec that indents the json, you can do:

    >>> json_codec = ValueCodecs.json(indent=2)
    >>> backend = dict()
    >>> interface = json_codec(backend)
    >>> interface['a'] = {'b': 2}  # we write a dict
    >>> print(backend['a'])  # written in backend with indent
    {
      "b": 2
    }


    """

    # TODO: Clean up module import polution?
    # TODO: Import all these in module instead of class
    # TODO: Figure out a way to import these dynamically, only if a particular codec is used
    # TODO: Figure out how to give codecs annotations that can actually be inspected!

    class default:
        """To contain default codecs. Is populated by @_add_default_codecs"""

    import pickle, json, gzip, bz2, base64 as b64, lzma, codecs, io
    from operator import methodcaller
    from dol.zipfiledol import (
        zip_compress,
        zip_decompress,
        tar_compress,
        tar_decompress,
    )

    str_to_bytes: ValueCodec[bytes, bytes] = value_wrap(str.encode, bytes.decode)
    stringio: ValueCodec[str, io.StringIO] = value_wrap(
        io.StringIO, methodcaller("read")
    )
    bytesio: ValueCodec[bytes, io.BytesIO] = value_wrap(
        io.BytesIO, methodcaller("read")
    )

    pickle: ValueCodec[Any, bytes] = value_wrap(pickle.dumps, pickle.loads)
    json: ValueCodec[dict, str] = value_wrap(json.dumps, json.loads)
    csv: ValueCodec[list, str] = value_wrap(csv_encode, csv_decode)
    csv_dict: ValueCodec[list, str] = value_wrap(csv_dict_encode, csv_dict_decode)

    base64: ValueCodec[bytes, bytes] = value_wrap(b64.b64encode, b64.b64decode)
    urlsafe_b64: ValueCodec[bytes, bytes] = value_wrap(
        b64.urlsafe_b64encode, b64.urlsafe_b64decode
    )
    codecs: ValueCodec[str, bytes] = value_wrap(codecs.encode, codecs.decode)

    # Note: Note clear if escaping or unescaping is the encoder or decoder here
    # I have never had the need for stores using it, so will omit for now
    # html: ValueCodec[str, str] = value_wrap(html.unescape, html.escape)

    # Compression
    zipfile: ValueCodec[bytes, bytes] = value_wrap(zip_compress, zip_decompress)
    gzip: ValueCodec[bytes, bytes] = value_wrap(gzip.compress, gzip.decompress)
    bz2: ValueCodec[bytes, bytes] = value_wrap(bz2.compress, bz2.decompress)
    tarfile: ValueCodec[bytes, bytes] = value_wrap(tar_compress, tar_decompress)
    lzma: ValueCodec[bytes, bytes] = value_wrap(
        lzma.compress, lzma.decompress, exclude=("format",)
    )

    import quopri, plistlib

    quopri: ValueCodec[bytes, bytes] = value_wrap(
        quopri.encodestring, quopri.decodestring
    )
    plistlib: ValueCodec[bytes, bytes] = value_wrap(
        plistlib.dumps, plistlib.loads, exclude=("fmt",)
    )

    # Any is really xml.etree.ElementTree.Element, but didn't want to import
    xml_etree: Codec[Any, bytes] = value_wrap(_xml_tree_encode, _xml_tree_decode)

    # TODO: Review value_wrap so it works with non-class functions like below
    #   See: https://github.com/i2mint/dol/discussions/41#discussioncomment-8015800

    single_nested_value: Codec[KT, dict[KT, VT]]

    def single_nested_value(key):
        """

        >>> d = {
        ...     1: {'en': 'one', 'fr': 'un', 'sp': 'uno'},
        ...     2: {'en': 'two', 'fr': 'deux', 'sp': 'dos'},
        ... }
        >>> en = ValueCodecs.single_nested_value('en')(d)
        >>> en[1]
        'one'
        >>> en[1] = 'ONE'
        >>> d[1]  # note that here d[1] is completely replaced (not updated)
        {'en': 'ONE'}
        """
        return ValueCodec(partial(single_nest_in_dict, key), itemgetter(key))

    tuple_of_dict: Codec[Iterable[VT], dict[KT, VT]]

    def tuple_of_dict(keys):
        """Get a tuple-view of dict values.

        >>> d = {
        ...     1: {'en': 'one', 'fr': 'un', 'sp': 'uno'},
        ...     2: {'en': 'two', 'fr': 'deux', 'sp': 'dos'},
        ... }
        >>> codec = ValueCodecs.tuple_of_dict(['fr', 'sp'])
        >>> codec.encoder(['deux', 'tre'])
        {'fr': 'deux', 'sp': 'tre'}
        >>> codec.decoder({'en': 'one', 'fr': 'un', 'sp': 'uno'})
        ('un', 'uno')
        >>> frsp = codec(d)
        >>> frsp[2]
        ('deux', 'dos')
        >>> ('deux', 'dos')
        ('deux', 'dos')
        >>> frsp[2] = ('DEUX', 'DOS')
        >>> frsp[2]
        ('DEUX', 'DOS')

        Note that writes completely replace the values in the backend dict,
        it doesn't update them:

        >>> d[2]
        {'fr': 'DEUX', 'sp': 'DOS'}

        See also `dol.KeyTemplate` for more general key-based views.
        """
        return ValueCodec(partial(nest_in_dict, keys), itemgetter(*keys))


from dol.util import invertible_maps


@_add_default_codecs
class KeyCodecs(CodecCollection):
    """
    A collection of key codecs
    """

    def affixed(prefix: str = "", suffix: str = ""):
        return affix_key_codec(prefix=prefix, suffix=suffix)

    def suffixed(suffix: str):
        return affix_key_codec(suffix=suffix)

    def prefixed(prefix: str):
        return affix_key_codec(prefix=prefix)

    def common_prefixed(keys: Iterable[str]):
        from dol.util import max_common_prefix

        prefix = max_common_prefix(keys)
        return KeyCodecs.prefixed(prefix)

    def mapped_keys(
        encoder: Mapping | Callable | None = None,
        decoder: Mapping | Callable | None = None,
    ):
        """
        A factory that creates a key codec that uses "explicit" mappings to encode
        and decode keys.

        The encoders and decoders can be an explicit mapping of a function.
        If the encoder is a mapping, the decoder is the inverse of that mapping.
        If given explicitly, this will be asserted.
        If not, the decoder will be computed by swapping the keys and values of the
        encoder and asserting that no values were lost in the process
        (that is, that the mappings are invertible).
        The statements above are true if you swap "encoder" and "decoder".

        >>> km = KeyCodecs.mapped_keys({'a': 1, 'b': 2})
        >>> km.encoder('a')
        1
        >>> km.decoder(1)
        'a'

        If the encoder is a function, the decoder must be an iterable of keys who will
        be used as arguments of the function to get the encoded key, and the decode
        will be the inverse of that mapping.
        The statement above is true if you swap "encoder" and "decoder".

        >>> km = KeyCodecs.mapped_keys(['a', 'b'], str.upper)
        >>> km.encoder('A')
        'a'
        >>> km.decoder('a')
        'A'
        """
        encoder, decoder = invertible_maps(encoder, decoder)
        return KeyCodec(
            encoder=encoder.__getitem__,
            decoder=decoder.__getitem__,
        )


def common_prefix_keys_wrap(s: Mapping):
    """Transforms keys of mapping to omit the longest prefix they have in common"""
    common_prefix_wrap = KeyCodecs.common_prefixed(s)
    return common_prefix_wrap(s)


# TODO: Here, I'd like to decorate with store_decorator, but KeyCodecs.mapped_keys
#  doesn't apply to a Mapping class, only an instance. We have to make the iteration
#  of keys to be inverted have lazy capabilities (only happen when the instance is made)
def add_invertible_key_decoder(store: Mapping, *, decoder: Callable):
    """Add a key decoder to a store (instance)"""
    return KeyCodecs.mapped_keys(store, decoder=decoder)(store)


# --------------------------------- KV Codecs ------------------------------------------


dflt_ext_mapping = {
    ".json": ValueCodecs.json,
    ".csv": ValueCodecs.csv,
    ".csv_dict": ValueCodecs.csv_dict,
    ".pickle": ValueCodecs.pickle,
    ".gz": ValueCodecs.gzip,
    ".bz2": ValueCodecs.bz2,
    ".lzma": ValueCodecs.lzma,
    ".zip": ValueCodecs.zipfile,
    ".tar": ValueCodecs.tarfile,
    ".xml": ValueCodecs.xml_etree,
}


def key_based_codec_factory(key_mapping: dict, key_func: Callable = identity_func):
    """A factory that creates a key codec that uses the key to determine the
    codec to use."""

    def encoder(key):
        return key_mapping[key_func(key)]

    def decoder(key):
        return key_mapping[key_func(key)]

    return ValueCodec(encoder, decoder)


class NotGiven:
    """A singleton to indicate that a value was not given"""

    def __repr__(self):
        return "NotGiven"


from typing import NewType


def key_based_value_trans(
    key_func: Callable[[KT], KT],
    value_trans_mapping,
    default_factory: Callable[[], Callable],
    k=NotGiven,
):
    """A factory that creates a value codec that uses the key to determine the
    codec to use.

    # a key_func that gets the extension of a file path

    >>> import json
    >>> from functools import partial
    >>> key_func = lambda k: os.path.splitext(k)[1]
    >>> value_trans_mapping = {'.json': json.loads, '.txt': bytes.decode}
    >>> default_factory = partial(ValueError, "No codec for this extension")
    >>> trans = key_based_value_trans(
    ...     key_func, value_trans_mapping, default_factory=lambda: identity_func
    ... )


    """
    if k is NotGiven:
        return partial(
            key_based_value_trans, key_func, value_trans_mapping, default_factory
        )
    value_trans_key = key_func(k)
    value_trans = value_trans_mapping.get(value_trans_key, default_factory, None)
    if value_trans is None:
        value_trans = default_factory()
    return value_trans


@_add_default_codecs
class KeyValueCodecs(CodecCollection):
    """
    A collection of key-value codecs that can be used with postget and preset kv_wraps.
    """

    def key_based(
        key_mapping: dict,
        key_func: Callable = identity_func,
        *,
        default: Callable | None = None,
    ):
        """A factory that creates a key-value codec that uses the key to determine the
        value codec to use."""

    def extension_based(
        ext_mapping: dict = dflt_ext_mapping,
        *,
        default: Callable | None = None,
    ):
        """A factory that creates a key-value codec that uses the file extension to
        determine the value codec to use."""
```

## misc.py

```python
"""
Functions to read from and write to misc sources
"""

# TODO: Completely redo this, using preset and postget and making it into a plugin
#  architecture. See
from functools import partial
import os
import json
import pickle
import csv
import gzip
from contextlib import suppress
from io import StringIO

from dol.filesys import Files
from dol.zipfiledol import FilesOfZip
from dol.util import imdict, ModuleNotFoundIgnore


def csv_fileobj(csv_data, *args, **kwargs):  # TODO: Use extended wraps func to inject
    fp = StringIO("")
    writer = csv.writer(fp)
    writer.writerows(csv_data, *args, **kwargs)
    fp.seek(0)
    return fp.read().encode()


def identity_method(x):
    return x


# TODO: Enhance default handling so users can have their own defaults (checking for local config file etc.)
# Note: If you're tempted to add third-party cases here (like yaml, pandas):
#   DO NOT!! Defaults must work only with builtins (or misc would be non-deterministic)
dflt_func_key = lambda self, k: os.path.splitext(k)[1]
dflt_dflt_incoming_val_trans = staticmethod(identity_method)

# TODO: Get rid of lambdas. Use methodcaller and partial instead
#  Requires solving https://github.com/i2mint/dol/issues/9 first
dflt_incoming_val_trans_for_key = {
    ".bin": identity_method,
    ".csv": lambda v: list(csv.reader(StringIO(v.decode()))),
    ".txt": lambda v: v.decode(),
    ".pkl": lambda v: pickle.loads(v),
    ".pickle": lambda v: pickle.loads(v),
    ".json": lambda v: json.loads(v),
    ".zip": FilesOfZip,
    ".gzip": gzip.decompress,
}

dflt_outgoing_val_trans_for_key = {
    ".bin": identity_method,
    ".csv": csv_fileobj,
    ".txt": lambda v: v.encode(),
    ".pkl": lambda v: pickle.dumps(v),
    ".pickle": lambda v: pickle.dumps(v),
    ".json": lambda v: json.dumps(v).encode(),
    ".gzip": gzip.compress,
    ".ini": lambda v: ConfigStore(
        v, interpolation=ConfigReader.ExtendedInterpolation()
    ),
}

# TODO: Change whole module to be proper plugin architecture
with suppress(ModuleNotFoundError, ImportError):
    from config2py import ConfigReader, ConfigStore

    dflt_incoming_val_trans_for_key[".ini"] = partial(
        ConfigStore,
        interpolation=ConfigReader.ExtendedInterpolation(),
    )

    dflt_outgoing_val_trans_for_key[".ini"] = partial(
        ConfigStore, interpolation=ConfigReader.ExtendedInterpolation()
    )


synset_of_ext = {".ini": {".cnf", ".conf", ".config"}, ".gzip": [".gz"]}
for _user_this, _for_these_extensions in synset_of_ext.items():
    for _d in [
        dflt_incoming_val_trans_for_key,
        dflt_outgoing_val_trans_for_key,
    ]:
        if _user_this in _d:
            for _ext in _for_these_extensions:
                _d[_ext] = _d[_user_this]


# TODO: Different misc objects (function, class, default instance) should be a aligned more


class MiscReaderMixin:
    """Mixin to transform incoming vals according to the key their under.
    Warning: If used as a subclass, this mixin should (in general) be placed before the store


    >>> # make a reader that will wrap a dict
    >>> class MiscReader(MiscReaderMixin, dict):
    ...     def __init__(self, d,
    ...                         incoming_val_trans_for_key=None,
    ...                         dflt_incoming_val_trans=None,
    ...                         func_key=None):
    ...         dict.__init__(self, d)
    ...         MiscReaderMixin.__init__(self, incoming_val_trans_for_key, dflt_incoming_val_trans, func_key)
    ...
    >>>
    >>> incoming_val_trans_for_key = dict(
    ...     MiscReaderMixin._incoming_val_trans_for_key,  # take the existing defaults...
    ...     **{'.bin': lambda v: [ord(x) for x in v.decode()], # ... override how to handle the .bin extension
    ...      '.reverse_this': lambda v: v[::-1]  # add a new extension (and how to handle it)
    ...     })
    >>>
    >>> import pickle
    >>> d = {
    ...     'a.bin': b'abc123',
    ...     'a.reverse_this': b'abc123',
    ...     'a.csv': b'event,year\\n Magna Carta,1215\\n Guido,1956',
    ...     'a.txt': b'this is not a text',
    ...     'a.pkl': pickle.dumps(['text', [str, map], {'a list': [1, 2, 3]}]),
    ...     'a.json': '{"str": "field", "int": 42, "float": 3.14, "array": [1, 2], "nested": {"a": 1, "b": 2}}',
    ... }
    >>>
    >>> s = MiscReader(d=d, incoming_val_trans_for_key=incoming_val_trans_for_key)
    >>> list(s)
    ['a.bin', 'a.reverse_this', 'a.csv', 'a.txt', 'a.pkl', 'a.json']
    >>> s['a.bin']
    [97, 98, 99, 49, 50, 51]
    >>> s['a.reverse_this']
    b'321cba'
    >>> s['a.csv']
    [['event', 'year'], [' Magna Carta', '1215'], [' Guido', '1956']]
    >>> s['a.pkl']
    ['text', [<class 'str'>, <class 'map'>], {'a list': [1, 2, 3]}]
    >>> s['a.json']
    {'str': 'field', 'int': 42, 'float': 3.14, 'array': [1, 2], 'nested': {'a': 1, 'b': 2}}
    """

    _func_key = lambda self, k: os.path.splitext(k)[1]
    _dflt_incoming_val_trans = staticmethod(identity_method)

    _incoming_val_trans_for_key = imdict(dflt_incoming_val_trans_for_key)

    def __init__(
        self,
        incoming_val_trans_for_key=None,
        dflt_incoming_val_trans=None,
        func_key=None,
    ):
        if incoming_val_trans_for_key is not None:
            self._incoming_val_trans_for_key = incoming_val_trans_for_key
        if dflt_incoming_val_trans is not None:
            self._dflt_incoming_val_trans = dflt_incoming_val_trans
        if func_key is not None:
            self._func_key = func_key

    def __getitem__(self, k):
        func_key = self._func_key(k)
        trans_func = self._incoming_val_trans_for_key.get(
            func_key, self._dflt_incoming_val_trans
        )
        return trans_func(super().__getitem__(k))


# import urllib
import urllib.request

DFLT_USER_AGENT = "Wget/1.16 (linux-gnu)"


def _is_dropbox_url(url):
    return url.startswith("http://www.dropbox.com") or url.startswith(
        "https://www.dropbox.com"
    )


def _bytes_from_dropbox(url, chk_size=1024, user_agent=DFLT_USER_AGENT):
    from io import BytesIO

    def _download_from_dropbox(url, file, chk_size=1024, user_agent=DFLT_USER_AGENT):
        def iter_content_and_copy_to(file):
            req = urllib.request.Request(url)
            req.add_header("user-agent", user_agent)
            with urllib.request.urlopen(req) as response:
                while True:
                    chk = response.read(chk_size)
                    if len(chk) > 0:
                        file.write(chk)
                    else:
                        break

        if not isinstance(file, str):
            iter_content_and_copy_to(file)
        else:
            with open(file, "wb") as _target_file:
                iter_content_and_copy_to(_target_file)

    with BytesIO() as file:
        _download_from_dropbox(url, file, chk_size=chk_size, user_agent=user_agent)
        file.seek(0)
        return file.read()


def url_to_bytes(url):
    if _is_dropbox_url(url):
        return _bytes_from_dropbox(url)
    else:
        with urllib.request.urlopen(url) as response:
            return response.read()


# TODO: I'd really like to reuse MiscReaderMixin here! There's a lot of potential.
# TODO: For more flexibility, the default store should probably be a UriReader (that doesn't exist yet)
#  If store argument of get_obj was a type instead of an instance, or if MiscReaderMixin was a transformer, if would
#  be easier -- but would it make their individual concerns mixed?
#   Also, preset and postget (trans.wrap_kvs(...)) now exist. Let's use them here.
def get_obj(
    k,
    store=Files(""),
    incoming_val_trans_for_key=imdict(dflt_incoming_val_trans_for_key),
    dflt_incoming_val_trans=identity_method,
    func_key=lambda k: os.path.splitext(k)[1],
):
    """A quick way to get an object, with default... everything (but the key, you know, a clue of what you want)"""
    if k.startswith("http://") or k.startswith("https://"):
        v = url_to_bytes(k)
    else:
        if isinstance(
            store, Files
        ):  # being extra careful to only do this if default local store
            # preprocessing the key if it starts with '.', '..', or '~'
            if k.startswith(".") or k.startswith(".."):
                k = os.path.abspath(k)
            elif k.startswith("~"):
                k = os.path.expanduser(k)
        v = store[k]
    trans_func = (incoming_val_trans_for_key or {}).get(
        func_key(k), dflt_incoming_val_trans
    )
    return trans_func(v)


# TODO: I'd really like to reuse MiscReaderMixin here! There's a lot of potential.
#  Same comment as for get_obj.
class MiscGetter:
    """
    An object to write (and only write) to a store (default local files) with automatic deserialization
    according to a property of the key (default: file extension).

    >>> from dol.misc import get_obj, misc_objs_get
    >>> import os
    >>> import json
    >>>
    >>> pjoin = lambda *p: os.path.join(os.path.expanduser('~'), *p)
    >>> path = pjoin('tmp.json')
    >>> d = {'a': {'b': {'c': [1, 2, 3]}}}
    >>> json.dump(d, open(path, 'w'))  # putting a json file there, the normal way, so we can use it later
    >>>
    >>> k = path
    >>> t = get_obj(k)  # if you'd like to use a function
    >>> assert t == d
    >>> tt = misc_objs_get[k]  # if you'd like to use an object (note: can get, but nothing else (no list, set, del, etc))
    >>> assert tt == d
    >>> t
    {'a': {'b': {'c': [1, 2, 3]}}}
    """

    def __init__(
        self,
        store=Files(""),
        incoming_val_trans_for_key=imdict(dflt_incoming_val_trans_for_key),
        dflt_incoming_val_trans=identity_method,
        func_key=lambda k: os.path.splitext(k)[1],
    ):
        self.store = store
        self.incoming_val_trans_for_key = incoming_val_trans_for_key
        self.dflt_incoming_val_trans = dflt_incoming_val_trans
        self.func_key = func_key

    def __getitem__(self, k):
        return get_obj(
            k,
            self.store,
            self.incoming_val_trans_for_key,
            self.dflt_incoming_val_trans,
            self.func_key,
        )

    def __iter__(self):
        # Disabling "manually" to avoid iteration falling back on __getitem__ with integers
        # To know more, see:
        #   https://stackoverflow.com/questions/37941523/pip-uninstall-no-files-were-found-to-uninstall
        #   https://www.python.org/dev/peps/pep-0234/

        raise NotImplementedError(
            "By default, there's no iteration in MiscGetter. "
            "But feel free to subclass if you "
            "have a particular sense of what the iteration should yield!"
        )


misc_objs_get = MiscGetter()

# TODO: Make this be more tightly couples with the actual default used in get_obj and MiscGetter (avoid misalignments)
misc_objs_get.dflt_incoming_val_trans_for_key = dflt_incoming_val_trans_for_key


class MiscStoreMixin(MiscReaderMixin):
    r"""Mixin to transform incoming and outgoing vals according to the key their under.
    Warning: If used as a subclass, this mixin should (in general) be placed before the store

    See also: preset and postget args from wrap_kvs decorator from dol.trans.

    >>> # Make a class to wrap a dict with a layer that transforms written and read values
    >>> class MiscStore(MiscStoreMixin, dict):
    ...     def __init__(self, d,
    ...                         incoming_val_trans_for_key=None, outgoing_val_trans_for_key=None,
    ...                         dflt_incoming_val_trans=None, dflt_outgoing_val_trans=None,
    ...                         func_key=None):
    ...         dict.__init__(self, d)
    ...         MiscStoreMixin.__init__(self, incoming_val_trans_for_key, outgoing_val_trans_for_key,
    ...                                 dflt_incoming_val_trans, dflt_outgoing_val_trans, func_key)
    ...
    >>>
    >>> outgoing_val_trans_for_key = dict(
    ...     MiscStoreMixin._outgoing_val_trans_for_key,  # take the existing defaults...
    ...     **{'.bin': lambda v: ''.join([chr(x) for x in v]).encode(), # ... override how to handle the .bin extension
    ...        '.reverse_this': lambda v: v[::-1]  # add a new extension (and how to handle it)
    ...     })
    >>> ss = MiscStore(d={},  # store starts empty
    ...                incoming_val_trans_for_key={},  # overriding incoming trans so we can see the raw data later
    ...                outgoing_val_trans_for_key=outgoing_val_trans_for_key)
    ...
    >>> # here's what we're going to write in the store
    >>> data_to_write = {
    ...      'a.bin': [97, 98, 99, 49, 50, 51],
    ...      'a.reverse_this': b'321cba',
    ...      'a.csv': [['event', 'year'], [' Magna Carta', '1215'], [' Guido', '1956']],
    ...      'a.txt': 'this is not a text',
    ...      'a.pkl': ['text', [str, map], {'a list': [1, 2, 3]}],
    ...      'a.json': {'str': 'field', 'int': 42, 'float': 3.14, 'array': [1, 2], 'nested': {'a': 1, 'b': 2}}}
    >>> # write this data in our store
    >>> for k, v in data_to_write.items():
    ...     ss[k] = v
    >>> list(ss)
    ['a.bin', 'a.reverse_this', 'a.csv', 'a.txt', 'a.pkl', 'a.json']
    >>> # Looking at the contents (what was actually stored/written)
    >>> for k, v in ss.items():
    ...     if k != 'a.pkl':
    ...         print(f"{k}: {v}")
    ...     else:  # need to verify pickle data differently, since printing contents is problematic in doctest
    ...         assert pickle.loads(v) == data_to_write['a.pkl']
    a.bin: b'abc123'
    a.reverse_this: b'abc123'
    a.csv: b'event,year\r\n Magna Carta,1215\r\n Guido,1956\r\n'
    a.txt: b'this is not a text'
    a.json: b'{"str": "field", "int": 42, "float": 3.14, "array": [1, 2], "nested": {"a": 1, "b": 2}}'

    """

    _dflt_outgoing_val_trans_for_key = staticmethod(identity_method)
    _outgoing_val_trans_for_key = dflt_outgoing_val_trans_for_key

    def __init__(
        self,
        incoming_val_trans_for_key=None,
        outgoing_val_trans_for_key=None,
        dflt_incoming_val_trans=None,
        dflt_outgoing_val_trans=None,
        func_key=None,
    ):
        super().__init__(incoming_val_trans_for_key, dflt_incoming_val_trans, func_key)
        if outgoing_val_trans_for_key is not None:
            self._outgoing_val_trans_for_key = outgoing_val_trans_for_key
        if dflt_outgoing_val_trans is not None:
            self._dflt_outgoing_val_trans = dflt_outgoing_val_trans

    def __setitem__(self, k, v):
        func_key = self._func_key(k)
        trans_func = self._outgoing_val_trans_for_key.get(
            func_key, self._dflt_outgoing_val_trans_for_key
        )
        return super().__setitem__(k, trans_func(v))


# TODO: I'd really like to reuse MiscStoreMixin here! There's a lot of potential.
#  If store argument of get_obj was a type instead of an instance, or if MiscReaderMixin was a transformer, if would
#  be easier -- but would it make their individual concerns mixed?
def set_obj(
    k,
    v,
    store=Files(""),
    outgoing_val_trans_for_key=imdict(dflt_outgoing_val_trans_for_key),
    func_key=lambda k: os.path.splitext(k)[1],
):
    """A quick way to get an object, with default...
    # everything (but the key, you know, a clue of what you want)"""
    if isinstance(store, Files) and store._prefix in {"", "/"}:
        k = os.path.abspath(os.path.expanduser(k))

    trans_func = outgoing_val_trans_for_key.get(
        func_key(k), dflt_outgoing_val_trans_for_key
    )
    store[k] = trans_func(v)


# TODO: I'd really like to reuse MiscReaderMixin here! There's a lot of potential.
#  Same comment as above.
class MiscGetterAndSetter(MiscGetter):
    """
    An object to read and write (and nothing else) to a store (default local) with automatic (de)serialization
    according to a property of the key (default: file extension).

    >>> from dol.misc import set_obj, misc_objs  # the function and the object
    >>> import json
    >>> import os
    >>>
    >>> pjoin = lambda *p: os.path.join(os.path.expanduser('~'), *p)
    >>>
    >>> d = {'a': {'b': {'c': [1, 2, 3]}}}
    >>> misc_objs[pjoin('tmp.json')] = d
    >>> filepath = os.path.expanduser('~/tmp.json')
    >>> assert misc_objs[filepath] == d  # yep, it's there, and can be retrieved
    >>> assert json.load(open(filepath)) == d  # in case you don't believe it's an actual json file
    >>>
    >>> # using pickle
    >>> misc_objs[pjoin('tmp.pkl')] = d
    >>> assert misc_objs[pjoin('tmp.pkl')] == d
    >>>
    >>> # using txt
    >>> misc_objs[pjoin('tmp.txt')] = 'hello world!'
    >>> assert misc_objs[pjoin('tmp.txt')] == 'hello world!'
    >>>
    >>> # using csv
    >>> misc_objs[pjoin('tmp.csv')] = [[1,2,3], ['a','b','c']]
    >>> assert misc_objs[pjoin('tmp.csv')] == [['1','2','3'], ['a','b','c']]  # yeah, well, not numbers, but you deal with it
    >>>
    >>> # using bin
    ... misc_objs[pjoin('tmp.bin')] = b'let us pretend these are bytes of an audio waveform'
    >>> assert misc_objs[pjoin('tmp.bin')] == b'let us pretend these are bytes of an audio waveform'

    """

    def __init__(
        self,
        store=Files(""),
        incoming_val_trans_for_key=imdict(dflt_incoming_val_trans_for_key),
        outgoing_val_trans_for_key=imdict(dflt_outgoing_val_trans_for_key),
        dflt_incoming_val_trans=identity_method,
        func_key=lambda k: os.path.splitext(k)[1],
    ):
        self.store = store
        self.incoming_val_trans_for_key = incoming_val_trans_for_key
        self.outgoing_val_trans_for_key = outgoing_val_trans_for_key
        self.dflt_incoming_val_trans = dflt_incoming_val_trans
        self.func_key = func_key

    def __setitem__(self, k, v):
        return set_obj(k, v, self.store, self.outgoing_val_trans_for_key, self.func_key)


misc_objs = MiscGetterAndSetter()
```

## mixins.py

```python
"""Mixins"""

import json
from dol.errors import (
    WritesNotAllowed,
    DeletionsNotAllowed,
    OverWritesNotAllowedError,
)


class SimpleJsonMixin:
    """simple json serialization.
    Useful to store and retrieve
    """

    _docsuffix = "Data is assumed to be a JSON string, and is loaded with json.loads and dumped with json.dumps"

    def _obj_of_data(self, data):
        return json.loads(data)

    def _data_of_obj(self, obj):
        return json.dumps(obj)


class IdentityKeysWrapMixin:
    """Transparent KeysWrapABC. Often placed in the mro to satisfy the KeysWrapABC need in a neutral way.
    This is useful in cases where the keys the persistence functions work with are the same as those you want to work
    with.
    """

    def _id_of_key(self, k):
        """
        Maps an interface identifier (key) to an internal identifier (_id) that is actually used to perform operations.
        Can also perform validation and permission checks.
        :param k: interface identifier of some data
        :return: internal identifier _id
        """
        return k

    def _key_of_id(self, _id):
        """
        The inverse of _id_of_key. Maps an internal identifier (_id) to an interface identifier (key)
        :param _id:
        :return:
        """
        return _id


class IdentityValsWrapMixin:
    """Transparent ValsWrapABC. Often placed in the mro to satisfy the KeysWrapABC need in a neutral way.
    This is useful in cases where the values can be persisted by __setitem__ as is (or the serialization is
    handled somewhere in the __setitem__ method.
    """

    def _data_of_obj(self, v):
        """
        Serialization of a python object.
        :param v: A python object.
        :return: The serialization of this object, in a format that can be stored by __getitem__
        """
        return v

    def _obj_of_data(self, data):
        """
        Deserialization. The inverse of _data_of_obj.
        :param data: Serialized data.
        :return: The python object corresponding to this data.
        """
        return data


class IdentityKvWrapMixin(IdentityKeysWrapMixin, IdentityValsWrapMixin):
    """Transparent Keys and Vals Wrap"""

    pass


from functools import partial

encode_as_utf8 = partial(str, encoding="utf-8")


class StringKvWrap(IdentityKvWrapMixin):
    def _obj_of_data(self, v):
        return encode_as_utf8(v)


class FilteredKeysMixin:
    """
    Filters __iter__ and __contains__ with (the boolean filter function attribute) _key_filt.
    """

    def __iter__(self):
        return filter(self._key_filt, super().__iter__())

    def __contains__(self, k) -> bool:
        """
        Check if collection of keys contains k.
        Note: This method iterates over all elements of the collection to check if k is present.
        Therefore it is not efficient, and in most cases should be overridden with a more efficient version.
        :return: True if k is in the collection, and False if not
        """
        return self._key_filt(k) and super().__contains__(k)


########################################################################################################################
# Mixins to disable specific operations


class ReadOnlyMixin:
    """Put this as your first parent class to disallow write/delete operations"""

    def __setitem__(self, k, v):
        raise WritesNotAllowed("You can't write with that Store")

    def __delitem__(self, k):
        raise DeletionsNotAllowed("You can't delete with that Store")

    def clear(self):
        raise DeletionsNotAllowed(
            "You can't delete (so definitely not delete all) with that Store"
        )

    def pop(self, k):
        raise DeletionsNotAllowed(
            "You can't delete (including popping) with that Store"
        )


from dol.util import copy_attrs


class OverWritesNotAllowedMixin:
    """Mixin for only allowing a write to a key if they key doesn't already exist.
    Note: Should be before the persister in the MRO.

    >>> class TestPersister(OverWritesNotAllowedMixin, dict):
    ...     pass
    >>> p = TestPersister()
    >>> p['foo'] = 'bar'
    >>> #p['foo'] = 'bar2'  # will raise error
    >>> p['foo'] = 'this value should not be stored' # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    dol.errors.OverWritesNotAllowedError: key foo already exists and cannot be overwritten.
        If you really want to write to that key, delete it before writing
    >>> p['foo']  # foo is still bar
    'bar'
    >>> del p['foo']
    >>> p['foo'] = 'this value WILL be stored'
    >>> p['foo']
    'this value WILL be stored'
    """

    @staticmethod
    def wrap(cls):
        # TODO: Consider moving to trans and making instances wrappable too
        class NoOverWritesClass(OverWritesNotAllowedMixin, cls): ...

        copy_attrs(NoOverWritesClass, cls, ("__name__", "__qualname__", "__module__"))
        return NoOverWritesClass

    def __setitem__(self, k, v):
        if self.__contains__(k):
            raise OverWritesNotAllowedError(
                "key {} already exists and cannot be overwritten. "
                "If you really want to write to that key, delete it before writing".format(
                    k
                )
            )
        return super().__setitem__(k, v)


########################################################################################################################
# Mixins to define mapping methods from others


class GetBasedContainerMixin:
    def __contains__(self, k) -> bool:
        """
        Check if collection of keys contains k.
        Note: This method actually fetches the contents for k, returning False if there's a key error trying to do so
        Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
        :return: True if k is in the collection, and False if not
        """
        try:
            self.__getitem__(k)
            return True
        except KeyError:
            return False


class IterBasedContainerMixin:
    def __contains__(self, k) -> bool:
        """
        Check if collection of keys contains k.
        Note: This method iterates over all elements of the collection to check if k is present.
        Therefore it is not efficient, and in most cases should be overridden with a more efficient version.
        :return: True if k is in the collection, and False if not
        """
        for collection_key in self.__iter__():
            if collection_key == k:
                return True
        return False  # return False if the key wasn't found


class IterBasedSizedMixin:
    def __len__(self) -> int:
        """
        Number of elements in collection of keys.
        Note: This method iterates over all elements of the collection and counts them.
        Therefore it is not efficient, and in most cases should be overridden with a more efficient version.
        :return: The number (int) of elements in the collection of keys.
        """
        # TODO: some other means to more quickly count files?
        # Note: Found that sum(1 for _ in self.__iter__()) was slower for small, slightly faster for big inputs.
        count = 0
        for _ in self.__iter__():
            count += 1
        return count


class IterBasedSizedContainerMixin(IterBasedSizedMixin, IterBasedContainerMixin):
    """
    An ABC that defines
        (a) how to iterate over a collection of elements (keys) (__iter__)
        (b) check that a key is contained in the collection (__contains__), and
        (c) how to get the number of elements in the collection
    This is exactly what the collections.abc.Collection (from which Keys inherits) does.
    The difference here, besides the "Keys" purpose-explicit name, is that Keys offers default
     __len__ and __contains__  definitions based on what ever __iter__ the concrete class defines.

    Keys is a collection (i.e. a Sized (has __len__), Iterable (has __iter__), Container (has __contains__).
    It's purpose is to serve as a collection of object identifiers in a key->obj mapping.
    The Keys class doesn't implement __iter__ (so needs to be subclassed with a concrete class), but
    offers mixin __len__ and __contains__ methods based on a given __iter__ method.
    Note that usually __len__ and __contains__ should be overridden to more, context-dependent, efficient methods.
    """

    pass


class HashableMixin:
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return hash(self) == hash(other)
```

## naming.py

```python
"""
This module is about generating, validating, and operating on (parametrized) fields (i.e. stings, e.g. paths).
"""

import re
import os
from functools import partial, wraps
from types import MethodType
from typing import Union

from dol.util import safe_compile
from dol.signatures import set_signature_of_func
from dol.errors import KeyValidationError, _assert_condition

assert_condition = partial(_assert_condition, err_cls=KeyValidationError)

path_sep = os.path.sep

base_validation_funs = {
    "be a": isinstance,
    "be in": lambda val, check_val: val in check_val,
    "be at least": lambda val, check_val: val >= check_val,
    "be more than": lambda val, check_val: val > check_val,
    "be no more than": lambda val, check_val: val <= check_val,
    "be less than": lambda val, check_val: val < check_val,
}

dflt_validation_funs = base_validation_funs
dflt_all_kwargs_should_be_in_validation_dict = False
dflt_ignore_misunderstood_validation_instructions = False

dflt_arg_pattern = r".+"

day_format = "%Y-%m-%d"
day_format_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")

capture_template = "({format})"
named_capture_template = "(?P<{name}>{format})"

fields_re = re.compile("(?<={)[^}]+(?=})")


def validate_kwargs(
    kwargs_to_validate,
    validation_dict,
    validation_funs=None,
    all_kwargs_should_be_in_validation_dict=False,
    ignore_misunderstood_validation_instructions=False,
):
    """
    Utility to validate a dict. It's main use is to validate function arguments (expressing the validation checks
    in validation_dict) by doing validate_kwargs(locals()), usually in the beginning of the function
    (to avoid having more accumulated variables than we need in locals())
    :param kwargs_to_validate: as the name implies...
    :param validation_dict: A dict specifying what to validate. Keys are usually name of variables (when feeding
        locals()) and values are dicts, themselves specifying check:check_val pairs where check is a string that
        points to a function (see validation_funs argument) and check_val is an object that the kwargs_to_validate
        value will be checked against.
    :param validation_funs: A dict of check:check_function(val, check_val) where check_function is a function returning
        True if val is valid (with respect to check_val).
    :param all_kwargs_should_be_in_validation_dict: If True, will raise an error if kwargs_to_validate contains
        keys that are not in validation_dict.
    :param ignore_misunderstood_validation_instructions: If True, will raise an error if validation_dict contains
        a key that is not in validation_funs (safer, since if you mistype a key in validation_dict, the function will
        tell you so!
    :return: True if all the validations passed.

    >>> validation_dict = {
    ...     'system': {
    ...         'be in': {'darwin', 'linux'}
    ...     },
    ...     'fv_version': {
    ...         'be a': int,
    ...         'be at least': 5
    ...     }
    ... }
    >>> validate_kwargs({'system': 'darwin'}, validation_dict)
    True
    >>> try:
    ...     validate_kwargs({'system': 'windows'}, validation_dict)
    ... except AssertionError as e:
    ...     assert str(e).startswith('system must be in')  # omitting the set because inconsistent order
    >>> try:
    ...     validate_kwargs({'fv_version': 9.9}, validation_dict)
    ... except AssertionError as e:
    ...     print(e)
    fv_version must be a <class 'int'>
    >>> try:
    ...     validate_kwargs({'fv_version': 4}, validation_dict)
    ... except AssertionError as e:
    ...     print(e)
    fv_version must be at least 5
    >>> validate_kwargs({'fv_version': 6}, validation_dict)
    True
    """
    validation_funs = dict(base_validation_funs or {}, **(validation_funs or {}))
    for (
        var,
        val,
    ) in kwargs_to_validate.items():  # for every (var, val) pair of kwargs
        if var in validation_dict:  # if var is in the validation_dict
            for check, check_val in validation_dict[
                var
            ].items():  # for every (key, val) of this dict
                if (
                    check in base_validation_funs
                ):  # if you have a validation check for it
                    if not validation_funs[check](val, check_val):  # check it's valid
                        raise AssertionError(
                            f"{var} must {check} {check_val}"
                        )  # and raise an error if not
                elif (
                    not ignore_misunderstood_validation_instructions
                ):  # should ignore if check not understood?
                    raise AssertionError(
                        "I don't know what to do with the validation check '{}'".format(
                            check
                        )
                    )
        elif (
            all_kwargs_should_be_in_validation_dict
        ):  # should all variables have checks?
            raise AssertionError("{} wasn't in the validation_dict")
    return True


def namedtuple_to_dict(nt):
    """

    >>> from collections import namedtuple
    >>> NT = namedtuple('MyTuple', ('foo', 'hello'))
    >>> nt = NT(1, 42)
    >>> nt
    MyTuple(foo=1, hello=42)
    >>> d = namedtuple_to_dict(nt)
    >>> d
    {'foo': 1, 'hello': 42}
    """
    return {field: getattr(nt, field) for field in nt._fields}


def dict_to_namedtuple(d, namedtuple_obj=None):
    """

    >>> from collections import namedtuple
    >>> NT = namedtuple('MyTuple', ('foo', 'hello'))
    >>> nt = NT(1, 42)
    >>> nt
    MyTuple(foo=1, hello=42)
    >>> d = namedtuple_to_dict(nt)
    >>> d
    {'foo': 1, 'hello': 42}
    >>> dict_to_namedtuple(d)
    NamedTupleFromDict(foo=1, hello=42)
    >>> dict_to_namedtuple(d, nt)
    MyTuple(foo=1, hello=42)
    """
    if namedtuple_obj is None:
        namedtuple_obj = "NamedTupleFromDict"
    if isinstance(namedtuple_obj, str):
        namedtuple_name = namedtuple_obj
        namedtuple_cls = namedtuple(namedtuple_name, tuple(d.keys()))
    elif isinstance(namedtuple_obj, tuple) and hasattr(namedtuple_obj, "_fields"):
        namedtuple_cls = namedtuple_obj.__class__
    elif isinstance(namedtuple_obj, type):
        namedtuple_cls = namedtuple_obj
    else:
        raise TypeError(
            f"Can't resolve the nametuple class specification: {namedtuple_obj}"
        )

    return namedtuple_cls(**d)


def update_fields_of_namedtuple(
    nt: tuple, *, name_of_output_type=None, remove_fields=(), **kwargs
):
    """Replace fields of namedtuple

    >>> from collections import namedtuple
    >>> NT = namedtuple('NT', ('a', 'b', 'c'))
    >>> nt = NT(1,2,3)
    >>> nt
    NT(a=1, b=2, c=3)
    >>> update_fields_of_namedtuple(nt, c=3000)  # replacing a single field
    NT(a=1, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, c=3000, a=1000)  # replacing two fields
    NT(a=1000, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, a=1000, c=3000)  # see that the original order doesn't change
    NT(a=1000, b=2, c=3000)
    >>> update_fields_of_namedtuple(nt, b=2000, d='hello')  # replacing one field and adding a new one
    UpdatedNT(a=1, b=2000, c=3, d='hello')
    >>> # Now let's try controlling the name of the output type, remove fields, and add new ones
    >>> update_fields_of_namedtuple(nt, name_of_output_type='NewGuy', remove_fields=('a', 'c'), hello='world')
    NewGuy(b=2, hello='world')
    """

    output_type_can_be_the_same_as_input_type = (not remove_fields) and set(
        kwargs.keys()
    ).issubset(nt._fields)
    d = dict(namedtuple_to_dict(nt), **kwargs)
    for f in remove_fields:
        d.pop(f)

    if output_type_can_be_the_same_as_input_type and name_of_output_type is None:
        return dict_to_namedtuple(d, nt.__class__)
    else:
        name_of_output_type = name_of_output_type or f"Updated{nt.__class__.__name__}"
        return dict_to_namedtuple(d, name_of_output_type)


empty_field_p = re.compile("{}")


def get_fields_from_template(template):
    """
    Get list from {item} items of template string
    :param template: a "template" string (a string with {item} items
    -- the kind that is used to mark token for str.format)
    :return: a list of the token items of the string, in the order they appear

    >>> get_fields_from_template('this{is}an{example}of{a}template')
    ['is', 'example', 'a']
    """
    # TODO: Need to use the string module, and need to auto-name the fields instead of refusing unnamed
    assert not empty_field_p.search(
        template
    ), "All fields must be named: That is, no empty {} allowed"
    return fields_re.findall(template)


# until_slash = "[^" + path_sep + "]+"
# until_slash_capture = '(' + until_slash + ')'


def mk_format_mapping_dict(format_dict, required_keys, sep=path_sep):
    until_sep = "[^" + re.escape(sep) + "]+"
    new_format_dict = format_dict.copy()
    for k in required_keys:
        if k not in new_format_dict:
            new_format_dict[k] = until_sep
    return new_format_dict


def mk_capture_patterns(mapping_dict):
    new_mapping_dict = dict()
    for k, v in mapping_dict.items():
        new_v = capture_template.format(format=v)
        new_mapping_dict[k] = new_v
    return new_mapping_dict


def mk_named_capture_patterns(mapping_dict):
    new_mapping_dict = dict()
    for k, v in mapping_dict.items():
        new_v = named_capture_template.format(name=k, format=v)
        new_mapping_dict[k] = new_v
    return new_mapping_dict


def template_to_pattern(mapping_dict, template):
    if mapping_dict:
        p = safe_compile(
            "{}".format(
                "|".join(["{" + re.escape(x) + "}" for x in list(mapping_dict.keys())])
            )
        )
        return p.sub(
            lambda x: mapping_dict[x.string[(x.start() + 1) : (x.end() - 1)]],
            template,
        )
    else:
        return template


def mk_extract_pattern(
    template, format_dict=None, named_capture_patterns=None, name=None
):
    format_dict = format_dict or {}
    named_capture_patterns = named_capture_patterns or mk_named_capture_patterns(
        format_dict
    )
    assert name is not None
    mapping_dict = dict(format_dict, **{name: named_capture_patterns[name]})
    p = safe_compile(
        "{}".format(
            "|".join(["{" + re.escape(x) + "}" for x in list(mapping_dict.keys())])
        )
    )

    return safe_compile(
        p.sub(
            lambda x: mapping_dict[x.string[(x.start() + 1) : (x.end() - 1)]],
            template,
        )
    )


# TODO: Is dependent on path sep -- separate concern
def mk_pattern_from_template_and_format_dict(template, format_dict=None, sep=path_sep):
    r"""Make a compiled regex to match template
    Args:
        template: A format string
        format_dict: A dict whose keys are template fields and values are regex strings to capture them
    Returns: a compiled regex

    >>> import os
    >>> p = mk_pattern_from_template_and_format_dict('{here}/and/{there}')
    >>> if os.name == 'nt':  # for windows
    ...     assert p == re.compile('(?P<here>[^\\\\]+)/and/(?P<there>[^\\\\]+)')
    ... else:
    ...     assert p == re.compile('(?P<here>[^/]+)/and/(?P<there>[^/]+)')
    >>> p = mk_pattern_from_template_and_format_dict('{here}/and/{there}', {'there': r'\d+'})
    >>> if os.name == 'nt':  # for windows
    ...     assert p == re.compile(r'(?P<here>[^\\\\]+)/and/(?P<there>\d+)')
    ... else:
    ...     assert p == re.compile(r'(?P<here>[^/]+)/and/(?P<there>\d+)')
    >>> type(p)
    <class 're.Pattern'>
    >>> p.match('HERE/and/1234').groupdict()
    {'here': 'HERE', 'there': '1234'}
    """
    format_dict = format_dict or {}

    fields = get_fields_from_template(template)
    format_dict = mk_format_mapping_dict(format_dict, fields, sep=sep)
    named_capture_patterns = mk_named_capture_patterns(format_dict)
    pattern = template_to_pattern(named_capture_patterns, template)
    try:
        return safe_compile(pattern)
    except Exception as e:
        raise ValueError(
            f"Got an error when attempting to re.compile('{pattern}'): "
            f"{type(e)}({e})"
        )


def mk_prefix_templates_dicts(template):
    fields = get_fields_from_template(template)
    prefix_template_dict_including_name = dict()
    none_and_fields = [None] + fields
    for name in none_and_fields:
        if name == fields[-1]:
            prefix_template_dict_including_name[name] = template
        else:
            if name is None:
                next_name = fields[0]
            else:
                next_name = fields[
                    1 + next(i for i, _name in enumerate(fields) if _name == name)
                ]
            p = "{" + next_name + "}"
            template_idx_of_next_name = re.search(p, template).start()
            prefix_template_dict_including_name[name] = template[
                :template_idx_of_next_name
            ]

    prefix_template_dict_excluding_name = dict()
    for i, name in enumerate(fields):
        prefix_template_dict_excluding_name[name] = prefix_template_dict_including_name[
            none_and_fields[i]
        ]
    prefix_template_dict_excluding_name[None] = template

    return (
        prefix_template_dict_including_name,
        prefix_template_dict_excluding_name,
    )


def mk_kwargs_trans(**trans_func_for_key):
    """Make a dict transformer from functions that depends solely on keys (of the dict to be transformed)
    Used to easily make process_kwargs and process_info_dict arguments for LinearNaming.
    """
    assert all(
        map(callable, trans_func_for_key.values())
    ), "all argument values must be callable"

    def key_based_val_trans(**kwargs):
        for k, v in kwargs.items():
            if k in trans_func_for_key:
                kwargs[k] = trans_func_for_key[k](v)
        return kwargs

    return key_based_val_trans


def _mk(self, *args, **kwargs):
    """
    Make a full name with given kwargs. All required name=val must be present (or infered by self.process_kwargs
    function.
    The required fields are in self.fields.
    Does NOT check for validity of the vals.
    :param kwargs: The name=val arguments needed to construct a valid name
    :return: an name
    """
    n = len(args) + len(kwargs)
    if n > self.n_fields:
        raise ValueError(
            f"You have too many arguments: (args, kwargs) is ({args},{kwargs})"
        )
    elif n < self.n_fields:
        raise ValueError(
            f"You have too few arguments: (args, kwargs) is ({args},{kwargs})"
        )
    kwargs = dict({k: v for k, v in zip(self.fields, args)}, **kwargs)
    if self.process_kwargs is not None:
        kwargs = self.process_kwargs(**kwargs)
    return self.template.format(**kwargs)


# from dol.trans import add_wrapper_method
#
# # @add_wrapper_method
class StrTupleDict:
    def __init__(
        self,
        template: str | tuple | list,
        format_dict=None,
        process_kwargs=None,
        process_info_dict=None,
        named_tuple_type_name="NamedTuple",
        sep: str = path_sep,
    ):
        r"""Converting from and to strings, tuples, and dicts.

        Args:
            template: The string format template
            format_dict: A {field_name: field_value_format_regex, ...} dict
            process_kwargs: A function taking the field=value pairs and producing a dict of processed
                {field: value,...} dict (where both fields and values could have been processed.
                This is useful when we need to process (format, default, etc.) fields, or their values,
                according to the other fields of values in the collection.
                A specification of {field: function_to_process_this_value,...} wouldn't allow the full powers
                we are allowing here.
            process_info_dict: A sort of converse of format_dict.
                This is a {field_name: field_conversion_func, ...} dict that is used to convert info_dict values
                before returning them.
            name_separator: Used

        >>> ln = StrTupleDict('/home/{user}/fav/{num}.txt',
        ...	                  format_dict={'user': '[^/]+', 'num': r'\d+'},
        ...	                  process_info_dict={'num': int},
        ...                   sep='/'
        ...	                 )
        >>> ln.is_valid('/home/USER/fav/123.txt')
        True
        >>> ln.is_valid('/home/US/ER/fav/123.txt')
        False
        >>> ln.is_valid('/home/US/ER/fav/not_a_number.txt')
        False
        >>> ln.mk('USER', num=123)  # making a string (with args or kwargs)
        '/home/USER/fav/123.txt'
        >>> # Note: but ln.mk('USER', num='not_a_number') would fail because num is not valid
        >>> ln.info_dict('/home/USER/fav/123.txt')  # note in the output, 123 is an int, not a string
        {'user': 'USER', 'num': 123}
        >>>
        >>> # Trying with template given as a tuple, and with different separator
        >>> ln = StrTupleDict(template=('first', 'last', 'age'),
        ...                   format_dict={'age': r'-*\d+'},
        ...                   process_info_dict={'age': int},
        ...                   sep=',')
        >>> ln.tuple_to_str(('Thor', "Odinson", 1500))
        'Thor,Odinson,1500'
        >>> ln.str_to_dict('Loki,Laufeyson,1070')
        {'first': 'Loki', 'last': 'Laufeyson', 'age': 1070}
        >>> ln.str_to_tuple('Odin,Himself,-1')
        ('Odin', 'Himself', -1)
        >>> ln.tuple_to_dict(('Odin', 'Himself', -1))
        {'first': 'Odin', 'last': 'Himself', 'age': -1}
        >>> ln.dict_to_tuple({'first': 'Odin', 'last': 'Himself', 'age': -1})
        ('Odin', 'Himself', -1)
        """
        if format_dict is None:
            format_dict = {}

        self.sep = sep

        if isinstance(template, str):
            self.template = template
        else:
            self.template = self.sep.join([f"{{{x}}}" for x in template])

        fields = get_fields_from_template(self.template)

        format_dict = mk_format_mapping_dict(format_dict, fields, self.sep)

        named_capture_patterns = mk_named_capture_patterns(format_dict)

        pattern = template_to_pattern(named_capture_patterns, self.template)
        pattern += "$"
        pattern = safe_compile(pattern)

        extract_pattern = {}
        for name in fields:
            extract_pattern[name] = mk_extract_pattern(
                self.template, format_dict, named_capture_patterns, name
            )

        if isinstance(process_info_dict, dict):
            _processor_for_kw = process_info_dict

            def process_info_dict(**info_dict):
                return {
                    k: _processor_for_kw.get(k, lambda x: x)(v)
                    for k, v in info_dict.items()
                }

        self.fields = fields
        self.n_fields = len(fields)
        self.format_dict = format_dict
        self.named_capture_patterns = named_capture_patterns
        self.pattern = pattern
        self.extract_pattern = extract_pattern
        self.process_kwargs = process_kwargs
        self.process_info_dict = process_info_dict

        def _mk(self, *args, **kwargs):
            """
            Make a full name with given kwargs. All required name=val must be present (or infered by self.process_kwargs
            function.
            The required fields are in self.fields.
            Does NOT check for validity of the vals.
            :param kwargs: The name=val arguments needed to construct a valid name
            :return: an name
            """
            n = len(args) + len(kwargs)
            if n > self.n_fields:
                raise ValueError(
                    f"You have too many arguments: (args, kwargs) is ({args},{kwargs})"
                )
            elif n < self.n_fields:
                raise ValueError(
                    f"You have too few arguments: (args, kwargs) is ({args},{kwargs})"
                )
            kwargs = dict({k: v for k, v in zip(self.fields, args)}, **kwargs)
            if self.process_kwargs is not None:
                kwargs = self.process_kwargs(**kwargs)
            return self.template.format(**kwargs)

        set_signature_of_func(_mk, ["self"] + self.fields)
        self.mk = MethodType(_mk, self)
        self.NamedTuple = namedtuple(named_tuple_type_name, self.fields)

    def is_valid(self, s: str):
        """Check if the name has the "upload format" (i.e. the kind of fields that are _ids of fv_mgc, and what
        name means in most of the iatis system.
        :param s: the string to check
        :return: True iff name has the upload format
        """
        return bool(self.pattern.match(s))

    def str_to_dict(self, s: str):
        """
        Get a dict with the arguments of an name (for example group, user, subuser, etc.)
        :param s:
        :return: a dict holding the argument fields and values
        """
        m = self.pattern.match(s)
        if m:
            info_dict = m.groupdict()
            if self.process_info_dict:
                return self.process_info_dict(**info_dict)
            else:
                return info_dict
        else:
            raise ValueError(f"Invalid string format: {s}")

    def str_to_tuple(self, s: str):
        info_dict = self.str_to_dict(s)
        return tuple(info_dict[x] for x in self.fields)

    def str_to_namedtuple(self, s: str):
        return self.dict_to_namedtuple(self.str_to_dict(s))

    def str_to_simple_str(self, s: str):
        return self.sep.join(self.str_to_tuple(s))

    def simple_str_to_str(self, ss: str):
        return self.tuple_to_str(ss.split(self.sep))

    def super_dict_to_str(self, d: dict):
        """Like dict_to_str, but the input dict can have extra keys that are not used by dict_to_str"""
        return self.mk(**{k: v for k, v in d.items() if k in self.fields})

    def dict_to_str(self, d: dict):
        return self.mk(**d)

    def dict_to_tuple(self, d):
        assert_condition(
            len(self.fields) == len(d),
            f"len(d)={len(d)} but len(fields)={len(self.fields)}",
        )
        return tuple(d[f] for f in self.fields)

    def dict_to_namedtuple(self, d):
        return self.NamedTuple(**d)

    def tuple_to_dict(self, t):
        assert_condition(
            len(self.fields) == len(t),
            f"len(d)={len(t)} but len(fields)={len(self.fields)}",
        )
        return {f: x for f, x in zip(self.fields, t)}

    def tuple_to_str(self, t):
        return self.mk(*t)

    def namedtuple_to_tuple(self, nt):
        return tuple(nt)

    def namedtuple_to_dict(self, nt):
        return {k: getattr(nt, k) for k in self.fields}

    def namedtuple_to_str(self, nt):
        return self.dict_to_str(self.namedtuple_to_dict(nt))

    def extract(self, field, s):
        """Extract a single item from an name
        :param field: field of the item to extract
        :param s: the string from which to extract it
        :return: the value for name
        """
        return self.extract_pattern[field].match(s).group(1)

    info_dict = str_to_dict  # alias
    info_tuple = str_to_tuple  # alias

    def replace_name_elements(self, s: str, **elements_kwargs):
        """Replace specific name argument values with others
        :param s: the string to replace
        :param elements_kwargs: the arguments to replace (and their values)
        :return: a new name
        """
        name_info_dict = self.info_dict(s)
        for k, v in elements_kwargs.items():
            name_info_dict[k] = v
        return self.mk(**name_info_dict)

    def _info_str(self):
        kv = self.__dict__.copy()
        exclude = [
            "process_kwargs",
            "extract_pattern",
            "prefix_pattern",
            "prefix_template_including_name",
            "prefix_template_excluding_name",
        ]
        for f in exclude:
            kv.pop(f)
        s = ""
        s += "  * {}: {}\n".format("template", kv.pop("template"))
        s += "  * {}: {}\n".format("template", kv.pop("sep"))
        s += "  * {}: {}\n".format("format_dict", kv.pop("format_dict"))

        for k, v in kv.items():
            if hasattr(v, "pattern"):
                v = v.pattern
            s += f"  * {k}: {v}\n"

        return s

    def _print_info_str(self):
        print(self._info_str())


# TODO: mk_prefix has wrong signature. Repair.
class StrTupleDictWithPrefix(StrTupleDict):
    r"""Converting from and to strings, tuples, and dicts, but with partial "prefix" specs allowed.

    Args:
        template: The string format template
        format_dict: A {field_name: field_value_format_regex, ...} dict
        process_kwargs: A function taking the field=value pairs and producing a dict of processed
            {field: value,...} dict (where both fields and values could have been processed.
            This is useful when we need to process (format, default, etc.) fields, or their values,
            according to the other fields of values in the collection.
            A specification of {field: function_to_process_this_value,...} wouldn't allow the full powers
            we are allowing here.
        process_info_dict: A sort of converse of format_dict.
            This is a {field_name: field_conversion_func, ...} dict that is used to convert info_dict values
            before returning them.
        name_separator: Used

    >>> ln = StrTupleDictWithPrefix('/home/{user}/fav/{num}.txt',
    ...	                  format_dict={'user': '[^/]+', 'num': r'\d+'},
    ...	                  process_info_dict={'num': int},
    ...                   sep='/'
    ...	                 )
    >>> ln.mk('USER', num=123)  # making a string (with args or kwargs)
    '/home/USER/fav/123.txt'
    >>> ####### prefix methods #######
    >>> ln.is_valid_prefix('/home/USER/fav/')
    True
    >>> ln.is_valid_prefix('/home/USER/fav/12')  # False because too long
    False
    >>> ln.is_valid_prefix('/home/USER/fav')  # False because too short
    False
    >>> ln.is_valid_prefix('/home/')  # True because just right
    True
    >>> ln.is_valid_prefix('/home/USER/fav/123.txt')  # full path, so output same as is_valid() method
    True
    >>>
    >>> ln.mk_prefix('ME')
    '/home/ME/fav/'
    >>> ln.mk_prefix(user='YOU', num=456)  # full specification, so output same as same as mk() method
    '/home/YOU/fav/456.txt'
    """

    @wraps(StrTupleDict.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        (
            self.prefix_template_including_name,
            self.prefix_template_excluding_name,
        ) = mk_prefix_templates_dicts(self.template)

        _prefix_pattern = "$|".join(
            [
                x.format(**self.format_dict)
                for x in sorted(
                    list(self.prefix_template_including_name.values()), key=len
                )
            ]
        )
        _prefix_pattern += "$"
        self.prefix_pattern = safe_compile(_prefix_pattern)

        def _mk_prefix(self, *args, **kwargs):
            """
            Make a prefix for an uploads name that has has the path up to the first None argument.
            :return: A string that is the prefix of a valid name
            """
            assert (
                len(args) + len(kwargs) <= self.n_fields
            ), "You have too many arguments"
            kwargs = dict({k: v for k, v in zip(self.fields, args)}, **kwargs)
            if self.process_kwargs is not None:
                kwargs = self.process_kwargs(**kwargs)

            # ascertain that no fields were skipped (we can leave fields out at the end, but not in the middle)
            a_name_was_skipped = False
            for name in self.fields:
                if name not in kwargs:
                    if a_name_was_skipped == True:
                        raise ValueError(
                            "You are making a PREFIX: This means you can't skip any fields. "
                            "Once a name is omitted, you need to omit all further fields. "
                            f"The name order is {self.fields}. You specified {tuple(kwargs.keys())}"
                        )
                    else:
                        a_name_was_skipped = True

            keep_kwargs = {}
            last_name = None
            for name in self.fields:
                if name in kwargs:
                    keep_kwargs[name] = kwargs[name]
                    last_name = name
                else:
                    break

            return self.prefix_template_including_name[last_name].format(**keep_kwargs)

        set_signature_of_func(_mk_prefix, [(s, None) for s in self.fields])
        self.mk_prefix = MethodType(_mk_prefix, self)

    def is_valid_prefix(self, s):
        """Check if name is a valid prefix.
        :param s: a string (that might or might not be a valid prefix)
        :return: True iff name is a valid prefix
        """
        return bool(self.prefix_pattern.match(s))


LinearNaming = StrTupleDictWithPrefix

from dol.base import Store
from collections import namedtuple
from dol.util import lazyprop


class ParametricKeyStore(Store):
    def __init__(self, store, keymap=None):
        super().__init__(store)
        self._keymap = keymap

    @property
    def _linear_naming(self):
        print("_linear_naming Deprecated: Use _keymap instead")
        return self._keymap


class StoreWithTupleKeys(ParametricKeyStore):
    def _id_of_key(self, key):
        return self._keymap.mk(*key)

    def _key_of_id(self, _id):
        return self._keymap.info_tuple(_id)


class StoreWithDictKeys(ParametricKeyStore):
    def _id_of_key(self, key):
        return self._keymap.mk(**key)

    def _key_of_id(self, _id):
        return self._keymap.info_dict(_id)


class StoreWithNamedTupleKeys(ParametricKeyStore):
    @lazyprop
    def NamedTupleKey(self):
        return namedtuple("NamedTupleKey", field_names=self._keymap.fields)

    def _id_of_key(self, key):
        return self._keymap.mk(*key)

    def _key_of_id(self, _id):
        return self.NamedTupleKey(*self._keymap.info_tuple(_id))


# def mk_parametric_key_store_cls(store_cls, key_type=tuple):
#     if key_type == tuple:
#         super_cls = StoreWithTupleKeys
#     elif key_type == dict:
#         super_cls = StoreWithDictKeys
#     else:
#         raise ValueError("key_type needs to be tuple or dict")
#
#     class A(super_cls, store_cls):
#         def __init__(self, rootdir, subpath='', format_dict=None, process_kwargs=None, process_info_dict=None,
#                      **extra_store_kwargs):
#
#             path_format = os.path.join(rootdir, subpath)
#             store = store_cls.__init__(self, path_format=path_format, **extra_store_kwargs)
#             linear_naming = LinearNaming()
#
#             # FilepathFormatKeys.__init__(self, path_format)


class NamingInterface:
    def __init__(
        self,
        params=None,
        validation_funs=None,
        all_kwargs_should_be_in_validation_dict=dflt_all_kwargs_should_be_in_validation_dict,
        ignore_misunderstood_validation_instructions=dflt_ignore_misunderstood_validation_instructions,
        **kwargs,
    ):
        if params is None:
            params = {}
        if validation_funs is None:
            validation_funs = dflt_validation_funs
        validation_dict = {
            var: info.get("validation", {}) for var, info in params.items()
        }
        default_dict = {var: info.get("default", None) for var, info in params.items()}
        arg_pattern = {
            var: info.get("arg_pattern", dflt_arg_pattern)
            for var, info in params.items()
        }
        named_arg_pattern = {
            var: "(?P<" + var + ">" + pat + ")" for var, pat in arg_pattern.items()
        }
        to_str = {
            var: info["to_str"] for var, info in params.items() if "to_str" in info
        }
        to_val = {
            var: info["to_val"] for var, info in params.items() if "to_val" in info
        }

        self.validation_dict = validation_dict
        self.default_dict = default_dict
        self.arg_pattern = arg_pattern
        self.named_arg_pattern = named_arg_pattern
        self.to_str = to_str
        self.to_val = to_val

        self.validation_funs = validation_funs
        self.all_kwargs_should_be_in_validation_dict = (
            all_kwargs_should_be_in_validation_dict
        )
        self.ignore_misunderstood_validation_instructions = (
            ignore_misunderstood_validation_instructions
        )

    def validate_kwargs(self, **kwargs):
        return validate_kwargs(
            kwargs_to_validate=kwargs,
            validation_dict=self.validation_dict,
            validation_funs=self.validation_funs,
            all_kwargs_should_be_in_validation_dict=self.all_kwargs_should_be_in_validation_dict,
            ignore_misunderstood_validation_instructions=self.ignore_misunderstood_validation_instructions,
        )

    def default_for(self, arg, **kwargs):
        default = self.default_dict[arg]
        if (
            not isinstance(default, dict)
            or "args" not in default
            or "func" not in default
        ):
            return default
        else:  # call the func on the default['args'] values given in kwargs
            args = {arg_: kwargs[arg_] for arg_ in default["args"]}
            return default["func"](*args)

    def str_kwargs_from(self, **kwargs):
        return {k: self.to_str[k](v) for k, v in kwargs.items() if k in self.to_str}

    def val_kwargs_from(self, **kwargs):
        return {k: self.to_val[k](v) for k, v in kwargs.items() if k in self.to_val}

    def name_for(self, **kwargs):
        raise NotImplementedError("Interface method: Method needs to be implemented")

    def info_for(self, **kwargs):
        raise NotImplementedError("Interface method: Method needs to be implemented")

    def is_valid_name(self, name):
        raise NotImplementedError("Interface method: Method needs to be implemented")


class BigDocTest:
    """

    # TODO: Fix this test (maybe test assertions aren't correct)
    #   This happened when we changed some re.compile to safe_compile
    # >>>
    # >>> e_name = BigDocTest.mk_e_naming()
    # >>> u_name = BigDocTest.mk_u_naming()
    # >>> e_sref = 's3://bucket-GROUP/example/files/USER/SUBUSER/2017-01-24/1485272231982_1485261448469'
    # >>> u_sref = "s3://uploads/GROUP/upload/files/USER/2017-01-24/SUBUSER/a_file.wav"
    # >>> u_name_2 = "s3://uploads/ANOTHER_GROUP/upload/files/ANOTHER_USER/2017-01-24/SUBUSER/a_file.wav"
    # >>>
    # >>> ####### is_valid(self, name): ######
    # >>> e_name.is_valid(e_sref)
    # True
    # >>> e_name.is_valid(u_sref)
    # False
    # >>> u_name.is_valid(u_sref)
    # True
    # >>>
    # >>> ####### is_valid_prefix(self, name): ######
    # >>> e_name.is_valid_prefix('s3://bucket-')
    # True
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP')
    # False
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP/example/')
    # False
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP/example/files')
    # False
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP/example/files/')
    # True
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP/example/files/USER/SUBUSER/2017-01-24/')
    # True
    # >>> e_name.is_valid_prefix('s3://bucket-GROUP/example/files/USER/SUBUSER/2017-01-24/0_0')
    # True
    # >>>
    # >>> ####### info_dict(self, name): ######
    # >>> e_name.info_dict(e_sref)  # see that utc_ms args were cast to ints
    # {'group': 'GROUP', 'user': 'USER', 'subuser': 'SUBUSER', 'day': '2017-01-24', 's_ums': 1485272231982, 'e_ums': 1485261448469}
    # >>> u_name.info_dict(u_sref)  # returns None (because self was made for example!
    # {'group': 'GROUP', 'user': 'USER', 'day': '2017-01-24', 'subuser': 'SUBUSER', 'filename': 'a_file.wav'}
    # >>> # but with a u_name, it will work
    # >>> u_name.info_dict(u_sref)
    # {'group': 'GROUP', 'user': 'USER', 'day': '2017-01-24', 'subuser': 'SUBUSER', 'filename': 'a_file.wav'}
    # >>>
    # >>> ####### extract(self, item, name): ######
    # >>> e_name.extract('group', e_sref)
    # 'GROUP'
    # >>> e_name.extract('user', e_sref)
    # 'USER'
    # >>> u_name.extract('group', u_name_2)
    # 'ANOTHER_GROUP'
    # >>> u_name.extract('user', u_name_2)
    # 'ANOTHER_USER'
    # >>>

    #
    # >>> ####### mk_prefix(self, *args, **kwargs): ######
    # >>> e_name.mk_prefix()
    # 's3://bucket-'
    # >>> e_name.mk_prefix(group='GROUP')
    # 's3://bucket-GROUP/example/files/'
    # >>> e_name.mk_prefix(group='GROUP', user='USER')
    # 's3://bucket-GROUP/example/files/USER/'
    # >>> e_name.mk_prefix(group='GROUP', user='USER', subuser='SUBUSER')
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/'
    # >>> e_name.mk_prefix(group='GROUP', user='USER', subuser='SUBUSER', day='0000-00-00')
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/0000-00-00/'
    # >>> e_name.mk_prefix(group='GROUP', user='USER', subuser='SUBUSER', day='0000-00-00',
    # ... s_ums=1485272231982)
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/0000-00-00/1485272231982_'
    # >>> e_name.mk_prefix(group='GROUP', user='USER', subuser='SUBUSER', day='0000-00-00',
    # ... s_ums=1485272231982, e_ums=1485261448469)
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/0000-00-00/1485272231982_1485261448469'
    # >>>
    # >>> u_name.mk_prefix()
    # 's3://uploads/'
    # >>> u_name.mk_prefix(group='GROUP')
    # 's3://uploads/GROUP/upload/files/'
    # >>> u_name.mk_prefix(group='GROUP', user='USER')
    # 's3://uploads/GROUP/upload/files/USER/'
    # >>> u_name.mk_prefix(group='GROUP', user='USER', day='DAY')
    # 's3://uploads/GROUP/upload/files/USER/DAY/'
    # >>> u_name.mk_prefix(group='GROUP', user='USER', day='DAY')
    # 's3://uploads/GROUP/upload/files/USER/DAY/'
    # >>> u_name.mk_prefix(group='GROUP', user='USER', day='DAY', subuser='SUBUSER')
    # 's3://uploads/GROUP/upload/files/USER/DAY/SUBUSER/'
    # >>>
    # >>> ####### mk(self, *args, **kwargs): ######
    # >>> e_name.mk(group='GROUP', user='USER', subuser='SUBUSER', day='0000-00-00',
    # ...             s_ums=1485272231982, e_ums=1485261448469)
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/0000-00-00/1485272231982_1485261448469'
    # >>> e_name.mk(group='GROUP', user='USER', subuser='SUBUSER', day='from_s_ums',
    # ...             s_ums=1485272231982, e_ums=1485261448469)
    # 's3://bucket-GROUP/example/files/USER/SUBUSER/2017-01-24/1485272231982_1485261448469'
    # >>>
    # >>> ####### replace_name_elements(self, *args, **kwargs): ######
    # >>> name = 's3://bucket-redrum/example/files/oopsy@domain.com/ozeip/2008-11-04/1225779243969_1225779246969'
    # >>> e_name.replace_name_elements(name, user='NEW_USER', group='NEW_GROUP')
    # 's3://bucket-NEW_GROUP/example/files/NEW_USER/ozeip/2008-11-04/1225779243969_1225779246969'
    """

    @staticmethod
    def process_info_dict_for_example(**info_dict):
        if "s_ums" in info_dict:
            info_dict["s_ums"] = int(info_dict["s_ums"])
        if "e_ums" in info_dict:
            info_dict["e_ums"] = int(info_dict["e_ums"])
        return info_dict

    @staticmethod
    def example_process_kwargs(**kwargs):
        from datetime import datetime

        epoch = datetime.utcfromtimestamp(0)
        second_ms = 1000.0

        def utcnow_ms():
            return (datetime.utcnow() - epoch).total_seconds() * second_ms

        # from ut.util.time import second_ms, utcnow_ms
        if "s_ums" in kwargs:
            kwargs["s_ums"] = int(kwargs["s_ums"])
        if "e_ums" in kwargs:
            kwargs["e_ums"] = int(kwargs["e_ums"])

        if "day" in kwargs:
            day = kwargs["day"]
            # get the day in the expected format
            if isinstance(day, str):
                if day == "now":
                    day = datetime.utcfromtimestamp(
                        int(utcnow_ms() / second_ms)
                    ).strftime(day_format)
                elif day == "from_s_ums":
                    assert "s_ums" in kwargs, "need to have s_ums argument"
                    day = datetime.utcfromtimestamp(
                        int(kwargs["s_ums"] / second_ms)
                    ).strftime(day_format)
                else:
                    assert day_format_pattern.match(day)
            elif isinstance(day, datetime):
                day = day.strftime(day_format)
            elif "s_ums" in kwargs:  # if day is neither a string nor a datetime
                day = datetime.utcfromtimestamp(
                    int(kwargs["s_ums"] / second_ms)
                ).strftime(day_format)

            kwargs["day"] = day

        return kwargs

    @staticmethod
    def mk_e_naming():
        return LinearNaming(
            template="s3://bucket-{group}/example/files/{user}/{subuser}/{day}/{s_ums}_{e_ums}",
            format_dict={"s_ums": r"\d+", "e_ums": r"\d+", "day": "[^/]+"},
            process_kwargs=BigDocTest.example_process_kwargs,
            process_info_dict=BigDocTest.process_info_dict_for_example,
        )

    @staticmethod
    def mk_u_naming():
        return LinearNaming(
            template="s3://uploads/{group}/upload/files/{user}/{day}/{subuser}/{filename}",
            format_dict={"day": "[^/]+", "filepath": ".+"},
        )


import os
from functools import wraps
from dol.trans import wrap_kvs, store_decorator

pjoin = os.path.join

KeyMapNames = namedtuple("KeyMaps", ["key_of_id", "id_of_key"])
KeyMaps = namedtuple("KeyMaps", ["key_of_id", "id_of_key"])


def _get_keymap_names_for_str_to_key_type(key_type):
    if not isinstance(key_type, str):
        key_type = {
            tuple: "tuple",
            namedtuple: "namedtuple",
            dict: "dict",
            str: "str",
        }.get(key_type, None)

    if key_type not in {"tuple", "namedtuple", "dict", "str"}:
        raise ValueError(f"Not a recognized key_type: {key_type}")

    return KeyMapNames(key_of_id=f"str_to_{key_type}", id_of_key=f"{key_type}_to_str")


def _get_method_for_str_to_key_type(keymap, key_type):
    kmn = _get_keymap_names_for_str_to_key_type(key_type)
    return KeyMaps(
        key_of_id=getattr(keymap, kmn.key_of_id),
        id_of_key=getattr(keymap, kmn.id_of_key),
    )


# TODO: Make this into a proper store decorator
@store_decorator
def mk_store_from_path_format_store_cls(
    store=None,
    *,
    subpath="",
    store_cls_kwargs=None,
    key_type=namedtuple,
    keymap=StrTupleDict,
    keymap_kwargs=None,
    name=None,
):
    """Wrap a store (instance or class) that uses string keys to make it into a store that uses a specific key format.

    Args:
        store: The instance or class to wrap
        subpath: The subpath (defining the subset of the data pointed at by the URI
        store_cls_kwargs:  # if store is a class, the kwargs that you would have given the store_cls to make itself
        key_type: The key type you want to interface with:
            dict, tuple, namedtuple, str or 'dict', 'tuple', 'namedtuple', 'str'
        keymap:  # the keymap instance or class you want to use to map keys
        keymap_kwargs:  # if keymap is a cls, the kwargs to give it (besides the subpath)
        name: The name to give the class the function will make here

    Returns: An instance of a wrapped class


    Example:
    ```
    # Get a (session, bt) indexed LocalJsonStore
    s = mk_store_from_path_format_store_cls(LocalJsonStore,
                                                   os.path.join(root_dir, 'd'),
                                                   subpath='{session}/d/{bt}',
                                                   keymap_kwargs=dict(process_info_dict={'session': int, 'bt': int}))
    ```
    """
    if isinstance(keymap, type):
        keymap = keymap(subpath, **(keymap_kwargs or {}))  # make the keymap instance

    km = _get_method_for_str_to_key_type(keymap, key_type)

    if isinstance(store, type):
        name = name or "KeyWrapped" + store.__name__
        _WrappedStoreCls = wrap_kvs(
            store, name=name, key_of_id=km.key_of_id, id_of_key=km.id_of_key
        )

        class WrappedStoreCls(_WrappedStoreCls):
            def __init__(self, root_uri):
                path_format = pjoin(root_uri, subpath)
                super().__init__(path_format, **(store_cls_kwargs or {}))

        return WrappedStoreCls
    else:
        name = name or "KeyWrapped" + store.__class__.__name__
        return wrap_kvs(
            store, name=name, key_of_id=km.key_of_id, id_of_key=km.id_of_key
        )


mk_tupled_store_from_path_format_store_cls = mk_store_from_path_format_store_cls

from string import Formatter


# TODO: Make .vformat (therefore .format) work with args and kwargs
# TODO: Make it not blow up and conserve spec (e.g. the 1.2f of {foo:1.2f}) when not specified
class PartialFormatter(Formatter):
    """A string formatter that won't complain if the fields are only partially formatted.
    But note that you will lose the spec part of your template (e.g. in {foo:1.2f}, you'll loose the 1.2f
    if not foo is given -- but {foo} will remain).

    >>> partial_formatter = PartialFormatter()
    >>> str_template = 'foo:{foo} bar={bar} a={a} b={b:0.02f} c={c}'
    >>> partial_formatter.format(str_template, bar="BAR", b=34)
    'foo:{foo} bar=BAR a={a} b=34.00 c={c}'

    Note: If you only need a formatting function (not the transformed formatting string), a simpler solution may be:
    ```
    import functools
    format_str = functools.partial(str_template.format, bar="BAR", b=34)
    ```
    See https://stackoverflow.com/questions/11283961/partial-string-formatting for more options and discussions.
    """

    def get_value(self, key, args, kwargs):
        try:
            return super().get_value(key, args, kwargs)
        except KeyError:
            return "{" + key + "}"

    def format_fields_set(self, s):
        return {x[1] for x in self.parse(s) if x[1]}

    def format_with_non_none_vals(self, format_string, **mapping):
        mapping = {k: v for k, v in mapping.items() if v is not None}
        return self.vformat(format_string, (), mapping)


partial_formatter = PartialFormatter()
```

## paths.py

```python
"""Module for path (and path-like) object manipulation


Examples::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(path_filter(lambda p, k, v: v == 2, d))
    [('a', 'b', 'd')]
    >>> path_get(d, ('a', 'b', 'd'))
    2
    >>> path_set(d, ('a', 'b', 'd'), 4)
    >>> d
    {'a': {'b': {'c': 1, 'd': 4}, 'e': 3}}
    >>> path_set(d, ('a', 'b', 'new_ab_key'), 42)
    >>> d
    {'a': {'b': {'c': 1, 'd': 4, 'new_ab_key': 42}, 'e': 3}}

"""

from functools import wraps, partial
from dataclasses import dataclass
from typing import (
    Optional,
    Union,
    Any,
    Tuple,
    Literal,
    KT,
    VT,
    TypeVar,
    TypeAlias,
    List,
    Dict,
)
from collections.abc import Callable, Mapping, Iterable, Sequence, Iterator, Generator

from operator import getitem
import os

from dol.base import Store
from dol.util import lazyprop, add_as_attribute_of, max_common_prefix, safe_compile
from dol.trans import (
    store_decorator,
    kv_wrap,
    add_path_access,
    filt_iter,
    wrap_kvs,
    add_missing_key_handling,
)
from dol.dig import recursive_get_attr


KeyValueGenerator = Generator[tuple[KT, VT], None, None]
Path = TypeVar("Path")
PathExtenderFunc = Callable[[Path, KT], Path]
PathExtenderSpec = Union[str, PathExtenderFunc]
NestedMapping: TypeAlias = Mapping[KT, Union[VT, "NestedMapping[KT, VT]"]]


def separator_based_path_extender(path: Path, key: KT, sep: str) -> Path:
    """
    Extends a given path with a new key using the specified separator.
    If the path is empty, the key is returned as is.
    """
    return f"{path}{sep}{key}" if path else key


def ensure_path_extender_func(path_extender: PathExtenderSpec) -> PathExtenderFunc:
    """
    Ensure that the path_extender is a function that takes a path and a key and returns
    a new path."""
    if isinstance(path_extender, str):
        return partial(separator_based_path_extender, sep=path_extender)
    return path_extender


def flattened_dict_items(
    d,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Path | None = None,
    visit_nested: Callable = lambda obj: isinstance(obj, Mapping),
) -> KeyValueGenerator:
    """
    Yield flattened key-value pairs from a nested dictionary.
    """
    path_extender = ensure_path_extender_func(sep)

    stable_kwargs = dict(sep=sep, visit_nested=visit_nested)

    for k, v in d.items():
        new_path = path_extender(parent_path, k)
        if visit_nested(v):
            yield from flattened_dict_items(v, parent_path=new_path, **stable_kwargs)
        else:
            yield new_path, v


def flatten_dict(
    d,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Path | None = None,
    visit_nested: Callable = lambda obj: isinstance(obj, Mapping),
    egress: Callable[[KeyValueGenerator], Mapping] = dict,
):
    r"""
    Flatten a nested dictionary into a flat one, using key-paths as keys.

    See also `leaf_paths` for a related function that returns paths to leaf values.

    Args:
        d: The dictionary to flatten
        sep: The separator to use for joining keys, or a function that takes a path and
            a key and returns a new path.
        parent_path: The path to the parent of the current dict
        visit_nested: A function that returns True if a value should be visited
        egress: A function that takes a generator of key-value pairs and returns a mapping

    >>> d = {'a': {'b': 2}, 'c': 3}
    >>> flatten_dict(d)
    {'a.b': 2, 'c': 3}
    >>> flatten_dict(d, sep='/')
    {'a/b': 2, 'c': 3}

    """
    return egress(
        flattened_dict_items(
            d, sep=sep, parent_path=parent_path, visit_nested=visit_nested
        )
    )


def leaf_paths(
    d: NestedMapping,
    sep: PathExtenderSpec = ".",
    *,
    parent_path: Path | None = None,
    egress: Callable[[KeyValueGenerator], Mapping] = dict,
) -> dict[KT, KT | Path]:
    """
    Get a dictionary of leaf paths of a nested dictionary.

    Given a nested dictionary, returns a similarly structured dictionary where each
    leaf value is replaced by its flattened path. The 'sep' parameter can be either
    a string or a callable.

    Original use case: You used flatten_dict to flatten a nested dictionary, referencing
    your values with paths, but maybe you'd like to know what the paths that your
    nested dictionary is going to flatten to are. This function does that.
    The output is a dict with the same keys and structure as the input, but the leaf
    values are replaced by the paths that would be used to access them in a flat dict.

    Args:
        d: The nested dictionary to get the leaf paths from
        sep: The separator to use for joining keys, or a function that takes a path and
            a key and returns a new path.
        parent_path: The path to the parent of the current dict
        egress: A function that takes a generator of key-value pairs and returns a mapping

    Example:
    >>> leaf_paths({'a': {'b': 2}, 'c': 3})
    {'a': {'b': 'a.b'}, 'c': 'c'}

    >>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep="/")
    {'a': {'b': 'a/b'}, 'c': 'c'}

    >>> leaf_paths({'a': {'b': 2}, 'c': 3}, sep=lambda p, k: f"{p}-{k}" if p else k)
    {'a': {'b': 'a-b'}, 'c': 'c'}
    """
    path_extender = ensure_path_extender_func(sep)

    return egress(_leaf_paths_recursive(d, path_extender, parent_path=parent_path))


def _leaf_paths_recursive(
    d: NestedMapping,
    path_extender: PathExtenderFunc,
    parent_path: Path | None = None,
    *,
    visit_nested: Callable[[Any], bool] = lambda x: isinstance(x, dict),
) -> KeyValueGenerator:
    """
    A recursive generator that yields (key, value) pairs.
    A helper for leaf_paths.
    """
    for k, v in d.items():
        current_path = path_extender(parent_path, k)
        if visit_nested(v):
            yield k, dict(_leaf_paths_recursive(v, path_extender, current_path))
        else:
            yield k, current_path


path_sep = os.path.sep


def raise_on_error(d: dict):
    raise


def return_none_on_error(d: dict):
    return None


def return_empty_tuple_on_error(d: dict):
    return ()


OnErrorType = Union[Callable[[dict], Any], str]


# TODO: Could extend OnErrorType to be a dict with error class keys and callables or
#  strings as values. Then, the error class could be used to determine the error
#  handling strategy.
def _path_get(
    obj: Any,
    path,
    on_error: OnErrorType = raise_on_error,
    *,
    path_to_keys: Callable[[Any], Iterable] = None,
    get_value: Callable = getitem,
    caught_errors=(KeyError, IndexError),
):
    """Get elements of a mapping through a path to be called recursively.

    >>> _path_get({'a': {'b': 2}}, 'a')
    {'b': 2}
    >>> _path_get({'a': {'b': 2}}, ['a', 'b'])
    2
    >>> _path_get({'a': {'b': 2}}, ['a', 'c'])
    Traceback (most recent call last):
        ...
    KeyError: 'c'
    >>> _path_get({'a': {'b': 2}}, ['a', 'c'], lambda x: x)
    {'obj': {'a': {'b': 2}}, 'path': ['a', 'c'], 'result': {'b': 2}, 'k': 'c', 'error': KeyError('c')}

    # >>> assert _path_get({'a': {'b': 2}}, ['a', 'c'], lambda x: x) == {
    # ...     'mapping': {'a': {'b': 2}},
    # ...     'path': ['a', 'c'],
    # ...     'result': {'b': 2},
    # ...     'k': 'c',
    # ...     'error': KeyError('c')
    # ... }

    """

    if path_to_keys is not None:
        keys = path_to_keys(path)
    else:
        keys = path

    result = obj

    for k in keys:
        try:
            result = get_value(result, k)
        except caught_errors as error:
            if callable(on_error):
                return on_error(
                    dict(
                        obj=obj,
                        path=path,
                        result=result,
                        k=k,
                        error=error,
                    )
                )
            elif isinstance(on_error, str):
                # use on_error as a message, raising the same error class
                raise type(error)(on_error)
            else:
                raise ValueError(
                    f"on_error should be a callable (input is a dict) or a string. "
                    f"Was: {on_error}"
                )
    return result


def split_if_str(obj, sep="."):
    if isinstance(obj, str):
        return obj.split(sep)
    return obj


def separate_keys_with_separator(obj, sep="."):
    return map(cast_to_int_if_numeric_str, split_if_str(obj, sep))


def getitem(obj, k):
    return obj[k]


def get_attr_or_item(obj, k):
    """
    If ``k`` is a string, tries to get ``k`` as an attribute of ``obj`` first,
    and if that fails, gets it as ``obj[k]``

    WARNING: The hardcoded priority choices of this function regarding when to try
    k as an item, index, or attribute, don't apply to every case, so you may want to
    use an explicit value getter to be more robust!

    # >>> d = {'a': [1, {'items': 2, '3': 33, 3: 42}]}
    >>> get_attr_or_item({'items': 2}, 'items')
    2

    But if "items" is not there as a key of the object, the attribute is found:

    >>> get_attr_or_item({'not_items': 2}, 'items')  # doctest: +ELLIPSIS
    <built-in method items of dict object...>

    Both integers and string integers will work to get an item if obj is not a Mapping.

    >>> get_attr_or_item([7, 21, 42], 2)
    42
    >>> get_attr_or_item([7, 21, 42], '2')
    42

    If you're dealling with a Mapping, you can get both integer and string keys, and
    if you have both types in your Mapping, you'll get the right one!

    >>> get_attr_or_item({2: 'numerical key', '2': 'string key'}, 2)
    'numerical key'
    >>> get_attr_or_item({2: 'numerical key', '2': 'string key'}, '2')
    'string key'

    If you don't have the numerical version, the string version will still find your
    numerical key.

    >>> get_attr_or_item({2: 'numerical key'}, '2')
    'numerical key'

    The opposite is not true though: If you ask for an integer key, it will not find
    a string version of it.

    >>> get_attr_or_item({'2': 'string key'}, 2) # +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: 2

    """
    if isinstance(k, str):
        if str.isnumeric(k) and not isinstance(obj, Mapping):
            # if k is the string of an integer and obj is not a Mapping,
            # consider k to be int index
            k = int(k)
        try:
            return obj[k]  # prioritize getitem (keys and indices)
        except (TypeError, KeyError, IndexError):
            if str.isnumeric(k):
                if isinstance(obj, Mapping):
                    return obj[int(k)]
                else:
                    raise
            else:
                return getattr(obj, k)  # try it as an attribute
    else:
        return obj[k]


def keys_and_indices_path(str_path, *, sep=".", index_pattern=r"\[(\d+)\]"):
    """
    Transforms a string path separated by a specified separator into a tuple
    of keys and indices. Bracketed indices are extracted as integers.

    This function is meant to be used in as the key_transformer argument of path_get etc.

    Args:
        path (str): The input path string, e.g., "a21-59c.message[2].user".
        sep (str): The separator used to split the path, default is '.'.
        index_pattern (str): The regular expression pattern to match bracketed indices

    Returns:
        tuple: A tuple representation of the path, e.g., ("a21-59c", "message", 2, "user").

    Example:

    >>> keys_and_indices_path("a21-59c.message[2].user")
    ('a21-59c', 'message', 2, 'user')
    """
    # Split the path by the specified separator
    parts = str_path.split(sep)
    search = re.compile(index_pattern).search

    def transformed_parts():
        for part in parts:
            # Extract bracketed indices
            match = search(part)
            if match:
                # yield the key (without the index) and the extracted index
                yield part[: match.start()]  # Key before the index
                yield int(match.group(1))  # Index as integer
            else:
                # yield the key as is if no index is present
                yield part

    return tuple(transformed_parts())


# ------------------------------------------------------------------------------
# key-path operations


# TODO: Needs a lot more documentation and tests to show how versatile it is
def path_get(
    obj: Any,
    path,
    on_error: OnErrorType = raise_on_error,
    *,
    sep: str | Callable | None = None,
    key_transformer=None,
    get_value: Callable = get_attr_or_item,
    caught_errors=(Exception,),
):
    """
    Get elements of a mapping through a path to be called recursively.

    :param obj: The object to get the path from
    :param path: The path to get
    :param on_error: The error handler to use (default: raise_on_error)
    :param sep: Determines a path is transforms into a tuple of keys.
        If it's a string, ``lambda path: path.split(sep)`` is used.
        If not, it should be a function which takes in a path object and returns an iterable of keys.
    :param key_transformer: A function to transform the keys of the path
    :param get_value: A function to get the value of a key in a mapping
    :param caught_errors: The errors to catch (default: Exception)

    It will

    - split a path into keys (if sep is given, or if path is a string, will use '.' as a separator by default)

    - if key_transformer is given, apply to each key

    - consider string keys that are numeric as ints (convenient for lists)

    - get items also as attributes (attributes are checked for first for string keys)

    - catch all exceptions (that are subclasses of ``Exception``)

    >>> class A:
    ...      an_attribute = 42
    >>> path_get([1, [4, 5, {'a': A}], 3], [1, 2, 'a', 'an_attribute'])
    42

    By default, if ``path`` is a string, it will be split on ``sep``,
    which is ``'.'`` by default.

    >>> path_get([1, [4, 5, {'a': A}], 3], '1.2.a.an_attribute')
    42

    Note: The underlying function is ``_path_get``, but `path_get` has defaults and
    flexible input processing for more convenience.

    Note: ``path_get`` contains some ready-made ``OnErrorType`` functions in its
    attributes. For example, see how we can make ``path_get`` have the same behavior
    as ``dict.get`` by passing ``path_get.return_none_on_error`` as ``on_error``:

    >>> dd = path_get({}, 'no.keys', on_error=path_get.return_none_on_error)
    >>> dd is None
    True

    For example, ``path_get.raise_on_error``,
    ``path_get.return_none_on_error``, and ``path_get.return_empty_tuple_on_error``.

    """
    if sep is None:
        if isinstance(path, str):
            sep = "."
        else:
            sep = lambda x: x

    if isinstance(sep, str):
        sep_string = sep
        sep = lambda path: path.split(sep_string)
    else:
        assert callable(sep), f"sep should be a separator string, or callable: {sep=}"

    # Transform the path_to_keys further by applying key_transformer to each individual
    # key that path_to_keys (should) give(s) you
    if key_transformer is not None:
        # apply key_transformer to each key that sep(path) gives you
        _sep = sep
        sep = lambda path: map(key_transformer, _sep(path))

    return _path_get(
        obj,
        path,
        on_error=on_error,
        path_to_keys=sep,
        get_value=get_value,
        caught_errors=caught_errors,
    )


path_get.split_if_str = split_if_str
path_get.separate_keys_with_separator = separate_keys_with_separator
path_get.get_attr_or_item = get_attr_or_item
path_get.get_item = getitem
path_get.get_attr = getattr
path_get.keys_and_indices_path = keys_and_indices_path


# TODO: Make a transition plan for inversion of (obj, paths) order (not aligned with path_get!!)
@add_as_attribute_of(path_get)
def paths_getter(
    paths,
    obj=None,
    *,
    egress=dict,
    on_error: OnErrorType = raise_on_error,
    sep: str | Callable | None = None,
    key_transformer=None,
    get_value: Callable = get_attr_or_item,
    caught_errors=(Exception,),
):
    """
    Returns (path, values) pairs of the given paths in the given object.
    This is the "fan-out" version of ``path_get``, specifically designed to
    get multiple paths, returning the (path, value) pairs in a dict (by default),
    or via any pairs aggregator (``egress``) function.

    Note: For reasons who's clarity is burried in historical legacy, the order of
    obj and path are the opposite of path_get.

    :param paths: The paths to get
    :param obj: The object to get the paths from
    :param egress: The egress function to use (default: dict)
    :param on_error: The error handler to use (default: raise_on_error)
    :param sep: The separator to use if the path is a string
    :param key_transformer: A function to transform the keys of the path
    :param get_value: A function to get the value of a key in a mapping
    :param caught_errors: The errors to catch (default: Exception)

    >>> obj = {'a': {'b': 1, 'c': 2}, 'd': 3}
    >>> paths = ['a.c', 'd']
    >>> paths_getter(paths, obj=obj)
    {'a.c': 2, 'd': 3}
    >>> path_extractor = paths_getter(paths)
    >>> path_extractor(obj)
    {'a.c': 2, 'd': 3}

    See that the paths are used as the keys of the returned dict.
    If you want to specify your own keys, you can simply specify `paths` as a dict
    whose keys are the keys you want, and whose values are the paths to get:

    >>> path_extractor_2 = paths_getter({'california': 'a.c', 'dreaming': 'd'})
    >>> path_extractor_2(obj)
    {'california': 2, 'dreaming': 3}

    """
    kwargs = dict(
        on_error=on_error,
        sep=sep,
        key_transformer=key_transformer,
        get_value=get_value,
        caught_errors=caught_errors,
    )
    if obj is None:
        return partial(paths_getter, paths, egress=egress, **kwargs)

    if isinstance(paths, Mapping):

        def pairs():
            for key, path in paths.items():
                yield key, path_get(obj, path=path, **kwargs)

    else:

        def pairs():
            for path in paths:
                yield path, path_get(obj, path=path, **kwargs)

    return egress(pairs())


@add_as_attribute_of(path_get)
def chain_of_getters(
    getters: Iterable[Callable], obj=None, k=None, *, caught_errors=(Exception,)
):
    """If ``k`` is a string, tries to get ``k`` as an attribute of ``obj`` first,
    and if that fails, gets it as ``obj[k]``"""
    if obj is None and k is None:
        return partial(chain_of_getters, getters, caught_errors=caught_errors)
    for getter in getters:
        try:
            return getter(obj, k)
        except caught_errors:
            pass


@add_as_attribute_of(path_get)
def cast_to_int_if_numeric_str(k):
    if isinstance(k, str) and str.isnumeric(k):
        return int(k)
    return k


@add_as_attribute_of(path_get)
def _raise_on_error(d: Any):
    """Raise the error that was caught."""
    raise


@add_as_attribute_of(path_get)
def _return_none_on_error(d: Any):
    """Return None if an error was caught."""
    return None


@add_as_attribute_of(path_get)
def _return_empty_tuple_on_error(d: Any):
    """Return an empty tuple if an error was caught."""
    return ()


@add_as_attribute_of(path_get)
def _return_new_dict_on_error(d: Any):
    """Return a new dict if an error was caught."""
    return dict()


from dol.explicit import KeysReader


# TODO: Nothing particular about paths here. It's just a collection of keys
# (see dol.explicit.ExplicitKeys) with a key_to_value function.
# TODO: Yet another "explicit" pattern, found in dol.explicit, dol.sources
# (e.g. ObjReader), and which can (but perhaps not should) really be completely
# implemented with a value decoder (the getter) in a wrap_kvs over a {k: k...} mapping.
class PathMappedData(KeysReader):
    """
    A collection of keys with a key_to_value function to lazy load values.

    `PathMappedData` is particularly useful in cases where you want to have a mapping
    that lazy-loads values for keys from an explicit collection.

    Keywords: Lazy-evaluation, Mapping

    Args:
        data: The mapping to extract data from
        paths: The paths to extract data from the mapping

    Example::

    >>> data = {
    ...     'a': {
    ...         'b': [{'c': 1}, {'c': 2}],
    ...         'd': 'bar'
    ...     }
    ... }
    >>> paths = ['a.d', 'a.b.0.c']
    >>>
    >>> d = PathMappedData(data, paths)
    >>> list(d)
    ['a.d', 'a.b.0.c']
    >>> d['a.d']
    'bar'
    >>> d['a.b.0.c']
    1

    Now, data does contain a key path for 'a.b.1.c':

    >>> d.getter(d.src, 'a.b.1.c')
    2

    But since we didn't mention it in our paths parameter, it will raise a KeyError
    if we try to access it via the `PathMappedData` object:

    >>> d['a.b.1.c']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: "Key a.b.1.c was not found....key_collection attribute)"

    """

    def __init__(
        self,
        src: Mapping,
        key_collection,
        getter: Callable[[Mapping, Path], VT] = path_get,
        *,
        key_to_value: Callable[[Path], VT] = None,
    ) -> None:
        super().__init__(src, key_collection, getter)

    # def __getitem__(self, path: Path) -> VT:
    #     if path in self:
    #         return self.getter(self.data, path)
    #     else:
    #         raise KeyError(f'Path not found (in .paths attribute): {path}')

    # def __iter__(self) -> Iterator[Path]:
    #     yield from self.paths

    # def __len__(self) -> int:
    #     return len(self.paths)

    # def __contains__(self, path: Path) -> bool:
    #     return path in self.paths


# Note: Purposely didn't include any path validation to favor efficiency.
# Validation such as:
# if not key_path or not isinstance(key_path, Iterable):
#     raise ValueError(
#         f"Not a valid key path (should be an iterable with at least one element:"
#         f" {key_path}"
#     )
# TODO: Add possibility of producing different mappings according to the path/level.
#  For example, the new_mapping factory could be a list of factories, one for each
#  level, and/or take a path as an argument.
def path_set(
    d: Mapping,
    key_path: Iterable[KT],
    val: VT,
    *,
    sep: str = ".",
    new_mapping: Callable[[], VT] = dict,
):
    """
    Sets a val to a path of keys.

    :param d: The mapping to set the value in
    :param key_path: The path of keys to set the value to
    :param val: The value to set
    :param sep: The separator to use if the path is a string
    :param new_mapping: callable that returns a new mapping to use when key is not found
    :return:

    >>> d = {'a': 1, 'b': {'c': 2}}
    >>> path_set(d, ['b', 'e'], 42)
    >>> d
    {'a': 1, 'b': {'c': 2, 'e': 42}}

    >>> input_dict = {
    ...   "a": {
    ...     "c": "val of a.c",
    ...     "b": 1,
    ...   },
    ...   "10": 10,
    ...   "b": {
    ...     "B": {
    ...       "AA": 3
    ...     }
    ...   }
    ... }
    >>>
    >>> path_set(input_dict, ('new', 'key', 'path'), 7)
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7}}}

    You can also use a string as a path, with a separator:

    >>> path_set(input_dict, 'new/key/old/path', 8, sep='/')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': {'path': 7, 'old': {'path': 8}}}}

    If you specify a string path and a non-None separator, the separator will be used
    to split the string into a list of keys. The default separator is ``sep='.'``.

    >>> path_set(input_dict, 'new.key', 'new val')
    >>> input_dict  # doctest: +NORMALIZE_WHITESPACE
    {'a': {'c': 'val of a.c', 'b': 1}, '10': 10, 'b': {'B': {'AA': 3}},
    'new': {'key': 'new val'}}

    You can also specify a different ``new_mapping`` factory, which will be used to
    create new mappings when a key is missing. The default is ``dict``.

    >>> from collections import OrderedDict
    >>> input_dict = {}
    >>> path_set(input_dict, 'new.key', 42, new_mapping=OrderedDict)
    >>> input_dict  # doctest: +ELLIPSIS
    {'new': OrderedDict(...'key'...42...)}

    """
    if isinstance(key_path, str) and sep is not None:
        key_path = key_path.split(sep)

    first_key, *remaining_keys = key_path
    if len(key_path) == 1:  # base case
        d[first_key] = val
    else:
        if first_key not in d:
            d[first_key] = new_mapping()
        path_set(d[first_key], remaining_keys, val)


# TODO: Nice to have: Edits can be a nested dict, not necessarily a flat path-value one.
Edits = Union[Mapping[Path, VT], Iterable[tuple[Path, VT]]]


def path_edit(d: Mapping, edits: Edits = ()) -> Mapping:
    """Make a series of (in place) edits to a Mapping, specifying `(path, value)` pairs.


    Args:
        d (Mapping): The mapping to edit.
        edits: An iterable of ``(path, value)`` tuples, or ``path: value`` Mapping.

    Returns:
        Mapping: The edited mapping.

    >>> d = {'a': 1}
    >>> path_edit(d, [(['b', 'c'], 2), ('d.e.f', 3)])
    {'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}

    Changes happened also inplace (so if you don't want that, make a deepcopy first):

    >>> d
    {'a': 1, 'b': {'c': 2}, 'd': {'e': {'f': 3}}}

    You can also pass a dict of edits.

    >>> path_edit(d, {'a': 4, 'd.e.f': 5})
    {'a': 4, 'b': {'c': 2}, 'd': {'e': {'f': 5}}}

    """

    if isinstance(edits, Mapping):
        edits = list(edits.items())
    for path, value in edits:
        path_set(d, path, value)
    return d


from dol.base import kv_walk


PT = TypeVar("PT")  # Path Type
PkvFilt = Callable[[PT, KT, VT], bool]


#
def path_filter(
    pkv_filt: PkvFilt,
    d: Mapping,
    *,
    leafs_only: bool = True,
    breadth_first: bool = False,
) -> Iterator[PT]:
    """Walk a dict, yielding paths to values that pass the ``pkv_filt``

    :param pkv_filt: A function that takes a path, key, and value, and returns
        ``True`` if the path should be yielded, and ``False`` otherwise
    :param d: The ``Mapping`` to walk (scan through)
    :param leafs_only: Whether to yield only paths to leafs (default), or to yield
        paths to all values that pass the ``pkv_filt``.
    :param breadth_first: Whether to perform breadth-first traversal
        (instead of the default depth-first traversal).
    :return: An iterator of paths to values that pass the ``pkv_filt``

    Example::

    >>> d = {'a': {'b': {'c': 1, 'd': 2}, 'e': 3}}
    >>> list(path_filter(lambda p, k, v: v == 2, d))
    [('a', 'b', 'd')]

    >>> mm = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'meaning of life'}},
    ...     'aaa': {'bbb': 314},
    ... }
    >>> return_path_if_int_leaf = lambda p, k, v: (p, v) if isinstance(v, int) else None
    >>> paths = list(path_filter(return_path_if_int_leaf, mm))
    >>> paths  # only the paths to the int leaves are returned
    [('a', 'b', 'c'), ('aaa', 'bbb')]

    The ``pkv_filt`` argument can use path, key, and/or value to define your search
    query. For example, let's extract all the paths that have depth at least 3.

    >>> paths = list(path_filter(lambda p, k, v: len(p) >= 3, mm))
    >>> paths
    [('a', 'b', 'c'), ('aa', 'bb', 'cc')]

    The rationale for ``path_filter`` yielding matching paths, and not values or keys,
    is that if you have the paths, you can than get the keys and values with them,
    using ``path_get``.

    >>> from functools import partial, reduce
    >>> path_get = lambda m, k: reduce(lambda m, k: m[k], k, m)
    >>> extract_paths = lambda m, paths: map(partial(path_get, m), paths)
    >>> vals = list(extract_paths(mm, paths))
    >>> vals
    [42, 'meaning of life']

    Note: pkv_filt is first to match the order of the arguments of the
    builtin filter function.
    """
    _leaf_yield = partial(_path_matcher_leaf_yield, pkv_filt, None)
    kwargs = dict(leaf_yield=_leaf_yield, breadth_first=breadth_first)
    if not leafs_only:
        kwargs["branch_yield"] = _leaf_yield
    walker = kv_walk(d, **kwargs)
    yield from filter(None, walker)


# backwards compatibility quasi-alias (arguments are flipped)
def search_paths(
    d: Mapping,
    pkv_filt: PkvFilt,
    *,
    leafs_only: bool = True,
    breadth_first: bool = False,
) -> Iterator[PT]:
    """backwards compatibility quasi-alias (arguments are flipped)
    Use path_filter instead, since search_paths will be deprecated.
    """
    return path_filter(pkv_filt, d, leafs_only=leafs_only, breadth_first=breadth_first)


def _path_matcher_leaf_yield(pkv_filt: PkvFilt, sentinel, p: PT, k: KT, v: VT):
    """Helper to make (picklable) leaf_yields for paths_matching (through partial)"""
    if pkv_filt(p, k, v):
        return p
    else:
        return sentinel


@add_as_attribute_of(path_filter)
def _mk_path_matcher(pkv_filt: PkvFilt, sentinel=None):
    """Make a leaf_yield that only yields paths that pass the pkv_filt,
    and a sentinel (by default, ``None``) otherwise"""
    return partial(_path_matcher_leaf_yield, pkv_filt, sentinel)


@add_as_attribute_of(path_filter)
def _mk_pkv_filt(
    filt: Callable[[PT | KT | VT], bool], kind: Literal["path", "key", "value"]
) -> PkvFilt:
    """pkv_filt based on a ``filt`` that matches EITHER path, key, or value."""
    return partial(_pkv_filt, filt, kind)


def _pkv_filt(
    filt: Callable[[PT | KT | VT], bool],
    kind: Literal["path", "key", "value"],
    p: PT,
    k: KT,
    v: VT,
):
    """Helper to make (picklable) pkv_filt based on a ``filt`` that matches EITHER
    path, key, or value."""
    if kind == "path":
        return filt(p)
    elif kind == "key":
        return filt(k)
    elif kind == "value":
        return filt(v)
    else:
        raise ValueError(f"Invalid kind: {kind}")


@dataclass
class KeyPath:
    """
    A key mapper that converts from an iterable key (default tuple) to a string
    (given a path-separator str)

    Args:
        path_sep: The path separator (used to make string paths from iterable paths and
            visa versa
        _path_type: The type of the outcoming (inner) path. But really, any function to
        convert from a list to
            the outer path type we want.

    With ``'/'`` as a separator:

    >>> kp = KeyPath(path_sep='/')
    >>> kp._key_of_id(('a', 'b', 'c'))
    'a/b/c'
    >>> kp._id_of_key('a/b/c')
    ('a', 'b', 'c')

    With ``'.'`` as a separator:

    >>> kp = KeyPath(path_sep='.')
    >>> kp._key_of_id(('a', 'b', 'c'))
    'a.b.c'
    >>> kp._id_of_key('a.b.c')
    ('a', 'b', 'c')
    >>> kp = KeyPath(path_sep=':::', _path_type=dict.fromkeys)
    >>> _id = dict.fromkeys('abc')
    >>> _id
    {'a': None, 'b': None, 'c': None}
    >>> kp._key_of_id(_id)
    'a:::b:::c'
    >>> kp._id_of_key('a:::b:::c')
    {'a': None, 'b': None, 'c': None}

    Calling a ``KeyPath`` instance on a store wraps it so we can have path access to
    it.

    >>> s = {'a': {'b': {'c': 42}}}
    >>> s['a']['b']['c']
    42
    >>> # Now let's wrap the store
    >>> s = KeyPath('.')(s)
    >>> s['a.b.c']
    42
    >>> s['a.b.c'] = 3.14
    >>> s['a.b.c']
    3.14
    >>> del s['a.b.c']
    >>> s
    {'a': {'b': {}}}

    Note: ``KeyPath`` enables you to read with paths when all the keys of the paths
    are valid (i.e. have a value), but just as with a ``dict``, it will not create
    intermediate nested values for you (as for example, you could make for yourself
    using  ``collections.defaultdict``).

    """

    path_sep: str = path_sep
    _path_type: type | Callable = tuple

    def _key_of_id(self, _id):
        if not isinstance(_id, str):
            return self.path_sep.join(_id)
        else:
            return _id

    def _id_of_key(self, k):
        return self._path_type(k.split(self.path_sep))

    def __call__(self, store):
        path_accessible_store = add_path_access(store, path_type=self._path_type)
        return kv_wrap(self)(path_accessible_store)


# ------------------------------------------------------------------------------


class PrefixRelativizationMixin:
    """
    Mixin that adds a intercepts the _id_of_key an _key_of_id methods, transforming absolute keys to relative ones.
    Designed to work with string keys, where absolute and relative are relative to a _prefix attribute
    (assumed to exist).
    The cannonical use case is when keys are absolute file paths, but we want to identify data through relative paths.
    Instead of referencing files through an absolute path such as
        /A/VERY/LONG/ROOT/FOLDER/the/file/we.want
    we can instead reference the file as
        the/file/we.want

    Note though, that PrefixRelativizationMixin can be used, not only for local paths,
    but when ever a string reference is involved.
    In fact, not only strings, but any key object that has a __len__, __add__, and subscripting.

    When subclassed, should be placed before the class defining _id_of_key an _key_of_id.
    Also, assumes that a (string) _prefix attribute will be available.

    >>> from dol.base import Store
    >>> from collections import UserDict
    >>>
    >>> class MyStore(PrefixRelativizationMixin, Store):
    ...     def __init__(self, store, _prefix='/root/of/data/'):
    ...         super().__init__(store)
    ...         self._prefix = _prefix
    ...
    >>> s = MyStore(store=dict())  # using a dict as our store
    >>> s['foo'] = 'bar'
    >>> assert s['foo'] == 'bar'
    >>> s['too'] = 'much'
    >>> assert list(s.keys()) == ['foo', 'too']
    >>> # Everything looks normal, but are the actual keys behind the hood?
    >>> s._id_of_key('foo')
    '/root/of/data/foo'
    >>> # see when iterating over s.items(), we get the interface view:
    >>> list(s.items())
    [('foo', 'bar'), ('too', 'much')]
    >>> # but if we ask the store we're actually delegating the storing to, we see what the keys actually are.
    >>> s.store.items()
    dict_items([('/root/of/data/foo', 'bar'), ('/root/of/data/too', 'much')])
    """

    _prefix_attr_name = "_prefix"

    @lazyprop
    def _prefix_length(self):
        return len(getattr(self, self._prefix_attr_name))

    def _id_of_key(self, k):
        return getattr(self, self._prefix_attr_name) + k

    def _key_of_id(self, _id):
        return _id[self._prefix_length :]


class PrefixRelativization(PrefixRelativizationMixin):
    """A key wrap that allows one to interface with absolute paths through relative paths.
    The original intent was for local files. Instead of referencing files through an absolute path such as:

        */A/VERY/LONG/ROOT/FOLDER/the/file/we.want*

    we can instead reference the file as:

        *the/file/we.want*

    But PrefixRelativization can be used, not only for local paths, but when ever a string reference is involved.
    In fact, not only strings, but any key object that has a __len__, __add__, and subscripting.
    """

    def __init__(self, _prefix=""):
        self._prefix = _prefix


class ExplicitKeysWithPrefixRelativization(PrefixRelativizationMixin, Store):
    """
    dol.base.Keys implementation that gets it's keys explicitly from a collection given at initialization time.
    The key_collection must be a collections.abc.Collection (such as list, tuple, set, etc.)

    >>> from dol.base import Store
    >>> s = ExplicitKeysWithPrefixRelativization(key_collection=['/root/of/foo', '/root/of/bar', '/root/for/alice'])
    >>> keys = Store(store=s)
    >>> 'of/foo' in keys
    True
    >>> 'not there' in keys
    False
    >>> list(keys)
    ['of/foo', 'of/bar', 'for/alice']
    """

    __slots__ = ("_key_collection",)

    def __init__(self, key_collection, _prefix=None):
        # TODO: Find a better way to avoid the circular import
        from dol.explicit import ExplicitKeys  # here to avoid circular imports

        if _prefix is None:
            _prefix = max_common_prefix(key_collection)
        store = ExplicitKeys(key_collection=key_collection)
        self._prefix = _prefix
        super().__init__(store=store)


@store_decorator
def mk_relative_path_store(
    store_cls=None,
    *,
    name=None,
    with_key_validation=False,
    prefix_attr="_prefix",
):
    """

    Args:
        store_cls: The base store to wrap (subclass)
        name: The name of the new store (by default 'RelPath' + store_cls.__name__)
        with_key_validation: Whether keys should be validated upon access (store_cls must have an is_valid_key method

    Returns: A new class that uses relative paths (i.e. where _prefix is automatically added to incoming keys,
        and the len(_prefix) first characters are removed from outgoing keys.

    >>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
    >>> # -- but don't just believe the static dogmas).
    >>> MyStore = mk_relative_path_store(dict)  # wrap our favorite store: A dict.
    >>> s = MyStore()  # make such a store
    >>> s._prefix = '/ROOT/'
    >>> s['foo'] = 'bar'
    >>> dict(s.items())  # gives us what you would expect
    {'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(s.store)
    {'/ROOT/foo': 'bar'}
    >>>
    >>> # The static way: Make a class that will integrate the _prefix at construction time.
    >>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
    ...     def __init__(self, _prefix, *args, **kwargs):
    ...         self._prefix = _prefix

    You can choose the name you want that prefix to have as an attribute (we'll still make
    a hidden '_prefix' attribute for internal use, but at least you can have an attribute with the
    name you want.

    >>> MyRelStore = mk_relative_path_store(dict, prefix_attr='rootdir')
    >>> s = MyRelStore()
    >>> s.rootdir = '/ROOT/'

    >>> s['foo'] = 'bar'
    >>> dict(s.items())  # gives us what you would expect
    {'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(s.store)
    {'/ROOT/foo': 'bar'}

    """
    # name = name or ("RelPath" + store_cls.__name__)
    # __module__ = __module__ or getattr(store_cls, "__module__", None)

    if name is not None:
        from warnings import warn

        warn(
            f"The use of name argumment is deprecated. Use __name__ instead",
            DeprecationWarning,
        )

    cls = type(store_cls.__name__, (PrefixRelativizationMixin, Store), {})

    @wraps(store_cls.__init__)
    def __init__(self, *args, **kwargs):
        Store.__init__(self, store=store_cls(*args, **kwargs))
        prefix = recursive_get_attr(self.store, prefix_attr, "")
        setattr(
            self, prefix_attr, prefix
        )  # TODO: Might need descriptor to enable assignment

    cls.__init__ = __init__

    if prefix_attr != "_prefix":
        assert not hasattr(store_cls, "_prefix"), (
            f"You already have a _prefix attribute, "
            f"but want the prefix name to be {prefix_attr}. "
            f"That's not going to be easy for me."
        )

        # if not hasattr(cls, prefix_attr):
        #     warn(f"You said you wanted prefix_attr='{prefix_attr}', "
        #          f"but {cls} (the wrapped class) doesn't have a '{prefix_attr}'. "
        #          f"I'll let it slide because perhaps the attribute is dynamic. But I'm warning you!!")

        @property
        def _prefix(self):
            return getattr(self, prefix_attr)

        cls._prefix = _prefix

    if with_key_validation:
        assert hasattr(store_cls, "is_valid_key"), (
            "If you want with_key_validation=True, "
            "you'll need a method called is_valid_key to do the validation job"
        )

        def _id_of_key(self, k):
            _id = super(cls, self)._id_of_key(k)
            if self.store.is_valid_key(_id):
                return _id
            else:
                raise KeyError(
                    f"Key not valid (usually because does not exist or access not permitted): {k}"
                )

        cls._id_of_key = _id_of_key

    # if __module__ is not None:
    #     cls.__module__ = __module__

    # print(callable(cls))

    return cls


# TODO: Intended to replace the init-less PrefixRelativizationMixin
#  (but should change name if so, since Mixins shouldn't have inits)
class RelativePathKeyMapper:
    def __init__(self, prefix):
        self._prefix = prefix
        self._prefix_length = len(self._prefix)

    def _id_of_key(self, k):
        return self._prefix + k

    def _key_of_id(self, _id):
        return _id[self._prefix_length :]


@store_decorator
def prefixless_view(store=None, *, prefix=None):
    key_mapper = RelativePathKeyMapper(prefix)
    return wrap_kvs(
        store, id_of_key=key_mapper._id_of_key, key_of_id=key_mapper._key_of_id
    )


def _fallback_startswith(iterable, prefix):
    """Returns True iff iterable starts with prefix.
    Compares the first items of iterable and prefix iteratively.
    It can be terribly inefficient though, so it's best to use it only when you have to.
    """
    iter_iterable = iter(iterable)
    iter_prefix = iter(prefix)

    for prefix_item in iter_prefix:
        try:
            # Get the next item from iterable
            item = next(iter_iterable)
        except StopIteration:
            # If we've reached the end of iterable, return False
            return False

        if item != prefix_item:
            # If any pair of items are unequal, return False
            return False

    # If we've checked every item in prefix without returning, return True
    return True


# TODO: Routing pattern. Make plugin architecture.
# TODO: Add faster option for lists and tuples that are sizable and sliceable
def _startswith(iterable, prefix):
    """Returns True iff iterable starts with prefix.
    If prefix is a string, `str.startswith` is used, otherwise, the function
    will compare the first items of iterable and prefix iteratively.

    >>> _startswith('apple', 'app')
    True
    >>> _startswith('crapple', 'app')
    False
    >>> _startswith([1,2,3,4], [1,2])
    True
    >>> _startswith([0, 1,2,3,4], [1,2])
    False
    >>> _startswith([1,2,3,4], [])
    True
    """
    if isinstance(prefix, str):
        return iterable.startswith(prefix)
    else:
        return _fallback_startswith(iterable, prefix)


def _prefix_filter(store, prefix: str):
    """Filter the store to have only keys that start with prefix"""
    return filt_iter(store, filt=partial(_startswith, prefix=prefix))


def _prefix_filter_with_relativization(store, prefix: str):
    """Filter the store to have only keys that start with prefix"""
    return prefixless_view(_prefix_filter(store, prefix), prefix=prefix)


@store_decorator
def add_prefix_filtering(store=None, *, relativize_prefix: bool = False):
    """Add prefix filtering to a store.

    >>> d = {'a/b': 1, 'a/c': 2, 'd/e': 3, 'f': 4}
    >>> s = add_prefix_filtering(d)
    >>> assert s['a/'] == {'a/b': 1, 'a/c': 2}

    Demo usage on a `Mapping` type:

    >>> from collections import UserDict
    >>> D = add_prefix_filtering(UserDict)
    >>> s = D(d)
    >>> assert s['a/'] == {'a/b': 1, 'a/c': 2}

    """
    __prefix_filter = _prefix_filter
    if relativize_prefix:
        __prefix_filter = _prefix_filter_with_relativization
    return add_missing_key_handling(store, missing_key_callback=__prefix_filter)


@store_decorator
def handle_prefixes(
    store=None,
    *,
    prefix=None,
    filter_prefix: bool = True,
    relativize_prefix: bool = True,
    default_prefix="",
):
    """A store decorator that handles prefixes.

    If aggregates several prefix-related functionalities. It will (by default)

    - Filter the store so that only the keys starting with given prefix are accessible.

    - Relativize the keys (provide a view where the prefix is removed from the keys)

    Args:
        store: The store to wrap
        prefix: The prefix to use. If None and the store is an instance (not type),
                will take the longest common prefix as the prefix.
        filter_prefix: Whether to filter out keys that don't start with the prefix
        relativize_prefix: Whether to relativize the prefix
        default_prefix: The default prefix to use if no prefix is given and the store
                        is a type (not instance)

    >>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
    >>> dd = handle_prefixes(d, prefix='/ROOT/of/')
    >>> dd['foo'] = 'bar'
    >>> dict(dd.items())  # gives us what you would expect
    {'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
    >>> dict(dd.store)  # but see where the underlying store actually wrote 'bar':
    {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}

    """
    if prefix is None:
        if isinstance(store, type):
            raise TypeError(
                f"I can only infer prefix from a store instance, not a type: {store}"
            )
        prefix = max_common_prefix(store, default=default_prefix)
    if filter_prefix:
        store = filt_iter(store, filt=lambda k: k.startswith(prefix))
    if relativize_prefix:
        store = prefixless_view(store, prefix=prefix)
    return store


# TODO: Enums introduce a ridiculous level of complexity here.
#  Learn them of remove them!!

from dol.naming import StrTupleDict
from enum import Enum


class PathKeyTypes(Enum):
    str = "str"
    dict = "dict"
    tuple = "tuple"
    namedtuple = "namedtuple"


path_key_type_for_type = {
    str: PathKeyTypes.str,
    dict: PathKeyTypes.dict,
    tuple: PathKeyTypes.tuple,
}

_method_names_for_path_type = {
    PathKeyTypes.str: {
        "_id_of_key": StrTupleDict.simple_str_to_str,
        "_key_of_id": StrTupleDict.str_to_simple_str,
    },
    PathKeyTypes.dict: {
        "_id_of_key": StrTupleDict.dict_to_str,
        "_key_of_id": StrTupleDict.str_to_dict,
    },
    PathKeyTypes.tuple: {
        "_id_of_key": StrTupleDict.tuple_to_str,
        "_key_of_id": StrTupleDict.str_to_tuple,
    },
    PathKeyTypes.namedtuple: {
        "_id_of_key": StrTupleDict.namedtuple_to_str,
        "_key_of_id": StrTupleDict.str_to_namedtuple,
    },
}


#
# def str_to_simple_str(self, s: str):
#     return self.sep.join(*self.str_to_tuple(s))
#
#
# def simple_str_to_str(self, ss: str):
#     self.tuple_to_str(self.si)


# TODO: Add key and id type validation
def str_template_key_trans(
    template: str,
    key_type: PathKeyTypes | type,
    format_dict=None,
    process_kwargs=None,
    process_info_dict=None,
    named_tuple_type_name="NamedTuple",
    sep: str = path_sep,
):
    """Make a key trans object that translates from a string _id to a dict, tuple, or namedtuple key (and back)"""

    assert (
        key_type in PathKeyTypes
    ), f"key_type was {key_type}. Needs to be one of these: {', '.join(PathKeyTypes)}"

    class PathKeyMapper(StrTupleDict): ...

    setattr(
        PathKeyMapper,
        "_id_of_key",
        _method_names_for_path_type[key_type]["_id_of_key"],
    )
    setattr(
        PathKeyMapper,
        "_key_of_id",
        _method_names_for_path_type[key_type]["_key_of_id"],
    )

    key_trans = PathKeyMapper(
        template,
        format_dict,
        process_kwargs,
        process_info_dict,
        named_tuple_type_name,
        sep,
    )

    return key_trans


str_template_key_trans.method_names_for_path_type = _method_names_for_path_type
str_template_key_trans.key_types = PathKeyTypes


# TODO: Merge with mk_relative_path_store
def rel_path_wrap(o, _prefix):
    """
    Args:
        o: An object to be wrapped
        _prefix: The _prefix to use for key wrapping (will remove it from outcoming keys and add to ingoing keys.

    >>> # The dynamic way (if you try this at home, be aware of the pitfalls of the dynamic way
    >>> # -- but don't just believe the static dogmas).
    >>> d = {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0}
    >>> dd = rel_path_wrap(d, '/ROOT/of/')
    >>> dd['foo'] = 'bar'
    >>> dict(dd.items())  # gives us what you would expect
    {'every/thing': 42, 'this/too': 0, 'foo': 'bar'}
    >>>  # but under the hood, the dict we wrapped actually contains the '/ROOT/' prefix
    >>> dict(dd.store)
    {'/ROOT/of/every/thing': 42, '/ROOT/of/this/too': 0, '/ROOT/of/foo': 'bar'}
    >>>
    >>> # The static way: Make a class that will integrate the _prefix at construction time.
    >>> class MyStore(mk_relative_path_store(dict)):  # Indeed, mk_relative_path_store(dict) is a class you can subclass
    ...     def __init__(self, _prefix, *args, **kwargs):
    ...         self._prefix = _prefix

    """

    from dol import kv_wrap

    trans_obj = RelativePathKeyMapper(_prefix)
    return kv_wrap(trans_obj)(o)


# mk_relative_path_store_cls = mk_relative_path_store  # alias

## Alternative to mk_relative_path_store that doesn't make lint complain (but the repr shows MyStore, not name)
# def mk_relative_path_store_alt(store_cls, name=None):
#     if name is None:
#         name = 'RelPath' + store_cls.__name__
#
#     class MyStore(PrefixRelativizationMixin, Store):
#         @wraps(store_cls.__init__)
#         def __init__(self, *args, **kwargs):
#             super().__init__(store=store_cls(*args, **kwargs))
#             self._prefix = self.store._prefix
#     MyStore.__name__ = name
#
#     return MyStore


## Alternative to StrTupleDict (staging here for now, but should replace when ready)

import re
import string
from collections import namedtuple
from functools import wraps


def _return_none_if_none_input(func):
    """Wraps a method function, making it return `None` if the input is `None`.

    (More precisely, it will return `None` if the first (non-instance) input is `None`.

    >>> class Foo:
    ...     @_return_none_if_none_input
    ...     def bar(self, x, y=1):
    ...         return x + y
    >>> foo = Foo()
    >>> foo.bar(2)
    3
    >>> assert foo.bar(None) is None
    >>> assert foo.bar(x=None) is None

    Note: On the other hand, this will not return `None`, but should:
    ``foo.bar(y=3, x=None)``. To achieve this, we'd need to look into the signature,
    which seems like overkill and I might not want that systematic overhead in my
    methods.
    """

    @wraps(func)
    def _func(self, *args, **kwargs):
        if args and args[0] is None:
            return None
        elif kwargs and next(iter(kwargs.values())) is None:
            return None
        else:
            return func(self, *args, **kwargs)

    return _func


from typing import Tuple
from collections.abc import Iterable

string_formatter = string.Formatter()


def string_unparse(parsing_result: Iterable[tuple[str, str, str, str]]):
    """The inverse of string.Formatter.parse

    Will ravel

    >>> import string
    >>> formatter = string.Formatter()
    >>> string_unparse(formatter.parse('literal{name!c:spec}'))
    'literal{name!c:spec}'
    """
    reconstructed = ""
    for literal_text, field_name, format_spec, conversion in parsing_result:
        reconstructed += literal_text
        if field_name is not None:
            field = f"{{{field_name}"
            if conversion:
                assert (
                    len(conversion) == 1
                ), f"conversion can only be a single character: {conversion=}"
                field += f"!{conversion}"
            if format_spec:
                field += f":{format_spec}"
            field += "}"
            reconstructed += field
    return reconstructed


def _field_names(string_template):
    """
    Returns the field names in a string template.

    >>> _field_names("{name} is {age} years old.")
    ('name', 'age')
    """
    parsing_result = string_formatter.parse(string_template)
    return tuple(
        field_name for _, field_name, _, _ in parsing_result if field_name is not None
    )


def identity(x):
    return x


from dol.trans import KeyCodec, filt_iter
from inspect import signature

# Codec = namedtuple('Codec', 'encoder decoder')
FieldTypeNames = Literal["str", "dict", "tuple", "namedtuple", "simple_str", "single"]


# TODO: Make and use _return_none_if_none_input or not?
# TODO: Change to dataclass with 3.10+ (to be able to do KW_ONLY)
# TODO: Should be refactored and generalized to be able to automatically handle
#   all combinations of FieldTypeNames (and possibly open-close these as well?)
#   It's a "path finder" meshed pattern.
# TODO: Do we really want to allow field_patterns to be included in the template (the `{name:pattern}` options)?
#  Normally, this is used for string GENERATION as `{name:format}`, which is still useful for us here too.
#  The counter argument is that the main usage of KeyTemplate is not actually
#  generation, but extraction. Further, the format language is not as general as simply
#  using a format_field = {field: cast_function, ...} argument.
#  My decision would be to remove any use of the `{name:X}` form in the base class,
#  and have classmethods specialized for short-hand versions that use `name:regex` or
#  `name:format`, ...
class KeyTemplate:
    r"""A class for parsing and generating keys based on a template.

    Args:
        template: A template string with fields to be extracted or filled in.
        field_patterns: A dictionary of field names and their regex patterns.
        simple_str_sep: A separator string for simple strings (i.e. strings without
            fields).
        namedtuple_type_name: The name of the namedtuple type to use for namedtuple
            fields.
        dflt_pattern: The default pattern to use for fields that don't have a pattern
            specified.
        to_str_funcs: A dictionary of field names and their functions to convert them
            to strings.
        from_str_funcs: A dictionary of field names and their functions to convert
            them from strings.

    Examples:

    >>> st = KeyTemplate(
    ...     'root/{name}/v_{version}.json',
    ...     field_patterns={'version': r'\d+'},
    ...     from_str_funcs={'version': int},
    ... )

    And now you have a template that can be used to convert between various
    representations of the template: You can extract fields from strings, generate
    strings from fields, etc.

    >>> st.str_to_dict("root/dol/v_9.json")
    {'name': 'dol', 'version': 9}
    >>> st.dict_to_str({'name': 'meshed', 'version': 42})
    'root/meshed/v_42.json'
    >>> st.dict_to_tuple({'name': 'meshed', 'version': 42})
    ('meshed', 42)
    >>> st.tuple_to_dict(('i2', 96))
    {'name': 'i2', 'version': 96}
    >>> st.str_to_tuple("root/dol/v_9.json")
    ('dol', 9)
    >>> st.tuple_to_str(('front', 11))
    'root/front/v_11.json'
    >>> st.str_to_namedtuple("root/dol/v_9.json")
    NamedTuple(name='dol', version=9)
    >>> st.str_to_simple_str("root/dol/v_9.json")
    'dol,9'
    >>> st_clone = st.clone(simple_str_sep='/')
    >>> st_clone.str_to_simple_str("root/dol/v_9.json")
    'dol/9'


    With ``st.key_codec``, you can make a ``KeyCodec`` for the given source (decoded)
    and target (encoded) types.
    A `key_codec` is a codec; it has an encoder and a decoder.

    >>> key_codec = st.key_codec('tuple', 'str')
    >>> encoder, decoder = key_codec
    >>> decoder('root/dol/v_9.json')
    ('dol', 9)
    >>> encoder(('dol', 9))
    'root/dol/v_9.json'

    If you have a ``Mapping``, you can use ``key_codec`` as a decorator to wrap
    the mapping with a key mappings.

    >>> store = {
    ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
    ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
    ... }
    >>>
    >>> accessor = key_codec(store)
    >>> list(accessor)
    [('meshed', 151), ('dol', 9)]
    >>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
    >>> list(store)
    ['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
    >>> store['root/i2/v_4.json']
    '{"downloads": 274, "type": "utility"}'

    Note: If your store contains keys that don't fit the format, key_codec will
    raise a ``ValueError``. To remedy this, you can use the ``st.filt_iter`` to
    filter out keys that don't fit the format, before you wrap the store with
    ``st.key_codec``.

    >>> store = {
    ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
    ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
    ...     'root/not/the/right/format': "something else"
    ... }
    >>> accessor = st.filt_iter('str')(store)
    >>> list(accessor)
    ['root/meshed/v_151.json', 'root/dol/v_9.json']
    >>> accessor = st.key_codec('tuple', 'str')(st.filt_iter('str')(store))
    >>> list(accessor)
    [('meshed', 151), ('dol', 9)]
    >>> accessor['dol', 9]
    '{"downloads": 132, "type": "utility"}'

    You can also ask any (handled) combination of field types:

    >>> key_codec = st.key_codec('tuple', 'dict')
    >>> key_codec.encoder(('i2', 96))
    {'name': 'i2', 'version': 96}
    >>> key_codec.decoder({'name': 'fantastic', 'version': 4})
    ('fantastic', 4)

    """

    _formatter = string_formatter

    def __init__(
        self,
        template: str,
        *,
        field_patterns: dict = None,
        to_str_funcs: dict = None,
        from_str_funcs: dict = None,
        simple_str_sep: str = ",",
        namedtuple_type_name: str = "NamedTuple",
        dflt_pattern: str = ".*",
        dflt_field_name: Callable[[str], str] = "i{:02.0f}_".format,
        normalize_paths: bool = False,
    ):
        self._init_kwargs = dict(
            template=template,
            field_patterns=field_patterns,
            to_str_funcs=to_str_funcs,
            from_str_funcs=from_str_funcs,
            simple_str_sep=simple_str_sep,
            namedtuple_type_name=namedtuple_type_name,
            dflt_pattern=dflt_pattern,
            dflt_field_name=dflt_field_name,
        )
        self._original_template = template
        self.simple_str_sep = simple_str_sep
        self.namedtuple_type_name = namedtuple_type_name
        self.dflt_pattern = dflt_pattern
        self.dflt_field_name = dflt_field_name

        (
            self.template,
            self._fields,
            _to_str_funcs,
            field_patterns_,
        ) = self._extract_template_info(template)

        self._field_patterns = dict(
            {field: self.dflt_pattern for field in self._fields},
            **dict(field_patterns_, **(field_patterns or {})),
        )
        self._to_str_funcs = dict(
            {field: str for field in self._fields},
            **dict(_to_str_funcs, **(to_str_funcs or {})),
        )
        self._from_str_funcs = dict(
            {field: identity for field in self._fields}, **(from_str_funcs or {})
        )
        self._n_fields = len(self._fields)
        self._regex = self._compile_regex(self.template, normalize_path=normalize_paths)

    def clone(self, **kwargs):
        return type(self)(**{**self._init_kwargs, **kwargs})

    clone.__signature__ = signature(__init__)

    def key_codec(
        self, decoded: FieldTypeNames = "tuple", encoded: FieldTypeNames = "str"
    ):
        r"""Makes a ``KeyCodec`` for the given source and target types.

        >>> st = KeyTemplate(
        ...     'root/{name}/v_{version}.json',
        ...     field_patterns={'version': r'\d+'},
        ...     from_str_funcs={'version': int},
        ... )

        A `key_codec` is a codec; it has an encoder and a decoder.

        >>> key_codec = st.key_codec('tuple', 'str')
        >>> encoder, decoder = key_codec
        >>> decoder('root/dol/v_9.json')
        ('dol', 9)
        >>> encoder(('dol', 9))
        'root/dol/v_9.json'

        If you have a ``Mapping``, you can use ``key_codec`` as a decorator to wrap
        the mapping with a key mappings.

        >>> store = {
        ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
        ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
        ... }
        >>>
        >>> accessor = key_codec(store)
        >>> list(accessor)
        [('meshed', 151), ('dol', 9)]
        >>> accessor['i2', 4] = '{"downloads": 274, "type": "utility"}'
        >>> list(store)
        ['root/meshed/v_151.json', 'root/dol/v_9.json', 'root/i2/v_4.json']
        >>> store['root/i2/v_4.json']
        '{"downloads": 274, "type": "utility"}'

        Note: If your store contains keys that don't fit the format, key_codec will
        raise a ``ValueError``. To remedy this, you can use the ``st.filt_iter`` to
        filter out keys that don't fit the format, before you wrap the store with
        ``st.key_codec``.

        """
        self._assert_field_type(decoded, "decoded")
        self._assert_field_type(encoded, "encoded")
        coder = getattr(self, f"{decoded}_to_{encoded}")
        decoder = getattr(self, f"{encoded}_to_{decoded}")
        return KeyCodec(coder, decoder)

    def filt_iter(self, field_type: FieldTypeNames = "str"):
        r"""
        Makes a store decorator that filters out keys that don't match the template
        given field type.

        >>> store = {
        ...     'root/meshed/v_151.json': '{"downloads": 41, "type": "productivity"}',
        ...     'root/dol/v_9.json': '{"downloads": 132, "type": "utility"}',
        ...     'root/not/the/right/format': "something else"
        ... }
        >>> filt = KeyTemplate('root/{pkg}/v_{version}.json')
        >>> filtered_store = filt.filt_iter('str')(store)
        >>> list(filtered_store)
        ['root/meshed/v_151.json', 'root/dol/v_9.json']

        """
        if isinstance(field_type, Mapping):
            # The user wants to filter a store with the default
            return self.filt_iter()(field_type)
        self._assert_field_type(field_type, "field_type")
        filt_func = getattr(self, f"match_{field_type}")
        return filt_iter(filt=filt_func)

    # @_return_none_if_none_input
    def str_to_dict(self, s: str) -> dict:
        r"""Parses the input string and returns a dictionary of extracted values.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json',
        ...     from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_dict('root/life/v_30.json')
        {'i01_': 'life', 'ver': 30}

        """
        if s is None:
            return None
        match = self._regex.match(s)
        if match:
            return {k: self._from_str_funcs[k](v) for k, v in match.groupdict().items()}
        else:
            raise ValueError(f"String '{s}' does not match the template.")

    # @_return_none_if_none_input
    def dict_to_str(self, params: dict) -> str:
        r"""Generates a string from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.dict_to_str({'i01_': 'life', 'ver': 42})
        'root/life/v_042.json'

        """
        if params is None:
            return None
        params = {k: self._to_str_funcs[k](v) for k, v in params.items()}
        return self.template.format(**params)

    # @_return_none_if_none_input
    def dict_to_tuple(self, params: dict) -> tuple:
        r"""Generates a tuple from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_tuple('root/life/v_42.json')
        ('life', 42)

        """
        if params is None:
            return None
        return tuple(params.get(field_name) for field_name in self._fields)

    # @_return_none_if_none_input
    def tuple_to_dict(self, param_vals: tuple) -> dict:
        r"""Generates a dictionary from the tuple values based on the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.tuple_to_dict(('life', 42))
        {'i01_': 'life', 'ver': 42}
        """
        if param_vals is None:
            return None
        return {
            field_name: value for field_name, value in zip(self._fields, param_vals)
        }

    # @_return_none_if_none_input
    def str_to_tuple(self, s: str) -> tuple:
        r"""Parses the input string and returns a tuple of extracted values.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_tuple('root/life/v_42.json')
        ('life', 42)
        """
        if s is None:
            return None
        return self.dict_to_tuple(self.str_to_dict(s))

    # @_return_none_if_none_input
    def tuple_to_str(self, param_vals: tuple) -> str:
        r"""Generates a string from the tuple values based on the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.tuple_to_str(('life', 42))
        'root/life/v_042.json'
        """
        if param_vals is None:
            return None
        return self.dict_to_str(self.tuple_to_dict(param_vals))

    # @_return_none_if_none_input
    def str_to_single(self, s: str) -> Any:
        r"""Parses the input string and returns a single value.

        >>> st = KeyTemplate(
        ...     r'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_single('root/life/v_42.json')
        42
        """
        if s is None:
            return None
        return self.str_to_tuple(s)[0]

    # @_return_none_if_none_input
    def single_to_str(self, k: Any) -> str:
        r"""Generates a string from the single value based on the template.

        >>> st = KeyTemplate(
        ...     r'root/life/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.single_to_str(42)
        'root/life/v_042.json'
        """
        if k is None:
            return None
        return self.tuple_to_str((k,))

    # @_return_none_if_none_input
    def dict_to_namedtuple(
        self,
        params: dict,
    ):
        r"""Generates a namedtuple from the dictionary values based on the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
        >>> App
        NamedTuple(i01_='life', ver=42)
        """
        if params is None:
            return None
        return namedtuple(self.namedtuple_type_name, params.keys())(**params)

    # @_return_none_if_none_input
    def namedtuple_to_dict(self, nt):
        r"""Converts a namedtuple to a dictionary.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.dict_to_namedtuple({'i01_': 'life', 'ver': 42})
        >>> st.namedtuple_to_dict(App)
        {'i01_': 'life', 'ver': 42}
        """
        if nt is None:
            return None
        return dict(nt._asdict())  # TODO: Find way that doesn't involve private method

    def str_to_namedtuple(self, s: str):
        r"""Converts a string to a namedtuple.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> App = st.str_to_namedtuple('root/life/v_042.json')
        >>> App
        NamedTuple(i01_='life', ver=42)
        """
        if s is None:
            return None
        return self.dict_to_namedtuple(self.str_to_dict(s))

    # @_return_none_if_none_input
    def str_to_simple_str(self, s: str):
        r"""Converts a string to a simple string (i.e. a simple character-delimited string).

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.str_to_simple_str('root/life/v_042.json')
        'life,042'
        >>> st_clone = st.clone(simple_str_sep='-')
        >>> st_clone.str_to_simple_str('root/life/v_042.json')
        'life-042'
        """
        if s is None:
            return None
        return self.simple_str_sep.join(
            self._to_str_funcs[k](v) for k, v in self.str_to_dict(s).items()
        )

    # @_return_none_if_none_input
    def simple_str_to_tuple(self, ss: str):
        r"""Converts a simple character-delimited string to a dict.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ...     simple_str_sep='-',
        ... )
        >>> st.simple_str_to_tuple('life-042')
        ('life', 42)
        """
        if ss is None:
            return None
        if self.simple_str_sep:
            field_values = ss.split(self.simple_str_sep)
        else:
            field_values = (ss,)
        if len(field_values) != self._n_fields:
            raise ValueError(
                f"String '{ss}' has does not have the right number of field values. "
                f"Expected {self._n_fields}, got {len(field_values)} "
                f"(namely: {field_values}.)"
            )
        return tuple(f(x) for f, x in zip(self._from_str_funcs.values(), field_values))

    # @_return_none_if_none_input
    def simple_str_to_str(self, ss: str):
        r"""Converts a simple character-delimited string to a string.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ...     simple_str_sep='-',
        ... )
        >>> st.simple_str_to_str('life-042')
        'root/life/v_042.json'
        """
        if ss is None:
            return None
        return self.tuple_to_str(self.simple_str_to_tuple(ss))

    def match_str(self, s: str) -> bool:
        r"""
        Returns True iff the string matches the template.

        >>> st = KeyTemplate(
        ...     r'root/{}/v_{ver:03.0f:\d+}.json', from_str_funcs={'ver': int},
        ... )
        >>> st.match_str('root/life/v_042.json')
        True
        >>> st.match_str('this/does/not_match')
        False
        """
        return self._regex.match(s) is not None

    def match_dict(self, params: dict) -> bool:
        return self.match_str(self.dict_to_str(params))
        # Note: Could do:
        #  return all(self._field_patterns[k].match(v) for k, v in params.items())
        # but not sure that's even quicker (given regex is compiled)

    def match_tuple(self, param_vals: tuple) -> bool:
        return self.match_str(self.tuple_to_str(param_vals))

    def match_namedtuple(self, params: namedtuple) -> bool:
        return self.match_str(self.namedtuple_to_str(params))

    def match_simple_str(self, params: str) -> bool:
        return self.match_str(self.simple_str_to_str(params))

    def _extract_template_info(self, template):
        r"""Extracts information from the template. Namely:

        - normalized_template: A template where each placeholder has a field name
        (if not given, dflt_field_name will be used, which by default is
        'i{:02.0f}_'.format)

        - field_names: The tuple of field names in the order they appear in template

        - to_str_funcs: A dict of field names and their corresponding to_str functions,
        which will be used to convert the field values to strings when generating a
        string.

        - field_patterns_: A dict of field names and their corresponding regex patterns,
        which will be used to extract the field values from a string.

        These four values are used in the init to compute the parameters of the
        instance.

        >>> st = KeyTemplate(r'{:03.0f}/{name::\w+}')
        >>> st.template
        '{i01_}/{name}'
        >>> st._fields
        ('i01_', 'name')
        >>> st._field_patterns
        {'i01_': '.*', 'name': '\\w+'}
        >>> st._regex.pattern
        '(?P<i01_>.*)/(?P<name>\\w+)'
        >>> to_str_funcs = st._to_str_funcs
        >>> to_str_funcs['i01_'](3)
        '003'
        >>> to_str_funcs['name']('life')
        'life'

        """

        field_names = []
        field_patterns_ = {}
        to_str_funcs = {}

        def parse_and_transform():
            for index, (literal_text, field_name, format_spec, conversion) in enumerate(
                self._formatter.parse(template), 1
            ):
                field_name = (
                    self.dflt_field_name(index) if field_name == "" else field_name
                )
                if field_name is not None:
                    field_names.append(field_name)  # remember the field name
                    # extract format and pattern information:
                    if ":" not in format_spec:
                        format_spec += ":"
                    to_str_func_format, pattern = format_spec.split(":")
                    if to_str_func_format:
                        to_str_funcs[field_name] = (
                            "{" + f":{to_str_func_format}" + "}"
                        ).format
                    field_patterns_[field_name] = pattern or self.dflt_pattern
                # At this point you should have a valid field_name and empty format_spec
                yield (
                    literal_text,
                    field_name,
                    "",
                    conversion,
                )

        normalized_template = string_unparse(parse_and_transform())
        return normalized_template, tuple(field_names), to_str_funcs, field_patterns_

    def _compile_regex(self, template, normalize_path=False):
        r"""Parses the template, generating regex for matching the template.
        Essentially, it weaves together the literal text parts and the format_specs
        parts, transformed into name-caputuring regex patterns.

        Note that the literal text parts are regex-escaped so that they are not
        interpreted as regex. For example, if the template is "{name}.txt", the
        literal text part is replaced with "\\.txt", to avoid that the "." is
        interpreted as a regex wildcard. This would otherwise match any character.
        Instead, the escaped dot is matched literally.
        See https://docs.python.org/3/library/re.html#re.escape for more information.

        >>> KeyTemplate('{}.ext')._regex.pattern
        '(?P<i01_>.*)\\.ext'
        >>> KeyTemplate('{name}.ext')._regex.pattern
        '(?P<name>.*)\\.ext'
        >>> KeyTemplate(r'{::\w+}.ext')._regex.pattern
        '(?P<i01_>\\w+)\\.ext'
        >>> KeyTemplate(r'{name::\w+}.ext')._regex.pattern
        '(?P<name>\\w+)\\.ext'
        >>> KeyTemplate(r'{:0.02f:\w+}.ext')._regex.pattern
        '(?P<i01_>\\w+)\\.ext'
        >>> KeyTemplate(r'{name:0.02f:\w+}.ext')._regex.pattern
        '(?P<name>\\w+)\\.ext'
        """

        def mk_named_capture_group(field_name):
            if field_name:
                return f"(?P<{field_name}>{self._field_patterns[field_name]})"
            else:
                return ""

        def generate_pattern_parts(template):
            parts = self._formatter.parse(template)
            for literal_text, field_name, _, _ in parts:
                yield re.escape(literal_text) + mk_named_capture_group(field_name)

        return safe_compile(
            "".join(generate_pattern_parts(template)), normalize_path=normalize_path
        )

    @staticmethod
    def _assert_field_type(field_type: FieldTypeNames, name="field_type"):
        if field_type not in FieldTypeNames.__args__:
            raise ValueError(
                f"{name} must be one of {FieldTypeNames}. Was: {field_type}"
            )
```

## recipes.py

```python
"""Recipes using dol"""

__all__ = ["search_paths"]

from dol.paths import path_filter as search_paths  # was recipe. Promoted to paths
```

## scrap/__init__.py

```python
"""A place to stage or soft-deprecate code"""
```

## scrap/new_store_wrap.py

```python
"""Ideas for a new store wrapping setup"""

from typing import KT, VT
from collections.abc import MutableMapping, Mapping, Iterable, Callable
from dol.util import inject_method


# Note: See dol.base.Store (and complete MappingWrap)
# TODO: Complete with wrap hooks (those of dol.base.Store and more)
class MappingWrap:
    def __init__(self, store: Mapping):
        self.store = store

    # mapping special methods forward to hidden methods

    def __iter__(self) -> Iterable[KT]:
        return self._iter()

    def __getitem__(self, k: KT) -> VT:
        return self._getitem(k)

    def __len__(self, k: KT) -> int:
        return self._len(k)

    def __contains__(self, k: KT) -> bool:
        return self._contains(k)

    def __setitem__(self, k: KT, v: VT):
        return self._setitem(k, v)

    def __delitem__(self, k: KT):
        return self._delitem(k)

    # default mapping hidden methods just forward to store
    # TODO: Add wrapping hooks (_obj_of_data, etc.  including some for filt_iter, etc.)
    def _iter(self) -> Iterable[KT]:
        return iter(self.store)

    def _getitem(self, k: KT) -> VT:
        return self.store[k]

    def _len(self, k: KT) -> int:
        return len(self.store)

    def _contains(self, k: KT) -> bool:
        return k in self.store

    def _setitem(self, k: KT, v: VT):
        return self.store.__setitem__(k, v)

    def _delitem(self, k: KT):
        return self.store.__delitem__(k)

    # util method to inject new

    def _inject_method(self, method_function, method_name=None):
        return inject_method(self, method_function, method_name)


def test_mapping_wrap():
    from dol.scrap.new_store_wrap import MappingWrap

    from dol import TextFiles, filt_iter
    import posixpath

    def filter_when_ending_with_slash(self, k, slash=posixpath.sep):
        if not k.endswith(slash):
            return self.store[k]
        else:
            return filt_iter(self.store, filt=lambda key: key.startswith(k))

    class SlashTriggersFilter(MappingWrap):
        def _getitem(self, k):
            return filter_when_ending_with_slash(self, k)

    from dol.tests.utils_for_tests import mk_test_store_from_keys

    s = mk_test_store_from_keys()
    assert dict(s) == {
        "pluto": "Content of pluto",
        "planets/mercury": "Content of planets/mercury",
        "planets/venus": "Content of planets/venus",
        "planets/earth": "Content of planets/earth",
        "planets/mars": "Content of planets/mars",
        "fruit/apple": "Content of fruit/apple",
        "fruit/banana": "Content of fruit/banana",
        "fruit/cherry": "Content of fruit/cherry",
    }

    ss = SlashTriggersFilter(s)
    assert list(ss) == [
        "pluto",
        "planets/mercury",
        "planets/venus",
        "planets/earth",
        "planets/mars",
        "fruit/apple",
        "fruit/banana",
        "fruit/cherry",
    ]

    sss = ss["planets/"]
    assert dict(sss) == {
        "planets/mercury": "Content of planets/mercury",
        "planets/venus": "Content of planets/venus",
        "planets/earth": "Content of planets/earth",
        "planets/mars": "Content of planets/mars",
    }
```

## scrap/store_factories.py

```python
"""Utils to make stores"""

from typing import KT, VT, Any, NewType
from collections.abc import Mapping, Callable, Iterator, Collection
import dataclasses
from dataclasses import dataclass
import operator
from functools import partial
from contextlib import suppress
from dol.base import KvReader

Getter = NewType("Getter", Callable[[Mapping, KT], VT])
Lister = NewType("Getter", Callable[[Mapping], Iterator[KT]])
Sizer = NewType("Sizer", Callable[[Mapping], int])
ContainmentChecker = NewType("ContainmentChecker", Callable[[Mapping], bool])
Setter = NewType("Setter", Callable[[Mapping, KT, VT], Any])
Deleter = NewType("Deleter", Callable[[Mapping, KT], Any])

# count_by_iteration: Sizer
# check_by_iteration: ContainmentChecker
# check_by_trying_to_get: ContainmentChecker


def count_by_iteration(collection: Collection) -> int:
    """
    Number of elements in collection of keys.
    Note: This method iterates over all elements of the collection and counts them.
    Therefore it is not efficient, and in most cases should be overridden with a more
    efficient method.
    """
    count = 0
    for _ in iter(collection):
        count += 1
    return count


# TODO: Put KT here because it's the main use, but could be VT.
#  Should probably be T = TypeVar('T')?
def check_by_iteration(collection: Collection[KT], x: KT) -> bool:
    """
    Check if collection of keys contains k.
    Note: Method loops through contents of collection to see if query element exists.
    Therefore it may not be efficient, and in most cases, a method specific to the case should be used.
    :return: True if k is in the collection, and False if not
    """
    for existing_x in iter(collection):
        if existing_x == x:
            return True
    return False


def check_by_trying_to_get(mapping: Mapping, x: KT, false_on_error=(KeyError,)) -> bool:
    """
    Check if mapping contains x.
    Note: This method tries to get x from the mapping, returning ``False`` if it fails.
    Therefore it may not be efficient, and in most cases,
    a method specific to the case should be used.
    :return: True if x is in the mapping, and False if not
    """
    try:
        _ = mapping[x]
        return True
    except false_on_error:
        return False


# def mk_shell_factory(cls):
#     """Make a factory for a shell class"""
#     src_field, *other_fields = cls.__dataclass_fields__.items()
#     return partial(cls, **dict(other_fields))


# def add_shell_factory(cls):
#     """Add a factory for a shell class"""
#     from i2 import FuncFactory
#     cls.factory = FuncFactory(cls)
#     return cls


# TODO: See dol.sources.AttrContainer. Make KvReaderShell subsume it, then refactor
# TODO: Make tools to (1) wrap types and (2) make recursion easy
# @add_shell_factory
@dataclass
class KvReaderShell(KvReader):
    """Wraps an object with a mapping interface

    See below how we can wrap a list with a mapping interface, and use it as a store.

    >>> arr = (1, 2, 3)
    >>> s = KvReaderShell(arr, getter=getattr, lister=dir)

    We defined the ``lister`` to yield the attributes of the list:

    >>> sorted(s)[:2]
    ['__add__', '__class__']

    We defined the ``getter`` to give us attributes:

    >>> callable(s['__add__'])
    True
    >>> s['__add__']((4, 5))
    (1, 2, 3, 4, 5)


    """

    src: Any
    getter: Getter = operator.getitem
    lister: Lister = iter
    with suppress(AttributeError):  # Note: dataclasses.KW_ONLY only >= 3.10
        _ = dataclasses.KW_ONLY
    sizer: Sizer = count_by_iteration
    is_contained: ContainmentChecker = check_by_iteration

    def __getitem__(self, k: KT) -> VT:
        return self.getter(self.src, k)

    def __iter__(self) -> Iterator[KT]:
        return iter(self.lister(self.src))

    def __len__(self):
        return self.sizer(self.src)

    def __contains__(self, k: KT):
        return self.is_contained(self.src, k)

    # @property
    # def factory(self):
    #     return mk_shell_factory(KvReaderShell)


@dataclass
class StoreShell(KvReaderShell):
    """Wraps an object with a mapping interface"""

    setter: Setter = operator.setitem
    deleter: Deleter = operator.delitem

    def __setitem__(self, k: KT, v: VT) -> Any:
        return self.setter(self.src, k, v)

    def __delitem__(self, k: KT) -> Any:
        return self.deleter(self.src, k)
```

## signatures.py

```python
"""Signature calculus: Tools to make it easier to work with function's signatures.

How to:

    - get names, kinds, defaults, annotations

    - make signatures flexibly

    - merge two or more signatures

    -
    - give a function a specific signature (with a choice of validations)

    - get an equivalent function with a different order of arguments

    - get an equivalent function with a subset of arguments (like partial)

    - get an equivalent function but with variadic *args and/or **kwargs replaced with
    non-variadic args (tuple) and kwargs (dict)

    - make an f(a) function in to a f(a, b=None) function with b ignored


Get names, kinds, defaults, annotations:

>>> def func(z, a: float=1.0, /, b=2, *, c: int=3):
...     pass
>>> sig = Sig(func)
>>> sig.names
['z', 'a', 'b', 'c']
>>> from inspect import Parameter
>>> assert sig.kinds == {
...     'z': Parameter.POSITIONAL_ONLY,
...     'a': Parameter.POSITIONAL_ONLY,
...     'b': Parameter.POSITIONAL_OR_KEYWORD,
...     'c': Parameter.KEYWORD_ONLY
... }
>>> # Note z is not in there (only defaulted params are included)
>>> sig.defaults
{'a': 1.0, 'b': 2, 'c': 3}
>>> sig.annotations
{'a': <class 'float'>, 'c': <class 'int'>}

Make signatures flexibly:

>>> Sig(func)
<Sig (z, a: float = 1.0, /, b=2, *, c: int = 3)>
>>> Sig(['a', 'b'])
<Sig (a, b)>
>>> Sig('x y z')
<Sig (x, y, z)>

Merge signatures.

>>> def foo(x): pass
>>> def bar(y: int, *, z=2): pass  # note the * (keyword only) will be lost!
>>> Sig(foo) + ['a', 'b'] + Sig(bar)
<Sig (x, a, b, y: int, z=2)>

Give a function a signature.

>>> @Sig('a b c')
... def func(*args, **kwargs):
...     print(args, kwargs)
>>> Sig(func)
<Sig (a, b, c)>


**Notes to the reader**

Both in the code and in the docs, we'll use short hands for parameter (argument) kind.

    - PK = Parameter.POSITIONAL_OR_KEYWORD

    - VP = Parameter.VAR_POSITIONAL

    - VK = Parameter.VAR_KEYWORD

    - PO = Parameter.POSITIONAL_ONLY

    - KO = Parameter.KEYWORD_ONLY

"""

from inspect import Signature, Parameter, signature, unwrap
import re
import sys
from typing import (
    Union,
    Any,
    Dict,
    Tuple,
    TypeVar,
    Literal,
    Optional,
    get_args,
)
from collections.abc import Callable, Iterable, Iterator, Mapping as MappingType
from typing import KT, VT, T
from types import FunctionType
from collections import defaultdict
from operator import eq, attrgetter

from functools import (
    cached_property,
    update_wrapper,
    partial,
    partialmethod,
    WRAPPER_ASSIGNMENTS,
    wraps as _wraps,
    update_wrapper as _update_wrapper,
)


def deprecation_of(func, old_name):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from warnings import warn

        warn(
            f"`{old_name}` is deprecated. Use `{func.__module__}.{func.__qualname__}` instead.",
            DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and
# kwdefaults

wrapper_assignments = (*WRAPPER_ASSIGNMENTS, "__defaults__", "__kwdefaults__")

update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)

_empty = Parameter.empty
empty = _empty

ParamsType = Iterable[Parameter]
ParamsAble = Union[ParamsType, Signature, MappingType[str, Parameter], Callable, str]
SignatureAble = Union[Signature, ParamsAble]
HasParams = Union[Iterable[Parameter], MappingType[str, Parameter], Signature, Callable]

# short hands for Parameter kinds
PK = Parameter.POSITIONAL_OR_KEYWORD
VP, VK = Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD
PO, KO = Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY
var_param_kinds = frozenset({VP, VK})
var_param_types = var_param_kinds  # Deprecate: for back-compatibility. Delete in 2021
var_param_kind_dflts_items = tuple({VP: (), VK: {}}.items())

DFLT_DEFAULT_CONFLICT_METHOD = "strict"
SigMergeOptions = Literal[None, "strict", "take_first", "fill_defaults_and_annotations"]

param_attributes = {"name", "kind", "default", "annotation"}


class InvalidSignature(SyntaxError, ValueError):
    """Raise when a signature is not valid"""


class FuncCallNotMatchingSignature(TypeError):
    """Raise when the call signature is not valid"""


class IncompatibleSignatures(ValueError):
    from pprint import pformat

    """Raise when two signatures are not compatible.
    (see https://github.com/i2mint/i2/discussions/76 for more information on signature
    compatibility)"""

    def __init__(self, *args, sig1=None, sig2=None, **kwargs):
        args = list(args or ("",))
        sig_pairs = None
        if sig1 and sig2:
            sig_pairs = SigPair(sig1, sig2)
            # add the signature differences to the error message
            args[0] += (
                f"\n----- Signature differences (not all differences necessarily "
                f"matter in your context): ----- \n{sig_pairs.diff_str()}"
            )
        super().__init__(*args, **kwargs)
        self.sig_pairs = sig_pairs


# TODO: Couldn't make this work. See https://www.python.org/dev/peps/pep-0562/
# deprecated_names = {'assure_callable', 'assure_signature', 'assure_params'}
#
#
# def __getattr__(name):
#     print(name)
#     if name in deprecated_names:
#         from warnings import warn
#         warn(f"{name} is deprecated (see code for new name -- look for aliases)",
#         DeprecationWarning)
#     raise AttributeError(f"module {__name__} has no attribute {name}")


def validate_signature(func: Callable) -> Callable:
    """
    Validates the signature of a function.

    >>> @validate_signature
    ... def has_valid_signature(x=Sig.empty, y=2):
    ...     pass
    >>> # all good, no errors raised
    >>>
    >>> @validate_signature  # doctest: +IGNORE_EXCEPTION_DETAIL
    ... def does_no_have_valid_signature(x=2, y=Sig.empty):
    ...     pass
    Traceback (most recent call last):
    ...
    i2.signatures.InvalidSignature: Invalid signature for function <function does_no_have_valid_signature at 0x106a72a70>: non-default argument follows default a
    rgument

    """
    try:
        Sig(func)  # to get errors if the signature is not valid
    except Exception as e:
        raise InvalidSignature(f"Invalid signature for function {func}: {e}")
    return func  # if all goes well, return the original function


def is_signature_error(e: BaseException) -> bool:
    """Check if an exception is a signature error"""
    return isinstance(InvalidSignature) or (
        isinstance(e, ValueError) and "no signature found" in str(e)
    )


def _param_sort_key(param):
    return (param.kind, param.kind == KO or param.default is not empty)


def sort_params(params):
    """

    :param params: An iterable of `Parameter` instances
    :return: A list of these instances sorted so as to obey the ``kind`` and ``default``
        order rules of python signatures.

    Note 1: It doesn't mean that these params constitute a valid signature together,
    since it doesn't verify rules like unicity of names and variadic kinds.

    Note 2: Though you can use ``sorted`` on an iterable of ``i2.signatures.Param``
    instances, know that even for sorting the three parameters below,
    the ``sort_params`` function is more than twice as fast.

    >>> from inspect import Parameter
    >>> sort_params(
    ...     [Parameter('a', kind=Parameter.POSITIONAL_OR_KEYWORD, default=1),
    ...     Parameter('b', kind=Parameter.POSITIONAL_ONLY),
    ...     Parameter('c', kind=Parameter.POSITIONAL_OR_KEYWORD)]
    ... )
    [<Parameter "b">, <Parameter "c">, <Parameter "a=1">]
    """
    return sorted(params, key=_param_sort_key)


def _return_none(o: object) -> None:
    return None


# (approximately) duplicated from i2.util to keep signatures.py standalone
def name_of_obj(
    o: object,
    *,
    base_name_of_obj: Callable = attrgetter("__name__"),
    caught_exceptions: tuple = (AttributeError,),
    default_factory: Callable = _return_none,
) -> str | None:
    """
    Tries to find the (or "a") name for an object, even if `__name__` doesn't exist.

    >>> name_of_obj(map)
    'map'
    >>> name_of_obj([1, 2, 3])
    'list'
    >>> name_of_obj(print)
    'print'
    >>> name_of_obj(lambda x: x)
    '<lambda>'
    >>> from functools import partial
    >>> name_of_obj(partial(print, sep=","))
    'print'
    >>> from functools import cached_property
    >>> class A:
    ...     @property
    ...     def prop(self):
    ...         return 1.0
    ...     @cached_property
    ...     def cached_prop(self):
    ...         return 2.0
    >>> name_of_obj(A.prop)
    'prop'
    >>> name_of_obj(A.cached_prop)
    'cached_prop'

    Note that ``name_of_obj`` uses the ``__name__`` attribute as its base way to get
    a name. You can customize this behavior though.
    For example, see that:

    >>> from inspect import Signature
    >>> name_of_obj(Signature.replace)
    'replace'

    If you want to get the fully qualified name of an object, you can do:

    >>> alt = partial(name_of_obj, base_name_of_obj=attrgetter('__qualname__'))
    >>> alt(Signature.replace)
    'Signature.replace'

    """
    try:
        return base_name_of_obj(o)
    except caught_exceptions:
        kwargs = dict(
            base_name_of_obj=base_name_of_obj,
            caught_exceptions=caught_exceptions,
            default_factory=default_factory,
        )
        if isinstance(o, (cached_property, partial, partialmethod)) and hasattr(
            o, "func"
        ):
            return name_of_obj(o.func, **kwargs)
        elif isinstance(o, property) and hasattr(o, "fget"):
            return name_of_obj(o.fget, **kwargs)
        elif hasattr(o, "__class__"):
            return name_of_obj(type(o), **kwargs)
        elif hasattr(o, "fset"):
            return name_of_obj(o.fset, **kwargs)
        return default_factory(o)


def ensure_callable(obj: SignatureAble):
    if isinstance(obj, Callable):
        return obj
    else:

        def f(*args, **kwargs):
            """Empty function made just to carry a signature"""

        f.__signature__ = ensure_signature(obj)
        return f


assure_callable = ensure_callable  # alias for backcompatibility


def ensure_signature(obj: SignatureAble) -> Signature:
    if isinstance(obj, Signature):
        return obj
    elif isinstance(obj, Callable):
        return _robust_signature_of_callable(obj)
    elif isinstance(obj, Iterable):
        params = ensure_params(obj)
        try:
            return Signature(parameters=params)
        except TypeError:
            raise TypeError(
                f"Don't know how to make that object into a Signature: {obj}"
            )
    elif isinstance(obj, Parameter):
        return Signature(parameters=(obj,))
    elif obj is None:
        return Signature(parameters=())
    # if you get this far...
    raise TypeError(f"Don't know how to make that object into a Signature: {obj}")


assure_signature = ensure_signature  # alias for backcompatibility


def ensure_param(p):
    if isinstance(p, Parameter):
        return p
    elif isinstance(p, dict):
        return Param(**p)
    elif isinstance(p, str):
        return Param(name=p)
    elif isinstance(p, Iterable):
        name, *r = p
        dflt_and_annotation = dict(zip(["default", "annotation"], r))
        return Param(name, PK, **dflt_and_annotation)
    else:
        raise TypeError(f"Don't know how to make {p} into a Parameter object")


def _params_from_mapping(mapping: MappingType):
    def gen():
        for k, v in mapping.items():
            if isinstance(v, MappingType):
                if "name" in v:
                    assert v["name"] == k, (
                        f"In a mapping specification of a params, "
                        f"either the 'name' of the val shouldn't be specified, "
                        f"or it should be the same as the key ({k}): "
                        f"{dict(mapping)}"
                    )
                    yield v
                else:
                    yield dict(name=k, **v)
            else:
                assert isinstance(v, Parameter) and v.name == k, (
                    f"In a mapping specification of a params, "
                    f"either the val should be a Parameter with the same name as the "
                    f"key ({k}), or it should be a mapping with a 'name' key "
                    f"with the same value as the key: {dict(mapping)}"
                )
                yield v

    return list(gen())


def _add_optional_keywords(sig, kwarg_and_defaults, kwarg_annotations=None):
    """
    Enhances a given signature with additional optional keyword-only arguments.

    Args:
        sig (Signature): The original function signature.
        kwarg_and_defaults (dict): A dictionary of keyword arguments and their default values.
        kwarg_annotations (dict, optional): A dictionary of keyword arguments and their type annotations.

    Returns:
        Signature: The enhanced function signature with additional keyword-only arguments.

    >>> from inspect import signature
    >>> def example_func(x, y): pass
    >>> original_sig = signature(example_func)
    >>> enhanced_sig = _add_optional_keywords(
    ...     original_sig, {'z': 3, 'verbose': False}, {'verbose': bool}
    ... )
    >>> str(enhanced_sig)
    '(x, y, *, z=3, verbose: bool = False)'

    Note:
        - Annotations for the additional keywords are optional.
        - All additional keywords are added as keyword-only arguments.
    """
    if isinstance(sig, Signature):
        sig = Sig(sig).merge_with_sig(
            Sig.from_objs(**kwarg_and_defaults), ch_to_all_pk=False
        )
        sig = sig.ch_kinds(**{k: Sig.KEYWORD_ONLY for k in kwarg_and_defaults})

        kwarg_annotations = kwarg_annotations or {}
        assert all(name in kwarg_and_defaults for name in kwarg_annotations), (
            "Some annotations were given for arguments that were not in kwarg_and_defaults:"
            f"\n{kwarg_and_defaults=}\n{kwarg_annotations=}"
        )
        sig = sig.ch_annotations(**kwarg_annotations)
        return sig
    else:
        func = sig  # assume it's a function
        # apply _add_optional_keywords to that function
        sig = Sig(func)
        sig = _add_optional_keywords(sig, kwarg_and_defaults, kwarg_annotations)
        # and inject the new signature into the function
        return sig(func)


def ensure_params(obj: ParamsAble = None):
    """Get an interable of Parameter instances from an object.

    :param obj:
    :return:

    From a callable:

    >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
    ...     ...
    >>> ensure_params(f)
    [<Parameter "w">, <Parameter "x: float = 1">, <Parameter "y=1">, <Parameter "z: int = 1">]

    From an iterable of strings, dicts, or tuples

    >>> ensure_params(
    ...     [
    ...         "xyz",
    ...         (
    ...             "b",
    ...             Parameter.empty,
    ...             int,
    ...         ),  # if you want an annotation without a default use Parameter.empty
    ...         (
    ...             "c",
    ...             2,
    ...         ),  # if you just want a default, make it the second element of your tup
    ...         dict(name="d", kind=Parameter.VAR_KEYWORD),
    ...     ]
    ... )  # all kinds are by default PK: Use dict to specify otherwise.
    [<Param "xyz">, <Param "b: int">, <Param "c=2">, <Param "**d">]


    If no input is given, an empty list is returned.

    >>> ensure_params()  # equivalent to ensure_params(None)
    []

    """
    # obj = inspect.unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))

    if obj is None:
        return []
    elif isinstance(obj, Signature):
        return list(obj.parameters.values())
    try:  # to get params from the builtin signature function
        return list(_robust_signature_of_callable(obj).parameters.values())
    except (TypeError, ValueError):
        if isinstance(obj, Iterable):
            if isinstance(obj, str):
                obj = [obj]
            # TODO: Can do better here! See attempt in _params_from_mapping:
            elif isinstance(obj, Mapping):
                obj = _params_from_mapping(obj)
                # obj = list(obj.values())
            else:
                obj = list(obj)
            if len(obj) == 0:
                return obj
            else:
                # TODO: put this in function that has more kind resolution power
                #  e.g. if a KEYWORD_ONLY arg was encountered, all subsequent
                #  have to be unless otherwise specified.
                return [ensure_param(p) for p in obj]
        else:
            if isinstance(obj, Parameter):
                obj = Signature([obj])
            elif isinstance(obj, Callable):
                obj = _robust_signature_of_callable(obj)
            elif obj is None:
                obj = {}
            if isinstance(obj, Signature):
                return list(obj.parameters.values())
        # if nothing above worked, perhaps you have a wrapped object? Try unwrapping until
        # you find a signature...
        if hasattr(obj, "__wrapped__"):
            obj = unwrap(obj, stop=(lambda f: hasattr(f, "__signature__")))
            return ensure_params(obj)
        else:  # if function didn't return at this point, it didn't find a match, so raise
            # a TypeError
            raise TypeError(
                f"Don't know how to make that object into an iterable of inspect.Parameter "
                f"objects: {obj}"
            )


assure_params = ensure_params  # alias for backcompatibility


class MissingArgValFor:
    """A simple class to wrap an argument name, indicating that it was missing somewhere.

    >>> MissingArgValFor("argname")
    MissingArgValFor("argname")
    """

    def __init__(self, argname: str):
        assert isinstance(argname, str)
        self.argname = argname

    def __repr__(self):
        return f'MissingArgValFor("{self.argname}")'


# TODO: Look into the handling of the Parameter.VAR_KEYWORD kind in params
def extract_arguments(
    params: ParamsAble,
    *,
    what_to_do_with_remainding="return",
    include_all_when_var_keywords_in_params=False,
    assert_no_missing_position_only_args=False,
    **kwargs,
):
    """Extract arguments needed to satisfy the params of a callable, dealing with the
    dirty details.

    Returns an (param_args, param_kwargs, remaining_kwargs) tuple where
    - param_args are the values of kwargs that are PO (POSITION_ONLY) as defined by
    params,
    - param_kwargs are those names that are both in params and not in param_args, and
    - remaining_kwargs are the remaining.

    Intended usage: When you need to call a function `func` that has some
    position-only arguments,
    but you have a kwargs dict of arguments in your hand. You can't just to `func(
    **kwargs)`.
    But you can (now) do
    ```
    args, kwargs, remaining = extract_arguments(kwargs, func)  # extract from kwargs
    what you need for func
    # ... check if remaing is empty (or not, depending on your paranoia), and then
    call the func:
    func(*args, **kwargs)
    ```
    (And if you doing that a lot: Do put it in a decorator!)

    See Also: extract_arguments.without_remainding

    The most frequent case you'll encounter is when there's no POSITION_ONLY args,
    your param_args will be empty
    and you param_kwargs will contain all the arguments that match params,
    in the order of these params.

    >>> from inspect import signature
    >>> def f(a, b, c=None, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((), {'a': 1, 'b': 2, 'c': 3, 'd': 4}, {'extra': 'stuff'})

    But sometimes you do have POSITION_ONLY arguments.
    What extract_arguments will do for you is return the value of these as the first
    element of
    the triple.

    >>> def f(a, b, c=None, /, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((1, 2, 3), {'d': 4}, {'extra': 'stuff'})

    Note above how we get `(1, 2, 3)`, the order defined by the func's signature,
    instead of `(2, 1, 3)`, the order defined by the kwargs.
    So it's the params (e.g. function signature) that determine the order, not kwargs.
    When using to call a function, this is especially crucial if we use POSITION_ONLY
    arguments.

    See also that the third output, the remaining_kwargs, as `{'extra': 'stuff'}` since
    it was not in the params of the function.
    Even if you include a VAR_KEYWORD kind of argument in the function, it won't change
    this behavior.

    >>> def f(a, b, c=None, /, d=0, **kws):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((1, 2, 3), {'d': 4}, {'extra': 'stuff'})

    This is because we don't want to assume that all the kwargs can actually be
    included in a call to the function behind the params.
    Instead, the user can chose whether to include the remainder by doing a:
    ```
    param_kwargs.update(remaining_kwargs)
    ```
    et voilà.

    That said, we do understand that it may be a common pattern, so we'll do that
    extra step for you
    if you specify `include_all_when_var_keywords_in_params=True`.

    >>> def f(a, b, c=None, /, d=0, **kws):
    ...     ...
    ...
    >>> extract_arguments(
    ...     f,
    ...     b=2,
    ...     a=1,
    ...     c=3,
    ...     d=4,
    ...     extra="stuff",
    ...     include_all_when_var_keywords_in_params=True,
    ... )
    ((1, 2, 3), {'d': 4, 'extra': 'stuff'}, {})

    If you're expecting no remainder you might want to just get the args and kwargs (
    not this third
    expected-to-be-empty remainder). You have two ways to do that, specifying:
        `what_to_do_with_remainding='ignore'`, which will just return the (args,
        kwargs) pair
        `what_to_do_with_remainding='assert_empty'`, which will do the same, but first
        assert the remainder is empty
    We suggest to use `functools.partial` to configure the `argument_argument` you need.

    >>> from functools import partial
    >>> arg_extractor = partial(
    ...     extract_arguments,
    ...     what_to_do_with_remainding="assert_empty",
    ...     include_all_when_var_keywords_in_params=True,
    ... )
    >>> def f(a, b, c=None, /, d=0, **kws):
    ...     ...
    ...
    >>> arg_extractor(f, b=2, a=1, c=3, d=4, extra="stuff")
    ((1, 2, 3), {'d': 4, 'extra': 'stuff'})

    And what happens if the kwargs doesn't contain all the POSITION_ONLY arguments?

    >>> def f(a, b, c=None, /, d=0):
    ...     ...
    ...
    >>> extract_arguments(f, b=2, d="is a kw arg", e="is not an arg at all")
    ((MissingArgValFor("a"), 2, MissingArgValFor("c")), {'d': 'is a kw arg'}, {'e': 'is not an arg at all'})


    A few more examples...

    Let's call `extract_arguments` with params being not a function,
    but, a Signature instance, a mapping whose values are Parameter instances,
    or an iterable of Parameter instances...

    >>> def func(a, b, /, c=None, *, d=0, **kws):
    ...     ...
    ...
    >>> sig = Signature.from_callable(func)
    >>> param_map = sig.parameters
    >>> param_iterable = param_map.values()
    >>> kwargs = dict(b=2, a=1, c=3, d=4, extra="stuff")
    >>> assert extract_arguments(sig, **kwargs) == extract_arguments(func, **kwargs)
    >>> assert extract_arguments(param_map, **kwargs) == extract_arguments(
    ...     func, **kwargs
    ... )
    >>> assert extract_arguments(param_iterable, **kwargs) == extract_arguments(
    ...     func, **kwargs
    ... )

    Edge case:
    No params specified? No problem. You'll just get empty args and kwargs. Everything
    in the remainder

    >>> extract_arguments(params=(), b=2, a=1, c=3, d=0)
    ((), {}, {'b': 2, 'a': 1, 'c': 3, 'd': 0})

    :param params: Specifies what PO arguments should be extracted.
        Could be a callable, Signature, iterable of Parameters...
    :param what_to_do_with_remainding:
        'return' (default): function will return `param_args`, `param_kwargs`,
        `remaining_kwargs`
        'ignore': function will return `param_args`, `param_kwargs`
        'assert_empty': function will assert that `remaining_kwargs` is empty and then
        return `param_args`, `param_kwargs`
    :param include_all_when_var_keywords_in_params=False,
    :param assert_no_missing_position_only_args=False,
    :param kwargs: The kwargs to extract the args from
    :return: A (param_args, param_kwargs, remaining_kwargs) tuple.
    """

    assert what_to_do_with_remainding in {"return", "ignore", "assert_empty"}
    assert isinstance(include_all_when_var_keywords_in_params, bool)
    assert isinstance(assert_no_missing_position_only_args, bool)

    params = ensure_params(params)
    if not params:
        return (), {}, {k: v for k, v in kwargs.items()}

    params_names = tuple(p.name for p in params)
    names_for_args = [p.name for p in params if p.kind == Parameter.POSITIONAL_ONLY]
    param_kwargs_names = [x for x in params_names if x not in set(names_for_args)]
    remaining_names = [x for x in kwargs if x not in params_names]

    param_args = tuple(kwargs.get(k, MissingArgValFor(k)) for k in names_for_args)
    param_kwargs = {k: kwargs[k] for k in param_kwargs_names if k in kwargs}
    remaining_kwargs = {k: kwargs[k] for k in remaining_names}

    if include_all_when_var_keywords_in_params:
        if (
            next(
                (p.name for p in params if p.kind == Parameter.VAR_KEYWORD),
                None,
            )
            is not None
        ):
            param_kwargs.update(remaining_kwargs)
            remaining_kwargs = {}

    if assert_no_missing_position_only_args:
        missing_argnames = tuple(
            x.argname for x in param_args if isinstance(x, MissingArgValFor)
        )
        assert (
            not missing_argnames
        ), f"There were some missing positional only argnames: {missing_argnames}"

    if what_to_do_with_remainding == "return":
        return param_args, param_kwargs, remaining_kwargs
    elif what_to_do_with_remainding == "ignore":
        return param_args, param_kwargs
    elif what_to_do_with_remainding == "assert_empty":
        assert (
            len(remaining_kwargs) == 0
        ), f"remaining_kwargs not empty: remaining_kwargs={remaining_kwargs}"
        return param_args, param_kwargs


extract_arguments_ignoring_remainder = partial(
    extract_arguments, what_to_do_with_remainding="ignore"
)
extract_arguments_asserting_no_remainder = partial(
    extract_arguments, what_to_do_with_remainding="assert_empty"
)

from collections.abc import Mapping
from typing import Optional
from collections.abc import Iterable


def function_caller(func, args, kwargs):
    return func(*args, **kwargs)


class Param(Parameter):
    """A thin wrap of Parameters: Adds shorter aliases to argument kinds and
    a POSITIONAL_OR_KEYWORD default to the argument kind to make it faster to make
    Parameter objects

    >>> list(map(Param, 'some quick arg params'.split()))
    [<Param "some">, <Param "quick">, <Param "arg">, <Param "params">]
    >>> from inspect import Signature
    >>> P = Param
    >>> Signature([P('x', P.PO), P('y', default=42, annotation=int), P('kw', P.KO)])
    <Signature (x, /, y: int = 42, *, kw)>
    """

    # aliases
    PK = Parameter.POSITIONAL_OR_KEYWORD
    PO = Parameter.POSITIONAL_ONLY
    KO = Parameter.KEYWORD_ONLY
    VP = Parameter.VAR_POSITIONAL
    VK = Parameter.VAR_KEYWORD

    def __init__(self, name, kind=PK, *, default=empty, annotation=empty):
        super().__init__(name, kind, default=default, annotation=annotation)

    def __lt__(self, other) -> bool:
        """Whether the self parameter can be before the other parameter in a signature.

        >>> Param('b') < Param('a', default=1)
        True
        >>> Param('b') > Param('a', default=1)
        False
        >>> Param('b', kind=Param.POSITIONAL_OR_KEYWORD) < Param('a', kind=Param.KEYWORD_ONLY)
        True
        >>> Param('b', kind=Param.POSITIONAL_OR_KEYWORD) > Param('a', kind=Param.KEYWORD_ONLY)
        False

        Note 1: The dual ``>`` operator is also infered.

        Note 2: This means that you can used ``sorted`` on an iterable of Param
        instances, but know that even for sorting the three parameters below,
        the ``sort_params`` function in the ``i2.signatures`` module is more than twice
        as fast.

        >>> sorted(
        ...     [Param('a', default=1),
        ...     Param('b', kind=Param.POSITIONAL_ONLY),
        ...     Param('c')]
        ... )
        [<Param "b">, <Param "c">, <Param "a=1">]
        """
        return (self.kind, self.default is not empty) < (
            other.kind,
            other.default is not empty,
        )


P = Param  # useful shorthand alias


def param_has_default_or_is_var_kind(p: Parameter):
    return p.default is not p.empty or p.kind in var_param_kinds


def parameter_to_dict(p: Parameter) -> dict:
    return dict(name=p.name, kind=p.kind, default=p.default, annotation=p.annotation)


WRAPPER_UPDATES = ("__dict__",)

# A default signature of (*no_sig_args, **no_sig_kwargs)
DFLT_SIGNATURE = signature(lambda *no_sig_args, **no_sig_kwargs: ...)


def _names_of_kind(sig):
    """Compute a tuple containing tuples of names for each kind

    >>> f = lambda a00, /, a11, a12, *a23, a34, a35, a36, **a47: None
    >>> _names_of_kind(Sig(f))
    (('a00',), ('a11', 'a12'), ('a23',), ('a34', 'a35', 'a36'), ('a47',))
    """
    d = defaultdict(list)
    for param in sig.params:
        d[param.kind].append(param.name)
    return tuple(tuple(d[kind]) for kind in range(5))


def maybe_first(items):
    return next(iter(items), None)


def name_of_var_kw_argument(sig):
    var_kw_list = [param.name for param in sig.params if param.kind == VK]
    result = maybe_first(var_kw_list)
    return result


def _map_action_on_cond(kvs, cond, expand):
    for k, v in kvs:
        if cond(
            k
        ):  # make a conditional on (k,v), use type KV, Iterable[KV], expand:KV -> Iterable[KV]
            yield from expand(v[k])  # expand should result in (k,v)
        else:
            yield k, v


def expand_nested_key(d, k):
    for key in d:
        if key == k and isinstance(d[k], dict) and k in d[k]:
            pass

    if k in d and len(d) >= 2:
        return d.items()

    if k in d and isinstance(d[k], dict) and k in d[k]:
        if len(d[k]) == 1:
            return expand_nested_key(d[k], k)
        else:
            return d[k].items()
    else:
        return d.items()


def flatten_if_var_kw(kvs, var_kw_name):
    cond = lambda k: k == var_kw_name
    expand = lambda k: k.items()
    # expand = lambda k: k.values()
    return _map_action_on_cond(kvs, cond, expand)


# TODO: See other signature operating functions below in this module:
#   Do we need them now that we have Sig?
#   Do we want to keep them and have Sig use them?
class Sig(Signature, Mapping):
    """A subclass of inspect.Signature that has a lot of extra api sugar,
    such as
        - making a signature for a variety of input types (callable,
            iterable of callables, parameter lists, strings, etc.)
        - has a dict-like interface
        - signature merging (with operator interfaces)
        - quick access to signature data
        - positional/keyword argument mapping.

    # Positional/Keyword argument mapping

    In python, arguments can be positional (args) or keyword (kwargs).
    ... sometimes both, sometimes a single one is imposed.
    ... and you have variadic versions of both.
    ... and you can have defaults or not.
    ... and all these different kinds have a particular order they must be in.
    It's is mess really. The flexibility is nice -- but still; a mess.

    You only really feel the mess if you try to do some meta-programming with your
    functions.
    Then, methods like `normalize_kind` can help you out, since you can enforce, and
    then assume, some stable interface to your functions.

    Two of the base methods for dealing with positional (args) and keyword (kwargs)
    inputs are:
        - `map_arguments`: Map some args/kwargs input to a keyword-only
            expression of the inputs. This is useful if you need to do some processing
            based on the argument names.
        - `mk_args_and_kwargs`: Translate a fully keyword expression of some
            inputs into an (args, kwargs) pair that can be used to call the function.
            (Remember, your function can have constraints, so you may need to do this.

    The usual pattern of use of these methods is to use `map_arguments`
    to map all the inputs to their corresponding name, do what needs to be done with
    that (example, validation, transformation, decoration...) and then map back to an
    (args, kwargs) pair than can actually be used to call the function.

    Examples of methods and functions using these:
    `call_forgivingly`, `tuple_the_args`, `map_arguments_from_variadics`, `extract_args_and_kwargs`,
    `source_arguments`, and `source_args_and_kwargs`.

    # Making a signature

    You can construct a `Sig` object from a callable,

    >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
    ...     ...
    >>> Sig(f)
    <Sig (w, /, x: float = 1, y=1, *, z: int = 1)>

    but also from any "ParamsAble" object. Such as...
    an iterable of Parameter instances, strings, tuples, or dicts:

    >>> Sig(
    ...     [
    ...         "a",
    ...         ("b", Parameter.empty, int),
    ...         ("c", 2),
    ...         ("d", 1.0, float),
    ...         dict(name="special", kind=Parameter.KEYWORD_ONLY, default=0),
    ...     ]
    ... )
    <Sig (a, b: int, c=2, d: float = 1.0, *, special=0)>
    >>>
    >>> Sig(
    ...     [
    ...         "a",
    ...         "b",
    ...         dict(name="args", kind=Parameter.VAR_POSITIONAL),
    ...         dict(name="kwargs", kind=Parameter.VAR_KEYWORD),
    ...     ]
    ... )
    <Sig (a, b, *args, **kwargs)>

    The parameters of a signature are like a matrix whose rows are the parameters,
    and the 4 columns are their properties: name, kind, default, and annotation
    (the two laste ones being optional).
    You get a row view when doing `Sig(...).parameters.values()`,
    but what if you want a column-view?
    Here's how:

    >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
    ...     ...
    >>>
    >>> s = Sig(f)
    >>> s.kinds  # doctest: +NORMALIZE_WHITESPACE
    {'w': <_ParameterKind.POSITIONAL_ONLY: 0>,
    'x': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
    'y': <_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
    'z': <_ParameterKind.KEYWORD_ONLY: 3>}

    >>> s.annotations
    {'x': <class 'float'>, 'z': <class 'int'>}
    >>> assert (
    ...     s.annotations == f.__annotations__
    ... )  # same as what you get in `__annotations__`
    >>>
    >>> s.defaults
    {'x': 1, 'y': 2, 'z': 3}
    >>> # Note that it's not the same as you get in __defaults__ though:
    >>> assert (
    ...     s.defaults != f.__defaults__ == (1, 2)
    ... )  # not 3, since __kwdefaults__ has that!

    We can sum (i.e. merge) and subtract (i.e. remove arguments) Sig instances.
    Also, Sig instance is callable. It has the effect of inserting it's signature in
    the input
    (in `__signature__`, but also inserting the resulting `__defaults__` and
    `__kwdefaults__`).
    One of the intents is to be able to do things like:

    >>> import inspect
    >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
    ...     ...
    >>> def g(i, w, /, j=2):
    ...     ...
    ...
    >>>
    >>> @Sig.from_objs(f, g, ["a", ("b", 3.14), ("c", 42, int)])
    ... def some_func(*args, **kwargs):
    ...     ...
    >>> inspect.signature(some_func)
    <Sig (w, i, /, a, x: float = 1, y=1, j=2, b=3.14, c: int = 42, *, z: int = 1)>
    >>>
    >>> sig = Sig(f) + g + ["a", ("b", 3.14), ("c", 42, int)] - "b" - ["a", "z"]
    >>> @sig
    ... def some_func(*args, **kwargs):
    ...     ...
    >>> inspect.signature(some_func)
    <Sig (w, i, x: float = 1, y=1, j=2, c: int = 42)>

    """

    # Adding parameter kinds as class attributes for usage convenience
    POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = Parameter.VAR_POSITIONAL
    KEYWORD_ONLY = Parameter.KEYWORD_ONLY
    VAR_KEYWORD = Parameter.VAR_KEYWORD

    def __init__(
        self,
        obj: ParamsAble = None,
        *,
        name=None,
        return_annotation=empty,
        __validate_parameters__=True,
    ):
        """Initialize a Sig instance.
        See Also: `ensure_params` to see what kind of objects you can make `Sig`s with.

        :param obj: A ParamsAble object, which could be:
            - a callable,
            - and iterable of Parameter instances
            - an iterable of strings (representing annotation-less, default-less)
            argument names,
            - tuples: (argname, default) or (argname, default, annotation),
            - dicts: ``{'name': REQUIRED,...}`` with optional `kind`, `default` and
            `annotation` fields
            - None (which will produce an argument-less Signature)

        >>> Sig(["a", "b", "c"])
        <Sig (a, b, c)>
        >>> Sig(
        ...     ["a", ("b", None), ("c", 42, int)]
        ... )  # specifying defaults and annotations
        <Sig (a, b=None, c: int = 42)>
        >>> import inspect
        >>> Sig(
        ...     ["a", ("b", inspect._empty, int)]
        ... )  # specifying an annotation without a default
        <Sig (a, b: int)>
        >>> Sig(["a", "b", "c"], return_annotation=str)  # specifying return annotation
        <Sig (a, b, c) -> str>
        >>> Sig('(a: int = 0, b: str = None, c: float = 3.14) -> str')
        <Sig (a: int = 0, b: str = None, c: float = 3.14) -> str>

        But you can always specify parameters the "long" way

        >>> Sig(
        ...     [inspect.Parameter(name="kws", kind=inspect.Parameter.VAR_KEYWORD)],
        ...     return_annotation=str,
        ... )
        <Sig (**kws) -> str>

        And note that:

        >>> Sig()
        <Sig ()>
        >>> Sig(None)
        <Sig ()>
        """
        if isinstance(obj, str):
            if re.match(r"^\(.*\)", obj):
                # This is a string representation of a signature
                # Dynamically create a function with the given signature then generate
                # the Sig object from this function.
                exec_env = dict()
                f_def = f"def f{obj}: pass"
                exec(f_def, exec_env)
                obj = exec_env["f"]
            else:
                obj = obj.split()

        if isinstance(obj, property):
            obj = obj.fget
        elif isinstance(obj, cached_property):
            obj = obj.func

        if (
            not isinstance(obj, Signature)
            and callable(obj)
            and return_annotation is empty
        ):
            return_annotation = _robust_signature_of_callable(obj).return_annotation
        # TODO: Catch errors and enhance error message with more what-to-do-about it
        #  message. For example,
        #  ValueError: wrong parameter order: positional or keyword parameter before
        #  positional-only parameter
        #  --> Here we could tell the user what pair of variables violated the rule
        super().__init__(
            ensure_params(obj),
            return_annotation=return_annotation,
            __validate_parameters__=__validate_parameters__,
        )
        self.names_of_kind = _names_of_kind(self)

        if len(self.names_of_kind[Parameter.VAR_POSITIONAL]) > 1:
            vps = self.names_of_kind[Parameter.VAR_POSITIONAL]
            raise InvalidSignature(f"You can't have several variadic keywords: {vps}")
        if len(self.names_of_kind[Parameter.VAR_KEYWORD]) > 1:
            vks = self.names_of_kind[Parameter.VAR_KEYWORD]
            raise InvalidSignature(f"You can't have several variadic keywords: {vks}")

        self.name = name or name_of_obj(obj)

    # TODO: Add params for more validation (e.g. arg number/name matching?)
    # TODO: Switch to ignore_incompatible_signatures=False when existing code is
    #   changed accordingly.
    def wrap(
        self,
        func: Callable,
        ignore_incompatible_signatures: bool = True,
        *,
        copy_function: bool | Callable = False,
    ):
        """Gives the input function the signature.

        This is similar to the `functools.wraps` function, but parametrized by a
        signature
        (not a callable). Also, where as both write to the input func's `__signature__`
        attribute, here we also write to
        - `__defaults__` and `__kwdefaults__`, extracting these from `__signature__`
            (functools.wraps doesn't do that at the time of writing this
            (see https://github.com/python/cpython/pull/21379)).
        - `__annotations__` (also extracted from `__signature__`)
        - does not write to `__module__`, `__name__`, `__qualname__`, `__doc__`
            (because again, we're basinig the injecton on a signature, not a function,
            so we have no name, doc, etc...)

        WARNING: The fact that you've modified the signature of your function doesn't
        mean that the decorated function will work as expected (or even work at all).
        See below for examples.

        >>> def f(w, /, x: float = 1, y=2, z: int = 3):
        ...     return w + x * y ** z
        >>> f(0, 1)  # 0 + 1 * 2 ** 3
        8
        >>> f.__defaults__
        (1, 2, 3)
        >>> assert 8 == f(0) == f(0, 1) == f(0, 1, 2) == f(0, 1, 2, 3)

        Now let's create a very similar function to f, but where:
        - w is not position-only
        - x annot is int instead of float, and doesn't have a default
        - z's default changes to 10

        >>> def g(w, x: int, y=2, z: int = 10):
        ...     return w + x * y ** z
        >>> s = Sig(g)
        >>> f = s.wrap(f)
        >>> import inspect
        >>> inspect.signature(f)  # see that
        <Sig (w, x: int, y=2, z: int = 10)>
        >>> # But (unlike with functools.wraps) here we get __defaults__ and
        __kwdefault__
        >>> f.__defaults__  # see that x has no more default & z's default is now 10
        (2, 10)
        >>> f(
        ...     0, 1
        ... )  # see that now we get a different output because using different defaults
        1024

        Remember that you are modifying the signature, not the function itself.
        Signature changes in defaults will indeed change the function's behavior.
        But changes in name or kind will only be reflected in the signature, and
        misalignment with the wrapped function will lead to unexpected results.

        >>> def f(w, /, x: float = 1, y=2, *, z: int = 3):
        ...     return w + x * y ** z
        >>> f(0)  # 0 + 1 * 2 ** 3
        8
        >>> f(0, 1, 2, 3)  # error expected!
        Traceback (most recent call last):
          ...
        TypeError: f() takes from 1 to 3 positional arguments but 4 were given

        But if you try to remove the argument kind constraint by just changing the
        signature, you'll fail.

        >>> def g(w, x: float = 1, y=2, z: int = 3):
        ...     return w + x * y ** z
        >>> f = Sig(g).wrap(f)
        >>> f(0)
        Traceback (most recent call last):
          ...
        TypeError: f() missing 1 required keyword-only argument: 'z'
        >>> f(0, 1, 2, 3)
        Traceback (most recent call last):
          ...
        TypeError: f() takes from 0 to 3 positional arguments but 4 were given

        TODO: Give more explanations why this is.
        """

        # TODO: Should we make copy_function=False the default,
        #  so as to not override decorated function itself by default?
        if copy_function:
            if isinstance(copy_function, bool):
                from i2.util import copy_func as copy_function
            else:
                assert callable(
                    copy_function
                ), f"copy_function must be a callable. This is not: {copy_function}"
            func = copy_function(func)

        # Analyze self and func signature to validate sanity
        _validate_sanity_of_signature_change(func, self, ignore_incompatible_signatures)

        # Change (mutate!) func, writing a new __signature__, __annotations__,
        # __defaults__ and __kwdefaults__
        func.__signature__ = Sig(
            self.parameters.values(), return_annotation=self.return_annotation
        )
        func.__annotations__ = self.annotations
        func.__defaults__, func.__kwdefaults__ = self._dunder_defaults_and_kwdefaults()

        # special case of functools.partial: need to tell .keywords about kwdefaults
        if isinstance(func, partial):
            # TODO: .args can't be modified -- write test to see if problem.
            #   If it is, consider returning a new partial with updated args & keywords.
            # wrapped_func.args = wrapped_func.args + wrapped_func.__defaults__
            func.keywords.update(func.__kwdefaults__)

        return func

    def __call__(self, func: Callable):
        """Gives the input function the signature.
        Just calls Sig.wrap so see docs of Sig.wrap (which contains examples and
        doctests).
        """
        return self.wrap(func)

    @classmethod
    def sig_or_default(cls, obj, default_signature=DFLT_SIGNATURE):
        """Returns a Sig instance, or a default signature if there was a ValueError
        trying to construct it.

        For example, `time.time` doesn't have a signature

        >>> import time
        >>> has_signature(time.time)
        False

        But we can tell `Sig` to give it the default one:

        >>> str(Sig.sig_or_default(time.time))
        '(*no_sig_args, **no_sig_kwargs)'

        That's the default signature, which should work for most purposes.
        You can also specify what the default should be though.

        >>> fake_signature = Sig(lambda *time_takes_no_arguments: ...)
        >>> str(Sig.sig_or_default(time.time, fake_signature))
        '(*time_takes_no_arguments)'

        Careful though. If you assign a signature to a function that is not aligned
        with that actually functioning of the function, bad things will happen.
        In this case, the actual signature of time is the empty signature:

        >>> str(Sig.sig_or_default(time.time, Sig(lambda: ...)))
        '()'

        """
        try:
            # (try to) return cls(obj) if obj is callable:
            if callable(obj):
                sig = cls(obj)
                # Check if we got our default signature (which means no real signature exists)
                if str(sig) == str(DFLT_SIGNATURE):
                    return Sig(default_signature)
                return sig
            else:
                raise TypeError(f"Object is not callable: {obj}")
        except ValueError:
            # if a ValueError is raised, return the default_signature
            return Sig(default_signature)

    @classmethod
    def sig_or_none(cls, obj):
        """Returns a Sig instance, or None if there was a ValueError trying to
        construct it.
        One use case is to be able to tell if an object has a signature or not.

        >>> robust_has_signature = lambda obj: bool(Sig.sig_or_none(obj))
        >>> robust_has_signature(robust_has_signature)  # an easy case
        True
        >>> robust_has_signature(
        ...     Sig
        ... )  # another easy one: This time, a type/class (which is callable, yes)
        True

        But here's where it get's interesting. `print`, a builtin, doesn't have a
        signature through inspect.signature.

        >>> has_signature(print)  # doctest: +SKIP
        False

        But we do get one with robust_has_signature

        >>> robust_has_signature(print)
        True

        """
        return cls.sig_or_default(obj, default_signature=None)

    def __bool__(self):
        return True

    def _positional_and_keyword_defaults(self):
        """Get ``{name: default, ...}`` dicts of positional and keyword defaults.

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     ...
        >>> pos_defaults, kw_defaults = Sig(foo)._positional_and_keyword_defaults()
        >>> pos_defaults
        {'y': 1}
        >>> kw_defaults
        {'z': 1}
        """
        ko_names = self.names_of_kind[KO]
        dflts = self.defaults
        return (
            {name: dflts[name] for name in dflts if name not in ko_names},
            {name: dflts[name] for name in dflts if name in ko_names},
        )

    def _dunder_defaults_and_kwdefaults(self):
        """Get the __defaults__, __kwdefaults__ (i.e. what would be the dunders baring
        these names in a python callable)

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     ...
        >>> __defaults__, __kwdefaults__ = Sig(foo)._dunder_defaults_and_kwdefaults()
        >>> __defaults__
        (1,)
        >>> __kwdefaults__
        {'z': 1}
        """

        pos_defaults, kw_defaults = self._positional_and_keyword_defaults()
        return (
            tuple(
                pos_defaults.values()
            ),  # as known as __defaults__ in python callables
            kw_defaults,  # as known as __kwdefaults__ in python callables
        )

    def to_signature_kwargs(self):
        """The dict of keyword arguments to make this signature instance.

        >>> def f(w, /, x: float = 2, y=1, *, z: int = 0) -> float:
        ...     ...
        >>> Sig(f).to_signature_kwargs()  # doctest: +NORMALIZE_WHITESPACE
        {'parameters':
            [<Parameter "w">,
            <Parameter "x: float = 2">,
            <Parameter "y=1">,
            <Parameter "z: int = 0">],
        'return_annotation': <class 'float'>}

        Note that this does NOT return:
        ```
                {'parameters': self.parameters,
                'return_annotation': self.return_annotation}
        ```
        which would not actually work as keyword arguments of ``Signature``.
        Yeah, I know. Don't ask me, ask the authors of `Signature`!

        Instead, `parammeters` will be ``list(self.parameters.values())``, which does
        work.

        """
        return {
            "parameters": list(self.parameters.values()),
            "return_annotation": self.return_annotation,
        }

    def to_simple_signature(self):
        """A builtin ``inspect.Signature`` instance equivalent (i.e. without the extra
        properties and methods)

        >>> def f(w, /, x: float = 2, y=1, *, z: int = 0):
        ...     ...
        >>> Sig(f).to_simple_signature()
        <Signature (w, /, x: float = 2, y=1, *, z: int = 0)>

        """
        return Signature(**self.to_signature_kwargs())

    def pair_with(self, other_sig) -> "SigPair":
        """Get an object that pairs with another signature for comparison, merging, etc.

        See `SigPair` for more details.
        """
        return SigPair(self, other_sig)

    def is_call_compatible_with(self, other_sig, *, param_comparator: Callable = None):
        """Return True if the signature is compatible with ``other_sig``. Meaning that
        all valid ways to call the signature are valid for ``other_sig``.
        """
        return is_call_compatible_with(
            self, other_sig, param_comparator=param_comparator
        )

    # TODO: Make these dunders open/close
    # def __le__(self, other_sig):
    #     """The "less than or equal" operator (<=).
    #     Return True if the signature is compatible with ``other_sig``. Meaning that
    #     all valid ways to call the signature are valid for ``other_sig``.
    #     """
    #     return self.is_call_compatible_with(other_sig)

    # def __ge__(self, other_sig):
    #     """The "greater than or equal" operator (>=).
    #     Return True if ``other_sig`` is compatible with the signature. Meaning that
    #     all valid ways to call ``other_sig`` are valid for the signature.
    #     """
    #     return other_sig <= self

    @classmethod
    def from_objs(
        cls,
        *objs,
        default_conflict_method: str = DFLT_DEFAULT_CONFLICT_METHOD,
        return_annotation=empty,
        **name_and_dflts,
    ):
        objs = list(objs)
        for name, default in name_and_dflts.items():
            objs.append([{"name": name, "kind": PK, "default": default}])
        if len(objs) > 0:
            first_obj, *objs = objs
            sig = cls(ensure_params(first_obj))
            for obj in objs:
                sig = sig.merge_with_sig(
                    obj, default_conflict_method=default_conflict_method
                )
                # sig = sig + obj
            return Sig(sig, return_annotation=return_annotation)
        else:  # if no objs are given
            return cls(return_annotation=return_annotation)  # return an empty signature

    @classmethod
    def from_params(cls, params):
        if isinstance(params, Parameter):
            params = (params,)
        return cls(params)

    @property
    def params(self):
        """Just list(self.parameters.values()), because that's often what we want.
        Why a Sig.params property when we already have a Sig.parameters property?

        Well, as much as is boggles my mind, it so happens that the Signature.parameters
        is a name->Parameter mapping, but the Signature argument `parameters`,
        though baring the same name,
        is expected to be a list of Parameter instances.

        So Sig.params is there to restore semantic consistence sanity.
        """
        return list(self.parameters.values())

    @property
    def names(self):
        return list(self.keys())

    @property
    def kinds(self):
        return {p.name: p.kind for p in self.values()}

    @property
    def defaults(self):
        """A ``{name: default,...}`` dict of defaults (regardless of kind)"""
        return {p.name: p.default for p in self.values() if p.default is not p.empty}

    @property
    def _defaults_(self):
        """What the ``__defaults__`` value would be for a func of the same signature"""
        return tuple(
            p.default
            for p in self.values()
            if (p.default is not p.empty and p.kind != KO)
        )

    @property
    def _kwdefaults_(self):
        """What the ``__kwdefaults__`` value would be for a func of the same signature"""
        return {
            p.name: p.default
            for p in self.values()
            if p.default is not p.empty and p.kind == KO
        }

    @property
    def annotations(self):
        """{arg_name: annotation, ...} dict of annotations of the signature.
        What `func.__annotations__` would give you.
        """
        return {
            p.name: p.annotation for p in self.values() if p.annotation is not p.empty
        }

    def detail_names_by_kind(self):
        return (
            self.names_of_kind[PO],
            self.names_of_kind[PK],
            next(iter(self.names_of_kind[VP]), None),
            self.names_of_kind[KO],
            next(iter(self.names_of_kind[VK]), None),
        )

    # TODO: Can be cleaned and generalized (include/exclude, function filter etc.)
    def get_names(self, spec, *, conserve_sig_order=True, allow_excess=False):
        """Return a tuple of names corresponding to the given spec.

        :param spec: An integer, string, or iterable of intergers and strings
        :param conserve_sig_order: Whether to order according to the signature
        :param allow_excess: Whether to allow items in spec that are not in signature

        >>> sig = Sig('a b c d e')
        >>> sig.get_names(0)
        ('a',)
        >>> sig.get_names([0, 2])
        ('a', 'c')
        >>> sig.get_names('b')
        ('b',)
        >>> sig.get_names([0, 'c', -1])
        ('a', 'c', 'e')

        See that by default the order of the signature is conserved:

        >>> sig.get_names('b e d')
        ('b', 'd', 'e')

        But you can change that default to conserve the order of the ``spec`` instead:

        >>> sig.get_names('b e d', conserve_sig_order=False)
        ('b', 'e', 'd')

        By default, you can't mention names that are not in signature.
        To allow this (making ``spec`` have "extract these" interpretation),
        set ``allow_excess=True``:

        >>> sig.get_names(['a', 'c', 'e', 'g', 'h'], allow_excess=True)
        ('a', 'c', 'e')

        """
        if isinstance(spec, str):
            spec = spec.split()
        elif isinstance(spec, int):
            spec = [spec]
        if isinstance(spec, Iterable):

            def find_names():
                names = self.names
                for item in spec:
                    if isinstance(item, int):
                        if item < len(names):
                            yield names[item]
                        elif not allow_excess:
                            raise IndexError(
                                f"There are only {len(names)} names in the signatures,"
                                f"but you asked for the index: {item}"
                            )
                    else:
                        if item in names:
                            yield item
                        elif not allow_excess:
                            raise ValueError(
                                f"No such param name in signatures: {item}"
                            )

            matched_names = tuple(find_names())
            if conserve_sig_order:
                _matched_names = tuple(x for x in self.names if x in matched_names)
                matched_names = _matched_names + tuple(
                    x for x in matched_names if x not in _matched_names
                )
            return matched_names
        else:
            raise TypeError(f"Unknown spec type: {spec}")

    def __iter__(self):
        return iter(self.parameters)

    def __len__(self):
        return len(self.parameters)

    # TODO: Return type inconsistent. When k is a string, returns Parameter,
    #  when an iterable of strings (or 'space separated argument names'),
    #  returns a signature. Could also return a single argument signatures.
    #  Behavior might be confusing. Pros/Cons? See if any current users of getitem,
    #  and switch to single arg signature return (that's consistent, and convenience
    #  of sig[argname] is weak (given sig.params[argname] does it)!)
    def __getitem__(self, k):
        if isinstance(k, int) or isinstance(k, slice):
            # TODO: Could extend slice handing to be able to use names as start and stop
            k = self.names[k]
        if isinstance(k, str):
            names = k.split()  # to handle 'multiple args in a string'
            if len(names) == 1:
                return self.parameters[k]
        else:
            assert isinstance(k, Iterable), f"key should be iterable, was: {k}"
            names = k
        params = [self[name] for name in names]
        return Sig.from_params(params)

    # TODO: Deprecate. Should use names_of_kind directly
    def names_for_kind(self, kind):
        """Get the arg names tuple for a given kind.
        Note, if you need to do this several times, or for several kinds, use
        ``names_of_kind`` property (a tuple) instead: It groups all names of kinds once,
        and caches the result.
        """
        from warnings import warn

        warn("Deprecated", DeprecationWarning)
        return self.names_of_kind[kind]

    # TODO: Consider using names_of_kind in other methods/properties

    @property
    def has_var_kinds(self):
        """
        >>> Sig(lambda x, *, y: None).has_var_kinds
        False
        >>> Sig(lambda x, *y: None).has_var_kinds
        True
        >>> Sig(lambda x, **y: None).has_var_kinds
        True
        """
        return bool(self.names_of_kind[VP]) or bool(self.names_of_kind[VK])
        # Old version:
        # return any(p.kind in var_param_kinds for p in self.values())

    @property
    def index_of_var_positional(self):
        """The index of the VAR_POSITIONAL param kind if any, and None if not.
        See also, Sig.index_of_var_keyword

        >>> assert Sig(lambda x, *y, z: 0).index_of_var_positional == 1
        >>> assert Sig(lambda x, /, y, **z: 0).index_of_var_positional == None
        """
        return next((i for i, p in enumerate(self.params) if p.kind == VP), None)

    @property
    def var_positional_name(self):
        idx = self.index_of_var_positional
        if idx is not None:
            return self.names[idx]
        # else returns None

    @property
    def has_var_positional(self):
        """
        Use index_of_var_positional or var_keyword_name directly when needing that
        information as well. This will avoid having to check the kinds list twice.
        """
        return any(p.kind == VP for p in self.values())

    @property
    def index_of_var_keyword(self):
        """The index of a VAR_KEYWORD param kind if any, and None if not.
        See also, Sig.index_of_var_positional

        >>> assert Sig(lambda **kwargs: 0).index_of_var_keyword == 0
        >>> assert Sig(lambda a, **kwargs: 0).index_of_var_keyword == 1
        >>> assert Sig(lambda a, *args, **kwargs: 0).index_of_var_keyword == 2

        And if there's none...

        >>> assert Sig(lambda a, *args, b=1: 0).index_of_var_keyword is None

        """
        last_arg_idx = len(self) - 1
        if last_arg_idx != -1:
            if self.params[last_arg_idx].kind == VK:
                return last_arg_idx
        # else returns None

    @property
    def var_keyword_name(self):
        idx = self.index_of_var_keyword
        if idx is not None:
            return self.names[idx]
        # else returns None

    @property
    def has_var_keyword(self):
        """
        Use index_of_var_keyword or var_keyword_name directly when needing that
        information as well. This will avoid having to check the kinds list twice.
        """
        return any(p.kind == VK for p in self.values())

    @property
    def required_names(self):
        """A tuple of required names, preserving the original signature order.

        A required name is that must be given in a function call, that is, the name of a
        paramater that doesn't have a default, and is not a variadic.

        That lost one is a frequent gotcha, so oo not fall in that gotcha that easily,
        we provide a property that contains what we need.

        >>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
        >>> Sig(f).required_names
        ('a00', 'a11', 'a12', 'a34')
        """
        # Note: This is quicker than using self.names_of_kind:
        return tuple(
            p.name
            for p in self.params
            if p.default is empty and p.kind not in var_param_kinds
        )

    @property
    def n_required(self):
        """The number of required arguments.
        A required argument is one that doesn't have a default, nor is VAR_POSITIONAL
        (*args) or VAR_KEYWORD (**kwargs).
        Note: Sometimes a minimum number of arguments in VAR_POSITIONAL and
        VAR_KEYWORD are in fact required,
        but we can't see this from the signature, so we can't tell you about that! You
        do the math.

        >>> f = lambda a00, /, a11, a12, *a23, a34, a35=1, a36='two', **a47: None
        >>> Sig(f).n_required
        4
        """
        return len(self.required_names)

    @property
    def positional_names(self):
        return self.names_of_kind[PO] + self.names_of_kind[PK]

    @property
    def keyword_names(self):
        return self.names_of_kind[PK] + self.names_of_kind[KO]

    def _transform_params(self, changes_for_name: dict):
        for name in self:
            if name in changes_for_name:
                p = changes_for_name[name]
                if isinstance(p, Parameter):
                    p = parameter_to_dict(p)
                yield self[name].replace(**p)
            else:
                # if name is not in params, just use existing param
                yield self[name]

    def modified(self, /, _allow_reordering=False, **changes_for_name):
        """Returns a modified (new) signature object.

        Note: This function doesn't modify the signature, but creates a modified copy
        of the signature.

        IMPORTANT WARNING: This is an advanced feature. Avoid wrapping a function with
        a modified signature, as this may not have the intended effect.

        >>> def foo(pka, *vpa, koa, **vka): ...
        >>> sig = Sig(foo)
        >>> sig
        <Sig (pka, *vpa, koa, **vka)>
        >>> assert sig.kinds['pka'] == PK

        Let's make a signature that is the same as sig, except that
            - `poa` is given a PO (POSITIONAL_ONLY) kind insteadk of PK
            - `koa` is given a default of None
            - the signature is given a return_annotation of str

        >>> new_sig = sig.modified(
        ...     pka={'kind': PO},
        ...     koa={'default': None},
        ...     return_annotation=str
        ... )
        >>> new_sig
        <Sig (pka, /, *vpa, koa=None, **vka) -> str>
        >>> assert new_sig.kinds['pka'] == PO  # now pos is of the PO kind!

        Here's an example of changing signature parameters in bulk.
        Here we change all kinds to be the friendly PK kind.

        >>> sig.modified(**{name: {'kind': PK} for name in sig.names})
        <Sig (pka, vpa, koa, vka)>

        Repetition of the above: This gives you a signature with all PK kinds.
        If you wrap a function with it, it will look like it has all PK kinds.
        But that doesn't mean you can actually use thenm as such.
        You'll need to modify (decorate further) your function further to reflect
        its new signature.

        On the other hand, if you decorate a function with a sig that adds or modifies
        defaults, these defaults will actually be used (unlike with `functools.wraps`).

        """
        new_return_annotation = changes_for_name.pop(
            "return_annotation", self.return_annotation
        )

        if _allow_reordering:
            params = sort_params(self._transform_params(changes_for_name))
        else:
            params = list(self._transform_params(changes_for_name))

        return Sig(params, name=self.name, return_annotation=new_return_annotation)

    def sort_params(self):
        """Returns a signature with the parameters sorted by kind and default presence."""
        sorted_params = sort_params(self.params)
        return type(self)(
            sorted_params, name=self.name, return_annotation=self.return_annotation
        )

    def ch_param_attrs(
        self, /, param_attr, *arg_new_vals, _allow_reordering=False, **kwargs_new_vals
    ):
        """Change a specific attribute of the params, returning a modified signature.
        This is a convenience method for the modified method when we're targetting
        a fixed param attribute: 'name', 'kind', 'default', or 'annotation'

        Instead of having to do this

        >>> def foo(a, *b, **c): ...
        >>> Sig(foo).modified(a={'name': 'A'}, b={'name': 'B'}, c={'name': 'C'})
        <Sig (A, *B, **C)>

        We can simply do this

        >>> Sig(foo).ch_param_attrs('name', a='A', b='B', c='C')
        <Sig (A, *B, **C)>

        One quite useful thing you can do with this is to set defaults, or set defaults
        where there are none. If you wrap your function with such a modified signature,
        you get a "curried" version of your function (called "partial" in python).
        (Note that the `functools.wraps` won't deal with defaults "correctly", but
        wrapping with `Sig` objects takes care of that oversight!)

        >>> def foo(a, b, c):
        ...     return a + b * c
        >>> special_foo = Sig(foo).ch_param_attrs('default', b=2, c=3)(foo)
        >>> Sig(special_foo)
        <Sig (a, b=2, c=3)>
        >>> special_foo(5)  # should be 5 + 2 * 3 == 11
        11


        # TODO: Would like to make this work (reordering)
        # Now, if you want to set a default for a but not b and c for example, you'll
        # get complaints:
        #
        # ```
        # ValueError: non-default argument follows default argument
        # ```
        #
        # will tell you.
        #
        # It's true. But if you're fine with rearranging the argument order,
        # `ch_param_attrs` can take care of that for you.
        # You'll have to tell it explicitly that you wish for this though, because
        # it's conservative.
        #
        # >>> # Note that for time being, Sig.wraps doesn't make a copy of the function
        # >>> #  so we need to redefine foo here@
        # >>> def foo(a, b, c):
        # ...     return a + b * c
        # >>> wrapper = Sig(foo).ch_param_attrs(
        # ... 'default', a=10, _allow_reordering=True
        # ... )
        # >>> another_foo = wrapper(foo)
        # >>> Sig(another_foo)
        # <Sig (b, c, a=10)>
        # >>> another_foo(2, 3)  # should be 10 + (2 * 3) =
        # 16

        """

        if not param_attr in param_attributes:
            raise ValueError(
                f"param_attr needs to be one of: {param_attributes}.",
                f" Was: {param_attr}",
            )
        all_pk_self = self.modified(
            _allow_reordering=True, **{name: {"kind": PK} for name in self.names}
        )
        new_attr_vals = all_pk_self.bind_partial(
            *arg_new_vals, **kwargs_new_vals
        ).arguments
        changes_for_name = {
            name: {param_attr: val} for name, val in new_attr_vals.items()
        }
        return self.modified(_allow_reordering=_allow_reordering, **changes_for_name)

    # Note: Oh, functools, why do you make currying so limited!
    # ch_names = partialmethod(ch_param_attrs, param_attr="name")
    # ch_kinds = partialmethod(ch_param_attrs, param_attr="kind", _allow_reordering=True)
    # ch_defaults = partialmethod(
    #     ch_param_attrs, param_attr="default", _allow_reordering=True
    # )
    # ch_annotations = partialmethod(ch_param_attrs, param_attr="annotation")

    def ch_names(self, /, **changes_for_name):
        argnames_not_in_sig = changes_for_name.keys() - self.keys()
        if argnames_not_in_sig:
            raise ValueError(
                f"argument names not in signature: {', '.join(argnames_not_in_sig)}"
            )
        return self.ch_param_attrs("name", **changes_for_name)

    def ch_kinds(self, /, _allow_reordering=True, **changes_for_name):
        return self.ch_param_attrs(
            "kind", _allow_reordering=_allow_reordering, **changes_for_name
        )

    def ch_kinds_to_position_or_keyword(self):
        return all_pk_signature(self)

    def ch_defaults(self, /, _allow_reordering=True, **changes_for_name):
        return self.ch_param_attrs(
            "default", _allow_reordering=_allow_reordering, **changes_for_name
        )

    def ch_annotations(self, /, **changes_for_name):
        return self.ch_param_attrs("annotation", **changes_for_name)

    def add_optional_keywords(
        self=None, /, kwarg_and_defaults=None, kwarg_annotations=None
    ):
        """Add optional keyword arguments to a signature.

        >>> @Sig.add_optional_keywords({"c": 2, "d": 3}, {"c": int})
        ... def foo(a, *, b=1, **kwargs):
        ...     return f"{a=}, {b=}, {kwargs=}"
        ...

        You can still call the function as before, and like before, any "extra" keyword
        arguments will be passed to kwargs:

        >>> foo(0, d=10)
        "a=0, b=1, kwargs={'d': 10}"

        The difference is that now the signature of `foo` now has `c` and `d`:

        >>> str(Sig(foo))
        '(a, *, c: int = 2, d=3, b=1, **kwargs)'

        """

        # Resolve arguments ( to be able to use this method as a decorator)
        if isinstance(self, dict):
            if kwarg_and_defaults is not None:
                kwarg_annotations = kwarg_and_defaults
                kwarg_and_defaults = None
            if kwarg_and_defaults is None:
                kwarg_and_defaults = self
            self = None

        # If self is None, a factory is returned
        if self is None:
            return partial(
                _add_optional_keywords,
                kwarg_and_defaults=kwarg_and_defaults,
                kwarg_annotations=kwarg_annotations,
            )
        else:  # if not, apply _add_optional_keywords to self
            return _add_optional_keywords(
                self, kwarg_and_defaults, kwarg_annotations=kwarg_annotations
            )

    # TODO: Make default_conflict_method be able to be a callable and get rid of string
    #  mapping complexity in merge_with_sig code
    def merge_with_sig(
        self,
        sig: ParamsAble,
        ch_to_all_pk: bool = False,
        *,
        default_conflict_method: SigMergeOptions = DFLT_DEFAULT_CONFLICT_METHOD,
    ):
        """Return a signature obtained by merging self signature with another signature.
        Insofar as it can, given the kind precedence rules, the arguments of self will
        appear first.

        :param sig: The signature to merge with.
        :param ch_to_all_pk: Whether to change all kinds of both signatures to PK (
        POSITIONAL_OR_KEYWORD)
        :return:

        >>> def func(a=None, *, b=1, c=2):
        ...     ...
        ...
        >>>
        >>> s = Sig(func)
        >>> s
        <Sig (a=None, *, b=1, c=2)>

        Observe where the new arguments ``d`` and ``e`` are placed,
        according to whether they have defaults and what their kind is:

        >>> s.merge_with_sig(["d", "e"])
        <Sig (d, e, a=None, *, b=1, c=2)>
        >>> s.merge_with_sig(["d", ("e", 4)])
        <Sig (d, a=None, e=4, *, b=1, c=2)>
        >>> s.merge_with_sig(["d", dict(name="e", kind=KO, default=4)])
        <Sig (d, a=None, *, b=1, c=2, e=4)>
        >>> s.merge_with_sig(
        ...     [dict(name="d", kind=KO), dict(name="e", kind=KO, default=4)]
        ... )
        <Sig (a=None, *, d, b=1, c=2, e=4)>

        If the kind of the params is not important, but order is, you can specify
        ``ch_to_all_pk=True``:

        >>> s.merge_with_sig(["d", "e"], ch_to_all_pk=True)
        <Sig (d, e, a=None, b=1, c=2)>
        >>> s.merge_with_sig([("d", 3), ("e", 4)], ch_to_all_pk=True)
        <Sig (a=None, b=1, c=2, d=3, e=4)>

        """
        if ch_to_all_pk:
            _self = Sig(all_pk_signature(self))
            _sig = Sig(all_pk_signature(ensure_signature(sig)))
        else:
            _self = self
            _sig = Sig(sig)

        # Validation of the signatures

        _msg = f"\nHappened during an attempt to merge {self} and {sig}"
        errors = {}

        # Check if both signatures have VAR_POSITIONAL parameters
        if _self.has_var_keyword and _sig.has_var_keyword:
            errors["var_positional_conflict"] = (
                f"Can't merge two signatures if they both have a VAR_POSITIONAL parameter: {_msg}"
            )

        # Check if both signatures have VAR_KEYWORD parameters
        if _self.has_var_keyword and _sig.has_var_keyword:
            errors["var_keyword_conflict"] = (
                f"Can't merge two signatures if they both have a VAR_KEYWORD parameter: {_msg}"
            )

        # Check if parameters with the same name have the same kind
        if not all(
            _self[name].kind == _sig[name].kind for name in _self.keys() & _sig.keys()
        ):
            errors["kind_mismatch"] = (
                "During a signature merge, if two names are the same, they must have the "
                f"**same kind**:\n\t{_msg}\n"
                "Tip: If you're trying to merge functions in some way, consider decorating "
                "them with a signature mapping that avoids the argument name clashing"
            )

        # Check if default_conflict_method is a valid SigMergeOption
        if default_conflict_method not in get_args(SigMergeOptions):
            errors["invalid_conflict_method"] = (
                "default_conflict_method should be one of: "
                f"{get_args(SigMergeOptions)}"
            )

        if errors:
            # TODO: Raise all errors at once?
            # TODO: Raise custom errors with more info?

            # raise the first error
            error_msg = next(iter(errors.values()))
            raise IncompatibleSignatures(error_msg, sig1=_self, sig2=_sig)

        if default_conflict_method == "take_first":
            _sig = _sig - set(_self.keys() & _sig.keys())
        elif default_conflict_method == "fill_defaults_and_annotations":
            _self = _fill_defaults_and_annotations(_self, _sig)
            _sig = _fill_defaults_and_annotations(_sig, _self)

        if not all(
            _self[name].default == _sig[name].default
            for name in _self.keys() & _sig.keys()
        ):
            # if default_conflict_method == 'take_first':
            #     _sig = _sig - set(_self.keys() & _sig.keys())
            # else:

            error_msg = (
                "During a signature merge, if two names are the same they must have the"
                f"**same default**:\n\t{_msg}\n"
                "Tip: If you're trying to merge functions in some way, consider "
                "decorating "
                "them with a signature mapping that avoids the argument name clashing."
                "You can also set ch_to_all_pk=True to ignore the kind of the "
                'parameters or change the default_conflict_method to "take_first",'
                "or another method that suits your needs."
            )

            raise IncompatibleSignatures(error_msg, sig1=_self, sig2=_sig)

        # assert all(
        #     _self[name].default == _sig[name].default
        #     for name in _self.keys() & _sig.keys()
        # ), (
        #     'During a signature merge, if two names are the same, they must have the '
        #     f'**same default**:\n\t{_msg}\n'
        #     "Tip: If you're trying to merge functions in some way, consider
        #     decorating "
        #     "them a signature mapping that "
        #     'avoids the argument name clashing'
        # )

        params = list(
            self._chain_params_of_signatures(
                _self.without_defaults,
                _sig.without_defaults,
                _self.with_defaults,
                _sig.with_defaults,
            )
        )
        params.sort(key=lambda p: p.kind)
        return self.__class__(params)

    def __add__(self, sig: ParamsAble):
        """Merge two signatures (casting all non-VAR kinds to POSITIONAL_OR_KEYWORD
        before hand)

        Important Notes:
        - The resulting Sig will loose it's return_annotation if it had one.
            This is to avoid making too many assumptions about how the sig sum will be
            used.
            If a return_annotation is needed (say, for composition, the last
            return_annotation
            summed), one can subclass Sig and overwrite __add__
        - POSITION_ONLY and KEYWORD_ONLY kinds will be replaced by
        POSITIONAL_OR_KEYWORD kind.
        This is to simplify the interface and code.
        If the user really wants to maintain those kinds, they can replace them back
        after the fact.

        >>> def f(w, /, x: float = 1, y=1, *, z: int = 1):
        ...     ...
        >>> def h(i, j, w):
        ...     ...  # has a 'w' argument, like f and g
        ...
        >>> def different(a, b: str, c=None):
        ...     ...  # No argument names in common with other functions

        >>> Sig(f) + Sig(different)
        <Sig (w, a, b: str, x: float = 1, y=1, z: int = 1, c=None)>
        >>> Sig(different) + Sig(f)
        <Sig (a, b: str, w, c=None, x: float = 1, y=1, z: int = 1)>

        The order of the first signature will take precedence over the second,
        but default-less arguments have to come before arguments with defaults.
         first, and Note the difference of the orders.

        >>> Sig(f) + Sig(h)
        <Sig (w, i, j, x: float = 1, y=1, z: int = 1)>
        >>> Sig(h) + Sig(f)
        <Sig (i, j, w, x: float = 1, y=1, z: int = 1)>

        The sum of two Sig's takes a safe-or-blow-up-now approach.
        If any of the arguments have different defaults or annotations, summing will
        raise an AssertionError.
        It's up to the user to decorate their input functions to express the default
        they actually desire.

        >>> def ff(w, /, x: float, y=1, *, z: int = 1):
        ...     ...  # just like f, but without the default for x
        >>> Sig(f) + Sig(ff)  # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
        Traceback (most recent call last):
        ...
        IncompatibleSignatures: During a signature merge, if two names are the same, they must
        have the **same default**
        ...

        >>> def hh(i, j, w=1):
        ...     ...  # like h, but w has a default
        ...
        >>> Sig(h) + Sig(hh)  # doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
        Traceback (most recent call last):
        ...
        IncompatibleSignatures: During a signature merge, if two names are the same, they must
        have the **same default**
        ...

        >>> Sig(f) + [
        ...     "w",
        ...     ("y", 1),
        ...     ("d", 1.0, float),
        ...     dict(name="special", kind=Parameter.KEYWORD_ONLY, default=0),
        ... ]
        <Sig (w, x: float = 1, y=1, z: int = 1, d: float = 1.0, special=0)>

        """
        return self.merge_with_sig(sig, ch_to_all_pk=True)

    def __radd__(self, sig: ParamsAble):
        """Adding on the right.
        The raison d'être for this is so that you can start your summing with any
        signature speccifying
         object that Sig will be able to resolve into a signature. Like this:

        >>> ["first_arg", ("second_arg", 42)] + Sig(lambda x, y: x * y)
        <Sig (first_arg, x, y, second_arg=42)>

        Note that the ``second_arg`` doesn't actually end up being the second argument
        because
        it has a default and x and y don't. But if you did this:

        >>> ["first_arg", ("second_arg", 42)] + Sig(lambda x=0, y=1: x * y)
        <Sig (first_arg, second_arg=42, x=0, y=1)>

        you'd get what you expect.

        Of course, we could have just obliged you to say ``Sig(['first_arg',
        ('second_arg', 42)])``
        explicitly and spare ourselves yet another method.
        The reason we made ``__radd__`` is so we can make it handle 0 + Sig(...),
        so that you can
        merge an iterable of signatures like this:

        >>> def f(a, b, c):
        ...     ...
        ...
        >>> def g(c, b, e):
        ...     ...
        ...
        >>> sigs = map(Sig, [f, g])
        >>> sum(sigs)
        <Sig (a, b, c, e)>

        Let's say, for whatever reason (don't ask me), you wanted to make a function
        that contains all the
        arguments of all the functions of ``os.path`` (that don't contain any var arg
        kinds).

        >>> import os.path
        >>> funcs = list(
        ...     filter(
        ...         callable,
        ...         (
        ...             getattr(os.path, a)
        ...             for a in dir(os.path)
        ...             if not a.startswith("_")
        ...         ),
        ...     )
        ... )
        >>> sigs = filter(lambda sig: not sig.has_var_kinds, map(Sig, funcs))
        >>> # Note: Skipping because not stable between python versions
        >>> sum(sigs)  # doctest: +SKIP
        <Sig (path, p, paths, m, filename, s, f1, f2, fp1, fp2, s1, s2, start=None)>
        """
        if sig == 0:  # so that we can do ``sum(iterable_of_sigs)``
            sig = Sig([])
        else:
            sig = Sig(sig)
        return sig.merge_with_sig(self)

    def remove_names(self, names):
        names = {p.name for p in ensure_params(names)}
        new_params = {
            name: p for name, p in self.parameters.items() if name not in names
        }
        return self.__class__(new_params, return_annotation=self.return_annotation)

    def add_params(self, params: Iterable):
        """Creates a new instance of Sig after merging the parameters of this signature
        with a list of new parameters. The new list of parameters is automatically
        sorted based on signature constraints given by kinds and default values.
        See Python native signature documentation for more details.

        >>> s = Sig('(a, /, b, *, c)')
        >>> s.add_params([
        ...     Param('kwargs', VK),
        ...     dict(name='d', kind=KO),
        ...     Param('args', VP),
        ...     'e',
        ...     Param('f', PO),
        ... ])
        <Sig (a, f, /, b, e, *args, c, d, **kwargs)>
        """

        def comparator(param):
            return (param.kind, param.kind == KO or param.default is not empty)

        new_params = self.params + [ensure_param(p) for p in params]
        new_params = sorted(new_params, key=comparator)
        return type(self)(new_params)

    def __sub__(self, sig):
        return self.remove_names(sig)

    @staticmethod
    def _chain_params_of_signatures(*sigs):
        """Yields Parameter instances taken from sigs without repeating the same name
        twice.

        >>> str(list(
        ...     Sig._chain_params_of_signatures(
        ...         Sig(lambda x, *args, y=1: ...),
        ...         Sig(lambda x, y, z, **kwargs: ...),
        ...     )
        ...   )
        ... )
        '[<Parameter "x">, <Parameter "*args">, <Parameter "y=1">, <Parameter "z">, <Parameter "**kwargs">]'

        """
        already_merged_names = set()
        for s in sigs:
            for p in s.parameters.values():
                if p.name not in already_merged_names:
                    yield p
                already_merged_names.add(p.name)

    @property
    def without_defaults(self):
        """Sub-signature containing only "required" (i.e. without defaults) parameters.

        >>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).without_defaults)
        ['a', 'b']
        """
        return self.__class__(
            p for p in self.values() if not param_has_default_or_is_var_kind(p)
        )

    @property
    def with_defaults(self):
        """Sub-signature containing only "not required" (i.e. with defaults) parameters.

        >>> list(Sig(lambda *args, a, b, x=1, y=1, **kwargs: ...).with_defaults)
        ['args', 'x', 'y', 'kwargs']
        """
        return self.__class__(
            p for p in self.values() if param_has_default_or_is_var_kind(p)
        )

    def normalize_kind(
        self,
        kind=PK,
        except_kinds=var_param_kinds,
        add_defaults_if_necessary=False,
        argname_to_default=None,
        allow_reordering=False,
    ):
        except_kinds = except_kinds or set()
        if add_defaults_if_necessary:
            if argname_to_default is None:

                def argname_to_default(argname):
                    return None

        def changed_params():
            there_was_a_default = False
            for p in self.parameters.values():
                if p.kind not in except_kinds:
                    if add_defaults_if_necessary:
                        if there_was_a_default and p.default is _empty:
                            p = p.replace(kind=kind, default=argname_to_default(p.name))
                        there_was_a_default = p.default is not _empty
                    else:
                        p = p.replace(kind=kind)
                yield p

        params = list(changed_params())
        try:
            return type(self)(params, return_annotation=self.return_annotation)
        except ValueError as e:
            if allow_reordering:
                return self.sort_params()
            else:
                raise

    def map_arguments(
        self,
        args: tuple = None,
        kwargs: dict = None,
        *,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
    ) -> dict:
        """Map arguments (args and kwargs) to the parameters of function's signature.

        When you need to manage how the arguments of a function are specified,
        you need to take care of
        multiple cases depending on whether they were specified as positional arguments
        (`args`) or keyword arguments (`kwargs`).

        The `map_arguments` (and it's sorta-inverse inverse,
        `mk_args_and_kwargs`)
        are there to help you manage this.

        If you could rely on the the fact that only `kwargs` were given it would
        reduce the complexity of your code.
        This is why we have the `all_pk_signature` function in `signatures.py`.

        We also need to have a means to make a `kwargs` only from the actual `(*args,
        **kwargs)` used at runtime.
        We have `Signature.bind` (and `bind_partial`) for that.

        But these methods will fail if there is extra stuff in the `kwargs`.
        Yet sometimes we'd like to have a `dict` that services several functions that
        will extract their needs from it.

        That's where  `Sig.map_arguments_from_variadics(*args, **kwargs)` is needed.
        :param args: The args the function will be called with.
        :param kwargs: The kwargs the function will be called with.
        :param apply_defaults: (bool) Whether to apply signature defaults to the
        non-specified argument names
        :param allow_partial: (bool) True iff you want to allow partial signature
        fulfillment.
        :param allow_excess: (bool) Set to True iff you want to allow extra kwargs
        items to be ignored.
        :param ignore_kind: (bool) Set to True iff you want to ignore the position and
        keyword only kinds,
            in order to be able to accept args and kwargs in such a way that there can
            be cross-over
            (args that are supposed to be keyword only, and kwargs that are supposed
            to be positional only)
        :return: An {param_name: arg_val, ...} dict

        See also the sorta-inverse of this function: mk_args_and_kwargs

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.map_arguments((11, 22, "you"), dict(z="zoo"))
        ...     == sig.map_arguments((11, 22), dict(y="you", z="zoo"))
        ...     == {"w": 11, "x": 22, "y": "you", "z": "zoo"}
        ... )

        By default, `apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> sig.map_arguments(args=(11,), kwargs={"x": 22})
        {'w': 11, 'x': 22}

        But if you specify `apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22}, apply_defaults=True
        ... )
        {'w': 11, 'x': 22, 'y': 'YY', 'z': 'ZZ'}

        By default, `ignore_excess=False`, so specifying kwargs that are not in the
        signature will lead to an exception.

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}
        ... )
        Traceback (most recent call last):
            ...
        TypeError: got an unexpected keyword argument 'not_in_sig'

        Specifying `allow_excess=True` will ignore such excess fields of kwargs.
        This is useful when you want to source several functions from a same dict.

        >>> sig.map_arguments(
        ...     args=(11,), kwargs={"x": 22, "not_in_sig": -1}, allow_excess=True
        ... )
        {'w': 11, 'x': 22}

        On the other side of `ignore_excess` you have `allow_partial` that will allow
        you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> sig.map_arguments(args=(), kwargs={"x": 22})
        Traceback (most recent call last):
        ...
        TypeError: missing a required argument: 'w'

        But if you specify `allow_partial=True`...

        >>> sig.map_arguments(
        ...     args=(), kwargs={"x": 22}, allow_partial=True
        ... )
        {'x': 22}

        That's a lot of control (eight combinations total), but not everything is
        controllable here:
        Position only and keyword only kinds need to be respected:

        >>> sig.map_arguments(args=(1, 2, 3, 4), kwargs={})
        Traceback (most recent call last):
        ...
        TypeError: too many positional arguments
        >>> sig.map_arguments(args=(), kwargs=dict(w=1, x=2, y=3, z=4))
        Traceback (most recent call last):
        ...
        TypeError:...'w'...

        But if you want to ignore the kind of parameter, just say so:

        >>> sig.map_arguments(
        ...     args=(1, 2, 3, 4), kwargs={}, ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        >>> sig.map_arguments(
        ...     args=(), kwargs=dict(w=1, x=2, y=3, z=4), ignore_kind=True
        ... )
        {'w': 1, 'x': 2, 'y': 3, 'z': 4}
        """

        def get_var_dflts():
            if self.has_var_positional:
                yield self.var_positional_name, ()
            if self.has_var_keyword:
                yield self.var_keyword_name, {}

        _args = args or ()
        _kwargs = kwargs or {}

        if ignore_kind:
            var_dflts = dict(get_var_dflts())
            sig = self.normalize_kind(kind=KO, except_kinds=None)
            sig = sig.ch_defaults(**var_dflts)
            for arg, p in zip(_args, sig.params):
                if p.name in _kwargs:
                    raise TypeError(f"multiple values for argument '{p.name}'")
                _kwargs[p.name] = arg
            _args = ()
        else:
            sig = self

        if not sig.has_var_positional and allow_excess:
            max_allowed_num_of_posisional_args = sum(
                k <= PK for k in sig.kinds.values()
            )
            _args = _args[:max_allowed_num_of_posisional_args]
        if not sig.has_var_keyword and allow_excess:
            _kwargs = {k: v for k, v in _kwargs.items() if k in sig}

        binder = sig.bind_partial if allow_partial else sig.bind
        b = binder(*_args, **_kwargs)
        if apply_defaults:
            b.apply_defaults()

        return b.arguments

    kwargs_from_args_and_kwargs = deprecation_of(
        map_arguments, "kwargs_from_args_and_kwargs"
    )

    def mk_args_and_kwargs(
        self,
        arguments: dict,
        *,
        apply_defaults=False,
        allow_partial=False,
        allow_excess=False,
        ignore_kind=False,
        args_limit: int | None = 0,
    ) -> tuple[tuple, dict]:
        """Extract args and kwargs such that func(*args, **kwargs) can be called,
        where func has instance's signature.

        :param arguments: The {param_name: arg_val,...} dict to process
        :param args_limit: How "far" in the params should args (positional arguments)
            be searched for.
            - args_limit==0: Take the minimum number possible of args (positional
                arguments). Only those that are position only or before a var-positional.
            - args_limit is None: Take the maximum number of args (positional arguments).
                The only kwargs (keyword arguments) you should have are keyword-only
                and var-keyword arguments.
            - args_limit positive integer: Take the args_limit first argument names
                (of signature) as args, and the rest as kwargs.

        >>> def foo(w, /, x: float, y=1, *, z: int = 1):
        ...     return ((w + x) * y) ** z
        >>> foo_sig = Sig(foo)
        >>> args, kwargs = foo_sig.mk_args_and_kwargs(
        ...     dict(w=4, x=3, y=2, z=1)
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        >>> assert foo(*args, **kwargs) == foo(4, 3, 2, z=1) == 14

        What about variadics?

        >>> def bar(a, /, b, *args, c=2, **kwargs):
        ...     pass
        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7))
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        You can also give the arguments in a different order:

        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(args=(3,4), kwargs=dict(d=6, e=7), b=2, c=5, a=1)
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        The `args_limit` begs explanation.
        Consider the signature of `def foo(w, /, x: float, y=1, *, z: int = 1): ...`
        for instance. We could call the function with the following (args, kwargs) pairs:
        - ((1,), {'x': 2, 'y': 3, 'z': 4})
        - ((1, 2), {'y': 3, 'z': 4})
        - ((1, 2, 3), {'z': 4})
        The two other combinations (empty args or empty kwargs) are not valid
        because of the / and * constraints.

        But when asked for an (args, kwargs) pair, which of the three valid options
        should be returned? This is what the `args_limit` argument controls.

        If `args_limit == 0`, the least args (positional arguments) will be returned.
        It's the default.

        >>> arguments = dict(w=4, x=3, y=2, z=1)
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=0)
        ((4,), {'x': 3, 'y': 2, 'z': 1})

        If `args_limit is None`, the least kwargs (keyword arguments) will be returned.

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=None)
        ((4, 3, 2), {'z': 1})

        If `args_limit` is a positive integer, the first `[args_limit]` arguments
        will be returned (not checking at all if this is valid!).

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=1)
        ((4,), {'x': 3, 'y': 2, 'z': 1})
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=2)
        ((4, 3), {'y': 2, 'z': 1})
        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=3)
        ((4, 3, 2), {'z': 1})

        Note that if you specify `args_limit` to be greater than the maximum of
        positional arguments, it behaves as if `args_limit` was `None`:

        >>> foo_sig.mk_args_and_kwargs(arguments, args_limit=4)
        ((4, 3, 2), {'z': 1})

        Note that 'args_limit''s behavior is consistent with list behvior in the sense
        that:

        >>> args = (0, 1, 2, 3)
        >>> args[:0]
        ()
        >>> args[:None]
        (0, 1, 2, 3)
        >>> args[2]
        2

        If variable positional arguments are present, `args_limit` is ignored and
        all positional arguments are returned as args.

        >>> Sig(bar).mk_args_and_kwargs(
        ...     dict(a=1, b=2, args=(3,4), c=5, kwargs=dict(d=6, e=7)),
        ...     args_limit=1
        ... )
        ((1, 2, 3, 4), {'c': 5, 'd': 6, 'e': 7})

        By default, only the arguments that were given in the `arguments` input will be
        returned in the (args, kwargs) output.
        If you also want to get those that have defaults (according to signature),
        you need to specify it with the `apply_defaults=True` argument.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3))
        ((4,), {'x': 3})
        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3), apply_defaults=True)
        ((4,), {'x': 3, 'y': 1, 'z': 1})

        By default, all required arguments must be given.
        Not doing so will lead to a `TypeError`.
        If you want to process your arguments anyway, specify `allow_partial=True`.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4))
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'x'
        >>> foo_sig.mk_args_and_kwargs(dict(w=4), allow_partial=True)
        ((4,), {})

        Specifying argument names that are not recognized by the signature will
        lead to a `TypeError`.
        If you want to avoid this (and just take from the input `kwargs` what ever you
        can), specify this with `allow_excess=True`.

        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'))
        Traceback (most recent call last):
            ...
        TypeError: Got unexpected keyword arguments: extra
        >>> foo_sig.mk_args_and_kwargs(dict(w=4, x=3, extra='stuff'),
        ...     allow_excess=True)
        ((4,), {'x': 3})

        See `map_arguments` (namely for the description of the arguments).
        """
        arguments = arguments or {}
        extra_arguments = set(arguments) - set(self.names)
        if extra_arguments and not allow_excess:
            raise TypeError(
                f"Got unexpected keyword arguments: {', '.join(extra_arguments)}"
            )
        _arguments = {p: arguments[p] for p in self.names if p in arguments}
        vp_args = _arguments.get(self.var_positional_name, ())
        vk_args = _arguments.get(self.var_keyword_name, {})
        if vp_args:
            # If there are var positional arguments, we ignore the args_limit
            args_limit = None

        pos, pks, kos = (
            self.names_of_kind[PO],
            self.names_of_kind[PK],
            self.names_of_kind[KO],
        )
        names_for_args = pos
        names_for_kwargs = kos
        if args_limit is None:
            # All the PKs go to args, so we have:
            # names_for_args == POs + PKs
            # names_for_kwargs == KOs
            names_for_args += pks
        else:
            # Take the [args_limit] first arguments (of signature) as args. The minimum
            # number of args is the number of POs. The maximum number of args is the
            # number of POs + PKs. The rest are kwargs.
            nb_of_positional_pks = min(max(args_limit - len(pos), 0), len(pks))
            names_for_args += pks[:nb_of_positional_pks]
            names_for_kwargs = pks[nb_of_positional_pks:] + names_for_kwargs

        args = tuple(_arguments[name] for name in names_for_args if name in _arguments)
        kwargs = {
            name: _arguments[name] for name in names_for_kwargs if name in _arguments
        }

        # Note that, at this stage, the variadics arguments are not yet in the args and
        # kwargs variables.
        # We first call map_arguments with the args and kwargs with no variadics to
        # validate that all the explicit arguments are valid and there is no missing
        # required argument.

        # In fact, imagine the following:

        # >>> def foo(a, *args):
        # ...     ...
        # >>> foo_sig = Sig(foo)
        # >>> foo_sig.mk_args_and_kwargs(arguments=dict(args=(1,)))

        # This should fail because `a` is missing in the arguments.
        # But if we included the variadics in the args, the value '1' would have been
        # mapped to `a` by `map_arguments` and the error would not have been caught.
        # Same logic for kwargs.

        __arguments = self.map_arguments(
            args,
            kwargs,
            apply_defaults=apply_defaults,
            allow_partial=allow_partial,
            # allow_excess=allow_excess,
            # ignore_kind=ignore_kind,
        )

        # Let's retrieve the args and kwargs from the output of `map_arguments`, because
        # some extra stuff might have been added (defaults). And let's also add the
        # variadics.
        pos_arguments = {
            name: arg for name, arg in __arguments.items() if name in names_for_args
        }
        kw_arguments = {
            name: arg for name, arg in __arguments.items() if name in names_for_kwargs
        }

        if ignore_kind:
            # If ignore_kind is True, return all arguments as kwargs
            args = ()
            d_vp_args = (
                {self.var_positional_name: vp_args} if self.has_var_positional else {}
            )
            d_vk_args = {self.var_keyword_name: vk_args} if self.has_var_keyword else {}
            kwargs = {**pos_arguments, **d_vp_args, **kw_arguments, **d_vk_args}
        else:
            args = tuple(pos_arguments.values()) + vp_args
            kwargs = dict(kw_arguments, **vk_args)

        return args, kwargs

    args_and_kwargs_from_kwargs = deprecation_of(
        mk_args_and_kwargs, "args_and_kwargs_from_kwargs"
    )

    def map_arguments_from_variadics(
        self,
        *args,
        _apply_defaults=False,
        _allow_partial=False,
        _allow_excess=False,
        _ignore_kind=False,
        **kwargs,
    ):
        """Convenience method that calls map_arguments from variadics

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> sig = Sig(foo)
        >>> assert (
        ...     sig.map_arguments_from_variadics(1, 2, 3, z=4)
        ...     == sig.map_arguments_from_variadics(1, 2, y=3, z=4)
        ...     == {"w": 1, "x": 2, "y": 3, "z": 4}
        ... )

        What about var positional and var keywords?

        >>> def bar(*args, **kwargs):
        ...     ...
        ...
        >>> Sig(bar).map_arguments_from_variadics(1, 2, y=3, z=4)
        {'args': (1, 2), 'kwargs': {'y': 3, 'z': 4}}

        Note that though `w` is a position only argument, you can specify `w=11` as
        a keyword argument too, using `_ignore_kind=True`:

        >>> Sig(foo).map_arguments_from_variadics(w=11, x=22, _ignore_kind=True)
        {'w': 11, 'x': 22}

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function
        (in view of being completed later).

        >>> Sig(foo).map_arguments_from_variadics(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError: missing a required argument: 'w'

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).map_arguments_from_variadics(x=3, y=2, _allow_partial=True)
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those arguments
        you input.

        >>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2)
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).map_arguments_from_variadics(4, x=3, y=2, _apply_defaults=True)
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}
        """
        return self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=_allow_excess,
            ignore_kind=_ignore_kind,
        )

    extract_kwargs = deprecation_of(map_arguments_from_variadics, "extract_kwargs")

    def extract_args_and_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _allow_excess=True,
        _apply_defaults=False,
        _args_limit=0,
        **kwargs,
    ):
        """Source the (args, kwargs) for the signature instance, ignoring excess
        arguments.

        >>> def foo(w, /, x: float, y=2, *, z: int = 1):
        ...     return w + x * y ** z
        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        The difference with map_arguments_from_variadics is that here the output is
        ready to be called by the function whose signature we have, since the
        position-only arguments will be returned as args.

        >>> foo(*args, **kwargs)
        10

        Note that though `w` is a position only argument, you can specify `w=4` as a
        keyword argument too (by default):

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).extract_args_and_kwargs(w=4, x=3, y=2, _ignore_kind=False)
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).extract_args_and_kwargs(x=3, y=2)
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        But if you specify `_allow_partial=True`...

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(
        ...     x=3, y=2, _allow_partial=True
        ... )
        >>> (args, kwargs) == ((), {"x": 3, "y": 2})
        True

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(4, x=3, y=2)
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> args, kwargs = Sig(foo).extract_args_and_kwargs(
        ...     4, x=3, y=2, _apply_defaults=True
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        True
        """
        arguments = self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=_allow_excess,
            ignore_kind=_ignore_kind,
        )
        return self.mk_args_and_kwargs(
            arguments,
            allow_partial=_allow_partial,
            args_limit=_args_limit,
        )

    def source_arguments(
        self,
        *args,
        _apply_defaults=False,
        _allow_partial=False,
        _ignore_kind=True,
        **kwargs,
    ):
        """Source the arguments for the signature instance, ignoring excess arguments.

        >>> def foo(w, /, x: float, y="YY", *, z: str = "ZZ"):
        ...     ...
        >>> Sig(foo).source_arguments(11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        Note that though `w` is a position only argument, you can specify `w=11` as a
        keyword argument too (by default):

        >>> Sig(foo).source_arguments(w=11, x=22, extra="keywords", are="ignored")
        {'w': 11, 'x': 22}

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).source_arguments(
        ...     w=11, x=22, extra="keywords", are="ignored", _ignore_kind=False
        ... )
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_arguments(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        But if you specify `_allow_partial=True`...

        >>> Sig(foo).source_arguments(
        ...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
        ... )
        {'x': 3, 'y': 2}

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> Sig(foo).source_arguments(4, x=3, y=2, extra="keywords", are="ignored")
        {'w': 4, 'x': 3, 'y': 2}

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> Sig(foo).source_arguments(
        ...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
        ... )
        {'w': 4, 'x': 3, 'y': 2, 'z': 'ZZ'}


        """
        return self.map_arguments(
            args,
            kwargs,
            apply_defaults=_apply_defaults,
            allow_partial=_allow_partial,
            allow_excess=True,
            ignore_kind=_ignore_kind,
        )

    source_kwargs = deprecation_of(source_arguments, "source_kwargs")

    def source_args_and_kwargs(
        self,
        *args,
        _ignore_kind=True,
        _allow_partial=False,
        _apply_defaults=False,
        _args_limit=0,
        **kwargs,
    ):
        """Source the (args, kwargs) for the signature instance, ignoring excess
        arguments.

        >>> def foo(w, /, x: float, y=2, *, z: int = 1):
        ...     return w + x * y ** z
        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> args, kwargs
        ((4,), {'x': 3, 'y': 2})

        The difference with source_arguments is that here the output is ready to be
        called by the
        function whose signature we have, since the position-only arguments will be
        returned as
        args.

        >>> foo(*args, **kwargs)
        10

        Note that though `w` is a position only argument, you can specify `w=4` as a
        keyword argument too (by default):

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     w=4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> assert (args, kwargs) == ((4,), {"x": 3, "y": 2})

        If you don't want to allow that, you can say `_ignore_kind=False`

        >>> Sig(foo).source_args_and_kwargs(
        ...     w=4, x=3, y=2, extra="keywords", are="ignored", _ignore_kind=False
        ... )
        Traceback (most recent call last):
          ...
        TypeError: ...'w'...

        You can use `_allow_partial` that will allow you, if
        set to `True`, to underspecify the params of a function (in view of being
        completed later).

        >>> Sig(foo).source_args_and_kwargs(x=3, y=2, extra="keywords", are="ignored")
        Traceback (most recent call last):
          ...
        TypeError:...'w'...

        But if you specify `_allow_partial=True`...

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     x=3, y=2, extra="keywords", are="ignored", _allow_partial=True
        ... )
        >>> (args, kwargs) == ((), {"x": 3, "y": 2})
        True

        By default, `_apply_defaults=False`, which will lead to only get those
        arguments you input.

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored"
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2})
        True

        But if you specify `_apply_defaults=True` non-specified non-require arguments
        will be returned with their defaults:

        >>> args, kwargs = Sig(foo).source_args_and_kwargs(
        ...     4, x=3, y=2, extra="keywords", are="ignored", _apply_defaults=True
        ... )
        >>> (args, kwargs) == ((4,), {"x": 3, "y": 2, "z": 1})
        True
        """
        arguments = self.source_arguments(
            *args,
            _apply_defaults=_apply_defaults,
            _allow_partial=_allow_partial,
            _ignore_kind=_ignore_kind,
            **kwargs,
        )
        return self.mk_args_and_kwargs(
            arguments,
            allow_partial=_allow_partial,
            args_limit=_args_limit,
        )

    @property
    def inject_into_keyword_variadic(self):
        """
        Decorator that uses signature to source the keyword variadic of target function.

        See replace_kwargs_using function for more details, including examples.

        >>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
        ...     return a + x + y + z
        >>> @Sig(apple).inject_into_keyword_variadic
        ... def sauce(a, b, c, **sauce_kwargs):
        ...     return b * c + apple(a, **sauce_kwargs)

        The function will works:

        >>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
        18

        But the signature now doesn't have the `**sauce_kwargs`, but more informative
        signature elements sourced from `apple`:

        >>> Sig(sauce)
        <Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>

        """
        return replace_kwargs_using(self)


def _fill_defaults_and_annotations(sig1: Sig, sig2: Sig):
    """Return the same signature as ``sig1``, but where empty param properties
    (default or annotation) were filled by the property found in ``sig2`` if it has a
    param of the same name

    >>> _fill_defaults_and_annotations(
    ...    Sig('(a, /, b: str, *, c=3)'), Sig('(a: float, b: int = 2, c=300)')
    ... )
    <Sig (a: float, /, b: str = 2, *, c=3)>

    """

    def filled_properties_of_sig1():
        alt_defaults = sig2.defaults
        alt_annotations = sig2.annotations
        for p in sig1.params:
            yield Parameter(
                p.name,
                p.kind,
                default=(
                    p.default
                    if p.default is not empty
                    else alt_defaults.get(p.name, empty)
                ),
                annotation=(
                    p.annotation
                    if p.annotation is not empty
                    else alt_annotations.get(p.name, empty)
                ),
            )

    return Sig(filled_properties_of_sig1())


def _validate_sanity_of_signature_change(
    func: Callable, new_sig: Sig, ignore_incompatible_signatures: bool = True
):
    func_pos, func_kw = Sig(func)._positional_and_keyword_defaults()
    self_pos, self_kw = new_sig._positional_and_keyword_defaults()
    # print(func_pos, func_kw )
    # print(self_pos, self_kw)

    pos_default_switching_to_kw = set(func_pos) & set(self_kw)
    kw_default_switching_to_pos = set(func_kw) & set(self_pos)

    # print(pos_default_switching_to_kw, kw_default_switching_to_pos)

    if not ignore_incompatible_signatures and (
        pos_default_switching_to_kw or kw_default_switching_to_pos
    ):
        raise IncompatibleSignatures(
            f"Changing both the kind and the default of a param will result to "
            f"unexpected behaviors if the function is not properly wrapped to do so."
            f"If you really want to do this, inject signature using the "
            f"`ignore_incompatible_signatures=True`"
            f"argument in `Sig.wrap(...)`. "
            f"Alternatively, you can use `i2.wrapper` tools to have more control "
            f"over function defaults and signatures."
            f"The function you were wrapping had signature: "
            f"{name_of_obj(func) or ''}{Sig(func)} and "
            f"the signature you wanted to inject was {new_sig.name or ''}{new_sig}",
            sig1=Sig(func),
            sig2=new_sig,
        )


########################################################################################
# Utils


def _signature_differences_str_for_error_msg(sig1, sig2):

    from pprint import pformat

    sig_diff = sig1.pair_with(sig2)

    sig1_name = f"{sig1.name}" if sig1.name else "sig1"
    sig2_name = f"{sig2.name}" if sig2.name else "sig2"

    return (
        "FYI: Here are the raw signature differences for {sig1_name} and {sig2_name} "
        f"(not all need to necessarily be resolved):\n{pformat(sig_diff)}"
    )


########################################################################################
# Recipes


def mk_sig_from_args(*args_without_default, **args_with_defaults):
    """Make a Signature instance by specifying args_without_default and
    args_with_defaults.

    >>> mk_sig_from_args("a", "b", c=1, d="bar")
    <Signature (a, b, c=1, d='bar')>
    """
    assert all(
        isinstance(x, str) for x in args_without_default
    ), "all default-less arguments must be strings"
    return Sig.from_objs(
        *args_without_default, **args_with_defaults
    ).to_simple_signature()


def _remove_variadics_from_sig(sig, ch_variadic_keyword_to_keyword=True):
    """Remove variadics from signature
    >>> def foo(a, *args, bar, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> sig = Sig(foo)
    >>> assert str(sig) == '(a, *args, bar, **kwargs)'
    >>> new_sig = _remove_variadics_from_sig(sig)
    >>> str(new_sig)=='(a, args=(), *, bar, kwargs={})'
    True

    Note that if there is not variadic positional arguments, the variadic keyword
    will still be a keyword-only kind.

    >>> def func(a, bar=None, **kwargs):
    ...     return f"{a=}, {bar=}, {kwargs=}"
    >>> nsig = _remove_variadics_from_sig(Sig(func))
    >>> assert str(nsig)=='(a, bar=None, *, kwargs={})'

    If the function has neither variadic kinds, it will remain untouched.

    >>> def func(a, /, b, *, c=3):
    ...     return a + b + c
    >>> sig = _remove_variadics_from_sig(Sig(func))

    >>> assert sig == Sig(func)


    If you only want the variadic positional to be handled, but leave leave any
    VARIADIC_KEYWORD kinds (**kwargs) alone, you can do so by setting
    `ch_variadic_keyword_to_keyword=False`.

    >>> def foo(a, *args, bar=None, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> assert str(Sig(_remove_variadics_from_sig(Sig(foo))))=='(a, args=(), *, bar=None, kwargs={})'
    """

    idx_of_vp = sig.index_of_var_positional
    var_keyword_argname = sig.var_keyword_name
    result_sig = sig
    if idx_of_vp is not None or var_keyword_argname is not None:
        params = sig.params
        if var_keyword_argname:  # if there's a VAR_KEYWORD argument
            if ch_variadic_keyword_to_keyword:
                i = sig.index_of_var_keyword
                # TODO: Reflect on pros/cons of having mutable {} default here:
                params[i] = params[i].replace(kind=Parameter.KEYWORD_ONLY, default={})

        try:  # TODO: Avoid this try catch. Look in advance for default ordering?
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK, default=())
            result_sig = Signature(params, return_annotation=sig.return_annotation)
        except ValueError:
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK)
            result_sig = Signature(params, return_annotation=sig.return_annotation)

    return result_sig


# TODO: Might want to make func be a positional-only argument, because if kwargs
#  contains a func key, we have a problem. But call_forgivingly is used broadly,
#  so must first test all dependents before making this change.
def call_forgivingly(func, *args, **kwargs):
    """
    Call function on given args and kwargs, but only taking what the function needs
    (not choking if they're extras variables)

    Tip: If you into trouble because your kwargs has a 'func' key,
    (which would then clash with the ``func`` param of call_forgivingly), then
    use `_call_forgivingly` instead, specifying args and kwargs as tuple and
    dict.

    >>> def foo(a, b: int = 0, c=None) -> int:
    ...     return "foo", (a, b, c)
    >>> call_forgivingly(
    ...     foo,  # the function you want to call
    ...     "input for a",  # meant for a -- the first (and only) argument foo requires
    ...     c=42,  # skiping b and giving c a non-default value
    ...     intruder="argument",  # but wait, this argument name doesn't exist! Oh no!
    ... )  # well, as it happens, nothing bad -- the intruder argument is just ignored
    ('foo', ('input for a', 0, 42))

    An example of what happens when variadic kinds are involved:

    >>> def bar(x, *args1, y=1, **kwargs1):
    ...     return x, args1, y, kwargs1
    >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    (1, (2, 3), 4, {'z': 5})

    # >>> def bar(x, y=1, **kwargs1):
    # ...     return x, y, kwargs1
    # >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    # (1, 4, {'z': 5})

    # >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)

    # >>> def bar(x, *args1, y=1):
    # ...     return x, args1, y
    # >>> call_forgivingly(bar, 1, 2, 3, y=4, z=5)
    # (1, (2, 3), {'z': 5})

    """
    return _call_forgivingly(func, args, kwargs)


# TODO: See if there's a more elegant way to do this
def _call_forgivingly(func, args, kwargs):
    """
    Helper for _call_forgivingly.
    """

    sig = Sig(func)
    arguments = sig.map_arguments(args, kwargs, allow_excess=True)
    _args, _kwargs = sig.mk_args_and_kwargs(arguments, args_limit=len(args))
    return func(*_args, **_kwargs)

    # sig = Sig(func)
    # variadic_kinds = {
    #     name: kind for name, kind in sig.kinds.items() if kind in [VP, VK]
    # }
    # if VP in variadic_kinds.values() and VK in variadic_kinds.values():
    #     _args = args
    #     _kwargs = kwargs
    # else:
    #     new_sig = sig - variadic_kinds.keys()
    #     _args, _kwargs = new_sig.source_args_and_kwargs(*args, _ignore_kind=False, **kwargs)
    #     for k, v in _kwargs.items():
    #         if k not in kwargs:
    #             _args = _args + (v,)
    #     _kwargs = {k: v for k, v in _kwargs.items() if k in kwargs}
    #     if VP in variadic_kinds.values():
    #         _args = args
    #     elif VK in variadic_kinds.values():
    #         _kwargs = dict(_kwargs, **kwargs)
    # return func(*_args, **_kwargs)


def call_somewhat_forgivingly(
    func, args, kwargs, enforce_sig: SignatureAble | None = None
):
    """Call function on given args and kwargs, but with controllable argument leniency.
    By default, the function will only pick from args and kwargs what matches it's
    signature, ignoring anything else in args and kwargs.

    But the real use of `call_somewhat_forgivingly` kicks in when you specify a
    `enforce_sig`: A signature (or any object that can be resolved into a signature
    through `Sig(enforce_sig)`) that will be used to bind the inputs, thus validating
    them against the `enforce_sig` signature (including extra arguments, defaults,
    etc.).

    `call_somewhat_forgivingly` helps you do this kind of thing systematically.

    >>> f = lambda a: a * 11
    >>> assert call_somewhat_forgivingly(f, (2,), {}) == f(2)

    In the above, we have no `enforce_sig`. The real use of call_somewhat_forgivingly
    is when we ask it to enforce a signature. Let's do this by specifying a function
    (no need for it to do anything: Only the signature is used.

    >>> g = lambda a, b=None: ...

    Calling `f` on it's normal set of inputs (one input in this case) gives you the
    same thing as `f`:

    >>> assert call_somewhat_forgivingly(f, (2,), {}, enforce_sig=g) == f(2)
    >>> assert call_somewhat_forgivingly(f, (), {'a': 2}, enforce_sig=g) == f(2)

    If you call with an extra positional argument, it will just be ignored.

    >>> assert call_somewhat_forgivingly(f, (2, 'ignored'), {}, enforce_sig=g) == f(2)

    If you call with a `b` keyword-argument (which matches `g`'s signature,
    it will also be ignored.

    >>> assert call_somewhat_forgivingly(
    ... f, (2,), {'b': 'ignored'}, enforce_sig=g
    ... ) == f(2)
    >>> assert call_somewhat_forgivingly(
    ...     f, (), {'a': 2, 'b': 'ignored'}, enforce_sig=g
    ... ) == f(2)

    But if you call with three positional arguments (one more than g allows),
    or call with a keyword argument that is not in `g`'s signature, it will
    raise a `TypeError`:

    >>> call_somewhat_forgivingly(f,
    ...     (2, 'ignored', 'does_not_fit_g_signature_anymore'), {}, enforce_sig=g
    ... )
    Traceback (most recent call last):
        ...
    TypeError: too many positional arguments
    >>> call_somewhat_forgivingly(f,
    ...     (2,), {'this_argname': 'is not in g'}, enforce_sig=g
    ... )
    Traceback (most recent call last):
        ...
    TypeError: got an unexpected keyword argument 'this_argname'

    """
    enforce_sig = Sig(enforce_sig or func)
    # Validate that args and kwargs are compatible with enforce_sig
    enforce_sig.bind(*args, **kwargs)
    return _call_forgivingly(func, args, kwargs)


def convert_to_PK(kinds):
    return {name: PK for name in kinds}


def kind_forgiving_func(func, kinds_modifier=convert_to_PK):
    """Wraps the func, changing the argument kinds according to kinds_modifier.
    The default behaviour is to change all kinds to POSITIONAL_OR_KEYWORD kinds.
    The original purpose of this function is to remove argument-kind restriction
    annoyances when doing functional manipulations such as:

    >>> from functools import partial
    >>> isinstance_of_str = partial(isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    Traceback (most recent call last):
      ...
    TypeError: isinstance() takes no keyword arguments

    Here, instead, we can just get a kinder version of the function and do what we
    want to do:

    >>> _isinstance = kind_forgiving_func(isinstance)
    >>> isinstance_of_str = partial(_isinstance, class_or_tuple=str)
    >>> isinstance_of_str('I am a string')
    True
    >>> isinstance_of_str(42)
    False

    See also: ``i2.signatures.all_pk_signature``

    """
    sig = Sig(func)
    kinds_modif = kinds_modifier(sig.kinds)
    _sig = sig.ch_kinds(**kinds_modif)

    @_sig
    @wraps(func)
    def _func(*args, **kwargs):
        _args, _kwargs = sig.extract_args_and_kwargs(
            *args, _allow_excess=False, **kwargs
        )
        return func(*_args, **_kwargs)

    # _func.__signature__ = sig
    return _func


# TODO: Should we protect from misuse with signature compatibility check?
def use_interface(interface_sig):
    """Use interface_sig as (enforced/validated) signature of the decorated function.
    That is, the decorated function will use the original function has the backend,
    the function actually doing the work, but with a frontend specified
    (in looks and in argument validation) `interface_sig`

    consider the situation where are functionality is parametrized by a
    function `g` taking two inputs, `a`, and `b`.
    Now you want to carry out this functionality using a function `f` that does what
    `g` should do, but doesn't use `a`, and doesn't even have it in it's arguments.

    The solution to this is to _adapt_ `f` to the `g` interface:
    ```
    def my_g(a, b):
        return f(a)
    ```
    and use `my_g`.

    >>> f = lambda a: a * 11
    >>> interface = lambda a, b=None: ...
    >>>
    >>> new_f = use_interface(interface)(f)

    See how only the first argument, or `a` keyword argument, is taken into account
    in `new_f`:

    >>> assert new_f(2) == f(2)
    >>> assert new_f(2, 3) == f(2)
    >>> assert new_f(2, b=3) == f(2)
    >>> assert new_f(b=3, a=2) == f(2)

    But if we add more positional arguments than `interface` allows,
    or any keyword arguments that `interface` doesn't recognize...

    >>> new_f(1,2,3)
    Traceback (most recent call last):
      ...
    TypeError: too many positional arguments
    >>> new_f(1, c=2)
    Traceback (most recent call last):
      ...
    TypeError: got an unexpected keyword argument 'c'
    """
    interface_sig = Sig(interface_sig)

    def interface_wrapped_decorator(func):
        @interface_sig
        def _func(*args, **kwargs):
            return call_somewhat_forgivingly(
                func, args, kwargs, enforce_sig=interface_sig
            )

        return _func

    return interface_wrapped_decorator


import inspect


def has_signature(obj, robust=False):
    """Check if an object has a signature -- i.e. is callable and inspect.signature(
    obj) returns something.

    This can be used to more easily get signatures in bulk without having to write
    try/catches:

    >>> from functools import partial
    >>> len(
    ...     list(
    ...         filter(
    ...             None,
    ...             map(
    ...                 partial(has_signature, robust=False),
    ...                 (Sig, print, map, filter, Sig.wrap),
    ...             ),
    ...         )
    ...     )
    ... )  # doctest: +SKIP
    2

    If robust is set to True, `has_signature` will use `Sig` to get the signature,
    so will return True in most cases.

    """
    if robust:
        return bool(Sig.sig_or_none(obj))
    else:
        try:
            return bool((callable(obj) or None) and signature(obj))
        except ValueError:
            return False


# TODO: Need to define and use this function more carefully.
#   Is the goal to remove positional? Remove variadics? Normalize the signature?
def all_pk_signature(callable_or_signature: Callable | Signature):
    """Changes all (non-variadic) arguments to be of the PK (POSITION_OR_KEYWORD) kind.

    Wrapping a function with the resulting signature doesn't make that function callable
    with PK kinds in itself.
    It just gives it a signature without position and keyword ONLY kinds.
    It should be used to wrap such a function that actually carries out the
    implementation though!

    >>> def foo(w, /, x: float, y=1, *, z: int = 1, **kwargs):
    ...     ...
    >>> def bar(*args, **kwargs):
    ...     ...
    ...
    >>> from inspect import signature
    >>> new_foo = all_pk_signature(foo)
    >>> Sig(new_foo)
    <Sig (w, x: float, y=1, z: int = 1, **kwargs)>
    >>> all_pk_signature(signature(foo))
    <Sig (w, x: float, y=1, z: int = 1, **kwargs)>

    But note that the variadic arguments *args and **kwargs remain variadic:

    >>> all_pk_signature(signature(bar))
    <Signature (*args, **kwargs)>

    It works with `Sig` too (since Sig is a Signature), and maintains it's other
    attributes (like name).

    >>> sig = all_pk_signature(Sig(bar))
    >>> sig
    <Sig (*args, **kwargs)>
    >>> sig.name
    'bar'

    See also: ``i2.signatures.kind_forgiving_func``

    """

    if isinstance(callable_or_signature, Signature):
        sig = callable_or_signature

        def changed_params():
            for p in sig.parameters.values():
                if p.kind not in var_param_kinds:
                    yield p.replace(kind=PK)
                else:
                    yield p

        new_sig = type(sig)(
            list(changed_params()), return_annotation=sig.return_annotation
        )
        for attrname, attrval in getattr(sig, "__dict__", {}).items():
            setattr(new_sig, attrname, attrval)
        return new_sig
    elif isinstance(callable_or_signature, Callable):
        func = callable_or_signature
        sig = all_pk_signature(Sig(func))
        return sig(func)


# Changed ch_signature_to_all_pk to all_pk_signature because ch_signature_to_all_pk
# was misleading: It doesn't change anything at all, it returns a constructed signature.
# It doesn't change all kinds to PK -- just the non-variadic ones.
ch_signature_to_all_pk = all_pk_signature  # alias for back-compatibility


def normalized_func(func):
    sig = Sig(func)

    def argument_values_tuple(args, kwargs):
        b = sig.bind(*args, **kwargs)
        arg_vals = dict(b.arguments)

        poa, pka, vpa, koa, vka = [], [], (), {}, {}

        for name, val in arg_vals.items():
            kind = sig.kinds[name]
            if kind == PO:
                poa.append(val)
            elif kind == PK:
                pka.append(val)
            elif kind == VP:
                vpa = val  # there can only be one VP!
            elif kind == KO:
                koa.update({name: val})
            elif kind == VK:
                vka = val  # there can only be one VK!
        return poa, pka, vpa, koa, vka

    def _args_and_kwargs(args, kwargs):
        poa, pka, vpa, koa, vka = argument_values_tuple(args, kwargs)

        _args = (*poa, *pka, *vpa)
        _kwargs = {**koa, **vka}

        return _args, _kwargs

    # @sig.modified(**{name: {'kind': PK} for name in sig.names})
    def _func(*args, **kwargs):
        # poa, pka, vpa, koa, vka = argument_values_tuple(args, kwargs)
        # print(poa, pka, vpa, koa, vka)
        _args, _kwargs = _args_and_kwargs(args, kwargs)
        return func(*_args, **_kwargs)

    return _func


def ch_variadics_to_non_variadic_kind(func, *, ch_variadic_keyword_to_keyword=True):
    """A decorator that will change a VAR_POSITIONAL (*args) argument to a tuple (args)
    argument of the same name.

    Essentially, given a `func(a, *b, c, **d)` function want to get a
    `new_func(a, b=(), c=None, d={})` that has the same functionality
    (in fact, calls the original `func` function behind the scenes), but without
    where the variadic arguments *b and **d are replaced with a `b` expecting an
    iterable (e.g. tuple/list) and `d` expecting a `dict` to contain the
    desired inputs.

    Besides this, the decorator tries to be as conservative as possible, making only
    the minimum changes needed to meet the goal of getting to a variadic-less
    interface. When it doubt, and error will be raised.

    >>> def foo(a, *args, bar, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> assert str(Sig(foo)) == '(a, *args, bar, **kwargs)'
    >>> wfoo = ch_variadics_to_non_variadic_kind(foo)
    >>> str(Sig(wfoo))
    '(a, args=(), *, bar, kwargs={})'

    And now to do this:

    >>> foo(1, 2, 3, bar=4, hello="world")
    "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

    We can do it like this instead:

    >>> wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world"))
    "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"

    Note, the outputs are the same. It's just the way we call our function that has
    changed.

    >>> assert wfoo(1, (2, 3), bar=4, kwargs=dict(hello="world")
    ... ) == foo(1, 2, 3, bar=4, hello="world")
    >>> assert wfoo(1, (2, 3), bar=4) == foo(1, 2, 3, bar=4)
    >>> assert wfoo(1, (), bar=4) == foo(1, bar=4)

    Note that if there is not variadic positional arguments, the variadic keyword
    will still be a keyword-only kind.

    >>> @ch_variadics_to_non_variadic_kind
    ... def func(a, bar=None, **kwargs):
    ...     return f"{a=}, {bar=}, {kwargs=}"
    >>> str(Sig(func))
    '(a, bar=None, *, kwargs={})'
    >>> assert func(1, bar=4, kwargs=dict(hello="world")
    ...     ) == "a=1, bar=4, kwargs={'hello': 'world'}"

    If the function has neither variadic kinds, it will remain untouched.

    >>> def func(a, /, b, *, c=3):
    ...     return a + b + c
    >>> ch_variadics_to_non_variadic_kind(func) == func
    True

    If you only want the variadic positional to be handled, but leave leave any
    VARIADIC_KEYWORD kinds (**kwargs) alone, you can do so by setting
    `ch_variadic_keyword_to_keyword=False`.
    If you'll need to use `ch_variadics_to_non_variadic_kind` in such a way
    repeatedly, we suggest you use `functools.partial` to not have to specify this
    configuration repeatedly.

    >>> from functools import partial
    >>> tuple_the_args = partial(ch_variadics_to_non_variadic_kind,
    ...     ch_variadic_keyword_to_keyword=False
    ... )
    >>> @tuple_the_args
    ... def foo(a, *args, bar=None, **kwargs):
    ...     return f"{a=}, {args=}, {bar=}, {kwargs=}"
    >>> Sig(foo)
    <Sig (a, args=(), *, bar=None, **kwargs)>
    >>> foo(1, (2, 3), bar=4, hello="world")
    "a=1, args=(2, 3), bar=4, kwargs={'hello': 'world'}"




    """
    if func is None:
        return partial(
            ch_variadics_to_non_variadic_kind,
            ch_variadic_keyword_to_keyword=ch_variadic_keyword_to_keyword,
        )
    sig = Sig(func)
    idx_of_vp = sig.index_of_var_positional
    var_keyword_argname = sig.var_keyword_name

    if idx_of_vp is not None or var_keyword_argname is not None:
        # If the function has any variadic (position or keyword)...

        @wraps(func)
        def variadic_less_func(*args, **kwargs):
            # extract from kwargs those inputs that need to be expressed positionally
            if ch_variadic_keyword_to_keyword:
                arguments = kwargs
            else:
                arguments = {k: v for k, v in kwargs.items() if k in sig}
                if sig.has_var_keyword:
                    arguments[sig.var_keyword_name] = {
                        k: v for k, v in kwargs.items() if k not in sig
                    }
            _args, _kwargs = sig.mk_args_and_kwargs(arguments, allow_partial=True)
            # print('COUCOU', kwargs, arguments)
            # add these to the existing args
            args = args + _args

            if idx_of_vp is not None:
                # separate the args that are positional, variadic, and after variadic
                a, _vp_args_, args_after_vp = (
                    args[:idx_of_vp],
                    args[idx_of_vp],
                    args[idx_of_vp + 1 :],
                )
                if args_after_vp:
                    raise FuncCallNotMatchingSignature(
                        "There should be only keyword arguments after the Variadic "
                        "args. "
                        f"Function was called with (positional={args}, keywords="
                        f"{_kwargs})"
                    )
            else:
                a, _vp_args_ = args, ()

            # extract from the remaining _kwargs, the dict corresponding to the
            # variadic keywords, if any, since these need to be **-ed later
            _var_keyword_kwargs = _kwargs.pop(var_keyword_argname, {})

            if ch_variadic_keyword_to_keyword:
                # an extra level of extraction is needed in this case
                # _var_keyword_kwargs = _var_keyword_kwargs.pop(var_keyword_argname, {})
                return func(*a, *_vp_args_, **_kwargs, **_var_keyword_kwargs)
            else:
                # call the original function with the unravelled args
                return func(*a, *_vp_args_, **_kwargs, **_var_keyword_kwargs)

        params = sig.params

        if var_keyword_argname:  # if there's a VAR_KEYWORD argument
            if ch_variadic_keyword_to_keyword:
                i = sig.index_of_var_keyword
                # TODO: Reflect on pros/cons of having mutable {} default here:
                params[i] = params[i].replace(kind=Parameter.KEYWORD_ONLY, default={})

        try:  # TODO: Avoid this try catch. Look in advance for default ordering?
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK, default=())
            variadic_less_func.__signature__ = Sig(
                # Note: Changed signature(func) to Sig(func) but don't know if the first
                #  was on purpose.
                params,
                return_annotation=Sig(func).return_annotation,
            )
        except ValueError:
            if idx_of_vp is not None:
                params[idx_of_vp] = params[idx_of_vp].replace(kind=PK)
            variadic_less_func.__signature__ = Sig(
                params, return_annotation=Sig(func).return_annotation
            )

        return variadic_less_func
    else:
        return func


tuple_the_args = partial(
    ch_variadics_to_non_variadic_kind, ch_variadic_keyword_to_keyword=False
)
tuple_the_args.__name__ = "tuple_the_args"
tuple_the_args.__doc__ = """
A decorator that will change a VAR_POSITIONAL (*args) argument to a tuple (args)
argument of the same name.
"""


def ch_func_to_all_pk(func):
    """Returns a decorated function where all arguments are of the PK kind.
    (PK: Positional_or_keyword)

    :param func: A callable
    :return:

    >>> def f(a, /, b, *, c=None, **kwargs):
    ...     return a + b * c
    ...
    >>> print(Sig(f))
    (a, /, b, *, c=None, **kwargs)
    >>> ff = ch_func_to_all_pk(f)
    >>> print(Sig(ff))
    (a, b, c=None, **kwargs)
    >>> ff(1, 2, 3)
    7
    >>>
    >>> def g(x, y=1, *args, **kwargs):
    ...     ...
    ...
    >>> print(Sig(g))
    (x, y=1, *args, **kwargs)
    >>> gg = ch_func_to_all_pk(g)
    >>> print(Sig(gg))
    (x, y=1, args=(), **kwargs)

    # >>> def h(x, *y, z):
    # ...     print(f"{x=}, {y=}, {z=}")
    # >>> h(1, 2, 3, z=4)
    # x=1, y=(2, 3), z=4
    # >>> hh = ch_func_to_all_pk(h)
    # >>> hh(1, (2, 3), z=4)
    # x=1, y=(2, 3), z=4
    """

    # _func = tuple_the_args(func)
    # sig = Sig(_func)
    #
    # @wraps(func)
    # def __func(*args, **kwargs):
    #     # b = Sig(_func).bind_partial(*args, **kwargs)
    #     # return _func(*b.args, **b.kwargs)
    #     args, kwargs = Sig(_func).extract_args_and_kwargs(
    #         *args, **kwargs, _ignore_kind=False
    #     )
    #     return _func(*args, **kwargs)
    #
    _func = tuple_the_args(func)
    sig = Sig(_func)

    @wraps(func)
    def __func(*args, **kwargs):
        args, kwargs = Sig(_func).extract_args_and_kwargs(
            *args,
            **kwargs,
            # _ignore_kind=False,
            # _allow_partial=True
        )
        return _func(*args, **kwargs)

    __func.__signature__ = all_pk_signature(sig)
    return __func


def copy_func(f):
    """Copy a function (not sure it works with all types of callables)"""
    g = FunctionType(
        f.__code__,
        f.__globals__,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    if hasattr(f, "__signature__"):
        g.__signature__ = f.__signature__
    return g


# TODO: Similar to other function in this module -- merge.
def params_of(obj: HasParams):
    if isinstance(obj, Signature):
        obj = list(obj.parameters.values())
    elif isinstance(obj, Mapping):
        obj = list(obj.values())
    elif callable(obj):
        obj = list(signature(obj).parameters.values())
    assert all(
        isinstance(p, Parameter) for p in obj
    ), "obj needs to be a Iterable[Parameter] at this point"
    return obj  # as is


########################################################################################################################
# TODO: Encorporate in Sig
def insert_annotations(s: Signature, /, *, return_annotation=empty, **annotations):
    """Insert annotations in a signature.
    (Note: not really insert but returns a copy of input signature)

    >>> from inspect import signature
    >>> s = signature(lambda a, b, c=1, d="bar": 0)
    >>> s
    <Signature (a, b, c=1, d='bar')>
    >>> ss = insert_annotations(s, b=int, d=str)
    >>> ss
    <Signature (a, b: int, c=1, d: str = 'bar')>
    >>> insert_annotations(s, b=int, d=str, e=list)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    AssertionError: These argument names weren't found in the signature: {'e'}
    """
    assert set(annotations) <= set(s.parameters), (
        f"These argument names weren't found in the signature: "
        f"{set(annotations) - set(s.parameters)}"
    )
    params = dict(s.parameters)
    for name, annotation in annotations.items():
        p = params[name]
        params[name] = Parameter(
            name=name, kind=p.kind, default=p.default, annotation=annotation
        )
    return Signature(params.values(), return_annotation=return_annotation)


def common_and_diff_argnames(func1: callable, func2: callable) -> dict:
    """Get list of argument names that are common to two functions, as well as the two
    lists of names that are different

    Args:
        func1: First function
        func2: Second function

    Returns: A dict with fields 'common', 'func1_not_func2', and 'func2_not_func1'

    >>> def f(t, h, i, n, k):
    ...     ...
    ...
    >>> def g(t, w, i, c, e):
    ...     ...
    ...
    >>> common_and_diff_argnames(f, g)
    {'common': ['t', 'i'], 'func1_not_func2': ['h', 'n', 'k'], 'func2_not_func1': ['w', 'c', 'e']}
    >>> common_and_diff_argnames(g, f)
    {'common': ['t', 'i'], 'func1_not_func2': ['w', 'c', 'e'], 'func2_not_func1': ['h', 'n', 'k']}
    """
    p1 = signature(func1).parameters
    p2 = signature(func2).parameters
    return {
        "common": [x for x in p1 if x in p2],
        "func1_not_func2": [x for x in p1 if x not in p2],
        "func2_not_func1": [x for x in p2 if x not in p1],
    }


dflt_name_for_kind = {
    Parameter.VAR_POSITIONAL: "args",
    Parameter.VAR_KEYWORD: "kwargs",
}

arg_order_for_param_tuple = ("name", "default", "annotation", "kind")


def set_signature_of_func(
    func, parameters, *, return_annotation=empty, __validate_parameters__=True
):
    """Set the signature of a function, with sugar.

    Args:
        func: Function whose signature you want to set
        signature: A list of parameter specifications. This could be an
        inspect.Parameter object or anything that
            the mk_param function can resolve into an inspect.Parameter object.
        return_annotation: Passed on to inspect.Signature.
        __validate_parameters__: Passed on to inspect.Signature.

    Returns:
        None (but sets the signature of the input function)

    >>> import inspect
    >>> def foo(*args, **kwargs):
    ...     pass
    ...
    >>> inspect.signature(foo)
    <Signature (*args, **kwargs)>
    >>> set_signature_of_func(foo, ["a", "b", "c"])
    >>> inspect.signature(foo)
    <Signature (a, b, c)>
    >>> set_signature_of_func(
    ...     foo, ["a", ("b", None), ("c", 42, int)]
    ... )  # specifying defaults and annotations
    >>> inspect.signature(foo)
    <Signature (a, b=None, c: int = 42)>
    >>> set_signature_of_func(
    ...     foo, ["a", "b", "c"], return_annotation=str
    ... )  # specifying return annotation
    >>> inspect.signature(foo)
    <Signature (a, b, c) -> str>
    >>> # But you can always specify parameters the "long" way
    >>> set_signature_of_func(
    ...     foo,
    ...     [inspect.Parameter(name="kws", kind=inspect.Parameter.VAR_KEYWORD)],
    ...     return_annotation=str,
    ... )
    >>> inspect.signature(foo)
    <Signature (**kws) -> str>

    """
    sig = Sig(
        parameters,
        return_annotation=return_annotation,
        __validate_parameters__=__validate_parameters__,
    )
    func.__signature__ = sig.to_simple_signature()
    # Not returning func so it's clear(er) that the function is transformed in place


# Pattern: (rewiring) wrapper of make_dataclass
# TODO: Is there a clean way for module to be populated by __name__ of caller module?
def sig_to_dataclass(
    sig: SignatureAble, *, cls_name=None, bases=(), module=None, **kwargs
):
    """
    Make a ``class`` (through ``make_dataclass``) from the given signature.

    :param sig: A ``SignatureAble``, that is, anything that ensure_signature can
        resolve into an ``inspect.Signature`` object, including a signature object
        itself, but also most callables, a list or params, etc.
    :param cls_name: The same as ``cls_name`` of ``dataclasses.make_dataclass``
    :param bases: The same as ``bases`` of ``dataclasses.make_dataclass``
    :param module: Set to module (usually ``__name__`` to specify ther module of
        caller) so that the class and instances can be pickle-able.
    :param kwargs: Passed on to ``dataclasses.make_dataclass``
    :return: A dataclass

    >>> def foo(a, /, b : int=2, *, c=3):
    ...     pass
    ...
    >>> K = sig_to_dataclass(foo, cls_name='K')
    >>> str(Sig(K))
    '(a, b: int = 2, c=3) -> None'
    >>> k = K(1,2,3)
    >>> (k.a, k.b, k.c)
    (1, 2, 3)

    Would also work with any of these (and more):

    >>> K = sig_to_dataclass(Sig(foo), cls_name='K')
    >>> K = sig_to_dataclass(Sig(foo).params, cls_name='K')

    Note: ``cls_name`` is not required (we'll try to figure out a good default for you),
    but it's advised to only use this convenience in extreme mode.
    Choosing your own name might make for a safer future if you're reusing your class.

    """
    from dataclasses import make_dataclass

    sig = ensure_signature(sig)
    cls_name = cls_name or getattr(sig, "name", "_made_by_sig_to_dataclass")
    params = ensure_params(sig)
    fields = [(p.name, p.annotation, p.default) for p in params]
    cls = make_dataclass(cls_name, fields, bases=bases, **kwargs)
    if module:
        cls.__module__ = module
    return cls


def replace_kwargs_using(sig: SignatureAble):
    """
    Decorator that replaces the variadic keyword argument of the target function using
    the `sig`, the signature of a source function.
    It essentially injects the difference between `sig` and the target function's
    signature into the target function's signature. That is, it replaces the
    variadic keyword argument (a.k.a. "kwargs") with those parameters that are in `sig`
    but not in the target function's signature.

    This is meant to be used when a `targ_func` (the function you'll apply the
    decorator to) has a variadict keyword argument that is just used to forward "extra"
    arguments to another function, and you want to make sure that the signature of the
    `targ_func` is consistent with the `sig` signature.
    (Also, you don't want to copy the signatures around manually.)

    In the following, `sauce` (the target function) has a variadic keyword argument,
    `sauce_kwargs`, that is used to forward extra arguments to `apple` (the source
    function).

    >>> def apple(a, x: int, y=2, *, z=3, **extra_apple_options):
    ...     return a + x + y + z
    >>> @replace_kwargs_using(apple)
    ... def sauce(a, b, c, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)

    The function will works:

    >>> sauce(1, 2, 3, x=4, z=5)  # func still works? Should be: 1 + 4 + 2 + 5 + 2 * 3
    18

    But the signature now doesn't have the `**sauce_kwargs`, but more informative
    signature elements sourced from `apple`:

    >>> Sig(sauce)
    <Sig (a, b, c, *, x: int, y=2, z=3, **extra_apple_options)>

    One thing to note is that the order of the arguments in the signature of `apple`
    may change to accomodate for the python parameter order rules
    (see https://docs.python.org/3/reference/compound_stmts.html#function-definitions).
    The new order will try to conserve the order of the original arguments of `sauce`
    in-so-far as it doesn't violate the python parameter order rules, though.
    See examples below:

    >>> @Sig.replace_kwargs_using(apple)
    ... def sauce(a, b=2, c=3, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)
    >>> Sig(sauce)
    <Sig (a, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>

    >>> @Sig.replace_kwargs_using(apple)
    ... def sauce(a=1, b=2, c=3, **sauce_kwargs):
    ...     return b * c + apple(a, **sauce_kwargs)
    >>> Sig(sauce)
    <Sig (a=1, b=2, c=3, *, x: int, y=2, z=3, **extra_apple_options)>

    """

    def decorator(targ_func):
        targ_func_sig = Sig(targ_func)  # function whose signature we're changing
        if targ_func_sig.has_var_keyword:
            # remove it from the signature of targ_sig (we're replacing it!)
            targ_func_sig = Sig(targ_func)[:-1]
        else:
            # if there is none, we shouldn't be using replace_kwargs_using!
            raise ValueError(
                f"Target function {targ_func} must have a variadict keyword argument"
            )

        src_sig = Sig(sig)  # signature we're using to replace kwargs of targ_func

        # Remove all params of src_sig that are in targ_func_sig
        # This is because if they're used, they will be bound to the non-variadic
        # target arguments, so there's no conflict: the target kind, default,
        # and annotation should be used not the source ones.
        src_sig -= targ_func_sig

        # make all parameters of src_sig keyword-only
        # (they're replacing variadic keywords after all!)
        # All? No -- a variadic keyword in the src_sig should remain so
        names_of_all_params_in_src_sig_that_are_not_variadic_keyword = [
            p.name for p in src_sig.params if p.kind != Parameter.VAR_KEYWORD
        ]
        n = len(names_of_all_params_in_src_sig_that_are_not_variadic_keyword)

        src_sig = src_sig.ch_kinds(
            **dict(
                zip(
                    names_of_all_params_in_src_sig_that_are_not_variadic_keyword,
                    [Parameter.KEYWORD_ONLY] * n,
                )
            )
        )

        new_sig = targ_func_sig.merge_with_sig(src_sig)
        return new_sig(targ_func)

    return decorator


Sig.replace_kwargs_using = replace_kwargs_using

#########################################################################################
# Manual construction of missing signatures
# ############################################################################


# TODO: Might want to monkey-patch inspect._signature_from_callable to use
#  sigs_for_sigless_builtin_name
def _robust_signature_of_callable(callable_obj: Callable) -> Signature:
    r"""Get the signature of a Callable, returning a custom made one for those
    builtins that don't have one

    >>> _robust_signature_of_callable(
    ...     _robust_signature_of_callable
    ... )  # has a normal signature
    <Signature (callable_obj: ...Callable) -> inspect.Signature>
    >>> s = _robust_signature_of_callable(print)  # has one that this module provides
    >>> assert isinstance(s, Signature)
    >>> # Will be: <Signature (*value, sep=' ', end='\n', file=<_io.TextIOWrapper
    name='<stdout>' mode='w' encoding='utf-8'>, flush=False)>
    >>> _robust_signature_of_callable(
    ...     slice
    ... )  # doesn't have one, so will return a blanket one
    <Signature (*no_sig_args, **no_sig_kwargs)>

    """
    # First check if we have a custom signature for this type/object
    # This is important for operator instances that might have generic signatures in Python 3.12+
    obj_name = getattr(callable_obj, "__name__", None)
    if obj_name in sigs_for_sigless_builtin_name:
        return sigs_for_sigless_builtin_name[obj_name] or DFLT_SIGNATURE

    type_name = getattr(type(callable_obj), "__name__", None)
    if type_name in sigs_for_type_name:
        return sigs_for_type_name[type_name] or DFLT_SIGNATURE

    # Try to get the signature normally
    try:
        return signature(callable_obj)
    except ValueError:
        # if all attempts fail, return the default signature
        return DFLT_SIGNATURE


def resolve_function(obj: T) -> T | Callable:
    """Get the underlying function of a property or cached_property

    Note that if all conditions fail, the object itself is returned.

    The problem this function solves is that sometimes there's a function behind an
    object, but it's not always easy to get to it. For example, in a class, you might
    want to get the source of the code decorated with ``@property``, a
    ``@cached_property``, or a ``partial`` function.

    Consider the following example:

    >>> from functools import cached_property, partial
    >>> class C:
    ...     @property
    ...     def prop(self):
    ...         pass
    ...     @cached_property
    ...     def cached_prop(self):
    ...         pass
    ...     partial_func = partial(partial)

    Note that ``prop`` is not callable, and you can't get its source.

    >>> import inspect
    >>> callable(C.prop)
    False
    >>> inspect.getsource(C.prop)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    TypeError: <property object at 0x...> is not a module, class, method, function, traceback, frame, or code object

    But if you grab the underlying function, you can get the source:

    >>> func = resolve_function(C.prop)
    >>> callable(func)
    True
    >>> isinstance(inspect.getsource(func), str)
    True

    Same goes with ``cached_property`` and ``partial``:

    >>> isinstance(inspect.getsource(resolve_function(C.cached_prop)), str)
    True
    >>> isinstance(inspect.getsource(resolve_function(C.partial_func)), str)
    True

    """
    if isinstance(obj, cached_property):
        return obj.func
    elif isinstance(obj, property):
        return obj.fget
    elif isinstance(obj, (partial, partialmethod)):
        return obj.func
    elif not callable(obj) and callable(wrapped := getattr(obj, "__wrapped__", None)):
        # If obj is not callable, but has a __wrapped__ attribute that is, return that
        return wrapped
    else:  # if not just return obj
        return obj


def dict_of_attribute_signatures(cls: type) -> dict[str, Signature]:
    """
    A function that extracts the signatures of all callable attributes of a class.

    :param cls: The class that holds the the ``(name, func)`` pairs we want to extract.
    :return: A dict of ``(name, signature(func))`` pairs extracted from class.

    One of the intended applications is to use ``dict_of_attribute_signatures`` as a
    decorator, like so:

    >>> @dict_of_attribute_signatures
    ... class names_and_signatures:
    ...     def foo(x: str, *, y=2) -> tuple: ...
    ...     def bar(z, /) -> float: ...
    >>> names_and_signatures
    {'foo': <Signature (x: str, *, y=2) -> tuple>, 'bar': <Signature (z, /) -> float>}
    """

    def gen():
        object_attr_names = set(vars(object))
        for attr_name, attr_val in vars(cls).items():
            if callable(attr_val):
                if attr_name not in object_attr_names:
                    # if the attr is a callable attribute that's not in all objects...
                    yield attr_name, signature(attr_val)

    return dict(gen())


@dict_of_attribute_signatures
class sigs_for_builtins:
    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        """__import__(name, globals=None, locals=None, fromlist=(), level=0) -> module"""

    def filter(function, iterable, /):
        """filter(function or None, iterable) --> filter object"""

    def map(func, iterable, /, *iterables):
        """map(func, *iterables) --> map object"""

    def print(*value, sep=" ", end="\n", file=sys.stdout, flush=False):
        """print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)"""

    def zip(*iterables):
        """
        zip(*iterables) --> A zip object yielding tuples until an input is exhausted.
        """

    def bool(x: Any, /) -> bool: ...

    def bytearray(iterable_of_ints: Iterable[int], /): ...

    def classmethod(function: Callable, /): ...

    def int(x, base=10, /): ...

    def iter(callable: Callable, sentinel=None, /): ...

    def next(iterator: Iterator, default=None, /): ...

    def staticmethod(function: Callable, /): ...

    def str(bytes_or_buffer, encoding=None, errors=None, /): ...

    def super(type_, obj=None, /): ...

    # def type(name, bases=None, dict=None, /):
    #     ...


sigs_for_builtins = dict(
    sigs_for_builtins,
    **{
        "__build_class__": None,
        # __build_class__(func, name, /, *bases, [metaclass], **kwds) -> class
        # "bool": None,
        # bool(x) -> bool
        "breakpoint": None,
        # breakpoint(*args, **kws)
        # "bytearray": None,
        # bytearray(iterable_of_ints) -> bytearray
        # bytearray(string, encoding[, errors]) -> bytearray
        # bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer
        # bytearray(int) -> bytes array of size given by the parameter initialized with
        # null bytes
        # bytearray() -> empty bytes array
        "bytes": None,
        # bytes(iterable_of_ints) -> bytes
        # bytes(string, encoding[, errors]) -> bytes
        # bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer
        # bytes(int) -> bytes object of size given by the parameter initialized with null
        # bytes
        # bytes() -> empty bytes object
        # "classmethod": None,
        # classmethod(function) -> method
        "dict": None,
        # dict() -> new empty dictionary
        # dict(mapping) -> new dictionary initialized from a mapping object's
        # dict(iterable) -> new dictionary initialized as if via:
        # dict(**kwargs) -> new dictionary initialized with the name=value pairs
        "dir": None,
        # dir([object]) -> list of strings
        "frozenset": None,
        # frozenset() -> empty frozenset object
        # frozenset(iterable) -> frozenset object
        "getattr": None,
        # getattr(object, name[, default]) -> value
        # "int": None,
        # int([x]) -> integer
        # int(x, base=10) -> integer
        # "iter": None,
        # iter(iterable) -> iterator
        # iter(callable, sentinel) -> iterator
        "max": None,
        # max(iterable, *[, default=obj, key=func]) -> value
        # max(arg1, arg2, *args, *[, key=func]) -> value
        "min": None,
        # min(iterable, *[, default=obj, key=func]) -> value
        # min(arg1, arg2, *args, *[, key=func]) -> value
        # "next": None,
        # next(iterator[, default])
        "range": None,
        # range(stop) -> range object
        # range(start, stop[, step]) -> range object
        "set": None,
        # set() -> new empty set object
        # set(iterable) -> new set object
        "slice": None,
        # slice(stop)
        # slice(start, stop[, step])
        # "staticmethod": None,
        # staticmethod(function) -> method
        # "str": None,
        # str(object='') -> str
        # str(bytes_or_buffer[, encoding[, errors]]) -> str
        # "super": None,
        # super() -> same as super(__class__, <first argument>)
        # super(type) -> unbound super object
        # super(type, obj) -> bound super object; requires isinstance(obj, type)
        # super(type, type2) -> bound super object; requires issubclass(type2, type)
        # "type": None,
        # type(object_or_name, bases, dict)
        # type(object) -> the object's type
        # type(name, bases, dict) -> a new type
        "vars": None,
        # vars([object]) -> dictionary
    },
)
# # Remove the None-valued elements (No, don't, because we distinguish
# # functions we listed but didn't associate a default signature, with those functions
# # we don't list at all.
# sigs_for_builtins = {
#     k: v for k, v in sigs_for_builtins.items() if v is not None
# }


# TODO: itemgetter, attrgetter and methodcaller use KT as their first argument, but
#  in reality both attrgetter and methodcaller are more restrictive: They need to be
#  valid attributes, therefore valid python identifiers. Any better typing for that?
# TODO: We take care of the MutableMapping dunders below, but some of these dunders
#  are not specific to MutableMapping. The signature used below is somewhat, but not
#  completely, specific to MutableMapping. For example, __contains__ is also defined
#  for the `set` type, but it's input is not called key, nor would the KT annotation
#  be completely correct. The signatures were sometimes made to be more general (such
#  as __setitem__ and __delitem__ returning an Any instead of None), but could be
#  made more (for example, annotating return of __iter__ as Iterator instead of
#  Iterator[KT]). We hope that the fact that all the signatures are positional-only
#  will at least mitigate the problem as far as name differences go.
@dict_of_attribute_signatures
class sigs_for_builtin_modules:
    """
    Below are the signatures, manually created to match those callables of the python
    standard library that don't have signatures (through ``inspect.signature``),
    """

    def __eq__(self, other, /) -> bool:
        """self.__eq__(other) <==> self==other"""

    def __ne__(self, other, /) -> bool:
        """self.__ne__(other) <==> self!=other"""

    def __iter__(self, /) -> Iterator[KT]:
        """self.__iter__() <==> iter(self)"""

    def __getitem__(self, key: KT, /) -> VT:
        """self.__getitem__(key) <==> self[key]"""

    def __len__(self, /) -> int:
        """self.__len__() <==> len(self)"""

    def __contains__(self, key: KT, /) -> bool:
        """self.__contains__(key) <==> key in self"""

    def __setitem__(self, key: KT, value: VT, /) -> Any:
        """self.__setitem__(key, value) <==> self[key] = value"""

    def __delitem__(self, key: KT, /) -> Any:
        """self.__delitem__(key) <==> del self[key]"""

    def itemgetter(item, /, *items) -> Callable[[Iterable[VT]], VT | tuple[VT]]:
        """itemgetter(item, ...) --> itemgetter object,"""

    def attrgetter(attr, /, *attrs) -> Callable[[Iterable[VT]], VT | tuple[VT]]:
        """attrgetter(item, ...) --> attrgetter object,"""

    def methodcaller(
        name: KT, /, *args: Iterable[VT], **kwargs: MappingType[str, Any]
    ) -> Callable[[Any], Any]:
        """methodcaller(name, ...) --> methodcaller object"""

    def partial(func: Callable, *args, **keywords) -> Callable:
        """``partial(func, *args, **keywords)`` - new function with partial application
        of the given arguments and keywords."""

    def partialmethod(func: Callable, *args, **keywords) -> Callable:
        """``functools.partialmethod(func, *args, **keywords)``"""


# Merge sigs_for_builtin_modules and sigs_for_builtins
sigs_for_sigless_builtin_name = dict(sigs_for_builtin_modules, **sigs_for_builtins)


@dict_of_attribute_signatures
class sigs_for_type_name:
    """
    Below are the signatures, manually created to match callable objects that are
    output by builtin functions or are instances of builtin classes, and that have no
    signatures (through ``inspect.signature``),
    """

    def itemgetter(iterable: Iterable[VT], /) -> VT | tuple[VT]: ...

    def attrgetter(iterable: Iterable[VT], /) -> VT | tuple[VT]: ...

    def methodcaller(obj: Any) -> Any: ...


############# Tools for testing #########################################################


def param_for_kind(
    name=None,
    kind="positional_or_keyword",
    with_default=False,
    annotation=Parameter.empty,
):
    """Function to easily and flexibly make inspect.Parameter objects for testing.

    It's annoying to have to compose parameters from scratch to testing things.
    This tool should help making it less annoying.

    >>> list(map(param_for_kind, param_kinds))
    [<Parameter "POSITIONAL_ONLY">, <Parameter "POSITIONAL_OR_KEYWORD">, <Parameter "VAR_POSITIONAL">, <Parameter "KEYWORD_ONLY">, <Parameter "VAR_KEYWORD">]
    >>> param_for_kind.positional_or_keyword()
    <Parameter "POSITIONAL_OR_KEYWORD">
    >>> param_for_kind.positional_or_keyword("foo")
    <Parameter "foo">
    >>> param_for_kind.keyword_only()
    <Parameter "KEYWORD_ONLY">
    >>> param_for_kind.keyword_only("baz", with_default=True)
    <Parameter "baz='dflt_keyword_only'">
    """
    name = name or f"{kind}"
    kind_obj = getattr(Parameter, str(kind).upper())
    kind = str(kind_obj).lower()
    default = (
        f"dflt_{kind}"
        if with_default and kind not in {"var_positional", "var_keyword"}
        else Parameter.empty
    )
    return Parameter(name=name, kind=kind_obj, default=default, annotation=annotation)


param_kinds = list(filter(lambda x: x.upper() == x, Parameter.__dict__))

for kind in param_kinds:
    lower_kind = kind.lower()
    setattr(param_for_kind, lower_kind, partial(param_for_kind, kind=kind))
    setattr(
        param_for_kind,
        "with_default",
        partial(param_for_kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, lower_kind),
        "with_default",
        partial(param_for_kind, kind=kind, with_default=True),
    )
    setattr(
        getattr(param_for_kind, "with_default"),
        lower_kind,
        partial(param_for_kind, kind=kind, with_default=True),
    )

########################################################################################
# Signature Comparison and Compatibility #
########################################################################################

Compared = TypeVar("Compared")
Comparison = TypeVar("Comparison")
Comparator = Callable[[Compared, Compared], Comparison]
Comparison.__doc__ = (
    "The return type of a Comparator. Typically a bool, or int, but can be anything."
    'In that sense it is more of a "collation" than I comparison'
)

# TODO: Make function that makes Comparator types according for different kinds of
#  compared types? (e.g. for comparing signatures, for comparing parameters, ...)
#  See HasAttr in https://github.com/i2mint/i2/blob/feb469acdc0bc8268877b400b9af6dda56de6292/i2/itypes.py#L164
#  for inspiration.
SignatureComparator = Callable[[Signature, Signature], Comparison]
ParamComparator = Callable[[Parameter, Parameter], Comparison]
CallableComparator = Callable[[Callable, Callable], Comparison]

ComparisonAggreg = Callable[[Iterable[Comparison]], Any]

CT = TypeVar("CT")  # some other Compared type (used to define KeyFunction
KeyFunction = Callable[[CT], Compared]
# KeyFunction.__doc__ = "Function that transforms one compared type to another"


def compare_signatures(func1, func2, signature_comparator: SignatureComparator = eq):
    return signature_comparator(Sig(func1), Sig(func2))


# TODO: Look into typing: Why does lint complain about this line of code?
def mk_func_comparator_based_on_signature_comparator(
    signature_comparator: SignatureComparator,
) -> CallableComparator:
    return partial(compare_signatures, signature_comparator=signature_comparator)


def _keyed_comparator(
    comparator: Comparator,
    key: KeyFunction,
    x: CT,
    y: CT,
) -> Comparison:
    """Apply a comparator after transforming inputs through a key function.

    >>> from operator import eq
    >>> parity = lambda x: x % 2
    >>> _keyed_comparator(eq, parity, 1, 3)
    True
    >>> _keyed_comparator(eq, parity, 1, 4)
    False
    """
    return comparator(key(x), key(y))


def keyed_comparator(
    comparator: Comparator,
    key: KeyFunction,
) -> Comparator:
    """Create a key-function enabled binary operator.

    In various places in python functionality is extended by allowing a key function.
    For example, the ``sorted`` function allows a key function to be passed, which is
    applied to each element before sorting. The keyed_comparator function allows a
    comparator to be extended in the same way. The returned comparator will apply the
    key function toeach input before applying the original comparator.

    >>> from operator import eq
    >>> parity = lambda x: x % 2
    >>> comparator = keyed_comparator(eq, parity)
    >>> list(map(comparator, [1, 1, 2, 2], [3, 4, 5, 6]))
    [True, False, False, True]
    """
    return partial(_keyed_comparator, comparator, key)


# For back-compatibility:
_key_function_enabled_operator = _keyed_comparator
_key_function_factory = keyed_comparator


# TODO: Show examples of how this can be used to produce precise error messages.
#  The way to do this is to have the attribute binary functions produce some info dicts
#  that can then be aggregated in aggreg to produce a final error message (or even
#  a final error object, which can even be raised) if there is indeed a mismatch at all.
#  Further more, we might want to make a function that will take a parametrized
#  param_binary_func and produce such a error raising function from it, using the
#  specific functions (extracted by Sig) to produce the error message.
def param_comparator(
    param1: Parameter,
    param2: Parameter,
    *,
    name: Comparator = eq,
    kind: Comparator = eq,
    default: Comparator = eq,
    annotation: Comparator = eq,
    aggreg: ComparisonAggreg = all,
) -> Comparison:
    """Compare two parameters.

    Note that by default, this function is strict, and will return False if
    any of the parameters are not equal. This is because the default
    aggregation function is `all` and the default comparison functions of the
    parameter's attributes are `eq` (meaning equality, not identity).

    But you can change that by passing different comparison functions and/or
    aggregation functions.

    In fact, the real purpose of this function is to be used as a factory of parameter
    binary functions, through parametrizing it with `functools.partial`.

    The parameter binary functions themselves are meant to be used to make signature
    binary functions.

    :param param1: first parameter
    :param param2: second parameter
    :param name: function to compare names
    :param kind: function to compare kinds
    :param default: function to compare defaults
    :param annotation: function to compare annotations
    :param aggreg: function to aggregate results

    >>> from inspect import Parameter
    >>> param1 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
    >>> param2 = Parameter('x', Parameter.POSITIONAL_OR_KEYWORD)
    >>> param_binary_func(param1, param2)
    True

    See https://github.com/i2mint/i2/issues/50#issuecomment-1381686812 for discussion.

    """
    return aggreg(
        (
            name(param1.name, param2.name),
            kind(param1.kind, param2.kind),
            default(param1.default, param2.default),
            annotation(param1.annotation, param2.annotation),
        )
    )


param_comparator: ParamComparator
param_binary_func = param_comparator  # back compatibility alias


def dflt1_is_empty_or_dflt2_is_not(dflt1, dflt2):
    """
    Why such a strange default comparison function?

    This is to be used as a default in is_call_compatible_with.

    Consider two functions func1 and func2 with a parameter p with default values
    dflt1 and dflt2 respectively.
    If dflt1 was not empty and dflt2 was, this would mean that func1 could be called
    without specifying p, but func2 couldn't.

    So to avoid this situation, we use dflt1_is_empty_or_dflt2_is_not as the default

    """
    return dflt1 is empty or dflt2 is not empty


# TODO: Implement annotation compatibility
def ignore_any_differences(x, y):
    return True


permissive_param_comparator = partial(
    param_comparator,
    name=ignore_any_differences,
    kind=ignore_any_differences,
    default=ignore_any_differences,
    annotation=ignore_any_differences,
)
permissive_param_comparator.__doc__ = """
Permissive version of param_comparator that ignores any differences of parameter 
attributes.

It is meant to be used with partial, but with a permissive base, contrary to the 
base param_comparator which requires strict equality (`eq`) for all attributes.
"""

dflt1_is_empty_or_dflt2_is_not_param_comparator = partial(
    permissive_param_comparator, default=dflt1_is_empty_or_dflt2_is_not
)


def return_tuple(x, y):
    return x, y


param_attribute_dict: ComparisonAggreg


def param_attribute_dict(name_kind_default_annotation: Iterable[Comparison]) -> dict:
    keys = ["name", "kind", "default", "annotation"]
    return {key: value for key, value in zip(keys, name_kind_default_annotation)}


param_comparison_dict = partial(
    param_comparator,
    name=return_tuple,
    kind=return_tuple,
    default=return_tuple,
    annotation=return_tuple,
    aggreg=param_attribute_dict,
)

param_comparison_dict.__doc__ = """
A ParamComparator that returns a dictionary with pairs parameter attributes.

>>> param1 = Sig('(a: int = 1)')['a']
>>> param2 = Sig('(a: str = 2)')['a']
>>> param_comparison_dict(param1, param2)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
{'name': ('a', 'a'), 'kind': ..., 'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
"""


def param_differences_dict(
    param1: Parameter,
    param2: Parameter,
    *,
    name: Comparator = eq,
    kind: Comparator = eq,
    default: Comparator = eq,
    annotation: Comparator = eq,
):
    """Makes a dictionary exibiting the differences between two parameters.

    >>> param1 = Sig('(a: int = 1)')['a']
    >>> param2 = Sig('(a: str = 2)')['a']
    >>> param_differences_dict(param1, param2)
    {'default': (1, 2), 'annotation': (<class 'int'>, <class 'str'>)}
    >>> param_differences_dict(param1, param2, default=lambda x, y: isinstance(x, type(y)))
    {'annotation': (<class 'int'>, <class 'str'>)}
    """
    equality_vector = param_comparator(
        param1,
        param2,
        name=name,
        kind=kind,
        default=default,
        annotation=annotation,
        aggreg=tuple,
    )
    comparison_dict = param_comparison_dict(param1, param2)
    return {
        key: comparison_dict[key]
        for key, equal in zip(comparison_dict, equality_vector)
        if not equal
    }


def defaults_are_the_same_when_not_empty(dflt1, dflt2):
    """
    Check if two defaults are the same when they are not empty.

    # >>> defaults_are_the_same_when_not_empty(1, 1)
    # True
    # >>> defaults_are_the_same_when_not_empty(1, 2)
    # False
    # >>> defaults_are_the_same_when_not_empty(1, None)
    # False
    # >>> defaults_are_the_same_when_not_empty(1, Parameter.empty)
    # True
    """
    return dflt1 is empty or dflt2 is empty or dflt1 == dflt2


def postprocess(egress: Callable):
    """A decorator that will process the output of the wrapped function with egress"""

    # Note: Vendorized version equivalent ones in i2.deco and i2.wrapper
    def postprocessed(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            original_output = func(*args, **kwargs)
            return egress(original_output)

        return wrapped_func

    return postprocessed


# TODO: It seems like param_comparator is really only used to compare parameters on defaults.
#   This may be due to the fact that is_call_compatible_with was developed independently
#   from the other general param_comparator functionality that was developed (see above)
#   The code of is_call_compatible_with should be reviwed and refactored to use general
#   tools.
@postprocess(
    all
)  # see "Use of postprocess" in https://github.com/i2mint/i2/discussions/63#discussioncomment-10394910
def is_call_compatible_with(
    sig1: Sig,
    sig2: Sig,
    *,
    param_comparator: ParamComparator | None = None,
) -> bool:
    """Return True if ``sig1`` is compatible with ``sig2``. Meaning that all valid ways
    to call ``sig1`` are valid for ``sig2``.

    :param sig1: The main signature.
    :param sig2: The signature to be compared with.
    :param param_comparator: The function used to compare two parameters

    >>> is_call_compatible_with(
    ...     Sig('(a, /, b, *, c)'),
    ...     Sig('(a, b, c)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('()'),
    ...     Sig('(a)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('()'),
    ...     Sig('(a=0)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, *, c)'),
    ...     Sig('(a, /, b, *, c)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('(a, /, *, c)'),
    ...     Sig('(a, /, b=0, *, c)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b)'),
    ...     Sig('(a, /, b, *, c)')
    ... )
    False
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b)'),
    ...     Sig('(a, /, b, *, c=0)')
    ... )
    True
    >>> is_call_compatible_with(
    ...     Sig('(a, /, b, *, c)'),
    ...     Sig('(*args, **kwargs)')
    ... )
    True
    """

    # Note: In case you're tempted to put this default function as an argument default,
    #  don't. Yes, it's preferable in many ways, but makes the "one source of truth"
    #  principle harder to maintain, since this default has to be the same anywhere
    #  the current function is called. Better signature/docs injection functionality
    #  would be warranted.
    #  See https://stackoverflow.com/questions/78874506/how-can-i-avoid-interface-repetition-in-python-function-signatures-and-docstring
    param_comparator = (
        param_comparator or dflt1_is_empty_or_dflt2_is_not_param_comparator
    )

    def validate_variadics():
        return (
            # sig1 can only have a VP if sig2 also has one
            (vp1 is None or vp2 is not None)
            and
            # sig1 can only have a VK if sig2 also has one
            (vk1 is None or vk2 is not None)
        )

    def validate_param_counts():
        # sig1 cannot have more positional params than sig2
        if len(ps1) > len(ps2) and not vp2:
            return False
        # sig1 cannot have keyword params that do not exist in sig2
        if len([n for n in ks1 if n not in ks2]) > 0 and not vk2:
            return False
        return True

    def validate_extra_params():
        # Any extra PO in sig2 must have a default value
        if len(pos1) < len(pos2) and not all(
            sig2.parameters[n].default is not empty for n in pos2[len(pos1) :]
        ):
            return False
        # Any extra PK in sig2 must have its corresponding PO or KO in sig1, or a
        # default value
        for i, n in enumerate(pks2):
            if (
                n not in pks1
                and len(pos1) <= len(pos2) + i
                and n not in kos1
                and sig2.parameters[n].default is empty
            ):
                return False
        # Any extra KO in sig2 must have a default value
        for n in kos2:
            if n not in kos1 and sig2.parameters[n].default == empty:
                return False
        return True

    def validate_param_positions():
        for i, n2 in enumerate(ps2):
            for j, n1 in enumerate(ks1):
                if n1 == n2:
                    if (
                        # It can be a PK in sig1 and a P (PO or PK) in sig2 only if
                        # its position in sig2 is >= to its position in sig1
                        (n1 in pks1 and i < len(pos1) + j)
                        or (
                            n1 in kos1
                            and (
                                # Cannot be a KO in sig1 and a PO in sig2
                                n2 in pos2
                                or
                                # It can be a KO in sig1 and a PK in sig2 only if its
                                # position in sig2 is > than the total number of POs
                                # and PKs in sig1
                                i < len(ps1)
                            )
                        )
                    ):
                        return False
        return True

    def validate_param_compatibility():
        # Every positional param in sig1 must be compatible with its
        # correspondant param in sig2 (at the same index).
        for i in range(len(ps1)):
            if i < len(ps2) and not param_comparator(sig1.params[i], sig2.params[i]):
                return False
        # Every keyword param in sig1 must be compatible with its
        # correspondant param in sig2 (with the same name).
        for n in ks1:
            if n in ks2 and not param_comparator(
                sig1.parameters[n], sig2.parameters[n]
            ):
                return False
        return True

    pos1, pks1, vp1, kos1, vk1 = sig1.detail_names_by_kind()
    ps1 = pos1 + pks1
    ks1 = pks1 + kos1
    pos2, pks2, vp2, kos2, vk2 = sig2.detail_names_by_kind()
    ps2 = pos2 + pks2
    ks2 = pks2 + kos2

    if vp1:
        sig1 -= vp1
    if vk1:
        sig1 -= vk1
    if vp2:
        sig2 -= vp2
    if vk2:
        sig2 -= vk2

    return (
        f()
        for f in [
            validate_variadics,
            validate_param_counts,
            validate_extra_params,
            validate_param_positions,
            validate_param_compatibility,
        ]
    )


from dataclasses import dataclass

from functools import cached_property
from dataclasses import dataclass
from inspect import Parameter


@dataclass
class SigPair:
    """
    Class that operates on a pair of signatures.

    For example, offers methods to compare two signatures in various ways.

    :param sig1: First signature or signature-able object.
    :param sig2: Second signature or signature-able object.

    >>> from pprint import pprint
    >>> def three(a, b: int, c=3): ...
    >>> def little(a, *, b=2, d=4) -> int: ...
    >>> def pigs(a, b) -> int: ...
    >>> sig_pair = SigPair(three, little)
    >>>
    >>> sig_pair.shared_names
    ['a', 'b']
    >>> sig_pair.names_missing_in_sig1
    ['d']
    >>> sig_pair.names_missing_in_sig2
    ['c']
    >>> sig_pair.param_comparison()
    False
    >>> pprint(sig_pair.diff())  # doctest: +NORMALIZE_WHITESPACE
    {'names_missing_in_sig1': ['d'],
    'names_missing_in_sig2': ['c'],
    'param_differences': {'b': {'annotation': (<class 'int'>,
                                                <class 'inspect._empty'>),
                                'default': (<class 'inspect._empty'>, 2),
                                'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                        <_ParameterKind.KEYWORD_ONLY: 3>)}},
    'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}

    Call compatibility says that any arguments leading to a valid call to a function
    having the first signature, will also lead to a valid call to a function having the
    second signature. This is not the case for the signatures of `three` and `little`:

    >>> sig_pair.are_call_compatible()
    False

    But we don't need to have equal signatures to have call compatibility. For example,

    >>> SigPair(three, lambda a, b=2, c=30: None).are_call_compatible()
    True

    Note that call-compatibility is not symmetric. For example, `pigs` is call
    compatible with `three`, since any arguments that are valid for `pigs` are valid
    for `three`:

    >>> SigPair(pigs, three).are_call_compatible()
    True

    But `three` is not call-compatible with `pigs` since `three` requires could include
    a `c` argument, which `pigs` would choke on.

    >>> SigPair(three, pigs).are_call_compatible()
    False

    """

    sig1: Callable | Sig
    sig2: Callable | Sig

    def __post_init__(self):
        self.sig1 = Sig(self.sig1)
        self.sig2 = Sig(self.sig2)

    @cached_property
    def shared_names(self):
        """
        List of names that are common to both signatures, in the order of sig1.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.shared_names
        ['b', 'c']
        """
        return [name for name in self.sig1.names if name in self.sig2.names]

    @cached_property
    def names_missing_in_sig2(self):
        """
        List of names that are in the sig1 signature but not in sig2.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.names_missing_in_sig2
        ['a']
        """
        return [name for name in self.sig1.names if name not in self.sig2.names]

    @cached_property
    def names_missing_in_sig1(self):
        """
        List of names that are in the sig2 signature but not in sig1.

        >>> sig1 = Sig(lambda a, b, c: None)
        >>> sig2 = Sig(lambda b, c, d: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.names_missing_in_sig1
        ['d']
        """
        return [name for name in self.sig2.names if name not in self.sig1.names]

    # TODO: Verify that the doctests are correct!
    def are_call_compatible(self, param_comparator=None) -> bool:
        """
        Check if the signatures are call-compatible.

        Returns True if sig1 can be used to call sig2 or vice versa.

        >>> sig1 = Sig(lambda a, b, c=3: None)
        >>> sig2 = Sig(lambda a, b: None)
        >>> comp = SigPair(sig1, sig2)
        >>> comp.are_call_compatible()
        False

        >>> comp = SigPair(sig2, sig1)
        >>> comp.are_call_compatible()
        True
        """
        return is_call_compatible_with(
            self.sig1, self.sig2, param_comparator=param_comparator
        )

    def param_comparison(self, comparator=param_comparator, aggregation=all) -> bool:
        """
        Compare parameters between the two signatures using the provided comparator function.

        :param comparator: A function to compare two parameters.
        :param aggregation: A function to aggregate the results of the comparisons.
        :return: Boolean result of the aggregated comparisons.

        >>> sig1 = Sig('(a, b: int, c=3)')
        >>> sig2 = Sig('(a, *, b=2, d=4)')
        >>> comp = SigPair(sig1, sig2)
        >>> comp.param_comparison()
        False
        """
        results = [
            comparator(self.sig1.parameters[name], self.sig2.parameters[name])
            for name in self.shared_names
        ]
        return aggregation(results)

    def param_differences(self) -> dict:
        """
        Get a dictionary of parameter differences between the two signatures.

        :return: A dict containing differences for each shared param that has any.

        >>> sig1 = Sig('(a, b: int, c=3)')
        >>> sig2 = Sig('(a, *, b=2, d=4)')
        >>> comp = SigPair(sig1, sig2)
        >>> result = comp.param_differences()
        >>> expected = {
        ...     'b': {
        ...         'kind': (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY),
        ...         'default': (Parameter.empty, 2),
        ...         'annotation': (int, Parameter.empty),
        ...     }
        ... }
        >>> result == expected
        True
        """

        def diff_pairs():
            for name in self.shared_names:
                diff_dict = param_differences_dict(
                    self.sig1.parameters[name], self.sig2.parameters[name]
                )
                if diff_dict:
                    yield name, diff_dict

        return dict(diff_pairs())

    def diff(self) -> dict:
        """
        Get a dictionary of differences between the two signatures.

        >>> from pprint import pprint
        >>> def three(a, b: int, c=3): ...
        >>> def little(a, *, b=2, d=4) -> int: ...
        >>> def pigs(a, b: int = 2) -> int: ...
        >>> pprint(SigPair(three, little).diff())  # doctest: +NORMALIZE_WHITESPACE
        {'names_missing_in_sig1': ['d'],
        'names_missing_in_sig2': ['c'],
        'param_differences': {'b': {'annotation': (<class 'int'>,
                                                    <class 'inspect._empty'>),
                                    'default': (<class 'inspect._empty'>, 2),
                                    'kind': (<_ParameterKind.POSITIONAL_OR_KEYWORD: 1>,
                                            <_ParameterKind.KEYWORD_ONLY: 3>)}},
        'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
        >>> pprint(SigPair(three, pigs).diff())  # doctest: +NORMALIZE_WHITESPACE
        {'names_missing_in_sig2': ['c'],
        'param_differences': {'b': {'default': (<class 'inspect._empty'>, 2)}},
        'return_annotation': (<class 'inspect._empty'>, <class 'int'>)}
        >>> pprint(SigPair(three, three).diff())
        {}
        """
        d = {
            key: value
            for key, value in {
                "names_missing_in_sig1": self.names_missing_in_sig1,
                "names_missing_in_sig2": self.names_missing_in_sig2,
                "param_differences": self.param_differences(),
            }.items()
            if value
        }
        # add the return_annotation difference, if any
        if self.sig1.return_annotation != self.sig2.return_annotation:
            d["return_annotation"] = (
                self.sig1.return_annotation,
                self.sig2.return_annotation,
            )
        return d

    def diff_str(self) -> str:
        """
        Get a string representation of the differences between the two signatures.
        """
        from pprint import pformat

        return pformat(self.diff())
```

## sources.py

```python
"""
This module contains key-value views of disparate sources.
"""

from typing import Union, Any
from collections.abc import Iterator, Mapping, Iterable, Callable
from operator import itemgetter
from itertools import groupby as itertools_groupby

from dol.base import KvReader, KvPersister
from dol.trans import cached_keys
from dol.caching import mk_cached_store
from dol.util import copy_attrs
from dol.signatures import Sig


# ignore_if_module_not_found = suppress(ModuleNotFoundError)
#
# with ignore_if_module_not_found:
#     # To install: pip install mongodol
#     from mongodol.stores import (
#         MongoStore,
#         MongoTupleKeyStore,
#         MongoAnyKeyStore,
#     )


def identity_func(x):
    return x


def inclusive_subdict(d, include):
    return {k: d[k] for k in d.keys() & include}


def exclusive_subdict(d, exclude):
    return {k: d[k] for k in d.keys() - exclude}


class NotUnique(ValueError):
    """Raised when an iterator was expected to have only one element, but had more"""


NoMoreElements = type("NoMoreElements", (object,), {})()


def unique_element(iterator):
    element = next(iterator)
    if next(iterator, NoMoreElements) is not NoMoreElements:
        raise NotUnique("iterator had more than one element")
    return element


KvSpec = Union[Callable, Iterable[Union[str, int]], str, int]


def _kv_spec_to_func(kv_spec: KvSpec) -> Callable:
    if isinstance(kv_spec, (str, int)):
        return itemgetter(kv_spec)
    elif isinstance(kv_spec, Iterable):
        return itemgetter(*kv_spec)
    elif kv_spec is None:
        return identity_func
    return kv_spec


# TODO: This doesn't work
# KvSpec.from = _kv_spec_to_func  # I'd like to be able to couple KvSpec and it's
# conversion function (even more: __call__ instead of from)


# TODO: Generalize to several layers
#   Need a general tool for flattening views.
#   What we're doing here is giving access to a nested/tree structure through a key-value
#   view where keys specify tree paths.
#   Should handle situations where number layers are not fixed in advanced,
#   but determined by some rules executed dynamically.
#   Related DirStore and kv_walk.
class FlatReader(KvReader):
    """Get a 'flat view' of a store of stores.
    That is, where keys are `(first_level_key, second_level_key)` pairs.
    This is useful, for instance, to make a union of stores (you'll get all the values).

    >>> readers = {
    ...     'fr': {1: 'un', 2: 'deux'},
    ...     'it': {1: 'uno', 2: 'due', 3: 'tre'},
    ... }
    >>> s = FlatReader(readers)
    >>> list(s)
    [('fr', 1), ('fr', 2), ('it', 1), ('it', 2), ('it', 3)]
    >>> s[('fr', 1)]
    'un'
    >>> s['it', 2]
    'due'
    """

    def __init__(self, readers):
        self._readers = readers

    def __iter__(self):
        # go through the first level paths:
        for first_level_path, reader in self._readers.items():
            for second_level_path in reader:  # go through the keys of the reader
                yield first_level_path, second_level_path

    def __getitem__(self, k):
        first_level_path, second_level_path = k
        return self._readers[first_level_path][second_level_path]


from collections import ChainMap


from collections import ChainMap
from typing import TypedDict, Union
from collections.abc import Callable, Iterable, Iterator, Mapping

from dol.base import KvPersister, KvReader
from dol.trans import wrap_kvs


class FanoutReader(KvReader):
    """Get a 'fanout view' of a store of stores.
    That is, when a key is requested, the key is passed to all the stores, and results
    accumulated in a dict that is then returned.

    param stores: A mapping of store keys to stores.
    param default: The value to return if the key is not in any of the stores.
    param get_existing_values_only: If True, only return values for stores that contain
        the key.

    Let's define the following sub-stores:

    >>> bytes_store = dict(
    ...     a=b'a',
    ...     b=b'b',
    ...     c=b'c',
    ... )
    >>> metadata_store = dict(
    ...     b=dict(x=2),
    ...     c=dict(x=3),
    ...     d=dict(x=4),
    ... )

    We can create a fan-out reader from these stores:

    >>> stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
    >>> reader = FanoutReader(stores)
    >>> reader['b']
    {'bytes_store': b'b', 'metadata_store': {'x': 2}}

    The reader returns a dict with the values from each store, keyed by the name of the
    store.

    We can also pass a default value to return if the key is not in the store:

    >>> reader = FanoutReader(
    ...     stores=stores,
    ...     default='no value in this store for this key',
    ... )
    >>> reader['a']
    {'bytes_store': b'a', 'metadata_store': 'no value in this store for this key'}

    If the key is not in any of the stores, a KeyError is raised:

    >>> reader['z']
    Traceback (most recent call last):
        ...
    KeyError: 'z'

    We can also pass `get_existing_values_only=True` to only return values for stores
    that contain the key:

    >>> reader = FanoutReader(
    ...     stores=stores,
    ...     get_existing_values_only=True,
    ... )
    >>> reader['a']
    {'bytes_store': b'a'}
    """

    def __init__(
        self,
        stores: Mapping[Any, Mapping],
        default: Any = None,
        *,
        get_existing_values_only: bool = False,
    ):
        if not isinstance(stores, Mapping):
            if isinstance(stores, Iterable):
                stores = dict(enumerate(stores))
            else:
                raise ValueError(
                    f"stores must be a Mapping or an Iterable, not {type(stores)}"
                )
        self._stores = stores
        self._default = default
        self._get_existing_values_only = get_existing_values_only

    @classmethod
    def from_variadics(cls, *args, **kwargs):
        """A way to create a fan-out store from a mix of args and kwargs, instead of a
        single dict.

        param args: sub-stores used to fan-out the data. These stores will be
            represented by their index in the tuple.
        param kwargs: sub-stores used to fan-out the data. These stores will be
            represented by their name in the dict. __init__ arguments can also be passed
            as kwargs (i.e. `default`, `get_existing_values_only`, and any other subclass
            specific arguments).

        Let's use the same sub-stores:

        >>> bytes_store = dict(
        ...     a=b'a',
        ...     b=b'b',
        ...     c=b'c',
        ... )
        >>> metadata_store = dict(
        ...     b=dict(x=2),
        ...     c=dict(x=3),
        ...     d=dict(x=4),
        ... )

        We can create a fan-out reader from these stores, using args:

        >>> reader = FanoutReader.from_variadics(bytes_store, metadata_store)
        >>> reader['b']
        {0: b'b', 1: {'x': 2}}

        The reader returns a dict with the values from each store, keyed by the index of
        the store in the `args` tuple.

        We can also create a fan-out reader passing the stores in kwargs:

        >>> reader = FanoutReader.from_variadics(
        ...     bytes_store=bytes_store,
        ...     metadata_store=metadata_store
        ... )
        >>> reader['b']
        {'bytes_store': b'b', 'metadata_store': {'x': 2}}

        This way, the returned value is keyed by the name of the store.

        We can also mix args and kwargs:

        >>> reader = FanoutReader.from_variadics(bytes_store, metadata_store=metadata_store)
        >>> reader['b']
        {0: b'b', 'metadata_store': {'x': 2}}

        Note that the order of the stores is determined by the order of the args and
        kwargs.
        """

        def extract_init_kwargs():
            for p in cls_sig.parameters:
                if p in kwargs:
                    yield p, kwargs.pop(p)

        cls_sig = Sig(cls)
        cls_kwargs = dict(extract_init_kwargs())
        stores = dict({i: store for i, store in enumerate(args)}, **kwargs)
        return cls(stores=stores, **cls_kwargs)

    @property
    def _keys(self):
        return ChainMap(*self._stores.values())

    def __getitem__(self, k):
        value_for_key_for_every_store = {
            store_key: store.get(k, self._default)
            for store_key, store in self._stores.items()
        }
        if all(v is self._default for v in value_for_key_for_every_store.values()):
            raise KeyError(k)
        if self._get_existing_values_only:
            value_for_key_for_every_store = {
                k: v
                for k, v in value_for_key_for_every_store.items()
                if v != self._default
            }
        return value_for_key_for_every_store

    def __iter__(self) -> Iterator:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, k) -> int:
        return k in self._keys


class FanoutPersister(FanoutReader, KvPersister):
    """
    A fanout persister is a fanout reader that can also set and delete items.

    param stores: A mapping of store keys to stores.
    param default: The value to return if the key is not in any of the stores.
    param get_existing_values_only: If True, only return values for stores that contain
        the key.
    param need_to_set_all_stores: If True, all stores must be set when setting a value.
        If False, only the stores that are set will be updated.
    param ignore_non_existing_store_keys: If True, ignore store keys from the value that
        are not in the persister. If False, a ValueError is raised.

    Let's create a persister from in-memory stores:

    >>> bytes_store = dict()
    >>> metadata_store = dict()
    >>> persister = FanoutPersister(
    ...     stores = dict(bytes_store=bytes_store, metadata_store=metadata_store)
    ... )

    The persister sets the values in each store, based on the store key in the value dict.

    >>> persister['a'] = dict(bytes_store=b'a', metadata_store=dict(x=1))
    >>> persister['a']
    {'bytes_store': b'a', 'metadata_store': {'x': 1}}

    By default, not all stores must be set when setting a value:

    >>> persister['b'] = dict(bytes_store=b'b')
    >>> persister['b']
    {'bytes_store': b'b', 'metadata_store': None}

    This allow to update a subset of the stores whithout having to set all the stores.

    >>> persister['a'] = dict(bytes_store=b'A')
    >>> persister['a']
    {'bytes_store': b'A', 'metadata_store': {'x': 1}}

    This behavior can be changed by passing `need_to_set_all_stores=True`:

    >>> persister_all_stores = FanoutPersister(
    ...     stores=dict(bytes_store=dict(), metadata_store=dict()),
    ...     need_to_set_all_stores=True,
    ... )
    >>> persister_all_stores['a'] = dict(bytes_store=b'a')
    Traceback (most recent call last):
        ...
    ValueError: All stores must be set when setting a value. Missing stores: {'metadata_store'}

    By default, if a store key from the value is not in the persister, a ValueError is
    raised:

    >>> persister['a'] = dict(
    ...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
    ... )
    Traceback (most recent call last):
        ...
    ValueError: The value contains some invalid store keys: {'other_store'}

    This behavior can be changed by passing `ignore_non_existing_store_keys=True`:

    >>> persister_ignore_non_existing_store_keys = FanoutPersister(
    ...     stores=dict(bytes_store=dict(), metadata_store=dict()),
    ...     ignore_non_existing_store_keys=True,
    ... )
    >>> persister_ignore_non_existing_store_keys['a'] = dict(
    ...     bytes_store=b'a', metadata_store=dict(y=1), other_store='some value'
    ... )
    >>> persister_ignore_non_existing_store_keys['a']
    {'bytes_store': b'a', 'metadata_store': {'y': 1}}

    Note that the value of the non-existing store key is ignored! So, be careful when
    using this option, to avoid losing data.

    Let's delete items now:

    >>> del persister['a']
    >>> 'a' in persister
    False

    The key as been deleted from all the stores:

    >>> 'a' in bytes_store
    False
    >>> 'a' in metadata_store
    False

    As expected, if the key is not in any of the stores, a KeyError is raised:

    >>> del persister['z']
    Traceback (most recent call last):
        ...
    KeyError: 'z'

    However, if the key is in some of the stores, but not in others, the key is deleted
    from the stores where it is present:

    >>> bytes_store=dict(a=b'a')
    >>> persister = FanoutPersister(
    ...     stores=dict(bytes_store=bytes_store, metadata_store=dict()),
    ... )
    >>> del persister['a']
    >>> 'a' in persister
    False
    >>> 'a' in bytes_store
    False
    """

    def __init__(
        self,
        stores: Mapping[Any, Mapping],
        default: Any = None,
        *,
        get_existing_values_only: bool = False,
        need_to_set_all_stores: bool = False,
        ignore_non_existing_store_keys: bool = False,
        **kwargs,
    ):
        super().__init__(
            stores=stores,
            default=default,
            get_existing_values_only=get_existing_values_only,
        )
        self._need_to_set_all_stores = need_to_set_all_stores
        self._ignore_non_existing_store_keys = ignore_non_existing_store_keys

    def __setitem__(self, k, v: Mapping):
        if self._need_to_set_all_stores and not set(self._stores).issubset(set(v)):
            missing_stores = set(self._stores) - set(v)
            raise ValueError(
                f"All stores must be set when setting a value. Missing stores: {missing_stores}"
            )
        if not self._ignore_non_existing_store_keys and not set(v).issubset(
            set(self._stores)
        ):
            invalid_store_keys = set(v) - set(self._stores)
            raise ValueError(
                f"The value contains some invalid store keys: {invalid_store_keys}"
            )
        for store_key, vv in v.items():
            if store_key in self._stores:
                self._stores[store_key][k] = vv

    def __delitem__(self, k):
        stores_to_delete_from = {
            store_key: store for store_key, store in self._stores.items() if k in store
        }
        if not stores_to_delete_from:
            raise KeyError(k)
        for store in stores_to_delete_from.values():
            del store[k]


NotFound = type("NotFound", (object,), {})()


@wrap_kvs(value_encoder=lambda self, v: {k: v for k in self._stores.keys()})
class CascadedStores(FanoutPersister):
    """
    A MutableMapping interface to a collection of stores that will write a value in
    all the stores it contains, read it from the first store it finds that has it, and
    write it back to all the stores up to the store where it found it.

    This is useful, for example, when you want to, say, write something to disk,
    and possibly to a remote backup or shared store, but also keep that value in memory.

    The name `CascadedStores` comes from "Cascaded Caches", which is a common pattern in
    caching systems
    (e.g. https://philipwalton.com/articles/cascading-cache-invalidation/)

    To demo this, let's create a couple of stores that print when they get a value:


    >>> from collections import UserDict
    >>> class LoggedDict(UserDict):
    ...     def __init__(self, name: str):
    ...        self.name = name
    ...        super().__init__()
    ...     def __getitem__(self, k):
    ...         print(f"Getting {k} from {self.name}")
    ...         return super().__getitem__(k)
    >>> cache = LoggedDict('cache')
    >>> disk = LoggedDict('disk')
    >>> remote = LoggedDict('remote')

    Now we can create a CascadedStores instance with these stores and write a
    value to it:

    >>> stores = CascadedStores([cache, disk, remote])
    >>> stores['f'] = 42

    See that it's in both stores:

    >>> cache['f']
    Getting f from cache
    42
    >>> disk['f']
    Getting f from disk
    42
    >>> remote['f']
    Getting f from remote
    42

    See how it reads from the first store only, because it found the `f` key there:

    >>> stores['f']
    Getting f from cache
    42

    Let's write something in disk only:

    >>> disk['g'] = 43

    Now if you ask for `g`, it won't find it in cache, but will find it in `disk`
    and return it.

    >>> stores['g']
    Getting g from disk
    43

    Here's the thing though. Now, `g` is also in `cache`:

    >>> cache
    {'f': 42, 'g': 43}

    But `remote` still only has `f`:

    >>> remote
    {'f': 42}


    """

    # Note: Need to overwrite FanoutPersister's getitem to not read values from all stores
    def __getitem__(self, k):
        """Returns the value of the first store for that key"""
        for store_ref, store in self._stores.items():
            if k in store:  # Check existence first, without triggering __getitem__
                v = store[k]  # Now get the value (will trigger logging)
                # value found, now let's write it to all the stores up to the store_ref
                for _store_ref, _store in self._stores.items():
                    if _store_ref != store_ref:
                        _store[k] = v
                    else:
                        break
                # now return the value
                return v
        raise KeyError(k)


class SequenceKvReader(KvReader):
    """
    A KvReader that sources itself in an iterable of elements from which keys and values
    will be extracted and grouped by key.

    >>> docs = [{'_id': 0, 's': 'a', 'n': 1},
    ...  {'_id': 1, 's': 'b', 'n': 2},
    ...  {'_id': 2, 's': 'b', 'n': 3}]
    >>>

    Out of the box, SequenceKvReader gives you enumerated integer indices as keys,
    and the sequence items as is, as vals

    >>> s = SequenceKvReader(docs)
    >>> list(s)
    [0, 1, 2]
    >>> s[1]
    {'_id': 1, 's': 'b', 'n': 2}
    >>> assert s.get('not_a_key') is None

    You can make it more interesting by specifying a val function to compute the vals
    from the sequence elements

    >>> s = SequenceKvReader(docs, val=lambda x: (x['_id'] + x['n']) * x['s'])
    >>> assert list(s) == [0, 1, 2]  # as before
    >>> list(s.values())
    ['a', 'bbb', 'bbbbb']

    But where it becomes more useful is when you specify a key as well.
    SequenceKvReader will then compute the keys with that function, group them,
    and return as the value, the list of sequence elements that match that key.

    >>> s = SequenceKvReader(docs,
    ...         key=lambda x: x['s'],
    ...         val=lambda x: {k: x[k] for k in x.keys() - {'s'}})
    >>> assert list(s) == ['a', 'b']
    >>> assert s['a'] == [{'_id': 0, 'n': 1}]
    >>> assert s['b'] == [{'_id': 1, 'n': 2}, {'_id': 2, 'n': 3}]

    The cannonical form of key and val is a function, but if you specify a str, int,
    or iterable thereof,
    SequenceKvReader will make an itemgetter function from it, for your convenience.

    >>> s = SequenceKvReader(docs, key='_id')
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == [{'_id': 1, 's': 'b', 'n': 2}]

    The ``val_postproc`` argument is ``list`` by default, but what if we don't specify
    any?
    Well then you'll get an unconsumed iterable of matches

    >>> s = SequenceKvReader(docs, key='_id', val_postproc=None)
    >>> assert isinstance(s[1], Iterable)

    The ``val_postproc`` argument specifies what to apply to this iterable of matches.
    For example, you can specify ``val_postproc=next`` to simply get the first matched
    element:


    >>> s = SequenceKvReader(docs, key='_id', val_postproc=next)
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == {'_id': 1, 's': 'b', 'n': 2}

    We got the whole dict there. What if we just want we didn't want the _id, which is
    used by the key, in our val?

    >>> from functools import partial
    >>> all_but_s = partial(exclusive_subdict, exclude=['s'])
    >>> s = SequenceKvReader(docs, key='_id', val=all_but_s, val_postproc=next)
    >>> assert list(s) == [0, 1, 2]
    >>> assert s[1] == {'_id': 1, 'n': 2}

    Suppose we want to have the pair of ('_id', 'n') values as a key, and only 's'
    as a value...

    >>> s = SequenceKvReader(docs, key=('_id', 'n'), val='s', val_postproc=next)
    >>> assert list(s) == [(0, 1), (1, 2), (2, 3)]
    >>> assert s[1, 2] == 'b'

    But remember that using ``val_postproc=next`` will only give you the first match
    as a val.

    >>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=next)
    >>> assert list(s) == ['a', 'b']
    >>> assert s['a'] == {'_id': 0, 'n': 1}
    >>> assert s['b'] == {'_id': 1, 'n': 2}   # note that only the first match is returned.

    If you do want to only grab the first match, but want to additionally assert
    that there is no more than one,
    you can specify this with ``val_postproc=unique_element``:

    >>> s = SequenceKvReader(docs, key='s', val=all_but_s, val_postproc=unique_element)
    >>> assert s['a'] == {'_id': 0, 'n': 1}
    >>> # The following should raise an exception since there's more than one match
    >>> s['b']  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    sources.NotUnique: iterator had more than one element

    """

    def __init__(
        self,
        sequence: Iterable,
        key: KvSpec = None,
        val: KvSpec = None,
        val_postproc=list,
    ):
        """Make a SequenceKvReader instance,

        :param sequence: The iterable to source the keys and values from.
        :param key: Specification of how to extract a key from an iterable element.
            If None, will use integer keys from key, val = enumerate(iterable).
            key can be a callable, a str or int, or an iterable of strs and ints.
        :param val: Specification of how to extract a value from an iterable element.
            If None, will use the element as is, as the value.
            val can be a callable, a str or int, or an iterable of strs and ints.
        :param val_postproc: Function to apply to the iterable of vals.
            Default is ``list``, which will have the effect of values being lists of all
            vals matching a key.
            Another popular choice is ``next`` which will have the effect of values
            being the first matched to the key
        """
        self.sequence = sequence
        if key is not None:
            self.key = _kv_spec_to_func(key)
        else:
            self.key = None
        self.val = _kv_spec_to_func(val)
        self.val_postproc = val_postproc or identity_func
        assert isinstance(self.val_postproc, Callable)

    def kv_items(self):
        if self.key is not None:
            for k, v in itertools_groupby(self.sequence, key=self.key):
                yield k, self.val_postproc(map(self.val, v))
        else:
            for i, v in enumerate(self.sequence):
                yield i, self.val(v)

    def __getitem__(self, k):
        for kk, vv in self.kv_items():
            if kk == k:
                return vv
        raise KeyError(f"Key not found: {k}")

    def __iter__(self):
        yield from map(itemgetter(0), self.kv_items())


@cached_keys
class CachedKeysSequenceKvReader(SequenceKvReader):
    """SequenceKvReader but with keys cached. Use this one if you will perform multiple
    accesses to only some of the keys of the store"""


@mk_cached_store
class CachedSequenceKvReader(SequenceKvReader):
    """SequenceKvReader but with the whole mapping cached as a dict. Use this one if
    you will perform multiple accesses to the store"""


# TODO: Basically same could be acheived with
#  wrap_kvs(obj_of_data=methodcaller('__call__'))
class FuncReader(KvReader):
    """Reader that seeds itself from a data fetching function list
    Uses the function list names as the keys, and their returned value as the values.

    For example: You have a list of urls that contain the data you want to have access
    to.
    You can write functions that bare the names you want to give to each dataset,
    and have the function fetch the data from the url, extract the data from the
    response and possibly prepare it (we advise minimally, since you can always
    transform from the raw source, but the opposite can be impossible).

    >>> def foo():
    ...     return 'bar'
    >>> def pi():
    ...     return 3.14159
    >>> s = FuncReader([foo, pi])
    >>> list(s)
    ['foo', 'pi']
    >>> s['foo']
    'bar'
    >>> s['pi']
    3.14159

    You might want to give your own names to the functions.
    You might even have to (because the callable you're using doesn't have a `__name__`).
    In that case, you can specify a ``{name: func, ...}`` dict instead of a simple
    iterable.

    >>> s = FuncReader({'FU': foo, 'Pie': pi})
    >>> list(s)
    ['FU', 'Pie']
    >>> s['FU']
    'bar'

    """

    def __init__(self, funcs: Mapping[str, Callable] | Iterable[Callable]):
        # TODO: assert no free arguments (arguments are allowed but must all have
        #  defaults)
        if isinstance(funcs, Mapping):
            self.funcs = dict(funcs)
        else:
            self.funcs = {func.__name__: func for func in funcs}

    def __contains__(self, k):
        return k in self.funcs

    def __iter__(self):
        yield from self.funcs

    def __len__(self):
        return len(self.funcs)

    def __getitem__(self, k):
        return self.funcs[k]()  # call the func


class FuncDag(FuncReader):
    def __init__(self, funcs, **kwargs):
        super().__init__(funcs)
        self._sig = {fname: Sig(func) for fname, func in self._func.items()}
        # self._input_names = sum(self._sig)

    def __getitem__(self, k):
        return self._func_of_name[k]()  # call the func


import os

psep = os.path.sep

ddir = lambda o: [x for x in dir(o) if not x.startswith("_")]


def not_underscore_prefixed(x):
    return not x.startswith("_")


def _path_to_module_str(path, root_path):
    assert path.endswith(".py")
    path = path[:-3]
    if root_path.endswith(psep):
        root_path = root_path[:-1]
    root_path = os.path.dirname(root_path)
    len_root = len(root_path) + 1
    path_parts = path[len_root:].split(psep)
    if path_parts[-1] == "__init__.py":
        path_parts = path_parts[:-1]
    return ".".join(path_parts)


# class SourceReader(KvReader):
#     def __getitem__(self, k):
#         return getsource(k)

# class NestedObjReader(ObjReader):
#     def __init__(self, obj, src_to_key, key_filt=None, ):


class ObjLoader:
    def __init__(self, data_of_key, obj_of_data=None):
        self.data_of_key = data_of_key
        if obj_of_data is not None or not callable(obj_of_data):
            raise TypeError("serializer must be None or a callable")
        self.obj_of_data = obj_of_data

    def __call__(self, k):
        if self.obj_of_data is not None:
            return self.obj_of_data(self.data_of_key(k))
        else:
            return self.data_of_key(k)


# TODO: See explicit.py module and FuncReader above for near duplicates!
# TODO: Add an obj_of_key argument to wrap_kvs? (Or should it be data_of_key?)
# TODO: Another near-duplicate found: dol.paths.PathMappedData
# Note: Older version commmented below
class ObjReader:
    """
    A reader that uses a specified function to get the contents for a given key.

    >>> # define a contents_of_key that reads stuff from a dict
    >>> data = {'foo': 'bar', 42: "everything"}
    >>> def read_dict(k):
    ...     return data[k]
    >>> pr = ObjReader(_obj_of_key=read_dict)
    >>> pr['foo']
    'bar'
    >>> pr[42]
    'everything'
    >>>
    >>> # define contents_of_key that reads stuff from a file given it's path
    >>> def read_file(path):
    ...     with open(path) as fp:
    ...         return fp.read()
    >>> pr = ObjReader(_obj_of_key=read_file)
    >>> file_where_this_code_is = __file__

    ``file_where_this_code_is`` should be the file where this doctest is written,
    therefore should contain what I just said:

    >>> 'therefore should contain what I just said' in pr[file_where_this_code_is]
    True

    """

    def __init__(self, _obj_of_key: Callable):
        self._obj_of_key = _obj_of_key

    @classmethod
    def from_composition(cls, data_of_key, obj_of_data=None):
        return cls(
            _obj_of_key=ObjLoader(data_of_key=data_of_key, obj_of_data=obj_of_data)
        )

    def __getitem__(self, k):
        try:
            return self._obj_of_key(k)
        except Exception as e:
            raise KeyError(
                "KeyError in {} when trying to __getitem__({}): {}".format(
                    e.__class__.__name__, k, e
                )
            )


# Pattern: Recursive navigation
# Note: Moved dev to independent package called "guide"
@cached_keys(keys_cache=set, name="Attrs")
class Attrs(ObjReader):
    """A simple recursive KvReader for the attributes of a python object.
    Keys are attr names, values are Attrs(attr_val) instances.

    Note: A more significant version of Attrs, along with many tools based on it,
    was moved to pypi package: guide.


        pip install guide
    """

    def __init__(self, obj, key_filt=not_underscore_prefixed, getattrs=dir):
        super().__init__(obj)
        self._key_filt = key_filt
        self.getattrs = getattrs

    @classmethod
    def module_from_path(
        cls, path, key_filt=not_underscore_prefixed, name=None, root_path=None
    ):
        import importlib.util

        if name is None:
            if root_path is not None:
                try:
                    name = _path_to_module_str(path, root_path)
                except Exception:
                    name = "fake.module.name"
        spec = importlib.util.spec_from_file_location(name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return cls(foo, key_filt)

    def __iter__(self):
        yield from filter(self._key_filt, self.getattrs(self.src))

    def __getitem__(self, k):
        return self.__class__(getattr(self.src, k), self._key_filt, self.getattrs)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.src}, {self._key_filt})"


Ddir = Attrs  # for back-compatibility, temporarily

import re


def _extract_first_identifier(string: str) -> str:
    m = re.match(r"\w+", string)
    if m:
        return m.group(0)
    else:
        return ""


def _dflt_object_namer(obj, dflt_name: str = "name_not_found"):
    return (
        getattr(obj, "__name__", None)
        or _extract_first_identifier(getattr(obj, "__doc__"))
        or dflt_name
    )


class AttrContainer:
    """Convenience class to hold Key-Val pairs as attribute-val pairs, with all the
    magic methods of mappings.

    On the other hand, you will not get the usuall non-dunders (non magic methods) of
    ``Mappings``. This is so that you can use tab completion to access only the keys
    the container has, and not any of the non-dunder methods like ``get``, ``items``,
    etc.

    >>> da = AttrContainer(foo='bar', life=42)
    >>> da.foo
    'bar'
    >>> da['life']
    42
    >>> da.true = 'love'
    >>> len(da)  # count the number of fields
    3
    >>> da['friends'] = 'forever'  # write as dict
    >>> da.friends  # read as attribute
    'forever'
    >>> list(da)  # list fields (i.e. keys i.e. attributes)
    ['foo', 'life', 'true', 'friends']
    >>> 'life' in da  # check containement
    True

    >>> del da['friends']  # delete as dict
    >>> del da.foo # delete as attribute
    >>> list(da)
    ['life', 'true']
    >>> da._source  # the hidden Mapping (here dict) that is wrapped
    {'life': 42, 'true': 'love'}

    If you don't specify a name for some objects, ``AttrContainer`` will use the
    ``__name__`` attribute of the objects:

    >>> d = AttrContainer(map, tuple, obj='objects')
    >>> list(d)
    ['map', 'tuple', 'obj']

    You can also specify a different way of auto naming the objects:

    >>> d = AttrContainer('an', 'example', _object_namer=lambda x: f"_{len(x)}")
    >>> {k: getattr(d, k) for k in d}
    {'_2': 'an', '_7': 'example'}

    .. seealso:: Objects in ``py2store.utils.attr_dict`` module
    """

    _source = None

    def __init__(
        self,
        *objects,
        _object_namer: Callable[[Any], str] = _dflt_object_namer,
        **named_objects,
    ):
        if objects:
            auto_named_objects = {_object_namer(obj): obj for obj in objects}
            self._validate_named_objects(auto_named_objects, named_objects)
            named_objects = dict(auto_named_objects, **named_objects)

        super().__setattr__("_source", {})
        for k, v in named_objects.items():
            setattr(self, k, v)

    @staticmethod
    def _validate_named_objects(auto_named_objects, named_objects):
        if not all(map(str.isidentifier, auto_named_objects)):
            raise ValueError(
                "All names produced by _object_namer should be valid python identifiers:"
                f" {', '.join(x for x in auto_named_objects if not x.isidentifier())}"
            )
        clashing_names = auto_named_objects.keys() & named_objects.keys()
        if clashing_names:
            raise ValueError(
                "Some auto named objects clashed with named ones: "
                f"{', '.join(clashing_names)}"
            )

    def __getitem__(self, k):
        return self._source[k]

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        delattr(self, k)

    def __iter__(self):
        return iter(self._source.keys())

    def __len__(self):
        return len(self._source)

    def __setattr__(self, k, v):
        self._source[k] = v
        super().__setattr__(k, v)

    def __delattr__(self, k):
        del self._source[k]
        super().__delattr__(k)

    def __contains__(self, k):
        return k in self._source

    def __repr__(self):
        return super().__repr__()


# TODO: Make it work with a store, without having to load and store the values explicitly.
class AttrDict(AttrContainer, KvPersister):
    """Convenience class to hold Key-Val pairs with both a dict-like and struct-like
    interface.

    The dict-like interface has just the basic get/set/del/iter/len
    (all "dunders": none visible as methods). There is no get, update, etc.
    This is on purpose, so that the only visible attributes
    (those you get by tab-completion for instance) are the those you injected.

    >>> da = AttrDict(foo='bar', life=42)

    You get the "keys as attributes" that you get with ``AttrContainer``:

    >>> da.foo
    'bar'

    But additionally, you get the extra ``Mapping`` methods:

    >>> list(da.keys())
    ['foo', 'life']
    >>> list(da.values())
    ['bar', 42]
    >>> da.get('foo')
    'bar'
    >>> da.get('not_a_key', 'default')
    'default'

    You can assign through key or attribute assignment:

    >>> da['true'] = 'love'
    >>> da.friends = 'forever'
    >>> list(da.items())
    [('foo', 'bar'), ('life', 42), ('true', 'love'), ('friends', 'forever')]


    etc.

    .. seealso:: Objects in ``py2store.utils.attr_dict`` module
    """


# class ObjReader(KvReader):
#     def __init__(self, obj):
#         self.src = obj
#         copy_attrs(
#             target=self,
#             source=self.src,
#             attrs=('__name__', '__qualname__', '__module__'),
#             raise_error_if_an_attr_is_missing=False,
#         )

#     def __repr__(self):
#         return f'{self.__class__.__qualname__}({self.src})'

#     @property
#     def _source(self):
#         from warnings import warn

#         warn('Deprecated: Use .src instead of ._source', DeprecationWarning, 2)
#         return self.src
```

## tests/__init__.py

```python

```

## tests/base_test.py

```python
"""Testing base.py objects"""

from typing import KT, VT, Tuple
from collections.abc import Iterable
import pytest
from dol import (
    MappingViewMixin,
    Store,
    wrap_kvs,
    filt_iter,
    cached_keys,
)
from dol.base import BaseItemsView, BaseKeysView, BaseValuesView
from dol.trans import take_everything


class WrappedDict(MappingViewMixin, dict):
    keys_iterated = False

    # you can modify the mapping object
    class KeysView(BaseKeysView):
        def __iter__(self) -> Iterable[KT]:
            self._mapping.keys_iterated = True
            return super().__iter__()

    # You can add functionality:
    class ValuesView(BaseValuesView):
        def distinct(self) -> Iterable[VT]:
            return set(super().__iter__())

    # you can modify existing functionality:
    class ItemsView(BaseItemsView):
        """Just like BaseKeysView, but yields the [key,val] pairs as lists instead of tuples"""

        def __iter__(self) -> Iterable[tuple[KT, VT]]:
            return map(list, super().__iter__())


@pytest.mark.parametrize(
    "source_dict, key_input_mapper, key_output_mapper, value_input_mapper, value_output_mapper, postget, key_filter",
    [
        ({"a": 1, "b": 2, "c": 3}, None, None, None, None, None, None),
        (
            {"a": 3, "b": 1, "c": 3},  # source_dict
            lambda k: k.lower(),  # key_input_mapper
            lambda k: k.upper(),  # key_output_mapper
            lambda v: v // 10,  # value_input_mapper
            lambda v: v * 10,  # value_output_mapper
            lambda k, v: f"{k}{v}",  # postget
            lambda k: k in {"a", "c"},  # key_filter
        ),
    ],
)
def test_mapping_views(
    source_dict,
    key_input_mapper,
    key_output_mapper,
    value_input_mapper,
    value_output_mapper,
    postget,
    key_filter,
):
    def assert_store_functionality(
        store,
        key_output_mapper=None,
        value_output_mapper=None,
        postget=None,
        key_filter=None,
        collection=list,
    ):
        key_output_mapper = key_output_mapper or (lambda k: k)
        value_output_mapper = value_output_mapper or (lambda v: v)
        postget = postget or (lambda k, v: v)
        key_filter = key_filter or (lambda k: True)
        assert collection(store) == collection(
            [key_output_mapper(k) for k in source_dict if key_filter(k)]
        )
        assert not store.keys_iterated
        assert collection(store.keys()) == collection(
            [key_output_mapper(k) for k in source_dict.keys() if key_filter(k)]
        )
        assert store.keys_iterated
        assert collection(store.values()) == collection(
            [
                postget(key_output_mapper(k), value_output_mapper(v))
                for k, v in source_dict.items()
                if key_filter(k)
            ]
        )
        assert sorted(store.values().distinct()) == sorted(
            {
                postget(key_output_mapper(k), value_output_mapper(v))
                for k, v in source_dict.items()
                if key_filter(k)
            }
        )
        assert collection(store.items()) == collection(
            [
                [
                    key_output_mapper(k),
                    postget(key_output_mapper(k), value_output_mapper(v)),
                ]
                for k, v in source_dict.items()
                if key_filter(k)
            ]
        )

    wd = WrappedDict(**source_dict)
    assert_store_functionality(wd)

    wwd = Store.wrap(WrappedDict(**source_dict))
    assert_store_functionality(wwd)

    WWD = Store.wrap(WrappedDict)
    wwd = WWD(**source_dict)
    assert_store_functionality(wwd)

    wwd = wrap_kvs(
        WrappedDict(**source_dict),
        id_of_key=key_input_mapper,
        key_of_id=key_output_mapper,
        data_of_obj=value_input_mapper,
        obj_of_data=value_output_mapper,
        postget=postget,
    )
    assert_store_functionality(
        wwd,
        key_output_mapper=key_output_mapper,
        value_output_mapper=value_output_mapper,
        postget=postget,
    )

    wwd = filt_iter(WrappedDict(**source_dict), filt=key_filter or take_everything)
    assert_store_functionality(wwd, key_filter=key_filter)

    wwd = cached_keys(WrappedDict(**source_dict), keys_cache=set)
    assert wwd._keys_cache == set(source_dict)
    assert isinstance(wwd.values().distinct(), set)
    assert_store_functionality(wwd, collection=sorted)


def test_wrap_kvs_vs_class_and_static_methods():
    """Adding wrap_kvs breaks methods when called from class

    That is, when you call Klass.method() (where method is a normal, class, or static)

    See issue "dol.base.Store.wrap breaks unbound method calls":
    https://github.com/i2mint/dol/issues/17

    """

    @Store.wrap
    class MyFiles:
        y = 2

        def normal_method(self, x=3):
            return self.y * x

        @classmethod
        def hello(cls):
            pass

        @staticmethod
        def hi():
            pass

    errors = []

    # This works fine!
    instance = MyFiles()
    assert instance.normal_method() == 6

    # But calling the method as a class...
    try:
        MyFiles.normal_method(instance)
    except Exception as e:
        print("method normal_method is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hello()
    except Exception as e:
        print("classmethod hello is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hi()
    except Exception as e:
        print("staticmethod hi is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    if errors:
        first_error, *_ = errors
        raise first_error
```

## tests/pickability_test.py

```python
"""
Test the pickability of stores when they're wrapped.
"""

import pytest
import pickle
from functools import partial

from dol.base import Store
from dol.trans import wrap_kvs, filt_iter, cached_keys

# TODO: Make it work


def test_pickling_w_dict():
    """To show that a dict pickles and unpickles just fine!"""
    s = {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_w_simple_store():
    s = Store({"a": 1, "b": 2})
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_store_wrap():
    D = Store.wrap(dict)
    d = {"a": 1, "b": 2}
    s = D(d)
    b = pickle.dumps(s)
    ss = pickle.loads(b)
    assert dict(s) == dict(ss)


def test_pickling_with_wrap_kvs_class():
    WrappedDict = wrap_kvs(key_of_id=add_tag, id_of_key=remove_tag)(dict)
    s = WrappedDict({"a": 1, "b": 2})
    assert_dict_of_unpickled_is_the_same(s)


# @pytest.mark.xfail
def test_pickling_with_wrap_kvs_instance():
    d = {"a": 1, "b": 2}
    s = wrap_kvs(d, key_of_id=add_tag, id_of_key=remove_tag)
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_filt_iter_class():
    filt_func = partial(is_below_max_len, max_len=3)
    WrappedDict = filt_iter(dict, filt=filt_func)
    s = WrappedDict({"a": 1, "bb": 2, "ccc": 3})
    assert dict(s) == {"a": 1, "bb": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_filt_iter_instance():
    d = {"a": 1, "bb": 2, "ccc": 3}
    filt_func = partial(is_below_max_len, max_len=3)
    s = filt_iter(d, filt=filt_func)
    assert dict(s) == {"a": 1, "bb": 2}
    assert_dict_of_unpickled_is_the_same(s)


# @pytest.mark.xfail
def test_pickling_with_cached_keys_class():
    WrappedDict = cached_keys(dict, keys_cache=sorted)
    s = WrappedDict({"b": 2, "a": 1})  # Note: b comes before a here
    assert list(s) == ["a", "b"]  # but here, things are sorted
    # assert list(dict(s)) == ['a', 'b']  # TODO: This fails! Why?
    assert list(dict(s.items())) == ["a", "b"]  # ... yet this one sees the cache
    assert dict(s.items()) == {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


def test_pickling_with_cached_keys_instance():
    d = {"b": 2, "a": 1}  # Note: b comes before a here
    s = cached_keys(d, keys_cache=sorted)
    assert list(s) == ["a", "b"]  # but here, things are sorted
    # assert list(dict(s)) == ['a', 'b']  # TODO: This fails! Why?
    assert list(dict(s.items())) == ["a", "b"]  # ... yet this one sees the cache
    assert dict(s.items()) == {"a": 1, "b": 2}
    assert_dict_of_unpickled_is_the_same(s)


# ------------------------ utils -------------------------------------------------------------------

pup = lambda obj: pickle.loads(pickle.dumps(obj))


def assert_dict_of_unpickled_is_the_same(original_obj):
    pickled = pickle._dumps(original_obj)
    unpickled = pickle.loads(pickled)
    assert dict(unpickled) == dict(original_obj)


def add_tag(k):
    return k + "__tag"


def remove_tag(k):
    assert k.endswith("__tag")
    return k[: -len("__tag")]


def is_below_max_len(x, max_len=3):
    return len(x) < max_len
```

## tests/scrap.py

```python
from dol import Store
import pickle

s = Store({"a": 1, "b": 2})
t = pickle.dumps(s)
ss = pickle.loads(t)

dict(s) == dict(ss)
```

## tests/test_appendable.py

```python
"""
Tests for appendable.py
"""

from dol.appendable import Extender, read_add_write


def test_extender():
    store = {"a": "pple"}
    # test normal extend
    a_extender = Extender(store, "a")
    a_extender.extend("sauce")
    assert store == {"a": "pplesauce"}
    # test creation (when key is not in store)
    b_extender = Extender(store, "b")
    b_extender.extend("anana")
    assert store == {"a": "pplesauce", "b": "anana"}
    # you can use the += operator too
    b_extender += " split"
    assert store == {"a": "pplesauce", "b": "anana split"}

    # test append
    # Need to define an append method that makes sense.
    # Here, with strings, we can just call extend.
    b_bis_extender = Extender(
        store, "b", append_method=lambda self, obj: self.extend(obj)
    )
    b_bis_extender.append("s")
    assert store == {"a": "pplesauce", "b": "anana splits"}
    # But if our "extend" values were lists, we'd need to have a different append method,
    # one that puts the single object into a list, so that its sum with the existing list
    # is a list.
    store = {"c": [1, 2, 3]}
    c_extender = Extender(
        store, "c", append_method=lambda self, obj: self.extend([obj])
    )
    c_extender.append(4)
    assert store == {"c": [1, 2, 3, 4]}
    # And if the values were tuples, we'd have to put the single object into a tuple.
    store = {"d": (1, 2, 3)}
    d_extender = Extender(
        store, "d", append_method=lambda self, obj: self.extend((obj,))
    )
    d_extender.append(4)
    assert store == {"d": (1, 2, 3, 4)}

    # Now, the default extend method is `read_add_write`, which retrieves the existing
    # value, sums it to the new value, and writes it back to the store.
    # If the values of your store have a sum defined (i.e. an `__add__` method),
    # **and** that sum method does what you want, then you can use the default
    # `extend_store_value` function.
    # O ye numpy users, beware! The sum of numpy arrays is an elementwise sum,
    # not a concatenation (you'd have to use `np.concatenate` for that).
    try:
        import numpy as np

        store = {"e": np.array([1, 2, 3])}
        e_extender = Extender(store, "e")
        e_extender.extend(np.array([4, 5, 6]))
        assert all(store["e"] == np.array([5, 7, 9]))
        # This is what the `extend_store_value` function is for: you can pass it a function
        # that does what you want.
        store = {"f": np.array([1, 2, 3])}

        def extend_store_value_for_numpy(store, key, iterable):
            store[key] = np.concatenate([store[key], iterable])

        f_extender = Extender(
            store, "f", extend_store_value=extend_store_value_for_numpy
        )
        f_extender.extend(np.array([4, 5, 6]))
        assert all(store["f"] == np.array([1, 2, 3, 4, 5, 6]))
        # WARNING: See that the `extend_store_value`` defined here doesn't accomodate for
        # the case where the key is not in the store. It is the user's responsibility to
        # handle that aspect in the `extend_store_value` they provide.
        # For your convenience, the `read_add_write` that is used as a default has
        # (and which **does** handle the non-existing key case by simply writing the value in
        # the store) has an `add_iterables` argument that can be set to whatever
        # makes sense for your use case.
        from functools import partial

        store = {"g": np.array([1, 2, 3])}
        extend_store_value_for_numpy = partial(
            read_add_write, add_iterables=lambda x, y: np.concatenate([x, y])
        )
        g_extender = Extender(
            store, "g", extend_store_value=extend_store_value_for_numpy
        )
        g_extender.extend(np.array([4, 5, 6]))
        assert all(store["g"] == np.array([1, 2, 3, 4, 5, 6]))
    except (ImportError, ModuleNotFoundError):
        pass
```

## tests/test_caching.py

```python
"""Test caching tools"""

import pytest
from functools import partial, cached_property
from collections import UserDict
from typing import Dict, Any

# Import the refactored implementations - adjust the import path as needed
from dol.caching import (
    cache_this,
    ExplicitKey,
    ApplyToMethodName,
    InstanceProp,
    ApplyToInstance,
    add_extension,
    cache_property_method,
)


def test_cache_property_method(capsys):
    """
    The objective of this test is to test the cache_property_method function
    over some edge cases. Namely, what happens if we use try to cache a method
    that is already decorated by a property, cached_property, or cache_this?
    """

    class TestClass:
        def normal_method(self):
            print("normal_method called")
            return 1

        @property
        def property_method(self):
            print("property_method called")
            return 2

        @cached_property
        def cached_property_method(self):
            print("cached_property_method called")
            return 3

        @cache_this
        def cache_this_method(self):
            print("cache_this_method called")
            return 4

    cache_property_method(
        TestClass,
        [
            "normal_method",
            "property_method",
            "cached_property_method",
            "cache_this_method",
        ],
    )

    obj = TestClass()

    # Test normal method
    assert obj.normal_method == 1
    captured = capsys.readouterr()
    assert "normal_method called" in captured.out

    assert obj.normal_method == 1
    captured = capsys.readouterr()
    assert "normal_method called" not in captured.out  # Should not print again

    # Test property method
    assert obj.property_method == 2
    captured = capsys.readouterr()
    assert "property_method called" in captured.out

    assert obj.property_method == 2
    captured = capsys.readouterr()
    assert "property_method called" not in captured.out  # Should not print again

    # Test cached_property method
    assert obj.cached_property_method == 3
    captured = capsys.readouterr()
    assert "cached_property_method called" in captured.out

    assert obj.cached_property_method == 3
    captured = capsys.readouterr()
    assert "cached_property_method called" not in captured.out  # Should not print again

    # Test cache_this method
    assert obj.cache_this_method == 4
    captured = capsys.readouterr()
    assert "cache_this_method called" in captured.out

    assert obj.cache_this_method == 4
    captured = capsys.readouterr()
    assert "cache_this_method called" not in captured.out  # Should not print again


# Utility classes for testing
class LoggedCache(UserDict):
    """Cache that logs get/set operations"""

    def __init__(self, name="cache"):
        super().__init__()
        self.name = name
        self.get_log = []
        self.set_log = []

    def __setitem__(self, key, value):
        self.set_log.append((key, value))
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        self.get_log.append(key)
        return super().__getitem__(key)


class MockValueCodecs:
    """Mock codec for testing"""

    class default:
        @staticmethod
        def pickle(store):
            """Mock function that would normally apply pickle encoding/decoding"""
            return store


# Test class with various key strategy examples
class TestClassWithKeyStrategies:
    def __init__(self):
        self.my_cache = LoggedCache("my_cache")
        self.key_name = "dynamic_key"
        self.compute_count = 0

    # Example using explicit key
    @cache_this(cache="my_cache", key=ExplicitKey("explicit_key"))
    def explicit_key_method(self):
        self.compute_count += 1
        return f"explicit_result_{self.compute_count}"

    # Example using function applied to method name
    @cache_this(
        cache="my_cache", key=ApplyToMethodName(lambda name: f"{name}_processed")
    )
    def method_name_key(self):
        self.compute_count += 1
        return f"method_name_result_{self.compute_count}"

    # Example using instance property
    @cache_this(cache="my_cache", key=InstanceProp("key_name"))
    def instance_prop_key(self):
        self.compute_count += 1
        return f"instance_prop_result_{self.compute_count}"

    # Example using function applied to instance
    @cache_this(
        cache="my_cache",
        key=ApplyToInstance(lambda instance: f"instance_{id(instance)}"),
    )
    def instance_func_key(self):
        self.compute_count += 1
        return f"instance_func_result_{self.compute_count}"

    # Example using string (implicit ExplicitKey)
    @cache_this(cache="my_cache", key="string_key")
    def string_key_method(self):
        self.compute_count += 1
        return f"string_key_result_{self.compute_count}"

    # Example using function with method name (implicit ApplyToMethodName)
    @cache_this(cache="my_cache", key=lambda name: f"{name}_func")
    def simple_func_key(self):
        self.compute_count += 1
        return f"simple_func_result_{self.compute_count}"

    # Example using function with instance (implicit ApplyToInstance)
    @cache_this(cache="my_cache", key=lambda self: f"self_{self.key_name}")
    def implicit_instance_func(self):
        self.compute_count += 1
        return f"implicit_instance_result_{self.compute_count}"

    # Example using a different cache
    def set_external_cache(self, external_cache):
        self.external_cache = external_cache

    @cache_this(cache="external_cache", key="external_key")
    def external_cache_method(self):
        self.compute_count += 1
        return f"external_result_{self.compute_count}"


# Test class for original use case with pickle extension
class PickleCached:
    def __init__(self):
        self._backend_store = LoggedCache("backend")
        self.cache = MockValueCodecs.default.pickle(self._backend_store)
        self.compute_count = 0

    @cache_this(cache="cache", key=ApplyToMethodName(lambda x: f"{x}.pkl"))
    def foo(self):
        self.compute_count += 1
        return f"foo_result_{self.compute_count}"


# Test class for data access pattern
class Dacc:
    def __init__(self):
        self.text_store = LoggedCache("text_store")
        self.json_store = LoggedCache("json_store")
        self.schema_description_key = "schema_description.txt"
        self.pricing_html_key = "pricing.html"
        self.schema_key = "schema.json"
        self.compute_count = 0

    @cache_this(cache="text_store", key=InstanceProp("schema_description_key"))
    def schema_description(self) -> str:
        self.compute_count += 1
        return f"schema_description_{self.compute_count}"

    @cache_this(cache="text_store", key=InstanceProp("pricing_html_key"))
    def pricing_page_html(self) -> str:
        self.compute_count += 1
        return f"pricing_html_{self.compute_count}"

    @cache_this(cache="json_store", key=InstanceProp("schema_key"))
    def schema(self) -> dict[str, Any]:
        self.compute_count += 1
        return {"version": f"schema_{self.compute_count}"}


# Tests for basic key strategies
class TestKeyStrategies:
    def test_explicit_key(self):
        """Test explicit key strategy"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.explicit_key_method
        assert result1 == "explicit_result_1"
        assert "explicit_key" in obj.my_cache
        assert obj.my_cache["explicit_key"] == "explicit_result_1"

        # Second access uses cached value
        result2 = obj.explicit_key_method
        assert result2 == "explicit_result_1"  # Same result, no recomputation
        assert obj.compute_count == 1  # Only computed once

    def test_method_name_key(self):
        """Test applying function to method name"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.method_name_key
        assert result1 == "method_name_result_1"
        assert "method_name_key_processed" in obj.my_cache
        assert obj.my_cache["method_name_key_processed"] == "method_name_result_1"

        # Second access uses cached value
        result2 = obj.method_name_key
        assert result2 == "method_name_result_1"
        assert obj.compute_count == 1

    def test_instance_prop_key(self):
        """Test using instance property as key"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.instance_prop_key
        assert result1 == "instance_prop_result_1"
        assert "dynamic_key" in obj.my_cache
        assert obj.my_cache["dynamic_key"] == "instance_prop_result_1"

        # Second access uses cached value
        result2 = obj.instance_prop_key
        assert result2 == "instance_prop_result_1"
        assert obj.compute_count == 1

        # Changing the key property forces recomputation with new key
        obj.key_name = "new_dynamic_key"
        result3 = obj.instance_prop_key
        assert result3 == "instance_prop_result_2"
        assert "new_dynamic_key" in obj.my_cache
        assert obj.compute_count == 2

    def test_instance_func_key(self):
        """Test applying function to instance"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.instance_func_key
        expected_key = f"instance_{id(obj)}"
        assert result1 == "instance_func_result_1"
        assert expected_key in obj.my_cache
        assert obj.my_cache[expected_key] == "instance_func_result_1"

        # Second access uses cached value
        result2 = obj.instance_func_key
        assert result2 == "instance_func_result_1"
        assert obj.compute_count == 1

    def test_string_key(self):
        """Test string key (implicit ExplicitKey)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.string_key_method
        assert result1 == "string_key_result_1"
        assert "string_key" in obj.my_cache
        assert obj.my_cache["string_key"] == "string_key_result_1"

        # Second access uses cached value
        result2 = obj.string_key_method
        assert result2 == "string_key_result_1"
        assert obj.compute_count == 1

    def test_simple_func_key(self):
        """Test function key (implicit ApplyToMethodName)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.simple_func_key
        assert result1 == "simple_func_result_1"
        assert "simple_func_key_func" in obj.my_cache
        assert obj.my_cache["simple_func_key_func"] == "simple_func_result_1"

        # Second access uses cached value
        result2 = obj.simple_func_key
        assert result2 == "simple_func_result_1"
        assert obj.compute_count == 1

    def test_implicit_instance_func(self):
        """Test function with instance (implicit ApplyToInstance)"""
        obj = TestClassWithKeyStrategies()

        # First access computes the value
        result1 = obj.implicit_instance_func
        assert result1 == "implicit_instance_result_1"
        assert "self_dynamic_key" in obj.my_cache
        assert obj.my_cache["self_dynamic_key"] == "implicit_instance_result_1"

        # Second access uses cached value
        result2 = obj.implicit_instance_func
        assert result2 == "implicit_instance_result_1"
        assert obj.compute_count == 1

        # Changing the key property forces recomputation with new key
        obj.key_name = "new_dynamic_key"
        result3 = obj.implicit_instance_func
        assert result3 == "implicit_instance_result_2"
        assert "self_new_dynamic_key" in obj.my_cache
        assert obj.compute_count == 2

    def test_external_cache(self):
        """Test using an external cache"""
        obj = TestClassWithKeyStrategies()
        external_cache = LoggedCache("external")
        obj.set_external_cache(external_cache)

        # First access computes the value
        result1 = obj.external_cache_method
        assert result1 == "external_result_1"
        assert "external_key" in external_cache
        assert external_cache["external_key"] == "external_result_1"

        # Second access uses cached value
        result2 = obj.external_cache_method
        assert result2 == "external_result_1"
        assert obj.compute_count == 1


# Tests for original pickle extension use case
class TestPickleCached:
    def test_pickle_cache(self):
        """Test the original pickle cache use case"""
        obj = PickleCached()

        # First access computes the value
        result1 = obj.foo
        assert result1 == "foo_result_1"
        assert "foo.pkl" in obj._backend_store
        assert obj._backend_store["foo.pkl"] == "foo_result_1"

        # Second access uses cached value
        result2 = obj.foo
        assert result2 == "foo_result_1"
        assert obj.compute_count == 1


# Tests for data access pattern
class TestDacc:
    def test_dacc_schema_description(self):
        """Test schema description with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.schema_description
        assert result1 == "schema_description_1"
        assert "schema_description.txt" in dacc.text_store
        assert dacc.text_store["schema_description.txt"] == "schema_description_1"

        # Second access uses cached value
        result2 = dacc.schema_description
        assert result2 == "schema_description_1"
        assert dacc.compute_count == 1

        # Changing the key property forces recomputation with new key
        dacc.schema_description_key = "new_schema_description.txt"
        result3 = dacc.schema_description
        assert result3 == "schema_description_2"
        assert "new_schema_description.txt" in dacc.text_store
        assert dacc.compute_count == 2

    def test_dacc_pricing_html(self):
        """Test pricing HTML with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.pricing_page_html
        assert result1 == "pricing_html_1"
        assert "pricing.html" in dacc.text_store
        assert dacc.text_store["pricing.html"] == "pricing_html_1"

        # Second access uses cached value
        result2 = dacc.pricing_page_html
        assert result2 == "pricing_html_1"
        assert dacc.compute_count == 1

    def test_dacc_schema(self):
        """Test schema with instance property key"""
        dacc = Dacc()

        # First access computes the value
        result1 = dacc.schema
        assert result1 == {"version": "schema_1"}
        assert "schema.json" in dacc.json_store
        assert dacc.json_store["schema.json"] == {"version": "schema_1"}

        # Second access uses cached value
        result2 = dacc.schema
        assert result2 == {"version": "schema_1"}
        assert dacc.compute_count == 1


# Tests for error cases
class TestErrorCases:
    def test_none_key_not_allowed(self):
        """Test that None keys are not allowed by default"""

        class TestNoneKey:
            def __init__(self):
                self.my_cache = {}

            @cache_this(cache="my_cache", key=lambda self: None)
            def none_key_method(self):
                return 42

        obj = TestNoneKey()
        with pytest.raises(TypeError, match="cannot be None"):
            obj.none_key_method

    def test_missing_cache_attribute(self):
        """Test error when cache attribute is missing"""

        class TestMissingCache:
            @cache_this(cache="nonexistent_cache", key="test_key")
            def test_method(self):
                return 42

        obj = TestMissingCache()
        with pytest.raises(TypeError, match="No attribute named 'nonexistent_cache'"):
            obj.test_method

    def test_invalid_cache_attribute(self):
        """Test error when cache attribute is not a MutableMapping"""

        class TestInvalidCache:
            def __init__(self):
                self.invalid_cache = "not a mapping"

            @cache_this(cache="invalid_cache", key="test_key")
            def test_method(self):
                return 42

        obj = TestInvalidCache()
        with pytest.raises(TypeError, match="is not a MutableMapping"):
            obj.test_method


# Tests for new CachedMethod functionality
class TestCachedMethodFunctionality:
    """Test the new method caching capabilities added to cache_this"""

    def test_basic_method_caching(self):
        """Test basic method caching with arguments"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache")
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute the result
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert "x=2;y=3" in obj.cache
        assert obj.cache["x=2;y=3"] == 6

        # Second call with same args should use cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1  # Should not increase

        # Call with different args should compute new result
        result3 = obj.process(4, 5)
        assert result3 == 20
        assert obj.call_count == 2
        assert "x=4;y=5" in obj.cache
        assert obj.cache["x=4;y=5"] == 20

    def test_method_caching_with_kwargs(self):
        """Test method caching with keyword arguments"""

        class Calculator:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache")
            def compute(self, x, y=10, z=None):
                self.call_count += 1
                return x + y + (z or 0)

        obj = Calculator()

        # Test with mixed positional and keyword args
        result1 = obj.compute(1, y=2, z=3)
        assert result1 == 6
        assert obj.call_count == 1
        assert "x=1;y=2;z=3" in obj.cache

        # Same call should use cache
        result2 = obj.compute(1, y=2, z=3)
        assert result2 == 6
        assert obj.call_count == 1

        # Different kwargs should compute new result
        result3 = obj.compute(1, y=5, z=3)
        assert result3 == 9
        assert obj.call_count == 2
        assert "x=1;y=5;z=3" in obj.cache

    def test_custom_key_function_for_methods(self):
        """Test method caching with custom key function"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache", key=lambda self, x, y: f"method__{x},{y}.pkl")
            def multiply(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute
        result1 = obj.multiply(3, 4)
        assert result1 == 12
        assert obj.call_count == 1
        assert "method__3,4.pkl" in obj.cache

        # Second call should use cache
        result2 = obj.multiply(3, 4)
        assert result2 == 12
        assert obj.call_count == 1

    def test_auto_detection_property_vs_method(self):
        """Test that cache_this correctly auto-detects properties vs methods"""

        class TestAutoDetect:
            def __init__(self):
                self.cache = {}
                self.property_calls = 0
                self.method_calls = 0

            @cache_this(cache="cache")
            def no_args_property(self):
                """This should be detected as a property"""
                self.property_calls += 1
                return "property_value"

            @cache_this(cache="cache")
            def with_args_method(self, x, y=5):
                """This should be detected as a method"""
                self.method_calls += 1
                return x + y

            @cache_this(cache="cache")
            def with_varargs(self, *args):
                """This should be detected as a method"""
                self.method_calls += 1
                return sum(args)

            @cache_this(cache="cache")
            def with_kwargs(self, **kwargs):
                """This should be detected as a method"""
                self.method_calls += 1
                return len(kwargs)

        obj = TestAutoDetect()

        # Test property detection
        result1 = obj.no_args_property
        assert result1 == "property_value"
        assert obj.property_calls == 1
        assert "no_args_property" in obj.cache

        # Second access should use cache
        result1_again = obj.no_args_property
        assert result1_again == "property_value"
        assert obj.property_calls == 1

        # Test method detection with regular args
        result2 = obj.with_args_method(10, y=15)
        assert result2 == 25
        assert obj.method_calls == 1
        assert "x=10;y=15" in obj.cache

        # Test method detection with varargs
        method_func = obj.with_varargs
        result3 = method_func(1, 2, 3, 4)
        assert result3 == 10
        assert obj.method_calls == 2

        # Test method detection with kwargs
        method_func2 = obj.with_kwargs
        result4 = method_func2(a=1, b=2, c=3)
        assert result4 == 3
        assert obj.method_calls == 3

    def test_as_property_override(self):
        """Test the as_property parameter to override auto-detection"""

        class TestOverride:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache", as_property=False)
            def property_as_method(self):
                # This has no args but we're forcing it to be treated as a method
                self.call_count += 1
                return "method_result"

        obj = TestOverride()

        # The property forced as method should return a callable
        method_func = obj.property_as_method
        result = method_func()
        assert result == "method_result"
        assert obj.call_count == 1

        # Calling again should use cache
        method_func2 = obj.property_as_method
        result2 = method_func2()
        assert result2 == "method_result"
        assert obj.call_count == 1  # Should not increase due to caching

    def test_method_caching_with_external_cache(self):
        """Test method caching with external cache dictionary"""

        external_cache = {}

        class DataProcessor:
            def __init__(self):
                self.call_count = 0

            @cache_this(cache=external_cache)
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute and cache externally
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert "x=2;y=3" in external_cache
        assert external_cache["x=2;y=3"] == 6

        # Second call should use external cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1

    def test_method_caching_with_pre_cache(self):
        """Test method caching with pre-cache functionality"""

        pre_cache_dict = {}

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache", pre_cache=pre_cache_dict)
            def process(self, x, y):
                self.call_count += 1
                return x * y

        obj = DataProcessor()

        # First call should compute and store in both caches
        result1 = obj.process(2, 3)
        assert result1 == 6
        assert obj.call_count == 1
        assert "x=2;y=3" in obj.cache

        # Second call should use pre-cache
        result2 = obj.process(2, 3)
        assert result2 == 6
        assert obj.call_count == 1

    def test_backward_compatibility_properties(self):
        """Test that existing property caching behavior is unchanged"""

        class TestClass:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache")
            def expensive_property(self):
                self.call_count += 1
                return 42

        obj = TestClass()

        # First access should compute the value
        result1 = obj.expensive_property
        assert result1 == 42
        assert obj.call_count == 1
        assert "expensive_property" in obj.cache

        # Second access should use cached value
        result2 = obj.expensive_property
        assert result2 == 42
        assert obj.call_count == 1  # Should not increase

    def test_method_with_complex_arguments(self):
        """Test method caching with complex argument types"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache")
            def process_data(self, data_list, multiplier=1, options=None):
                self.call_count += 1
                total = sum(data_list) * multiplier
                if options and options.get("double"):
                    total *= 2
                return total

        obj = DataProcessor()

        # Test with list and dict arguments
        result1 = obj.process_data([1, 2, 3], multiplier=2, options={"double": True})
        assert result1 == 24  # (1+2+3) * 2 * 2 = 24
        assert obj.call_count == 1

        # Same call should use cache
        result2 = obj.process_data([1, 2, 3], multiplier=2, options={"double": True})
        assert result2 == 24
        assert obj.call_count == 1

    def test_method_caching_thread_safety_simulation(self):
        """Test that method caching maintains thread safety patterns"""

        class ThreadSafeProcessor:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            @cache_this(cache="cache")
            def compute(self, x):
                # Simulate computation that might be called from multiple threads
                self.call_count += 1
                return x**2

        obj = ThreadSafeProcessor()

        # Multiple calls with same arguments should only compute once
        results = []
        for _ in range(5):
            results.append(obj.compute(10))

        assert all(r == 100 for r in results)
        assert obj.call_count == 1  # Should only compute once despite multiple calls

    def test_cache_false_behavior_for_methods(self):
        """Test that cache=False works for methods by creating a property"""

        class TestClass:
            def __init__(self):
                self.call_count = 0

            @cache_this(cache=False)
            def no_cache_property(self):
                # Note: when cache=False, methods without args become properties
                self.call_count += 1
                return 42

        obj = TestClass()

        # Each access should compute the result (no caching)
        result1 = obj.no_cache_property
        assert result1 == 42
        assert obj.call_count == 1

        result2 = obj.no_cache_property
        assert result2 == 42
        assert obj.call_count == 2  # Should increase each time

    def test_error_cases_for_methods(self):
        """Test error handling for method caching"""

        class TestErrorCases:
            def __init__(self):
                self.invalid_cache = "not a mapping"

            @cache_this(cache="invalid_cache")
            def method_with_invalid_cache(self, x):
                return x

            @cache_this(cache="nonexistent_cache")
            def method_with_missing_cache(self, x):
                return x

        obj = TestErrorCases()

        # Test invalid cache type
        with pytest.raises(TypeError, match="is not a MutableMapping"):
            obj.method_with_invalid_cache(5)

        # Test missing cache attribute
        with pytest.raises(TypeError, match="No attribute named 'nonexistent_cache'"):
            obj.method_with_missing_cache(5)

    def test_method_caching_integration_with_key_strategies(self):
        """Test that method caching works well with existing key strategies when forced as property"""

        class IntegrationTest:
            def __init__(self):
                self.cache = {}
                self.call_count = 0

            # Force a method with args to be treated as property (edge case)
            @cache_this(
                cache="cache", key=ExplicitKey("forced_property"), as_property=True
            )
            def forced_property_method(self):
                # This method will be treated as property despite having the potential for args
                self.call_count += 1
                return "forced_property_result"

        obj = IntegrationTest()

        # Should work as a property
        result1 = obj.forced_property_method
        assert result1 == "forced_property_result"
        assert obj.call_count == 1
        assert "forced_property" in obj.cache

        # Second access should use cache
        result2 = obj.forced_property_method
        assert result2 == "forced_property_result"
        assert obj.call_count == 1

    def test_comprehensive_requirements_example(self):
        """Test the exact example from the original requirements"""

        class DataProcessor:
            def __init__(self):
                self.cache = {}

            # This should work as before (property)
            @cache_this(cache="cache")
            def expensive_property(self):
                return "compute_once_result"

            # This should now work (method with args)
            @cache_this(cache="cache")
            def process(self, x, y):
                return x * y

            # With custom key function
            @cache_this(cache="cache", key=lambda self, x, y: f"method__{x},{y}.pkl")
            def multiply(self, x, y):
                return x * y

        obj = DataProcessor()

        # Test property caching
        result1 = obj.expensive_property
        assert result1 == "compute_once_result"

        # Test method caching
        result2 = obj.process(3, 4)
        assert result2 == 12

        # Test different args
        result3 = obj.process(5, 6)
        assert result3 == 30

        # Test custom key function
        result4 = obj.multiply(2, 3)
        assert result4 == 6

        # Verify cache contents and keys
        expected_keys = ["expensive_property", "x=3;y=4", "x=5;y=6", "method__2,3.pkl"]
        for key in expected_keys:
            assert key in obj.cache


class TestStackingCacheDecorators:
    """Test stacking multiple cache_this decorators for cascaded caching"""

    def test_basic_stacking(self):
        """Test basic stacking of two cache_this decorators"""
        trace = []
        cache = dict()
        disk = dict()

        class A:
            @cache_this(cache=cache)
            @cache_this(cache=disk)
            def f(self):
                trace.append("In f method")
                return 42

        a = A()

        # First access: should compute the value
        result = a.f
        assert result == 42
        assert trace == ["In f method"], "Method should be called once"
        assert "f" in cache, "Outer cache should have the value"
        assert "f" in disk, "Inner cache should also have the value"

        # Second access: should use the outer cache
        result2 = a.f
        assert result2 == 42
        assert trace == ["In f method"], "Method should not be called again"

    def test_pre_existing_value_in_inner_cache(self):
        """Test that pre-existing values in inner cache are propagated to outer cache"""
        trace = []
        cache = dict()
        disk = dict(g=99)  # Pre-existing value in disk

        class B:
            @cache_this(cache=cache)
            @cache_this(cache=disk)
            def g(self):
                trace.append("In g method")
                return 1  # Would return 1 if computed

        b = B()

        # First access: should get value from disk, not compute
        result = b.g
        assert result == 99, "Should get value from inner cache (disk)"
        assert trace == [], "Method should not be called (got value from disk)"
        assert "g" in cache, "Outer cache should be populated with value from disk"
        assert cache["g"] == 99, "Outer cache should have the value from disk"

        # Second access: should use outer cache
        result2 = b.g
        assert result2 == 99
        assert trace == [], "Method still should not be called"

    def test_triple_stacking(self):
        """Test stacking three levels of caching"""
        trace = []
        memory = dict()
        disk = dict()
        remote = dict()

        class C:
            @cache_this(cache=memory)
            @cache_this(cache=disk)
            @cache_this(cache=remote)
            def compute(self):
                trace.append("Computing")
                return "result"

        c = C()

        # First access: should compute and populate all caches
        result = c.compute
        assert result == "result"
        assert trace == ["Computing"]
        assert "compute" in memory
        assert "compute" in disk
        assert "compute" in remote

        # Second access: should use memory (outermost cache)
        trace.clear()
        result2 = c.compute
        assert result2 == "result"
        assert trace == [], "Should use memory cache"

    def test_stacking_with_instance_specific_caches(self):
        """Test stacking with instance-specific cache attributes"""
        trace = []
        shared_disk = dict()

        class D:
            def __init__(self, name):
                self.name = name
                self.cache = dict()

            @cache_this(cache="cache")
            @cache_this(cache=shared_disk)
            def process(self):
                trace.append(f"Processing {self.name}")
                return f"{self.name}_result"

        d1 = D("d1")
        d2 = D("d2")

        # First instance
        result1 = d1.process
        assert result1 == "d1_result"
        assert trace == ["Processing d1"]
        assert "process" in d1.cache
        assert "process" in shared_disk

        # Second instance: gets value from shared disk cache
        # This is correct cascading behavior - the shared cache has it
        result2 = d2.process
        assert result2 == "d1_result", "Gets value from shared disk cache"
        assert len(trace) == 1, "Method not called again (got from shared disk)"
        assert "process" in d2.cache, "Value propagated to d2's instance cache"

    def test_stacking_with_methods(self):
        """Test that stacking works with methods that take arguments"""
        cache = dict()
        disk = dict()

        class E:
            @cache_this(cache=cache)
            @cache_this(cache=disk)
            def multiply(self, x, y):
                return x * y

        e = E()

        # First call with arguments
        result1 = e.multiply(3, 4)
        assert result1 == 12

        # Cache should have the result with argument-based key
        assert len(cache) > 0
        assert len(disk) > 0

        # Second call with same arguments should use cache
        result2 = e.multiply(3, 4)
        assert result2 == 12

        # Different arguments should compute again
        result3 = e.multiply(5, 6)
        assert result3 == 30

    def test_issue_example(self):
        """Test the exact example from the issue description"""
        trace = []
        cache = dict()
        disk = dict(g=8)  # Pre-existing value

        class A:
            @cache_this(cache=cache)
            @cache_this(cache=disk)
            def f(self):
                trace.append("In f method")
                return 42

            @cache_this(cache=cache)
            @cache_this(cache=disk)
            def g(self):
                trace.append("In g method")
                return 99  # Would return 99 if computed

        a = A()

        # Test f (not in any cache)
        assert a.f == 42
        assert trace == ["In f method"]
        assert cache["f"] == 42
        assert disk["f"] == 42

        # Access f again
        assert a.f == 42
        assert trace == ["In f method"], "f method should not be called again"

        # Test g (pre-existing in disk)
        assert a.g == 8
        assert trace == ["In f method"], "g method should NEVER be called"
        assert cache["g"] == 8, "Value from disk should be in cache"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
```

## tests/test_dol_tools.py

```python
"""Test the tools module."""

# ------------------------------------------------------------------------------
# cache_method
import pytest
from unittest import mock
import tempfile
import os
from pathlib import Path

from dol import tools


class TestConfirmOverwrite:
    def test_confirm_overwrite_no_existing_key(self):
        # When key doesn't exist, should return value unchanged
        mapping = {}
        result = tools.confirm_overwrite(mapping, "key", "value")
        assert result == "value"

    def test_confirm_overwrite_same_value(self):
        # When key exists with same value, should return value unchanged
        mapping = {"key": "value"}
        result = tools.confirm_overwrite(mapping, "key", "value")
        assert result == "value"

    @mock.patch("builtins.input", return_value="new_value")
    def test_confirm_overwrite_confirmed(self, mock_input):
        # When user confirms overwrite, should return new value
        mapping = {"key": "old_value"}
        result = tools.confirm_overwrite(mapping, "key", "new_value")
        assert result == "new_value"
        mock_input.assert_called_once()

    @mock.patch("builtins.input", return_value="wrong_input")
    @mock.patch("builtins.print")
    def test_confirm_overwrite_rejected(self, mock_print, mock_input):
        # When user doesn't confirm, should return existing value
        mapping = {"key": "old_value"}
        result = tools.confirm_overwrite(mapping, "key", "new_value")
        assert result == "old_value"
        mock_input.assert_called_once()
        mock_print.assert_called_once()


class TestStoreAggregate:
    def test_store_aggregate_with_dict(self):
        content_store = {
            "file1.py": '"""Module docstring."""',
            "file2.py": "def foo(): pass",
        }
        result = tools.store_aggregate(
            content_store=content_store, kv_to_item=lambda k, v: f"{k}: {v}"
        )
        assert "file1.py: " in result
        assert "file2.py: " in result

    def test_store_aggregate_with_filters(self):
        content_store = {
            "file1.py": '"""Module docstring."""',
            "file2.txt": "Plain text",
            "file3.py": "def foo(): pass",
        }
        result = tools.store_aggregate(
            content_store=content_store,
            key_filter=lambda k: k.endswith(".py"),
            aggregator=", ".join,
        )
        assert "file1.py" in result
        assert "file3.py" in result
        assert "file2.txt" not in result

    def test_store_aggregate_with_custom_aggregator(self):
        content_store = {
            "file1": "content1",
            "file2": "content2",
        }
        result = tools.store_aggregate(
            content_store=content_store, aggregator=lambda items: len(list(items))
        )
        assert result == 2

    def test_store_aggregate_with_file_output(self):
        content_store = {
            "file1": "content1",
            "file2": "content2",
        }
        with tempfile.TemporaryDirectory() as tmpdir:

            class TestConvertToNumericalIfPossible:
                def test_convert_integer_string(self):
                    result = tools.convert_to_numerical_if_possible("123")
                    assert result == 123
                    assert isinstance(result, int)

                def test_convert_float_string(self):
                    result = tools.convert_to_numerical_if_possible("123.45")
                    assert result == 123.45
                    assert isinstance(result, float)

                def test_non_numerical_string(self):
                    result = tools.convert_to_numerical_if_possible("hello")
                    assert result == "hello"
                    assert isinstance(result, str)

                def test_empty_string(self):
                    result = tools.convert_to_numerical_if_possible("")
                    assert result == ""
                    assert isinstance(result, str)

                def test_infinity_string(self):
                    result = tools.convert_to_numerical_if_possible("infinity")
                    assert result == float("inf")
                    assert isinstance(result, float)

            class TestAskUserForValueWhenMissing:
                @mock.patch("builtins.input", return_value="user_value")
                def test_ask_user_when_missing_with_input(self, mock_input):
                    from dol.base import Store

                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(Store(store))

                    # Access the key which triggers __missing__
                    value = wrapped_store["missing_key"]

                    # After __missing__ is called, the value should be stored
                    assert store == {"missing_key": "user_value"}
                    assert value == "user_value"
                    mock_input.assert_called_once()

                @mock.patch("builtins.input", return_value="123")
                def test_ask_user_with_preprocessor(self, mock_input):
                    from dol.base import Store

                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(
                        Store(store),
                        value_preprocessor=tools.convert_to_numerical_if_possible,
                    )

                    # Access the key which triggers __missing__
                    value = wrapped_store["missing_key"]

                    # After __missing__ is called, the value should be stored
                    assert store == {"missing_key": 123}
                    assert value == 123
                    mock_input.assert_called_once()

                @mock.patch("builtins.input", return_value="")
                def test_ask_user_with_empty_input(self, mock_input):
                    from dol.base import Store

                    # Set up a mock for __missing__ to verify it gets called
                    store = {}
                    store_obj = Store(store)
                    original_missing = store_obj.__missing__

                    store_obj.__missing__ = mock.MagicMock(side_effect=original_missing)
                    wrapped_store = tools.ask_user_for_value_when_missing(store_obj)

                    # This should raise KeyError since we're returning empty string
                    with pytest.raises(KeyError):
                        wrapped_store["missing_key"]

                    # Verify the original __missing__ was called
                    store_obj.__missing__.assert_called_once()

                @mock.patch("builtins.input", return_value="user_value")
                def test_custom_message(self, mock_input):
                    from dol.base import Store

                    custom_msg = "Custom message for {k}:"
                    store = {}
                    wrapped_store = tools.ask_user_for_value_when_missing(
                        Store(store), on_missing_msg=custom_msg
                    )

                    wrapped_store["missing_key"]

                    # Check that input was called with the custom message
                    mock_input.assert_called_once_with(
                        custom_msg + " Value for missing_key:\n"
                    )

            class TestISliceStoreAdvanced:
                def test_islice_with_step(self):
                    original = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
                    sliceable = tools.iSliceStore(original)

                    # With step=2, should get every other value
                    assert list(sliceable[0:5:2]) == [1, 3, 5]

                def test_islice_with_negative_indices(self):
                    original = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
                    sliceable = tools.iSliceStore(original)

                    # Test with negative indices
                    assert list(sliceable[-3:-1]) == [3, 4]
                    assert list(sliceable[-3:]) == [3, 4, 5]

                def test_islice_single_item_access(self):
                    original = {"a": 1, "b": 2, "c": 3}
                    sliceable = tools.iSliceStore(original)

                    # To get a single item by position
                    item = next(sliceable[1:2])
                    assert item == 2

                def test_islice_out_of_bounds(self):
                    original = {"a": 1, "b": 2, "c": 3}
                    sliceable = tools.iSliceStore(original)

                    # Slicing beyond the end should just stop at the end
                    assert list(sliceable[2:10]) == [3]
                    # Empty slice when start is beyond length
                    assert list(sliceable[10:20]) == []

            class TestForestAdvanced:
                def test_forest_with_custom_filters(self):
                    d = {
                        "apple": {"kind": "fruit", "color": "red", "count": 5},
                        "banana": {"kind": "fruit", "color": "yellow", "count": 3},
                    }

                    # Only include keys that don't start with 'c'
                    forest = tools.Forest(
                        d,
                        is_leaf=lambda k, v: not isinstance(v, dict),
                        get_node_keys=lambda v: [
                            k for k in v.keys() if not k.startswith("c")
                        ],
                        get_src_item=lambda src, k: src[k],
                    )

                    assert list(forest["apple"]) == ["kind", "color"]
                    assert "count" not in list(forest["apple"])

                def test_forest_with_list_source(self):
                    # Test with a list as the source
                    lst = [
                        {"name": "item1", "value": 10},
                        {"name": "item2", "value": 20},
                        {"name": "item3", "value": 30},
                    ]

                    forest = tools.Forest(
                        lst,
                        is_leaf=lambda k, v: not isinstance(v, dict),
                        get_node_keys=lambda v: list(v.keys()),
                        get_src_item=lambda src, k: src[k],
                        forest_type=list,
                    )

                    # Since it's a list, we access by index
                    assert forest[0]["name"] == "item1"
                    assert forest[1]["value"] == 20
                    assert list(forest) == [0, 1, 2]

                def test_forest_with_leaf_transform(self):
                    d = {"a": "1", "b": "2", "c": "3"}

                    # Apply a transformation to leaf values
                    forest = tools.Forest(
                        d,
                        is_leaf=lambda k, v: True,  # all values are leaves
                        get_node_keys=lambda v: list(v.keys()),
                        get_src_item=lambda src, k: src[k],
                        leaf_trans=int,  # Convert string values to integers
                    )

                    assert forest["a"] == 1
                    assert forest["b"] == 2
                    assert isinstance(forest["c"], int)

    def test_islice_store_slicing(self):
        original = {"foo": "bar", "hello": "world", "alice": "bob"}
        sliceable = tools.iSliceStore(original)

        assert list(sliceable[0:2]) == ["bar", "world"]
        assert list(sliceable[-2:]) == ["world", "bob"]
        assert list(sliceable[:-1]) == ["bar", "world"]


class TestForest:
    def test_forest_with_dict(self):
        d = {
            "apple": {
                "kind": "fruit",
                "types": {"granny": {"color": "green"}, "fuji": {"color": "red"}},
            },
            "banana": {"kind": "fruit"},
        }

        forest = tools.Forest(
            d,
            is_leaf=lambda k, v: not isinstance(v, dict),
            get_node_keys=lambda v: list(v.keys()),
            get_src_item=lambda src, k: src[k],
        )

        assert list(forest) == ["apple", "banana"]
        assert forest["apple"]["kind"] == "fruit"
        assert list(forest["apple"]["types"]) == ["granny", "fuji"]
        assert forest["apple"]["types"]["granny"]["color"] == "green"

    def test_forest_to_dict(self):
        d = {
            "apple": {
                "kind": "fruit",
                "types": {
                    "granny": {"color": "green"},
                },
            }
        }

        forest = tools.Forest(
            d,
            is_leaf=lambda k, v: not isinstance(v, dict),
            get_node_keys=lambda v: list(v.keys()),
            get_src_item=lambda src, k: src[k],
        )

        dict_result = forest.to_dict()
        assert dict_result == d
```

## tests/test_edge_cases.py

```python
from dol.base import Store
import pytest


@pytest.mark.skip(reason="edge case that we will try to address later")
def test_simple_store_wrap_unbound_method_delegation():
    # What does Store.wrap do? It wraps classes or instances in such a way that
    # mapping methods (like __iter__, __getitem__, __setitem__, __delitem__, etc.)
    # are intercepted and transformed, but other methods are not (they stay as they
    # were).

    # This test is about the "stay as they were" part, so let's start with a simple
    # class that has a method that we want to keep untouched.
    class K:
        def pass_through(self):
            return "hi"

    # wrapping an instance
    instance_of_k = K()
    assert instance_of_k.pass_through() == "hi"
    wrapped_instance_of_k = Store.wrap(instance_of_k)
    assert wrapped_instance_of_k.pass_through() == "hi"

    # wrapping a class
    WrappedK = Store.wrap(K)

    instance_of_wrapped_k = WrappedK()
    assert instance_of_wrapped_k.pass_through() == "hi"

    # Everything seems fine, but the problem creeps up when we try to use these methods
    # through an "unbound call".
    # This is when you call the method from a class, feeding an instance.
    # With the original class, this works:
    assert K.pass_through(K()) == "hi"

    # But this gives us an error on the wrapped class
    assert WrappedK.pass_through(WrappedK()) == "hi"  # error
    # or even this:
    assert WrappedK.pass_through(K()) == "hi"  # error


@pytest.mark.skip(reason="edge case that we will try to address later")
def test_store_wrap_unbound_method_delegation():
    """Making sure `dol.base.Store.wrap` doesn't break unbound method calls.

    That is, when you call Klass.method() (where method is a normal, class, or static)

    https://github.com/i2mint/dol/issues/17
    """

    @Store.wrap
    class MyFiles:
        y = 2

        def normal_method(self, x=3):
            return self.y * x

        @classmethod
        def hello(cls):
            print("hello")

        @staticmethod
        def hi():
            print("hi")

    errors = []

    # This works fine!
    instance = MyFiles()
    assert instance.normal_method() == 6

    # But calling the method as a class...
    try:
        MyFiles.normal_method(instance)
    except Exception as e:
        print("method normal_method is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hello()
    except Exception as e:
        print("classmethod hello is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    try:
        MyFiles.hi()
    except Exception as e:
        print("staticmethod hi is broken by wrap_kvs decorator")
        print(f"{type(e).__name__}: {e}")
        errors.append(e)

    if errors:
        first_error, *_ = errors
        raise first_error
```

## tests/test_fanout_stores.py

```python
import pytest
from dol.sources import FanoutPersister
from dol.trans import wrap_kvs


@pytest.mark.parametrize(
    "value_encoder,value_decoder,persister_args,persister_kwargs,data",
    [
        (
            # Tuple fan-out store (separate values)
            lambda v: {k: v for k, v in enumerate(v)},
            lambda v: tuple(vv for vv in v.values()),
            (dict(), dict(), dict()),
            dict(get_existing_values_only=True),
            [
                ("k1", (1, 2, 3), None),
                ("k2", (1, 2), None),
                ("k3", (1, 2, 3, 4), ValueError),
            ],
        ),
        (
            # Salary fan-out store (computed values)
            lambda v: dict(net=v * 0.8, tax=v * 0.2),  # Gross to breakdown
            lambda v: sum(v.values()),  # Breakdown to gross
            (),
            dict(net=dict(), tax=dict()),
            [
                ("Peter", 3000, None),
                ("Paul", 5000, None),
                ("Jack", 10000, None),
            ],
        ),
    ],
)
def test_mk_custom_fanout_store(
    value_encoder, value_decoder, persister_args, persister_kwargs, data
):
    store = wrap_kvs(
        FanoutPersister.from_variadics(
            *persister_args,
            **persister_kwargs,
        ),
        data_of_obj=value_encoder,
        obj_of_data=value_decoder,
    )
    for k, v, error in data:

        def test_data():
            store[k] = v
            assert store[k] == v
            del store[k]
            assert k not in store

        if error is None:
            test_data()
        else:
            with pytest.raises(error):
                test_data()
```

## tests/test_filesys.py

```python
"""Test filesys objects."""

import os
from functools import partial
import tempfile
from pathlib import Path
from collections.abc import Mapping
import pytest

from dol.tests.utils_for_tests import mk_test_store_from_keys, mk_tmp_local_store
from dol.filesys import mk_dirs_if_missing, TextFiles, process_path


# --------------------------------------------------------------------------------------
# Utils


def all_folder_paths_under_folder(rootpath: str, include_rootpath=False):
    """Return all folder paths under folderpath."""
    from pathlib import Path

    rootpath = Path(rootpath)
    folderpaths = (str(p) for p in rootpath.glob("**/") if p.is_dir())
    if not include_rootpath:
        folderpaths = filter(lambda x: x != str(rootpath), folderpaths)
    return folderpaths


def delete_all_folders_under_folder(rootpath: str, include_rootpath=False):
    """Delete all folders under folderpath."""
    import shutil

    rootpath = Path(rootpath)
    if Path(rootpath).is_dir():
        for p in all_folder_paths_under_folder(
            rootpath, include_rootpath=include_rootpath
        ):
            p = Path(p)
            if p.is_dir():
                shutil.rmtree(p)


def empty_directory(s, path_must_include=("test_mk_dirs_if_missing",)):
    if isinstance(path_must_include, str):
        path_must_include = (path_must_include,)

    if not all(substr in s for substr in path_must_include):
        raise ValueError(
            f"Path '{s}' does not include any of the substrings: {path_must_include}.\n"
            "This is a safeguard. For your safety, I will delete nothing!"
        )

    import os, shutil

    try:
        for item in os.scandir(s):
            if item.is_dir():
                shutil.rmtree(item.path)
            else:
                os.remove(item.path)
    except FileNotFoundError:
        pass


# TODO: Should have a more general version of this that works with any MutableMapping
#  store as target store (instead of dirpath).
#  That's easy -- but then we need to also be able to make a filesys target store
#  that it works with (need to make folders on write, etc.)
def populate_folder(dirpath, contents: Mapping):
    """Populate a folder with the given (Mapping) contents."""
    for key, content in contents.items():
        path = os.path.join(dirpath, key)
        if isinstance(content, Mapping):
            os.makedirs(path, exist_ok=True)
            populate_folder(path, content)
        else:
            if isinstance(content, str):
                data_type = "s"
            elif isinstance(content, bytes):
                data_type = "b"
            else:
                raise ValueError(f"Unsupported type: {type(content)}")
            with open(path, "w" + data_type) as f:
                f.write(content)


# --------------------------------------------------------------------------------------
# Tests


def test_process_path():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "foo/bar")

        output_path = process_path(temp_path)
        assert output_path == temp_path
        assert not os.path.exists(output_path)

        output_path = process_path(temp_path, expanduser=False)
        assert output_path == temp_path
        assert not os.path.exists(output_path)

        with pytest.raises(AssertionError):
            output_path = process_path(temp_path, assert_exists=True)

        output_path = process_path(temp_path, ensure_dir_exists=True)
        assert output_path == temp_path
        assert os.path.exists(output_path)

        output_path = process_path(temp_path, assert_exists=True)
        assert output_path == temp_path
        assert os.path.exists(output_path)

        # If path doesn't end with a (system file separator) slash, add one:
        output_path = process_path(temp_path, ensure_endswith_slash=True)
        assert output_path == temp_path + os.path.sep

        # If path ends with a (system file separator) slash, remove it.
        output_path = process_path(
            temp_path + os.path.sep, ensure_does_not_end_with_slash=True
        )
        assert output_path == temp_path


def test_json_files():
    from dol import JsonFiles, Jsons
    from pathlib import Path
    import os

    t = mk_tmp_local_store("test_mk_dirs_if_missing", make_dirs_if_missing=False)
    empty_directory(t.rootdir, path_must_include="test_mk_dirs_if_missing")
    rootdir = t.rootdir

    s = JsonFiles(rootdir)
    s["foo"] = {"bar": 1}
    assert s["foo"] == {"bar": 1}
    foo_path = Path(os.path.join(rootdir, "foo"))
    assert foo_path.is_file(), "Should have created a file"
    assert (
        foo_path.read_text() == '{\n    "bar": 1\n}'
    ), "Should be json encoded, with indent, so on multiple lines"

    ss = Jsons(rootdir)
    assert "foo" not in ss, "foo should be filtered out because no .json extension"
    ss["apple"] = {"crumble": True}
    assert "apple" in ss
    assert "apple" in set(ss)  # which is different than 'apple' in ss
    assert ss["apple"] == {"crumble": True}
    apple_path = Path(os.path.join(rootdir, "apple.json"))
    assert apple_path.is_file(), "Should have created a file (with .json extension)"
    assert (
        apple_path.read_text() == '{\n    "crumble": true\n}'
    ), "Should be json encoded, with indent, so on multiple lines"


def test_mk_dirs_if_missing():
    s = mk_tmp_local_store("test_mk_dirs_if_missing", make_dirs_if_missing=False)
    empty_directory(s.rootdir, path_must_include="test_mk_dirs_if_missing")
    with pytest.raises(KeyError):
        s["this/path/does/not/exist"] = "hello"
    ss = mk_dirs_if_missing(s)
    ss["this/path/does/not/exist"] = "hello"  # this should work now
    assert ss["this/path/does/not/exist"] == "hello"

    # # It works on classes too:
    # TextFilesWithAutoMkdir = mk_tmp_local_store(TextFiles)
    # sss = TextFilesWithAutoMkdir(s.rootdir)
    # assert sss["another/path/that/does/not/exist"] == "hello"


def test_subfolder_stores():
    import os
    import tempfile
    from dol import Files
    from dol.tests.utils_for_tests import mk_tmp_local_store

    # from dol.kv_codecs import KeyCodecs
    # from pathlib import Path

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the folder structure and contents
        data = {
            "folder1": {
                "subfolder": {
                    "apple.p": b"pie",
                },
                "day.doc": b"time",
            },
            "folder2": {
                "this.txt": b"that",
                "over.json": b"there",
            },
        }

        # Create the directory structure in the temporary directory
        populate_folder(temp_dir, data)

        # Now import the function to be tested
        from dol.filesys import subfolder_stores

        # Invoke subfolder_stores with the temporary directory
        stores = subfolder_stores(
            root_folder=temp_dir,
            max_levels=None,
            include_hidden=False,
            relative_paths=True,
            slash_suffix=False,
            folder_to_store=Files,
        )

        # Collect the keys (subfolder paths)
        store_keys = set(stores.keys())

        # Expected subfolder paths (relative to temp_dir)
        expected_subfolders = {
            "folder1",
            os.path.join("folder1", "subfolder"),
            "folder2",
        }

        # Assert that the discovered subfolders match the expected ones
        assert (
            store_keys == expected_subfolders
        ), f"Expected {expected_subfolders}, got {store_keys}"

        # Test that the stores can access the files in their respective folders
        # Testing folder1
        folder1_store = stores["folder1"]
        assert isinstance(folder1_store, Files)
        assert set(folder1_store.keys()) == {"day.doc", "subfolder/apple.p"}
        assert folder1_store["day.doc"] == b"time"

        # Testing folder1/subfolder
        subfolder_store = stores[os.path.join("folder1", "subfolder")]
        assert isinstance(subfolder_store, Files)
        assert set(subfolder_store.keys()) == {"apple.p"}
        assert subfolder_store["apple.p"] == b"pie"

        # Testing folder2
        folder2_store = stores["folder2"]
        assert isinstance(folder2_store, Files)
        assert set(folder2_store.keys()) == {"this.txt", "over.json"}
        assert folder2_store["this.txt"] == b"that"
        assert folder2_store["over.json"] == b"there"
```

## tests/test_kv_codecs.py

```python
"""Test tools.py module."""

from dol.kv_codecs import ValueCodecs
import inspect


def test_kvcodec_user_story_01():
    # See https://github.com/i2mint/dol/discussions/44#discussioncomment-7598805

    # Say you have a source backend that has pickles of some lists-of-lists-of-strings,
    # using the .pkl extension,
    # and you want to copy this data to a target backend, but saving them as gzipped
    # csvs with the csv.gz extension.

    import pickle

    src_backend = {
        "file_1.pkl": pickle.dumps([["A", "B", "C"], ["one", "two", "three"]]),
        "file_2.pkl": pickle.dumps([["apple", "pie"], ["one", "two"], ["hot", "cold"]]),
    }
    targ_backend = dict()

    from dol import ValueCodecs, KeyCodecs, Pipe

    src_wrap = Pipe(KeyCodecs.suffixed(".pkl"), ValueCodecs.pickle())
    targ_wrap = Pipe(
        KeyCodecs.suffixed(".csv.gz"),
        ValueCodecs.csv() + ValueCodecs.str_to_bytes() + ValueCodecs.gzip(),
    )
    src = src_wrap(src_backend)
    targ = targ_wrap(targ_backend)

    targ.update(src)

    # From the point of view of src and targ, you see the same thing:

    assert list(src) == list(targ) == ["file_1", "file_2"]
    assert src["file_1"] == targ["file_1"] == [["A", "B", "C"], ["one", "two", "three"]]

    # But the backend of targ is different:

    src_backend["file_1.pkl"]
    # b'\x80\x04\x95\x19\x00\x00\x00\x00\x00\x00\x00]\x94(]\x94(K\x01K\x02K\x03e]\x94(K\x04K\x05K\x06ee.'
    targ_backend["file_1.csv.gz"]
    # b'\x1f\x8b\x08\x00*YWe\x02\xff3\xd41\xd21\xe6\xe52\xd11\xd51\xe3\xe5\x02\x00)4\x83\x83\x0e\x00\x00\x00'


def _test_codec(codec, obj, encoded=None, decoded=None):
    """Test codec by encoding and decoding obj and comparing to encoded and decoded."""
    if encoded is None:
        # diagnosis mode: Just return the encoded value
        return codec.encoder(obj)
    else:
        if decoded is None:
            decoded = obj
        assert (
            codec.encoder(obj) == encoded
        ), f"Expected {codec.encoder(obj)=} to equal {encoded=}"
        assert (
            codec.decoder(encoded) == decoded
        ), f"Expected {codec.decoder(encoded)=} to equal {decoded=}"


def _test_codec_part(codec, obj, encoded, slice_):
    """Test codec but only testing equality on part of the encoded value.
    This is useful for testing codecs that have a header or footer that is not
    deterministic. For example, gzip has a header that has a timestamp.
    Also, it's useful for when the encoded value is very long and you don't want
    to write it out in the test.
    """
    encoded_actual = codec.encoder(obj)
    assert encoded_actual[slice_] == encoded[slice_]
    assert codec.decoder(encoded_actual) == obj


def test_value_codecs():
    # ----------------- Test codec value wrapper -----------------

    json_codec = ValueCodecs.json()
    # Say you have a backend store with a mapping interface. Say a dict:
    backend = dict()
    # If you call json_codec on this
    interface = json_codec(backend)
    # you'll get json-value a transformation interface
    # That is, when you write a dict (set 'a' to be the dict {'b': 2}):
    interface["a"] = {"b": 2}
    # What's actually written in the backend is the json string:
    assert backend == {"a": '{"b": 2}'}
    # but this json is decoded to a dict when read from interface
    assert interface["a"] == {"b": 2}

    # ----------------- Test encoders and decoders -----------------

    json_codec = ValueCodecs.json()
    # You can get the (encoder, decoder) pair list this:
    encoder, decoder = json_codec
    # Or like this:
    encoder = json_codec.encoder
    decoder = json_codec.decoder
    # See them encode and decode:
    assert encoder({"b": 2}) == '{"b": 2}'
    assert decoder('{"b": 2}') == {"b": 2}

    # And now let's test the many codecs we have:
    # assert sorted(ValueCodecs()) == [
    #     'base64',
    #     'bz2',
    #     'codecs',
    #     'csv',
    #     'csv_dict',
    #     'gzip',
    #     'json',
    #     'lzma',
    #     'pickle',
    #     'tarfile',
    #     'urlsafe_b64',
    #     'xml_etree',
    #     'zipfile',
    # ]

    _test_codec(
        ValueCodecs.pickle(),
        [1, 2, 3],
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00]\x94(K\x01K\x02K\x03e.",
    )

    _test_codec(
        ValueCodecs.pickle(protocol=2), [1, 2, 3], b"\x80\x02]q\x00(K\x01K\x02K\x03e."
    )

    assert str(inspect.signature(ValueCodecs.pickle)) == (
        "(obj, data, protocol=None, fix_imports=True, buffer_callback=None, "
        "encoding='ASCII', errors='strict', buffers=())"
    )  # NOTE: May change according to python version. This is 3.8

    _test_codec(
        ValueCodecs.json(),
        {"water": "fire", "earth": "air"},
        '{"water": "fire", "earth": "air"}',
    )

    # test_codec(
    #     ValueCodecs.json(sort_keys=True),
    #     {'water': 'fire', 'earth': 'air'},
    #     '{"earth": "air", "water": "fire"}',  # <-- see how the keys are sorted here!
    #     {'earth': 'air', 'water': 'fire'}
    # )

    _test_codec(ValueCodecs.base64(), b"\xfc\xfd\xfe", b"/P3+")

    _test_codec(ValueCodecs.urlsafe_b64(), b"\xfc\xfd\xfe", b"_P3-")

    _test_codec_part(
        ValueCodecs.gzip(),
        b"hello",
        b"\x1f\x8b\x08\x00t\x85Se\x02\xff\xcbH\xcd\xc9\xc9\x07\x00\x86\xa6\x106\x05\x00\x00\x00",
        slice(10, -8),
    )

    _test_codec(
        ValueCodecs.bz2(),
        b"hello",
        b'BZh91AY&SY\x191e=\x00\x00\x00\x81\x00\x02D\xa0\x00!\x9ah3M\x073\x8b\xb9"\x9c(H\x0c\x98\xb2\x9e\x80',
    )

    _test_codec_part(ValueCodecs.tarfile(), b"hello", b"data.bin", slice(0, 8))

    _test_codec_part(
        ValueCodecs.lzma(),
        b"hello",
        b"\xfd7zXZ",
        slice(0, 4),
    )

    _test_codec_part(
        ValueCodecs.zipfile(),
        b"hello",
        b"PK\x03\x04\x14\x00\x00\x00\x08\x00",
        slice(0, 10),
    )

    _test_codec(ValueCodecs.codecs(), "hello", b"hello")
    # _test_codec(ValueCodecs.plistlib, {'a': 1, 'b': 2}, b'<?xml version="1.0" ...')

    _test_codec(
        ValueCodecs.csv(), [["a", "b", "c"], ["1", "2", "3"]], "a,b,c\r\n1,2,3\r\n"
    )

    _test_codec(
        ValueCodecs.csv(delimiter=";", lineterminator="\n"),
        [["a", "b", "c"], ["1", "2", "3"]],
        "a;b;c\n1;2;3\n",
    )

    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"]),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
        [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}],
    )

    # See that you don't get back when you started with. The ints aren't ints anymore!
    # You can resolve this by using the fieldcasts argument
    # (that's our argument -- not present in builtin csv module).
    # I should be a function (that transforms a dict to the one you want) or
    # list or tuple of the same size as the row (that specifies the cast function for each field)
    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"], fieldcasts=[int] * 2),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
    )

    _test_codec(
        ValueCodecs.csv_dict(fieldnames=["a", "b"], fieldcasts={"b": float}),
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "a,b\r\n1,2\r\n3,4\r\n",
        [{"a": "1", "b": 2.0}, {"a": "3", "b": 4.0}],
    )

    xml_encoder, xml_decoder = ValueCodecs.xml_etree()
    xml_string = '<html><body><h1 style="color:blue;">Hello, World!</h1></body></html>'
    tree = xml_encoder(xml_string)
    assert tree.tag == "html"
    assert len(tree) == 1
    assert tree[0].tag == "body"
    assert tree[0].attrib == {}
    assert len(tree[0]) == 1
    assert tree[0][0].tag == "h1"
    assert tree[0][0].attrib == {"style": "color:blue;"}

    # Let's change that attribute
    tree[0][0].attrib["style"] = "color:red;"
    # ... and write it back to a string
    new_xml_string = xml_decoder(tree, encoding="unicode")
    assert (
        new_xml_string
        == '<html><body><h1 style="color:red;">Hello, World!</h1></body></html>'
    )
```

## tests/test_paths.py

```python
"""Tests for paths.py"""

from dol.paths import KeyTemplate, path_get
import pytest


def test_path_get():
    # NOTE: The following examples test the current default behavior, but that doesn't
    # mean that this default behavior is the best behavior. I've made the choice of
    # aligning (though not completely) with the behavior of glom, which is a great
    # library for getting values from nested data structures (and I recommend to use
    # glom instead of path_get if you need more features than path_get provides).
    # On the other hand, I'm not sure glom's default choices are the best either.
    # I would vote for a more restrictive, but explicit, so predictable behavior by
    # default. That said, it makes path_get more annoying to use out of the box.

    path_get({"a": {"1": {"4": "c"}}}, "a.1.4") == "c"
    # When a path is given as a string, it is split on '.' and each element is used
    # as a key. So 'a.1.4' is equivalent to ['a', '1', '4']
    # Here, each key of the path acts as a key into a Mapping
    path_get({"a": {"1": {"4": "c"}}}, "a.1.4") == "c"
    # But see next, how '1' is actually interpreted as an integer, not a string, since
    # it's indexing into a list, and '4' is interpreted as a string, since it's
    # key-ing into a dict (a Mapping).
    path_get({"a": [7, {"4": "c"}]}, "a.1.4") == "c"
    # 4 would take on the role of an integer index if we replace the {'4': 'c'} Mapping
    # with a list.
    path_get({"a": [7, [0, 1, 2, 3, "c"]]}, "a.1.4") == "c"

    # Using an integer key in a mapping, is also not a problem:
    path_get({"a": [7, {4: "c"}]}, "a.1.4")

    # If you specify an integer key (here, 4) and your mapping (here {"4": "c"})
    # doesn't have one, on the other hand, you'll get a KeyError
    with pytest.raises(KeyError):
        path_get({"a": [7, {"4": "c"}]}, ("a", 1, 4))

    # To get out of such situations, you can either specify a different (functional)
    # `sep` argument, or specify a different `get_value` function. Here we use
    # path_get.chain_of_getters function to create a getter that tries a sequence of
    # getters, in order, until one succeeds.
    getter = path_get.chain_of_getters(
        [getattr, lambda obj, k: obj[k], lambda obj, k: obj[int(k)]]
    )
    path_get({"a": [7, {4: "c"}]}, ("a", 1, "4"), get_value=getter)


def test_string_template_template_construction():
    assert KeyTemplate("{}.ext").template == "{i01_}.ext"
    assert KeyTemplate("{name}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{::\w+}.ext").template == "{i01_}.ext"
    assert KeyTemplate(r"{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{name::\w+}.ext").template == "{name}.ext"
    assert KeyTemplate("{name:0.02f}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{name:0.02f:\w+}.ext").template == "{name}.ext"
    assert KeyTemplate(r"{:0.02f:\w+}.ext").template == "{i01_}.ext"


def test_string_template_regex():
    assert KeyTemplate("{}.ext")._regex.pattern == r"(?P<i01_>.*)\.ext"
    assert KeyTemplate("{name}.ext")._regex.pattern == r"(?P<name>.*)\.ext"
    assert KeyTemplate(r"{::\w+}.ext")._regex.pattern == r"(?P<i01_>\w+)\.ext"
    assert KeyTemplate(r"{name::\w+}.ext")._regex.pattern == r"(?P<name>\w+)\.ext"
    assert KeyTemplate(r"{:0.02f:\w+}.ext")._regex.pattern == r"(?P<i01_>\w+)\.ext"
    assert KeyTemplate(r"{name:0.02f:\w+}.ext")._regex.pattern == r"(?P<name>\w+)\.ext"


def test_string_template_simple():
    from dol.paths import KeyTemplate
    from collections import namedtuple

    st = KeyTemplate(
        r"root/{}/v_{version:03.0f:\d+}.json",
        from_str_funcs={"version": int},
    )

    assert st.str_to_dict("root/life/v_42.json") == {"i01_": "life", "version": 42}
    assert st.dict_to_str({"i01_": "life", "version": 42}) == "root/life/v_042.json"
    assert st.dict_to_tuple({"i01_": "life", "version": 42}) == ("life", 42)
    assert st.tuple_to_dict(("life", 42)) == {"i01_": "life", "version": 42}
    assert st.str_to_tuple("root/life/v_42.json") == ("life", 42)
    assert st.tuple_to_str(("life", 42)) == "root/life/v_042.json"

    assert st.str_to_simple_str("root/life/v_42.json") == "life,042"
    st_clone = st.clone(simple_str_sep="-")
    assert st_clone.str_to_simple_str("root/life/v_42.json") == "life-042"
    assert st_clone.simple_str_to_str("life-42") == "root/life/v_042.json"

    from collections import namedtuple

    VersionedFile = st.dict_to_namedtuple({"i01_": "life", "version": 42})
    assert VersionedFile == namedtuple("VersionedFile", ["i01_", "version"])("life", 42)
    assert st.namedtuple_to_dict(VersionedFile) == {"i01_": "life", "version": 42}
```

## tests/test_trans.py

```python
"""Test trans.py functionality."""

from dol.trans import filt_iter, redirect_getattr_to_getitem


def test_filt_iter():
    # Demo regex filter on a class
    contains_a = filt_iter.regex(r"a")
    # wrap the dict type with this
    filtered_dict = contains_a(dict)
    # now make a filtered_dict
    d = filtered_dict(apple=1, banana=2, cherry=3)
    # and see that keys not containing "a" are filtered out
    assert dict(d) == {"apple": 1, "banana": 2}

    # With this regex filt_iter, we made two specialized versions:
    # One filtering prefixes, and one filtering suffixes
    is_test = filt_iter.prefixes("test")  # Note, you can also pass a list of prefixes
    d = {"test.txt": 1, "report.doc": 2, "test_image.jpg": 3}
    dd = is_test(d)
    assert dict(dd) == {"test.txt": 1, "test_image.jpg": 3}

    is_text = filt_iter.suffixes([".txt", ".doc", ".pdf"])
    d = {"test.txt": 1, "report.doc": 2, "image.jpg": 3}
    dd = is_text(d)
    assert dict(dd) == {"test.txt": 1, "report.doc": 2}


def test_redirect_getattr_to_getitem():

    # Applying it to a class

    ## ... with the @decorator syntax
    @redirect_getattr_to_getitem
    class MyDict(dict):
        pass

    d1 = MyDict(a=1, b=2)
    assert d1.a == 1
    assert d1.b == 2
    assert list(d1) == ["a", "b"]

    ## ... as a decorator factory
    D = redirect_getattr_to_getitem()(dict)
    d2 = D(a=1, b=2)
    assert d2.a == 1
    assert d2.b == 2
    assert list(d2) == ["a", "b"]

    # Applying it to an instance

    ## ... as a decorator
    backend_d = dict(a=1, b=2)

    d3 = redirect_getattr_to_getitem(backend_d)
    assert d3.a == 1
    assert d3.b == 2
    assert list(d3) == ["a", "b"]

    ## ... as a decorator factory
    d4 = redirect_getattr_to_getitem()(backend_d)
    assert d4.a == 1
    assert d4.b == 2
    assert list(d4) == ["a", "b"]
```

## tests/utils_for_tests.py

```python
"""Utils for tests."""

from dol import TextFiles
import os
from functools import partial


_dflt_keys = (
    "pluto",
    "planets/mercury",
    "planets/venus",
    "planets/earth",
    "planets/mars",
    "fruit/apple",
    "fruit/banana",
    "fruit/cherry",
)


def mk_test_store_from_keys(
    keys=_dflt_keys,
    *,
    mk_store=dict,
    obj_of_key=lambda k: f"Content of {k}",
    empty_store_before_writing=False,
):
    """Make some test data for a store from a list of keys.

    None of the arguments are required, for the convenience of getting test stores
    quickly:

    >>> store = mk_test_store_from_keys()
    >>> store = mk_test_store_from_keys.for_local()  # makes files in temp local dir
    >>> store = mk_test_store_from_keys(keys=['one', 'two', 'three'])
    """
    if isinstance(mk_store, str):
        mk_store = mk_tmp_local_store(mk_store)
    store = mk_store()
    if empty_store_before_writing:
        for k in store:
            del store[k]
    for k in keys:
        store[k] = obj_of_key(k)
    return store


def mk_tmp_local_store(
    tmp_name="temp_local_store", mk_store=TextFiles, make_dirs_if_missing=True
):
    from dol import temp_dir, mk_dirs_if_missing

    store = mk_store(temp_dir(tmp_name))
    if make_dirs_if_missing:
        store = mk_dirs_if_missing(store)
    return store


mk_test_store_from_keys.for_local = partial(
    mk_test_store_from_keys, mk_store=mk_tmp_local_store
)
```

## tools.py

```python
"""
Various tools to add functionality to stores
"""

from dol.trans import store_decorator

NoSuchKey = type("NoSuchKey", (), {})

# ------------ useful trans functions to be used with wrap_kvs etc. ---------------------
# TODO: Consider typing or decorating functions to indicate their role (e.g. id_of_key,
#   key_of_id, data_of_obj, obj_of_data, preset, postget...)


_dflt_confirm_overwrite_user_input_msg = (
    "The key {k} already exists and has value {existing_v}. "
    "If you want to overwrite it with {v}, confirm by typing {v} here: "
)


# TODO: Parametrize user messages (bring to interface)
# role: preset
def confirm_overwrite(
    mapping, k, v, user_input_msg=_dflt_confirm_overwrite_user_input_msg
):
    """A preset function you can use in wrap_kvs to ask the user to confirm if
    they're writing a value in a key that already has a different value under it.

    >>> from dol.trans import wrap_kvs
    >>> d = {'a': 'apple', 'b': 'banana'}
    >>> d = wrap_kvs(d, preset=confirm_overwrite)

    Overwriting ``a`` with the same value it already has is fine (not really an
    over-write):

    >>> d['a'] = 'apple'

    Creating new values is also fine:

    >>> d['c'] = 'coconut'
    >>> assert d == {'a': 'apple', 'b': 'banana', 'c': 'coconut'}

    But if we tried to do ``d['a'] = 'alligator'``, we'll get a user input request:

    .. code-block::

        The key a already exists and has value apple.
        If you want to overwrite it with alligator, confirm by typing alligator here:

    And we'll have to type `alligator` and press RETURN to make the write go through.

    """
    if (existing_v := mapping.get(k, NoSuchKey)) is not NoSuchKey and existing_v != v:
        user_input = input(user_input_msg.format(k=k, v=v, existing_v=existing_v))
        if user_input != v:
            print(f"--> User confirmation failed: I won't overwrite {k}")
            # this will have the effect of rewriting the same value that's there already:
            return existing_v
    return v


# -------------------------------- Aggregate a store -----------------------------------

from typing import (
    Optional,
    KT,
    VT,
    Tuple,
    Union,
    TypeVar,
    Any,
)
from collections.abc import Callable, Mapping, Iterable
import os
from functools import partial
from pathlib import Path
from dol.trans import wrap_kvs
from dol.filesys import Files


def decode_as_latin1(b: bytes) -> str:
    return b.decode("latin1")


def markdown_section(k: KT, v: VT) -> str:
    return f"## {k}\n\n{v.strip()}\n\n"


def save_string_to_filepath(filepath: str, string: str):
    filepath = Path(filepath).expanduser().absolute()
    filepath.write_text(string)
    return string


def identity(x):
    return x


Latin1TextFiles = wrap_kvs(Files, value_decoder=decode_as_latin1)

Item = TypeVar("Item")
Aggregate = TypeVar("Aggregate")


def store_aggregate(
    content_store: Mapping[KT, VT] | str,  # Path to the folder or dol store
    *,
    kv_to_item: Callable[
        [KT, VT], Item
    ] = markdown_section,  # Function to convert key-value pairs to text
    aggregator: Callable[
        [Iterable[Item]], Aggregate
    ] = "\n\n".join,  # How to aggregate the item's into an aggregate
    egress: (
        Callable[[Aggregate], Any] | str
    ) = identity,  # function to apply to the aggregate before returning
    key_filter: Callable[[KT], bool] | None = None,  # Filter function for keys
    value_filter: Callable[[VT], bool] | None = None,  # Filter function for values
    kv_filter: None | (
        Callable[[tuple[KT, VT]], bool]
    ) = None,  # Filter function for key-value pairs
    local_store_factory: Callable[
        [str], Mapping[KT, VT]
    ] = Latin1TextFiles,  # Factory function for the local store
) -> Any:
    r'''
    Create an aggregate object of a store's (a Mapping of strings) content

    The function is written to be able to aggregate the keys and/or values of a store,
    no matter their type, and concatenate them into an object of arbitrary type.
    That said, the defaults are setup assuming the store's keys and values are text,
    and you want to concatenate them into a single string.
    This is useful, for example, when you have several files in a folder,
    and you want to create a single text/markdown file with all the content therein.

    This function filters content from a given content store, converts the key-value
    pairs to items (usually text), and (if you specify a filepath as the `egress`)
    saves the aggregate (text) before returning it.

    Args:
        content_store (Union[Mapping[KT, VT], str]): Path to the folder or dol store to read from.
        kv_to_item (Callable[[KT, VT], Item]):
            Function to convert key-value pairs to an Item (usually a string).
        aggregator (Callable[[Iterable[Item]], Aggregate]):
            The function that will aggregate the items that `kv_to_item` produces.
            Defaults to '\n\n'.join.
        egress (Union[Callable[[Aggregate], Any], str]):
            The function that will be called on the aggregate before returning it.
            Defaults to identity.
            Note that if you provide a string, the function will save the aggregate
            text to a file, assuming it is indeed text.
        key_filter (Optional[Callable[[KT], bool]]):
            Optional filter for keys. Defaults to None (no filtering).
        value_filter (Optional[Callable[[VT], bool]]):
            Optional filter for values. Defaults to None (no filtering).
        kv_filter (Optional[Callable[[Tuple[KT, VT]], bool]]):
            Optional filter for key-value pairs. Defaults to None (no filtering).
        local_store_factory (Callable[[str], Mapping[KT, VT]]): Factory function for the local store,
            used only if `content_store` is an existing folder path. Defaults to Latin1TextFiles.

    Returns:
        Any: Usually the aggregate object, which is usually the concatenated text.

    Normally, you'd specify your content store by specifying a root folder
    (the function will create a Mapping-view of the contents of the folder for you),
    or make a content store yourself (a Mapping object providing the key-value pairs).

    To provide a small example, we'll take a dict as our content store:

    >>> content_store = {
    ...     'file1.py': '"""Module docstring."""',
    ...     'file2.py': 'def foo(): pass',
    ...     'file3.py': '"""Another docstring."""',
    ...     'file4.md': 'Markdown content here.',
    ...     'file5.py': '"""If I mention file5.py, I will be excluded."""',
    ... }

    Define the filters:

    >>> key_filter = lambda k: k.endswith('.py')  # Only include keys that end with '.py'
    >>> value_filter = lambda v: v.startswith(
    ...     '"""'
    ... )  # Only include values that start with """ (marking a module docstring)
    >>> kv_filter = (
    ...     lambda kv: kv[0] not in kv[1]
    ... )  # Exclude key-value pairs where the value mentions the key

    Call the function with the provided filters and settings

    >>> result = store_aggregate(
    ...     content_store=content_store,  # The content_store dict
    ...     kv_to_item="{} -> {}".format,  # Format key-value pairs as "key -> value"
    ...     key_filter=key_filter,  # Key filter: Include only .py files
    ...     value_filter=value_filter,  # Value filter: Include only values starting with """
    ...     kv_filter=kv_filter,  # KV filter: Exclude if value contains the key
    ...     aggregator=', '.join,
    ...     egress='~/test.md'
    ... )
    >>> result
    'file1.py -> """Module docstring.""", file3.py -> """Another docstring."""'

    Here, you got the string as the result. If you want to save it to a file,
    you can provide the save_filepath argument, and it will save the text to the file,
    and return the save_filepath to you (which )

    Recipe: You can do a lot with the `kv_to_text` argument. For example, if your
    content store doesn't have string keys or values, you can always extract whatever
    information you need from them to produce the text that will represent that item.
    '''
    # Convert content_store to a dol store if it's a directory path
    if isinstance(content_store, str) and os.path.isdir(content_store):
        content_store = local_store_factory(content_store)

    if isinstance(egress, str):
        save_filepath = egress
        if save_filepath.startswith("~"):
            save_filepath = os.path.expanduser(save_filepath)
        # make an egress that will save the string to a file (then return the string)
        egress = partial(save_string_to_filepath, save_filepath)

    # Define default filters if not provided
    key_filter = key_filter or (lambda key: True)
    value_filter = value_filter or (lambda value: True)
    kv_filter = kv_filter or (lambda kv: True)

    def actual_kv_filter(kv):
        k, v = kv
        return key_filter(k) and value_filter(v) and kv_filter(kv)

    # Create the string by applying filters and kv_to_text conversion
    filtered_kv_pairs = filter(actual_kv_filter, content_store.items())
    aggregate = aggregator(kv_to_item(k, v) for k, v in filtered_kv_pairs)

    return egress(aggregate)


# --------------------------------------- Misc ------------------------------------------


from functools import partial
from typing import Any, Union
from collections.abc import Iterable, Callable
from itertools import islice

from dol.base import KvReader
from dol.base import Store  # For the ask_user_for_value_when_missing function

_dflt_ask_user_for_value_when_missing_msg = (
    "No such key was found. You can enter a value for it here "
    "or simply hit enter to leave the slot empty"
)


def convert_to_numerical_if_possible(s: str):
    """To be used with ``ask_user_for_value_when_missing`` ``value_preprocessor`` arg

    >>> convert_to_numerical_if_possible("123")
    123
    >>> convert_to_numerical_if_possible("123.4")
    123.4
    >>> convert_to_numerical_if_possible("one")
    'one'

    Border case: The strings "infinity" and "inf" actually convert to a valid float.

    >>> convert_to_numerical_if_possible("infinity")
    inf
    """
    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            pass
    return s


@store_decorator
def ask_user_for_value_when_missing(
    store=None,
    *,
    value_preprocessor: Callable | None = None,
    on_missing_msg: str = _dflt_ask_user_for_value_when_missing_msg,
):
    """Wrap a store so if a value is missing when the user asks for it, they will be
    given a chance to enter the value they want to write.

    :param store: The store (instance or class) to wrap
    :param value_preprocessor: Function to transform the user value before trying to
        write it (bearing in mind all user specified values are strings)
    :param on_missing_msg: String that will be displayed to prompt the user to enter a
        value
    :return:
    """

    store = Store.wrap(store)

    def __missing__(self, k):
        user_value = input(on_missing_msg + f" Value for {k}:\n")

        if user_value:
            if value_preprocessor:
                user_value = value_preprocessor(user_value)
            self[k] = user_value
        else:
            super(type(self), self).__missing__(k)

    store.__missing__ = __missing__
    return store


class iSliceStore(Mapping):
    """
    Wraps a store to make a reader that acts as if the store was a list
    (with integer keys, and that can be sliced).
    I say "list", but it should be noted that the behavior is more that of range,
    that outputs an element of the list
    when keying with an integer, but returns an iterable object (a range) if sliced.

    Here, a map object is returned when the sliceable store is sliced.

    >>> s = {'foo': 'bar', 'hello': 'world', 'alice': 'bob'}
    >>> sliceable_s = iSliceStore(s)

    The read-only functionalities of the underlying mapping are still available:

    >>> list(sliceable_s)
    ['foo', 'hello', 'alice']
    >>> 'hello' in sliceable_s
    True
    >>> sliceable_s['hello']
    'world'

    But now you can get slices as well:

    >>> list(sliceable_s[0:2])
    ['bar', 'world']
    >>> list(sliceable_s[-2:])
    ['world', 'bob']
    >>> list(sliceable_s[:-1])
    ['bar', 'world']

    Now, you can't do `sliceable_s[1]` because `1` isn't a valid key.
    But if you really wanted "item number 1", you can do:

    >>> next(sliceable_s[1:2])
    'world'

    Note that `sliceable_s[i:j]` is an iterable that needs to be consumed
    (here, with list) to actually get the data. If you want your data in a different
    format, you can use `dol.trans.wrap_kvs` for that.

    >>> from dol import wrap_kvs
    >>> ss = wrap_kvs(sliceable_s, obj_of_data=list)
    >>> ss[1:3]
    ['world', 'bob']
    >>> sss = wrap_kvs(sliceable_s, obj_of_data=sorted)
    >>> sss[1:3]
    ['bob', 'world']
    """

    def __init__(self, store):
        self.store = store

    def _get_islice(self, k: slice):
        start, stop, step = k.start, k.stop, k.step

        assert (step is None) or (step > 0), "step of slice can't be negative"
        negative_start = start is not None and start < 0
        negative_stop = stop is not None and stop < 0
        if negative_start or negative_stop:
            n = self.__len__()
            if negative_start:
                start = n + start
            if negative_stop:
                stop = n + stop

        return islice(self.store.keys(), start, stop, step)

    def __getitem__(self, k):
        if not isinstance(k, slice):
            return self.store[k]
        else:
            return map(self.store.__getitem__, self._get_islice(k))

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __contains__(self, k):
        return k in self.store


Src = Any
Key = Any
Val = Any

key_error_flag = type("KeyErrorFlag", (), {})()


def _isinstance(obj, class_or_tuple):
    """Same as builtin isinstance, but without the position only limitation
    that prevents from partializing class_or_tuple"""
    return isinstance(obj, class_or_tuple)


def type_check_if_type(filt):
    if isinstance(filt, type) or isinstance(filt, tuple):
        class_or_tuple = filt
        filt = partial(_isinstance, class_or_tuple=class_or_tuple)
    return filt


def return_input(x):
    return x


# TODO: Try dataclass
# TODO: Generalize forest_type to include mappings too
# @dataclass
class Forest(KvReader):
    """Provides a key-value forest interface to objects.

    A `<tree https://en.wikipedia.org/wiki/Tree_(data_structure)>`_
    is a nested data structure. A tree has a root, which is the parent of children,
    who themselves can be parents of further subtrees, or not; in which case they're
    called leafs.
    For more information, see
    `<wikipedia on trees https://en.wikipedia.org/wiki/Tree_(data_structure)>`_

    Here we allow one to construct a tree view of any python object, using a
    key-value interface to the parent-child relationship.

    A forest is a collection of trees.

    Arguably, a dictionnary might not be the most impactful example to show here, since
    it is naturally a tree (therefore a forest), and naturally key-valued: But it has
    the advantage of being easy to demo with.
    Where Forest would really be useful is when you (1) want to give a consistent
    key-value interface to the many various forms that trees and forest objects come
    in, or even more so when (2) your object's tree/forest structure is not obvious,
    so you need to "extract" that view from it (plus give it a consistent key-value
    interface, so that you can build an ecosystem of tools around it.

    Anyway, here's our dictionary example:

    >>> d = {
    ...     'apple': {
    ...         'kind': 'fruit',
    ...         'types': {
    ...             'granny': {'color': 'green'},
    ...             'fuji': {'color': 'red'}
    ...         },
    ...         'tasty': True
    ...     },
    ...     'acrobat': {
    ...         'kind': 'person',
    ...         'nationality': 'french',
    ...         'brave': True,
    ...     },
    ...     'ball': {
    ...         'kind': 'toy'
    ...     }
    ... }

    Must of the time, you'll want to curry ``Forest`` to make an ``object_to_forest``
    constructor for a given class of objects. In the case of dictionaries as the one
    above, this might look like this:

    >>> from functools import partial
    >>> a_forest = partial(
    ...     Forest,
    ...     is_leaf=lambda k, v: not isinstance(v, dict),
    ...     get_node_keys=lambda v: [vv for vv in iter(v) if not vv.startswith('b')],
    ...     get_src_item=lambda src, k: src[k]
    ... )
    >>>
    >>> f = a_forest(d)
    >>> list(f)
    ['apple', 'acrobat']

    Note that we specified in ``get_node_keys``that we didn't want to include items
    whose keys start with ``b`` as valid children. Therefore we don't have our
    ``'ball'`` in the list above.

    Note below which nodes are themselves ``Forests``, and whic are leafs:

    >>> ff = f['apple']
    >>> isinstance(ff, Forest)
    True
    >>> list(ff)
    ['kind', 'types', 'tasty']
    >>> ff['kind']
    'fruit'
    >>> fff = ff['types']
    >>> isinstance(fff, Forest)
    True
    >>> list(fff)
    ['granny', 'fuji']

    """

    def __init__(
        self,
        src: Src,
        *,
        get_node_keys: Callable[[Src], Iterable[Key]],
        get_src_item: Callable[[Src, Key], bool],
        is_leaf: Callable[[Key, Val], bool],
        forest_type: type | Callable = list,
        leaf_trans: Callable[[Val], Any] = return_input,
    ):
        """Initialize a ``Forest``

        :param src: The source of the ``Forest``. This could be any object you want.
            The following arguments should know how to handle it.
        :param get_node_keys: How to get the keys of the children of ``src``.
        :param get_src_item: How to get the value of a child of ``src`` from its key
        :param is_leaf: Determines if a ``(k, v)`` pair (child) is a leaf.
        :param forest_type: The type of a forest. Used both to determine if an object
            (must be iterable) is to be considered a forest (i.e. an iterable of sources
            that are roots of trees
        :param leaf_trans:
        """
        self.src = src
        self.get_node_keys = get_node_keys
        self.get_src_item = get_src_item
        self.is_leaf = is_leaf
        self.leaf_trans = leaf_trans
        if isinstance(forest_type, type):
            self.is_forest = isinstance(src, forest_type)
        else:
            self.is_forest = forest_type(src)
        self._forest_maker = partial(
            type(self),
            get_node_keys=get_node_keys,
            get_src_item=get_src_item,
            is_leaf=is_leaf,
            forest_type=forest_type,
            leaf_trans=leaf_trans,
        )

    def is_forest_type(self, obj):
        return isinstance(obj, list)

    def __iter__(self):
        if not self.is_forest:
            yield from self.get_node_keys(self.src)
        else:
            for i, _ in enumerate(self.src):
                yield i

    def __getitem__(self, k):
        if self.is_forest:
            assert isinstance(k, int), (
                f"When the src is a forest, you should key with an "
                f"integer. The key was {k}"
            )
            v = next(
                islice(self.src, k, k + 1), key_error_flag
            )  # TODO: raise KeyError if
            if v is key_error_flag:
                raise KeyError(f"No value for {k=}")
        else:
            v = self.get_src_item(self.src, k)
        if self.is_leaf(k, v):
            return self.leaf_trans(v)
        else:
            return self._forest_maker(v)

    def to_dict(self):
        def gen():
            for k, v in self.items():
                if isinstance(v, Forest):
                    yield k, v.to_dict()
                else:
                    yield k, v

        return dict(gen())

    def __repr__(self):
        return f"{type(self).__name__}({self.src})"
```

## trans.py

```python
"""Transformation/wrapping tools"""

from functools import wraps, partial, reduce
import types
import re
from inspect import signature, Parameter
from typing import Union, Optional, Any, Generic
from collections.abc import Iterable, Collection, Callable
from dataclasses import dataclass
from warnings import warn
from collections.abc import Iterable
from collections.abc import (
    KeysView as BaseKeysView,
    ValuesView as BaseValuesView,
    ItemsView as BaseItemsView,
)

from dol.errors import SetattrNotAllowed
from dol.base import Store, KvReader, AttrNames, kv_walk
from dol.util import (
    safe_compile,
    lazyprop,
    attrs_of,
    wraps,
    Pipe,
    LiteralVal,
    num_of_args,
)
from dol.signatures import Sig, KO


########################################################################################################################
# Internal Utils


def double_up_as_factory(decorator_func):
    """Repurpose a decorator both as it's original form, and as a decorator factory.
    That is, from a decorator that is defined do ``wrapped_func = decorator(func, **params)``,
    make it also be able to do ``wrapped_func = decorator(**params)(func)``.

    Note: You'll only be able to do this if all but the first argument are keyword-only,
    and the first argument (the function to decorate) has a default of ``None`` (this is for your own good).
    This is validated before making the "double up as factory" decorator.

    >>> @double_up_as_factory
    ... def decorator(func=None, *, multiplier=2):
    ...     def _func(x):
    ...         return func(x) * multiplier
    ...     return _func
    ...
    >>> def foo(x):
    ...     return x + 1
    ...
    >>> foo(2)
    3
    >>> wrapped_foo = decorator(foo, multiplier=10)
    >>> wrapped_foo(2)
    30
    >>>
    >>> multiply_by_3 = decorator(multiplier=3)
    >>> wrapped_foo = multiply_by_3(foo)
    >>> wrapped_foo(2)
    9
    >>>
    >>> @decorator(multiplier=3)
    ... def foo(x):
    ...     return x + 1
    ...
    >>> foo(2)
    9

    Note that to be able to use double_up_as_factory, your first argument (the object to be wrapped) needs to default
    to None and be the only argument that is not keyword-only (i.e. all other arguments need to be keyword only).

    >>> @double_up_as_factory
    ... def decorator_2(func, *, multiplier=2):
    ...     '''Should not be able to be transformed with double_up_as_factory'''
    Traceback (most recent call last):
      ...
    AssertionError: First argument of the decorator function needs to default to None. Was <class 'inspect._empty'>
    >>> @double_up_as_factory
    ... def decorator_3(func=None, multiplier=2):
    ...     '''Should not be able to be transformed with double_up_as_factory'''
    Traceback (most recent call last):
      ...
    AssertionError: All arguments (besides the first) need to be keyword-only

    """

    def validate_decorator_func(decorator_func):
        first_param, *other_params = signature(decorator_func).parameters.values()
        assert first_param.default is None, (
            f"First argument of the decorator function needs to default to None. "
            f"Was {first_param.default}"
        )
        assert all(
            p.kind in {p.KEYWORD_ONLY, p.VAR_KEYWORD} for p in other_params
        ), f"All arguments (besides the first) need to be keyword-only"
        return True

    validate_decorator_func(decorator_func)

    @wraps(decorator_func)
    def _double_up_as_factory(wrapped=None, **kwargs):
        if wrapped is None:  # then we want a factory
            return partial(decorator_func, **kwargs)
        else:
            return decorator_func(wrapped, **kwargs)

    return _double_up_as_factory


def _all_but_first_arg_are_keyword_only(func):
    """

    >>> def foo(a, *, b, c=2): ...
    >>> _all_but_first_arg_are_keyword_only(foo)
    True
    >>> def bar(a, b, *, c=2): ...
    >>> _all_but_first_arg_are_keyword_only(bar)
    False
    """
    kinds = (p.kind for p in signature(func).parameters.values())
    _ = next(kinds)  # consume first item, and all remaining should be KEYWORD_ONLY
    return all(kind == Parameter.KEYWORD_ONLY for kind in kinds)


# TODO: Separate the wrapper_assignments injection (and possibly make these not show up at the interface?)
# FIXME: doctest line numbers not shown correctly when wrapped by store_decorator!
def store_decorator(func):
    """Helper to make store decorators.

    You provide a class-decorating function ``func`` that takes a store type (and possibly additional params)
    and returns another decorated store type.

    ``store_decorator`` takes that ``func`` and provides an enhanced class decorator specialized for stores.
    Namely it will:
    - Add ``__module__``, ``__qualname__``, ``__name__`` and ``__doc__`` arguments to it
    - Copy the aforementioned arguments to the decorated class, or copy the attributes of the original if not specified.
    - Output a decorator that can be used in four different ways: a class/instance decorator/factory.

    By class/instance decorator/factory we mean that if ``A`` is a class, ``a`` an instance of it,
    and ``deco`` a decorator obtained with ``store_decorator(func)``,
    we can use ``deco`` to
    - class decorator: decorate a class
    - class decorator factory: make a function that decorates classes
    - instance decorator: decorate an instance of a store
    - instancce decorator factor: make a function that decorates instances of stores

    For example, say we have the following ``deco`` that we made with ``store_decorator``:

    >>> @store_decorator
    ... def deco(cls=None, *, x=1):
    ...     # do stuff to cls, or a copy of it...
    ...     cls.x = x  # like this for example
    ...     return cls

    And a class that has nothing to it:

    >>> class A: ...

    Nammely, it doesn't have an ``x``

    >>> hasattr(A, 'x')
    False

    We make a ``decorated_A`` with ``deco`` (class decorator example)

    >>> t = deco(A, x=42)
    >>> assert isinstance(t, type)

    and we see that we now have an ``x`` and it's 42

    >>> hasattr(A, 'x')
    True
    >>> A.x
    42

    But we could have also made a factory to decorate ``A`` and anything else that comes our way.

    >>> paint_it_42 = deco(x=42)
    >>> decorated_A = paint_it_42(A)
    >>> assert decorated_A.x == 42
    >>> class B:
    ...     x = 'destined to disappear'
    >>> assert paint_it_42(B).x == 42

    To be fair though, you'll probably see the factory usage appear in the following form,
    where the class is decorated at definition time.

    >>> @deco(x=42)
    ... class B:
    ...     pass
    >>> assert B.x == 42

    If your exists already, and you want to keep it as is (with the same name), you can
    use subclassing to transform a copy of ``A`` instead, as below.
    Also note in the following example, that ``deco`` was used without parentheses,
    which is equivalent to ``@deco()``,
    and yes, store_decorator makes that possible to, as long as your params have defaults

    >>> @deco
    ... class decorated_A(A):
    ...     pass
    >>> assert decorated_A.x == 1
    >>> assert A.x == 42

    Finally, you can also decorate instances:

    >>> class A: ...
    >>> a = A()
    >>> hasattr(a, 'x')
    False
    >>> b = deco(a); assert b.x == 1; # b has an x and it's 1
    >>> b = deco()(a); assert b.x == 1; # b has an x and it's 1
    >>> b = deco(a, x=42); assert b.x == 42  # b has an x and it's 42
    >>> b = deco(x=42)(a); assert b.x == 42; # b has an x and it's 42

    WARNING: Note though that the type of ``b`` is not the same type as ``a``

    >>> isinstance(b, a.__class__)
    False

    No, ``b`` is an instance of a ``dol.base.Store``, which is a class containing an
    instance of a store (here, ``a``).

    >>> type(b)
    <class 'dol.base.Store'>
    >>> b.store == a
    True

    Now, here's some more example, slightly closer to real usage

    >>> from dol.trans import store_decorator
    >>> from inspect import signature
    >>>
    >>> def rm_deletion(store=None, *, msg='Deletions not allowed.'):
    ...     name = getattr(store, '__name__', 'Something') + '_w_sommething'
    ...     assert isinstance(store, type), f"Should be a type, was {type(store)}: {store}"
    ...     wrapped_store = type(name, (store,), {})
    ...     wrapped_store.__delitem__ = lambda self, k: msg
    ...     return wrapped_store
    ...
    >>> remove_deletion = store_decorator(rm_deletion)

    See how the signature of the wrapper has some extra inputs that were injected (__module__, __qualname__, etc.):

    >>> print(str(signature(remove_deletion)))
    (store=None, *, msg='Deletions not allowed.', __module__=None, __name__=None, __qualname__=None, __doc__=None, __annotations__=None, __defaults__=None, __kwdefaults__=None)

    Using it as a class decorator factory (the most common way):

    As a class decorator "factory", without parameters (and without ()):

    >>> from collections import UserDict
    >>> @remove_deletion
    ... class WD(UserDict):
    ...     "Here's the doc"
    ...     pass
    >>> wd = WD(x=5, y=7)
    >>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'Deletions not allowed.'
    >>> assert wd.__doc__ == "Here's the doc"

    As a class decorator "factory", with parameters:

    >>> @remove_deletion(msg='No way. I do not trust you!!')
    ... class WD(UserDict): ...
    >>> wd = WD(x=5, y=7)
    >>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'

    The __doc__ is empty:

    >>> assert WD.__doc__ == None

    But we could specify a doc if we wanted to:

    >>> @remove_deletion(__doc__="Hi, I'm a doc.")
    ... class WD(UserDict):
    ...     "This is the original doc, that will be overritten"
    >>> assert WD.__doc__ == "Hi, I'm a doc."


    The class decorations above are equivalent to the two following:

    >>> WD = remove_deletion(UserDict)
    >>> wd = WD(x=5, y=7)
    >>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'Deletions not allowed.'
    >>>
    >>> WD = remove_deletion(UserDict, msg='No way. I do not trust you!!')
    >>> wd = WD(x=5, y=7)
    >>> assert wd == UserDict(x=5, y=7)  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'

    But we can also decorate instances. In this case they will be wrapped in a Store class
    before being passed on to the actual decorator.

    >>> d = UserDict(x=5, y=7)
    >>> wd = remove_deletion(d)
    >>> assert wd == d  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'Deletions not allowed.'
    >>>
    >>> d = UserDict(x=5, y=7)
    >>> wd = remove_deletion(d, msg='No way. I do not trust you!!')
    >>> assert wd == d  # same as far as dict comparison goes
    >>> assert wd.__delitem__('x') == 'No way. I do not trust you!!'

    """

    # wrapper_assignments = ('__module__', '__qualname__', '__name__', '__doc__', '__annotations__')
    wrapper_assignments = (
        "__module__",
        "__name__",
        "__qualname__",
        "__doc__",
        "__annotations__",
        "__defaults__",
        "__kwdefaults__",
    )

    @wraps(func)
    def _func_wrapping_store_in_cls_if_not_type(store, **kwargs):
        specials = dict()
        for a in wrapper_assignments:
            v = kwargs.pop(a, getattr(store, a, None))
            if v is not None:
                specials[a] = v

        if not isinstance(store, type):
            store_instance = store
            # StoreWrap = type(
            #     'StoreWrap', (Store,), {}
            # )  # a copy of Store, so Store isn't transformed directly

            WrapperStore = func(Store, **kwargs)
            r = WrapperStore(store_instance)
        else:
            assert _all_but_first_arg_are_keyword_only(func), (
                "To use decorating_store_cls, all but the first of your function's arguments need to be all keyword only. "
                f"The signature was {func.__qualname__}{signature(func)}"
            )
            r = func(store, **kwargs)

        for k, v in specials.items():
            if v is not None:
                setattr(r, k, v)

        return r

    # Two standard attributes for storing the original function are func and __wrapped__
    _func_wrapping_store_in_cls_if_not_type.func = func
    _func_wrapping_store_in_cls_if_not_type.__wrapped__ = func

    # @wraps(func)
    wrapper_sig = Sig(func).merge_with_sig(
        [dict(name=a, default=None, kind=KO) for a in wrapper_assignments],
        ch_to_all_pk=False,
    )

    # TODO: Re-use double_up_as_factory here
    @wrapper_sig
    def wrapper(store=None, **kwargs):
        if store is None:  # then we want a factory
            return partial(_func_wrapping_store_in_cls_if_not_type, **kwargs)
        else:
            wrapped_store_cls = _func_wrapping_store_in_cls_if_not_type(store, **kwargs)

            return wrapped_store_cls

    # Make sure the wrapper (yes, also the wrapper) has the same key dunders as the func
    for a in wrapper_assignments:
        v = getattr(func, a, None)
        if v is not None:
            setattr(wrapper, a, v)

    return wrapper


def ensure_set(x):
    if isinstance(x, str):
        x = [x]
    return set(x)


def get_class_name(cls, dflt_name=None):
    name = getattr(cls, "__qualname__", None)
    if name is None:
        name = getattr(getattr(cls, "__class__", object), "__qualname__", None)
        if name is None:
            if dflt_name is not None:
                return dflt_name
            else:
                raise ValueError(f"{cls} has no name I could extract")
    return name


def store_wrap(obj):
    if isinstance(obj, type):

        @wraps(type(obj), updated=())  # added this: test
        class StoreWrap(Store):
            @wraps(obj.__init__)
            def __init__(self, *args, **kwargs):
                persister = obj(*args, **kwargs)
                super().__init__(persister)

        return StoreWrap
    else:
        return Store(obj)


def _is_bound(method):
    return hasattr(method, "__self__")


def _first_param_is_an_instance_param(params):
    return len(params) > 0 and list(params)[0] in self_names


# TODO: Add validation of func: That all but perhaps 1 argument (not counting self)
#  has a default
def _has_unbound_self(func):
    """

    Args:
        func:

    Returns:

    >>> def f1(x): ...
    >>> assert _has_unbound_self(f1) == 0
    >>>
    >>> def f2(self, x): ...
    >>> assert _has_unbound_self(f2) == 1
    >>>
    >>> f3 = lambda self, x: True
    >>> assert _has_unbound_self(f3) == 1
    >>>
    >>> class A:
    ...     def bar(self, x): ...
    ...     def foo(dacc, x): ...
    >>> a = A()
    >>>
    >>> _has_unbound_self(a.bar)
    0
    >>> _has_unbound_self(a.foo)
    0
    >>> _has_unbound_self(A.bar)
    1
    >>> _has_unbound_self(A.foo)
    0
    >>>
    """
    try:
        params = signature(func).parameters
    except ValueError:
        # If there was a problem getting the signature, assume it's a signature-less builtin (so not a bound method)
        return False
    if len(params) == 0:
        # no argument, so we can't be wrapping anything!!!
        raise ValueError(
            "The function has no parameters, so I can't guess which one you want to wrap"
        )
    elif (
        not isinstance(func, type)
        and not _is_bound(func)
        and _first_param_is_an_instance_param(params)
    ):
        return True
    else:
        return False


def transparent_key_method(self, k):
    return k


def mk_kv_reader_from_kv_collection(
    kv_collection, name=None, getitem=transparent_key_method
):
    """Make a KvReader class from a Collection class.

    Args:
        kv_collection: The Collection class
        name: The name to give the KvReader class (by default, it will be kv_collection.__qualname__ + 'Reader')
        getitem: The method that will be assigned to __getitem__. Should have the (self, k) signature.
            By default, getitem will be transparent_key_method, returning the key as is.
            This default is useful when you want to delegate the actual getting to a _obj_of_data wrapper.

    Returns: A KvReader class that subclasses the input kv_collection
    """

    name = name or kv_collection.__qualname__ + "Reader"
    reader_cls = type(name, (kv_collection, KvReader), {"__getitem__": getitem})
    return reader_cls


def raise_disabled_error(functionality):
    def disabled_function(*args, **kwargs):
        raise ValueError(f"{functionality} is disabled")

    return disabled_function


def disable_delitem(o):
    if hasattr(o, "__delitem__"):
        o.__delitem__ = raise_disabled_error("deletion")
    return o


def disable_setitem(o):
    if hasattr(o, "__setitem__"):
        o.__setitem__ = raise_disabled_error("writing")
    return o


def mk_read_only(o):
    return disable_delitem(disable_setitem(o))


def is_iterable(x):
    return isinstance(x, Iterable)


def add_ipython_key_completions(store):
    """Add tab completion that shows you the keys of the store.
    Note: ipython already adds local path listing automatically,
     so you'll still get those along with your valid store keys.
    """

    def _ipython_key_completions_(self):
        return self.keys()

    if isinstance(store, type):
        store._ipython_key_completions_ = _ipython_key_completions_
    else:
        setattr(
            store,
            "_ipython_key_completions_",
            types.MethodType(_ipython_key_completions_, store),
        )
    return store


from dol.util import copy_attrs
from dol.errors import OverWritesNotAllowedError


def disallow_overwrites(store, *, error_msg=None, disable_deletes=True):
    assert isinstance(store, type), "store needs to be a type"
    if hasattr(store, "__setitem__"):

        def __setitem__(self, k, v):
            if k in self:
                raise OverWritesNotAllowedError(
                    "key {} already exists and cannot be overwritten. "
                    "If you really want to write to that key, delete it before writing".format(
                        k
                    )
                )
            return super(type(self), self).__setitem__(k, v)


class OverWritesNotAllowedMixin:
    """Mixin for only allowing a write to a key if they key doesn't already exist.
    Note: Should be before the persister in the MRO.

    >>> class TestPersister(OverWritesNotAllowedMixin, dict):
    ...     pass
    >>> p = TestPersister()
    >>> p['foo'] = 'bar'
    >>> #p['foo'] = 'bar2'  # will raise error
    >>> p['foo'] = 'this value should not be stored' # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    dol.errors.OverWritesNotAllowedError: key foo already exists and cannot be overwritten.
        If you really want to write to that key, delete it before writing
    >>> p['foo']  # foo is still bar
    'bar'
    >>> del p['foo']
    >>> p['foo'] = 'this value WILL be stored'
    >>> p['foo']
    'this value WILL be stored'
    """

    @staticmethod
    def wrap(cls):
        # TODO: Consider moving to trans and making instances wrappable too
        class NoOverWritesClass(OverWritesNotAllowedMixin, cls): ...

        copy_attrs(NoOverWritesClass, cls, ("__name__", "__qualname__", "__module__"))
        return NoOverWritesClass

    def __setitem__(self, k, v):
        if self.__contains__(k):
            raise OverWritesNotAllowedError(
                "key {} already exists and cannot be overwritten. "
                "If you really want to write to that key, delete it before writing".format(
                    k
                )
            )
        return super().__setitem__(k, v)


# TODO: Should replace wrapper=Store with wrapper=Store.wrap, to depend on a function,
#  not a class that has a ``wrap`` method. But need to first
#  investigate how wrap_kvs uses that argument to not break anything.
# TODO: It seems like at this point, we could merge store_decorator and _wrap_store
#  to dispose of the need for a separate class_trans_maker function and the body
#  of the decorators to be the smelly _wrap_store(class_trans_maker, locals(), ...)
def _wrap_store(
    class_trans_maker,
    class_trans_kwargs: dict,
    wrapper_arg_name="wrapper",
    store_arg_name="store",
):
    wrapper = class_trans_kwargs.pop(wrapper_arg_name, None) or Store
    store = class_trans_kwargs.pop(store_arg_name)
    class_trans = partial(class_trans_maker, **class_trans_kwargs)
    return wrapper.wrap(store, class_trans=class_trans)


@store_decorator
def insert_hash_method(
    store=None,
    *,
    hash_method: Callable[[Any], int] = id,
):
    """Make a store hashable using the specified ``hash_method``.
    Will add (or overwrite) a ``__hash__`` method to the store that uses the
    hash_method to hash the store and an ``__eq__`` method that compares the store to
    another based on the hash_method.

    The ``hash_method``, which will be used as the ``__hash__`` method of a class
    should return an integer value that represents the hash of the object.
    To remain sane, the hash value must be the same for an object every time the
    ``__hash__`` method is called during the lifetime of the object, and objects
    that compare equal (using the __eq__ method) must have the same hash value.

    It's also important that the hash function has the property of being deterministic
    and returning a hash value that is uniformly distributed across the range of
    possible integers for the given data. This is important for the hash table
    data structure to work efficiently.

    See `This issue <https://github.com/i2mint/dol/issues/7>`_ for further information.

    >>> d = {1: 2}  # not hashable!
    >>> dd = insert_hash_method(d)
    >>> assert isinstance(hash(dd), int)  # now hashable!

    It looks the same:

    >>> dd
    {1: 2}

    But don't be fooled: dd is not equal to the original ``d`` (since
    insert_hash_method``overwrote the ``__eq__`` method to compare based on the
    hash value):

    >>> d == dd
    False

    But if you cast both to dicts and then compare, you'll be using the key and value
    based comparison of dicts, which makes these two equal.

    >>> dict(d) == dict(dd)
    True

    The default ``hash_method`` is ``id``, so two hashable wrappers won't be equal
    to eachother:

    >>> insert_hash_method(d) == insert_hash_method(d)
    False

    In the following we show two things: That you can specify your own custom
    ``hash_method``, and that you can use ``insert_hash_method`` to wrap classes

    >>> class D(dict):
    ...     pass
    >>> DD = insert_hash_method(D, hash_method=lambda x: 42)
    >>> hash(DD(d))
    42

    You can also use it as a decorator, without arguments,

    >>> @insert_hash_method
    ... class E(dict):
    ...     pass
    >>> assert isinstance(hash(E({1: 2})), int)

    or with arguments (which you must specify as keyword arguments):

    >>> @insert_hash_method(hash_method=lambda x: sum(x.values()))
    ... class F(dict):
    ...     pass
    >>> hash(F({1: 2, 3: 4}))
    6

    """
    return _wrap_store(_insert_hash_method, locals())


def _is_hashable(store):
    return getattr(store, "__hash__", None) is not None


def _insert_hash_method(store, hash_method):
    class hashable_cls(store):
        """A hashable wrapper to stores"""

        def __hash__(self):
            return hash_method(self)

        def __eq__(self, other):
            return hash_method(self) == hash_method(other)

    return hashable_cls


########################################################################################################################
# Caching keys


# TODO: If a read-one-by-one (vs the current read all implementation) is necessary one day,
#   see https://github.com/zahlman/indexify/blob/master/src/indexify.py for ideas
#   but probably buffered (read by chunks) version of the later is better.
@store_decorator
def cached_keys(
    store=None,
    *,
    keys_cache: Callable | Collection = list,
    iter_to_container=None,  # deprecated: use keys_cache instead
    cache_update_method="update",
    name: str = None,  # TODO: might be able to be deprecated since included in store_decorator
    __module__=None,  # TODO: might be able to be deprecated since included in store_decorator
) -> Callable | KvReader:
    """Make a class that wraps input class's __iter__ becomes cached.

    Quite often we have a lot of keys, that we get from a remote data source, and don't want to have to ask for
    them again and again, having them be fetched, sent over the network, etc.
    So we need caching.

    But this caching is not the typical read caching, since it's __iter__ we want to cache, and that's a generator.
    So we'll implement a store class decorator specialized for this.

    The following decorator, when applied to a class (that has an __iter__), will perform the __iter__ code, consuming
    all items of the generator and storing them in _keys_cache, and then will yield from there every subsequent call.

    It is assumed, if you're using the cached_keys transformation, that you're dealing with static data
    (or data that can be considered static for the life of the store -- for example, when conducting analytics).
    If you ever need to refresh the cache during the life of the store, you can to delete _keys_cache like this:
    ```
    del your_store._keys_cache
    ```
    Once you do that, the next time you try to ask something about the contents of the store, it will actually do
    a live query again, as for the first time.

    Note: The default keys_cache is list though in many cases, you'd probably should use set, or an explicitly
    computer set instead. The reason list is used as the default is because (1) we didn't want to assume that
    order did not matter (maybe it does to you) and (2) we didn't want to assume that your keys were hashable.
    That said, if you're keys are hashable, and order does not matter, use set. That'll give you two things:
    (a) your `key in store` checks will be faster (O(1) instead of O(n)) and (b) you'll enforce unicity of keys.

    Know also that if you precompute the keys you want to cache with a container that has an update
    method (by default `update`) your cache updates will be faster and if the container you use has
    a `remove` method, you'll be able to delete as well.

    Args:
        store: The store instance or class to wrap (must have an __iter__), or None if you want a decorator.
        keys_cache: An explicit collection of keys
        iter_to_container: The function that will be applied to existing __iter__() and assigned to cache.
            The default is list. Another useful one is the sorted function.
        cache_update_method: Name of the keys_cache update method to use, if it is an attribute of keys_cache.
            Note that this cache_update_method will be used only
                if keys_cache is an explicit iterable and has that attribute
                if keys_cache is a callable and has that attribute.
            The default None
        name: The name of the new class

    Returns:
        If store is:
            None: Will return a decorator that can be applied to a store
            a store class: Will return a wrapped class that caches it's keys
            a store instance: Will return a wrapped instance that caches it's keys

        The instances of such key-cached classes have some extra attributes:
            _explicit_keys: The actual cache. An iterable container
            update_keys_cache: Is called if a user uses the instance to mutate the store (i.e. write or delete).

    You have two ways of caching keys:
    - By providing the explicit list of keys you want cache (and use)
    - By providing a callable that will iterate through your store and collect an explicit list of keys

    Let's take a simple dict as our original store.

    >>> source = dict(c=3, b=2, a=1)

    Specify an iterable, and it will be used as the cached keys

    >>> cached = cached_keys(source, keys_cache='bc')
    >>> list(cached.items())  # notice that the order you get things is also ruled by the cache
    [('b', 2), ('c', 3)]

    Specify a callable, and it will apply it to the existing keys to make your cache

    >>> list(cached_keys(source, keys_cache=sorted))
    ['a', 'b', 'c']

    You can use the callable keys_cache specification to filter as well!
    Oh, and let's demo the fact that if you don't specify the store, it will make a store decorator for you:

    >>> cache_my_keys = cached_keys(keys_cache=lambda keys: list(filter(lambda k: k >= 'b', keys)))
    >>> d = cache_my_keys(source)  # used as to transform an instance
    >>> list(d)
    ['c', 'b']

    Let's use that same `cache_my_keys` to decorate a class instead:

    >>> cached_dict = cache_my_keys(dict)
    >>> d = cached_dict(c=3, b=2, a=1)
    >>> list(d)
    ['c', 'b']

    Note that there's still an underlying store (dict) that has the data:

    >>> repr(d)  # repr isn't wrapped, so you can still see your underlying dict
    "{'c': 3, 'b': 2, 'a': 1}"

    And yes, you can still add elements,

    >>> d['z'] = 26
    >>> list(d.items())
    [('c', 3), ('b', 2), ('z', 26)]

    do bulk updates,

    >>> d.update({'more': 'of this'}, more_of='that')
    >>> list(d.items())
    [('c', 3), ('b', 2), ('z', 26), ('more', 'of this'), ('more_of', 'that')]

    and delete...

    >>> del d['more']
    >>> list(d.items())
    [('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]

    But careful! Know what you're doing if you try to get creative. Have a look at this:

    >>> d['a'] = 100  # add an 'a' item
    >>> d.update(and_more='of that')  # update to add yet another item
    >>> list(d.items())
    [('c', 3), ('b', 2), ('z', 26), ('more_of', 'that')]

    Indeed: No 'a' or 'and_more'.

    Now... they were indeed added. Or to be more precise, the value of the already existing a was changed,
    and a new ('and_more', 'of that') item was indeed added in the underlying store:

    >>> repr(d)
    "{'c': 3, 'b': 2, 'a': 100, 'z': 26, 'more_of': 'that', 'and_more': 'of that'}"

    But you're not seeing it.

    Why?

    Because you chose to use a callable keys_cache that doesn't have an 'update' method.
    When your _keys_cache attribute (the iterable cache) is not updatable itself, the
    way updates work is that we iterate through the underlying store (where the updates actually took place),
    and apply the keys_cache (callable) to that iterable.

    So what happened here was that you have your new 'a' and 'and_more' items, but your cached version of the
    store doesn't see it because it's filtered out. On the other hand, check out what happens if you have
    an updateable cache.

    Using `set` instead of `list`, after the `filter`.

    >>> cache_my_keys = cached_keys(keys_cache=set)
    >>> d = cache_my_keys(source)  # used as to transform an instance
    >>> sorted(d)  # using sorted because a set's order is not always the same
    ['a', 'b', 'c']
    >>> d['a'] = 100
    >>> d.update(and_more='of that')  # update to add yet another item
    >>> sorted(d.items())
    [('a', 100), ('and_more', 'of that'), ('b', 2), ('c', 3)]

    This example was to illustrate a more subtle aspect of cached_keys. You would probably deal with
    the filter concern in a different way in this case. But the rope is there -- it's your choice on how
    to use it.

    And here's some more examples if that wasn't enough!

    >>> # Lets cache the keys of a dict.
    >>> cached_dict = cached_keys(dict)
    >>> d = cached_dict(a=1, b=2, c=3)
    >>> # And you get a store that behaves as expected (but more speed and RAM)
    >>> list(d)
    ['a', 'b', 'c']
    >>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
    [('a', 1), ('b', 2), ('c', 3)]

    This is where the keys are stored:

    >>> d._keys_cache
    ['a', 'b', 'c']

    >>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
    >>> sorted_dict = cached_keys(dict, keys_cache=list)
    >>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
    >>> list(s)  # keys will be in the order they were defined
    ['b', 'a', 'c']
    >>> sorted_dict = cached_keys(dict, keys_cache=sorted)
    >>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
    >>> list(s)  # keys will be sorted
    ['a', 'b', 'c']
    >>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
    >>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
    >>> list(s)  # keys will be sorted according to their length
    ['c', 'aa', 'bbb']

    If you change the keys (adding new ones with __setitem__ or update, or removing with pop or popitem)
    then the cache is recomputed (the first time you use an operation that iterates over keys)

    >>> d.update(d=4)  # let's add an element (try d['d'] = 4 as well)
    >>> list(d)
    ['a', 'b', 'c', 'd']
    >>> d['e'] = 5
    >>> list(d.items())  # whether you iterate with .keys(), .values(), or .items()
    [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]

    >>> @cached_keys
    ... class A:
    ...     def __iter__(self):
    ...         yield from [1, 2, 3]
    >>> # Note, could have also used this form: AA = cached_keys(A)
    >>> a = A()
    >>> list(a)
    [1, 2, 3]
    >>> a._keys_cache = ['a', 'b', 'c']  # changing the cache, to prove that subsequent listing will read from there
    >>> list(a)  # proof:
    ['a', 'b', 'c']
    >>>

    >>> # Let's demo the iter_to_container argument. The default is "list", which will just consume the iter in order
    >>> sorted_dict = cached_keys(dict, keys_cache=list)
    >>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
    >>> list(s)  # keys will be in the order they were defined
    ['b', 'a', 'c']
    >>> sorted_dict = cached_keys(dict, keys_cache=sorted)
    >>> s = sorted_dict({'b': 3, 'a': 2, 'c': 1})
    >>> list(s)  # keys will be sorted
    ['a', 'b', 'c']
    >>> sorted_dict = cached_keys(dict, keys_cache=lambda x: sorted(x, key=len))
    >>> s = sorted_dict({'bbb': 3, 'aa': 2, 'c': 1})
    >>> list(s)  # keys will be sorted according to their length
    ['c', 'aa', 'bbb']
    """

    return _wrap_store(
        _cached_keys, dict(locals(), name=store.__qualname__ + "Wrapped")
    )

    # TODO: Replaced the following with the above:
    # arguments = {k: v for k, v in locals().items() if k != 'arguments'}
    # store = arguments.pop('store')
    # class_trans = partial(_cached_keys, **arguments)
    # arguments['name'] = arguments['name'] or store.__qualname__ + 'Wrapped'
    #
    # return Store.wrap(store, class_trans=class_trans)


def _cached_keys(
    store,
    keys_cache: Callable | Collection = list,
    iter_to_container=None,  # deprecated: use keys_cache instead
    cache_update_method="update",
    name: str = None,  # TODO: might be able to be deprecated since included in store_decorator
    __module__=None,  # TODO: might be able to be deprecated since included in store_decorator
):
    if iter_to_container is not None:
        assert callable(iter_to_container)
        warn(
            "The argument name 'iter_to_container' is being deprecated in favor "
            "of the more general 'keys_cache'"
        )
        # assert keys_cache == iter_to_container

    assert isinstance(
        store, type
    ), f"store_cls must be a type, was a {type(store)}: {store}"

    # name = name or 'IterCached' + get_class_name(store_cls)
    name = name or get_class_name(store)
    __module__ = __module__ or getattr(store, "__module__", None)

    class cached_cls(store):
        _keys_cache = None

    cached_cls.__name__ = name

    # cached_cls = type(name, (store_cls,), {"_keys_cache": None})

    # The following class is not the class that will be returned, but the class from which we'll take the methods
    #   that will be copied in the class that will be returned.
    # @_define_keys_values_and_items_according_to_iter
    class CachedIterMethods:
        _explicit_keys = False
        _updatable_cache = False
        _iter_to_container = None
        if hasattr(keys_cache, cache_update_method):
            _updatable_cache = True
        if is_iterable(
            keys_cache
        ):  # if keys_cache is iterable, it is the cache instance itself.
            _keys_cache = keys_cache
            _explicit_keys = True
        elif callable(keys_cache):
            # if keys_cache is not iterable, but callable, we'll use it to make the keys_cache from __iter__
            _iter_to_container = keys_cache

            @lazyprop
            def _keys_cache(self):
                # print(iter_to_container)
                return keys_cache(
                    super(cached_cls, self).__iter__()
                )  # TODO: Should it be iter(super(...)?

        @property
        def _iter_cache(self):  # for back-compatibility
            warn(
                "The new name for `_iter_cache` is `_keys_cache`. Start using that!",
                DeprecationWarning,
            )
            return self._keys_cache

        def __iter__(self):
            # If _keys_cache is None, then we haven't iterated yet, so we'll do it now.
            # This if clause was commmented out 3 years ago, and recently uncommented.
            if getattr(self, "_keys_cache", None) is None:
                self._keys_cache = keys_cache(super(cached_cls, self).__iter__())
            yield from self._keys_cache

        def __len__(self):
            return len(self._keys_cache)

        def __contains__(self, k):
            return k in self._keys_cache

        # The write and update stuff ###################################################################

        if _updatable_cache:

            def update_keys_cache(self, keys):
                """updates the keys by calling the"""
                update_func = getattr(self._keys_cache, cache_update_method)
                update_func(self._keys_cache, keys)

            update_keys_cache.__doc__ = (
                "Updates the _keys_cache by calling its {} method"
            )
        else:

            def update_keys_cache(self, keys):
                """Updates the _keys_cache by deleting the attribute"""
                try:
                    del self._keys_cache
                    # print('deleted _keys_cache')
                except AttributeError:
                    pass

        def __setitem__(self, k, v):
            super(cached_cls, self).__setitem__(k, v)
            # self.store[k] = v
            if (
                k not in self
            ):  # just to avoid deleting the cache if we already had the key
                self.update_keys_cache((k,))
                # Note: different construction performances: (k,)->10ns, [k]->38ns, {k}->50ns

        def update(self, other=(), **kwds):
            # print(other, kwds)
            # super(cached_cls, self).update(other, **kwds)
            super_setitem = super(cached_cls, self).__setitem__
            for k in other:
                # print(k, other[k])
                super_setitem(k, other[k])
                # self.store[k] = other[k]
            self.update_keys_cache(other)

            for k, v in kwds.items():
                # print(k, v)
                super_setitem(k, v)
                # self.store[k] = v
            self.update_keys_cache(kwds)

        def __delitem__(self, k):
            self._keys_cache.remove(k)
            super(cached_cls, self).__delitem__(k)

    # And this is where we add all the needed methods (for example, no __setitem__ won't be added if the original
    #   class didn't have one in the first place.
    special_attrs = {
        "update_keys_cache",
        "_keys_cache",
        "_explicit_keys",
        "_updatable_cache",
    }
    for attr in special_attrs | (
        AttrNames.KvPersister & attrs_of(cached_cls) & attrs_of(CachedIterMethods)
    ):
        setattr(cached_cls, attr, getattr(CachedIterMethods, attr))

    if __module__ is not None:
        cached_cls.__module__ = __module__

    if hasattr(store, "__doc__"):
        cached_cls.__doc__ = store.__doc__

    return cached_cls


cache_iter = cached_keys  # TODO: Alias, partial it and make it more like the original, for back compatibility.


@store_decorator
def catch_and_cache_error_keys(
    store=None,
    *,
    errors_caught=Exception,
    error_callback=None,
    use_cached_keys_after_completed_iter=True,
):
    """Store that will cache keys as they're accessed, separating those that raised errors and those that didn't.
    Getting a key will still through an error, but the access attempts will be collected in an ._error_keys attribute.
    Successfful attemps will be stored in _keys_cache.
    Retrieval iteration (items() or values()) will on the other hand, skip the error (while still caching it).
    If the iteration completes (and use_cached_keys_after_completed_iter), the use_cached_keys flag is turned on,
    which will result in the store now getting it's keys from the _keys_cache.

    >>> @catch_and_cache_error_keys(
    ...     error_callback=lambda store, key, err: print(f"Error with {key} key: {err}"))
    ... class Blacklist(dict):
    ...     _black_list = {'black', 'list'}
    ...
    ...     def __getitem__(self, k):
    ...         if k not in self._black_list:
    ...             return super().__getitem__(k)
    ...         else:
    ...             raise KeyError(f"Nope, that's from the black list!")
    >>>
    >>> s = Blacklist(black=7,  friday=20, frenzy=13)
    >>> list(s)
    ['black', 'friday', 'frenzy']
    >>> list(s.items())
    Error with black key: "Nope, that's from the black list!"
    [('friday', 20), ('frenzy', 13)]
    >>> sorted(s)  # sorting to get consistent output
    ['frenzy', 'friday']


    See that? First we had three keys, then we iterated and got only 2 items (fortunately,
    we specified an ``error_callback`` so we ccould see that the iteration actually
    dropped a key).

    That's strange. And even stranger is the fact that when we list our keys again,
    we get only two.

    You don't like it? Neither do I. But

    - It's not a completely outrageous behavior -- if you're talking to live data, it
        often happens that you get more, or less, from one second to another.

    - This store isn't meant to be long living, but rather meant to solve the problem of
        skiping items that are problematic (for example, malformatted files),
        with a trace of what was skipped and what's valid (in case we need to iterate
        again and don't want to bear the hit of requesting values for keys we already
        know are problematic.

    Here's a little peep of what is happening under the hood.
    Meet ``_keys_cache`` and ``_error_keys`` sets (yes, unordered -- so know it) that are meant
    to acccumulate valid and problematic keys respectively.

    >>> s = Blacklist(black=7,  friday=20, frenzy=13)
    >>> list(s)
    ['black', 'friday', 'frenzy']
    >>> s._keys_cache, s._error_keys
    (set(), set())
    >>> s['friday']
    20
    >>> s._keys_cache, s._error_keys
    ({'friday'}, set())
    >>> s['black']
    Traceback (most recent call last):
      ...
    KeyError: "Nope, that's from the black list!"
    >>> s._keys_cache, s._error_keys
    ({'friday'}, {'black'})

    But see that we still have the full list:

    >>> list(s)
    ['black', 'friday', 'frenzy']

    Meet ``use_cached_keys``: He's the culprit. It's a flag that indicates whether
    we should be using the cached keys or not. Obviously, it'll start off being
    ``False``:

    >>> s.use_cached_keys
    False

    Now we could set it to ``True`` manually to change the mode.
    But know that this switch happens automatically (UNLESS you specify otherwise by
    saying:``use_cached_keys_after_completed_iter=False``) when ever you got through a
    VALUE-PRODUCING iteration (i.e. entirely consuming `items()` or `values()`).

    >>> sorted(s.values())  # sorting to get consistent output
    Error with black key: "Nope, that's from the black list!"
    [13, 20]

    """

    assert isinstance(
        store, type
    ), f"store_cls must be a type, was a {type(store)}: {store}"

    # assert isinstance(store, Mapping), f"store_cls must be a Mapping.
    #  Was not. mro is {store.mro()}: {store}"

    # class cached_cls(store):
    #     _keys_cache = None
    #     _error_keys = None

    from dol.base import MappingViewMixin

    # The following class is not the class that will be returned,
    #   but the class from which we'll take the methods
    #   that will be copied in the class that will be returned.
    class CachedKeyErrorsStore(MappingViewMixin, store):
        @wraps(store.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._error_keys = set()
            self._keys_cache = set()
            self.use_cached_keys = False
            self.use_cached_keys_after_completed_iter = (
                use_cached_keys_after_completed_iter
            )
            self.errors_caught = errors_caught
            self.error_callback = error_callback

        def __getitem__(self, k):
            if self.use_cached_keys:
                return super().__getitem__(k)
            else:
                try:
                    v = super().__getitem__(k)
                    self._keys_cache.add(k)
                    return v
                except self.errors_caught as err:
                    self._error_keys.add(k)
                    # if self.error_callback is not None:
                    #     self.error_callback(store, k, err)
                    raise

        def __iter__(self):
            # if getattr(self, '_keys_cache', None) is None:
            #    self._keys_cache = iter_to_container(super(cached_cls, self).__iter__())
            if self.use_cached_keys:
                yield from self._keys_cache
                if self.use_cached_keys_after_completed_iter:
                    self.use_cached_keys = True
            else:
                yield from super().__iter__()

        def __len__(self):
            if self.use_cached_keys:
                return len(self._keys_cache)
            else:
                return super().__len__()

        class ItemsView(BaseKeysView):
            def __iter__(self):
                m = self._mapping
                if m.use_cached_keys:
                    for k in m._keys_cache:
                        yield k, m[k]
                else:
                    for k in m:
                        try:
                            yield k, m[k]
                        except m.errors_caught as err:
                            if m.error_callback is not None:
                                m.error_callback(store, k, err)
                if m.use_cached_keys_after_completed_iter:
                    m.use_cached_keys = True

            def __contains__(self, k):
                m = self._mapping
                if m.use_cached_keys:
                    return k in m._keys_cache
                else:
                    return k in m

        class ValuesView(BaseValuesView):
            def __iter__(self):
                m = self._mapping
                if m.use_cached_keys:
                    yield from (m[k] for k in m._keys_cache)
                else:
                    yield from (v for k, v in m.items())

        def __contains__(self, k):
            if self.use_cached_keys:
                return k in self._keys_cache
            else:
                return super().__contains__(k)

    return CachedKeyErrorsStore


def iterate_values_and_accumulate_non_error_keys(
    store, cache_keys_here: list, errors_caught=Exception, error_callback=None
):
    for k in store:
        try:
            v = store[k]
            cache_keys_here.append(k)
            yield v
        except errors_caught as err:
            if error_callback is not None:
                error_callback(store, k, err)


########################################################################################################################
# Filtering iteration


def take_everything(key):
    return True


FiltFunc = Callable[[Any], bool]


# Note: The full definition of filt_iter includes some attributes that will be added
#   to it below (seach for FiltIter.__dict__.items())
@store_decorator
def filt_iter(
    store=None,
    *,
    filt: Callable | Iterable = take_everything,
    name=None,
    __module__=None,  # TODO: might be able to be deprecated since included in store_decorator
):
    """Make a wrapper that will transform a store (class or instance thereof) into a sub-store (i.e. subset of keys).

    Args:
        filt: A callable or iterable:
            callable: Boolean filter function. A func taking a key and and returns True iff the key should be included.
            iterable: The collection of keys you want to filter "in"
        name: The name to give the wrapped class

    Returns: A wrapper (that then needs to be applied to a store instance or class.

    >>> filtered_dict = filt_iter(filt=lambda k: (len(k) % 2) == 1)(dict)  # keep only odd length keys
    >>>
    >>> s = filtered_dict({'a': 1, 'bb': object, 'ccc': 'a string', 'dddd': [1, 2]})
    >>>
    >>> list(s)
    ['a', 'ccc']
    >>> 'a' in s  # True because odd (length) key
    True
    >>> 'bb' in s  # False because odd (length) key
    False
    >>> assert s.get('bb', None) == None
    >>> len(s)
    2
    >>> list(s.keys())
    ['a', 'ccc']
    >>> list(s.values())
    [1, 'a string']
    >>> list(s.items())
    [('a', 1), ('ccc', 'a string')]
    >>> s.get('a')
    1
    >>> assert s.get('bb') is None
    >>> s['x'] = 10
    >>> list(s.items())
    [('a', 1), ('ccc', 'a string'), ('x', 10)]
    >>> try:
    ...     s['xx'] = 'not an odd key'
    ...     raise ValueError("This should have failed")
    ... except KeyError:
    ...     pass
    """

    return _wrap_store(_filt_iter, dict(locals(), name=store.__qualname__ + "Wrapped"))


# TODO: Factor out the method injection pattern (e.g. __getitem__, __setitem__
#  and __delitem__ are nearly identical)
def _filt_iter(store_cls: type, filt, name, __module__):
    assert isinstance(store_cls, type), f"store_cls must be a type: {store_cls}"

    if not callable(filt):  # if filt is not a callable...
        # ... assume it's the collection of keys you want and make a filter function
        # to filter those "in".
        assert isinstance(filt, Iterable), "filt should be a callable, or an iterable"
        keys_that_should_be_filtered_in = set(filt)

        def filt(k):
            return k in keys_that_should_be_filtered_in

    def __iter__(self):
        yield from filter(filt, super(store_cls, self).__iter__())

    store_cls.__iter__ = __iter__

    def __len__(self):
        c = 0
        for _ in self.__iter__():
            c += 1
        return c

    store_cls.__len__ = __len__

    def __contains__(self, k):
        if filt(k):
            return super(store_cls, self).__contains__(k)
        else:
            return False

    store_cls.__contains__ = __contains__
    if hasattr(store_cls, "__getitem__"):

        def __getitem__(self, k):
            if filt(k):
                return super(store_cls, self).__getitem__(k)
            else:
                raise KeyError(f"Key not in store: {k}")

        store_cls.__getitem__ = __getitem__

    if hasattr(store_cls, "get"):

        def get(self, k, default=None):
            if filt(k):
                return super(store_cls, self).get(k, default)
            else:
                return default

        store_cls.get = get
    if hasattr(store_cls, "__setitem__"):

        def __setitem__(self, k, v):
            if filt(k):
                return super(store_cls, self).__setitem__(k, v)
            else:
                raise KeyError(f"Key not in store: {k}")

        store_cls.__setitem__ = __setitem__
    if hasattr(store_cls, "__delitem__"):

        def __delitem__(self, k):
            if filt(k):
                return super(store_cls, self).__delitem__(k)
            else:
                raise KeyError(f"Key not in store: {k}")

        store_cls.__delitem__ = __delitem__
    return store_cls


def filter_regex(regex, *, return_search_func=False):
    r"""Make a filter that returns True if a string matches the given regex

    >>> is_txt = filter_regex(r'.*\.txt')
    >>> is_txt("test.txt")
    True
    >>> is_txt("report.doc")
    False

    """
    if isinstance(regex, str):
        regex = safe_compile(regex)
    if return_search_func:
        return regex.search
    else:
        pipe = Pipe(regex.search, bool)
        pipe.regex = regex
        return pipe


def filter_suffixes(suffixes):
    """Make a filter that returns True if a string ends with one of the given suffixes

    >>> ends_with_txt = filter_suffixes('.txt')
    >>> ends_with_txt("test.txt")
    True
    >>> ends_with_txt("report.doc")
    False
    >>> is_text = filter_suffixes(['.txt', '.doc', '.pdf'])
    >>> is_text("test.txt")
    True
    >>> is_text("report.doc")
    True
    >>> is_text("image.jpg")
    False

    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    return filter_regex("(" + "|".join(map(re.escape, suffixes)) + ")" + "$")


def filter_prefixes(prefixes):
    """Make a filter that returns True if a string starts with one of the given prefixes

    >>> starts_with_test = filter_prefixes('test')
    >>> starts_with_test("test.txt")
    True
    >>> starts_with_test("report.doc")
    False
    >>> is_test_or_report = filter_prefixes(['test', 'report'])
    >>> is_test_or_report("test.txt")
    True
    >>> is_test_or_report("report.doc")
    True
    >>> is_test_or_report("image.jpg")
    False

    """
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return filter_regex("^" + "|".join(map(re.escape, prefixes)))


class FiltIter:
    def __init__(self, *args, **kwargs):
        raise ValueError(
            "This class is not meant to be instantiated, but only act as a collection "
            "of functions to make mapping filtering decorators."
        )

    def regex(regex):
        """Make a mapping-filtering decorator that filters keys with a regex.

        :param regex: A regex string or compiled regex

        >>> contains_a = FiltIter.regex(r'a')
        >>> d = {'apple': 1, 'banana': 2, 'cherry': 3}
        >>> dd = contains_a(d)
        >>> dict(dd)
        {'apple': 1, 'banana': 2}
        """
        return filt_iter(filt=filter_regex(regex))

    def prefixes(prefixes):
        """Make a mapping-filtering decorator that filters keys with a prefixes.

        :param prefixes: A string or iterable of strings that are the prefixes to filter

        >>> is_test = FiltIter.prefixes('test')
        >>> d = {'test.txt': 1, 'report.doc': 2, 'test_image.jpg': 3}
        >>> dd = is_test(d)
        >>> dict(dd)
        {'test.txt': 1, 'test_image.jpg': 3}
        """
        return filt_iter(filt=filter_prefixes(prefixes))

    def suffixes(suffixes):
        """Make a mapping-filtering decorator that filters keys with a suffixes.

        :param suffixes: A string or iterable of strings that are the suffixes to filter

        >>> is_text = FiltIter.suffixes(['.txt', '.doc', '.pdf'])
        >>> d = {'test.txt': 1, 'report.doc': 2, 'image.jpg': 3}
        >>> dd = is_text(d)
        >>> dict(dd)
        {'test.txt': 1, 'report.doc': 2}
        """
        return filt_iter(filt=filter_suffixes(suffixes))


# TODO: Is there a better way to solve the tab-completion problem
# Here we add attributes to filt_iter via explicit statements assigning to None before
# looping through FilterIter and adding the actual functions, because Tab-completion
# wasn't working with the loop alone
filt_iter.regex = None
filt_iter.prefixes = None
filt_iter.suffixes = None

# add all the functions in FiltIter as attributes of filt_iter, so they're ready to use
for filt_name, filt_func in FiltIter.__dict__.items():
    if not filt_name.startswith("_"):
        # filt_func.__name__ = filt_name
        filt_func.__doc__ = (filt_func.__doc__ or "").replace("FiltIter", "filt_iter")
        setattr(filt_iter, filt_name, filt_func)


########################################################################################################################
# Wrapping keys and values

self_names = frozenset(["self", "store", "mapping"])


# TODO: Consider deprecation. Besides the name arg (whose usefulness is doubtful), this is just Store.wrap
def kv_wrap_persister_cls(persister_cls, name=None):
    """Make a class that wraps a persister into a dol.base.Store,

    Args:
        persister_cls: The persister class to wrap

    Returns: A Store wrapping the persister (see dol.base)

    >>> A = kv_wrap_persister_cls(dict)
    >>> a = A()
    >>> a['one'] = 1
    >>> a['two'] = 2
    >>> a['three'] = 3
    >>> list(a.items())
    [('one', 1), ('two', 2), ('three', 3)]
    >>> assert hasattr(a, '_obj_of_data')  # for example, it has this magic method
    >>> # If you overwrite the _obj_of_data method, you'll transform outcomming values with it.
    >>> # For example, say the data you stored were minutes, but you want to get then in secs...
    >>> a._obj_of_data = lambda data: data * 60
    >>> list(a.items())
    [('one', 60), ('two', 120), ('three', 180)]
    >>>
    >>> # And if you want to have class that has this weird "store minutes, retrieve seconds", you can do this:
    >>> class B(kv_wrap_persister_cls(dict)):
    ...     def _obj_of_data(self, data):
    ...         return data * 60
    >>> b = B()
    >>> b.update({'one': 1, 'two': 2, 'three': 3})  # you can write several key-value pairs at once this way!
    >>> list(b.items())
    [('one', 60), ('two', 120), ('three', 180)]
    >>> # Warning! Advanced under-the-hood chat coming up.... Note this:
    >>> print(b)
    {'one': 1, 'two': 2, 'three': 3}
    >>> # What?!? Well, remember, printing an object calls the objects __str__, which usually calls __repr__
    >>> # The wrapper doesn't wrap those methods, since they don't have consistent behaviors.
    >>> # Here you're getting the __repr__ of the underlying dict store, without the key and value transforms.
    >>>
    >>> # Say you wanted to transform the incoming minute-unit data, converting to secs BEFORE they were stored...
    >>> class C(kv_wrap_persister_cls(dict)):
    ...     def _data_of_obj(self, obj):
    ...         return obj * 60
    >>> c = C()
    >>> c.update(one=1, two=2, three=3)  # yet another way you can write multiple key-vals at once
    >>> list(c.items())
    [('one', 60), ('two', 120), ('three', 180)]
    >>> print(c)  # but notice that unlike when we printed b, here the stored data is actually transformed!
    {'one': 60, 'two': 120, 'three': 180}
    >>>
    >>> # Now, just to demonstrate key transformation, let's say that we need internal (stored) keys to be upper case,
    >>> # but external (the keys you see when listed) ones to be lower case, for some reason...
    >>> class D(kv_wrap_persister_cls(dict)):
    ...     _data_of_obj = staticmethod(lambda obj: obj * 60)  # to demonstrated another way of doing this
    ...     _key_of_id = lambda self, _id: _id.lower()  # note if you don't specify staticmethod, 1st arg must be self
    ...     def _id_of_key(self, k):  # a function definition like you're used to
    ...         return k.upper()
    >>> d = D()
    >>> d['oNe'] = 1
    >>> d.update(TwO=2, tHrEE=3)
    >>> list(d.items())  # you see clean lower cased keys at the interface of the store
    [('one', 60), ('two', 120), ('three', 180)]
    >>> # but internally, the keys are all upper case
    >>> print(d)  # equivalent to print(d.store), so keys and values not wrapped (values were transformed before stored)
    {'ONE': 60, 'TWO': 120, 'THREE': 180}
    >>>
    >>> # On the other hand, careful, if you gave the data directly to D, you wouldn't get that.
    >>> d = D({'one': 1, 'two': 2, 'three': 3})
    >>> print(d)
    {'one': 1, 'two': 2, 'three': 3}
    >>> # Thus is because when you construct a D with the dict, it initializes the dicts data with it directly
    >>> # before the key/val transformers are in place to do their jobs.
    """

    cls = Store.wrap(persister_cls)

    # TODO: The whole name and qualname thing -- is it really necessary, correct, what we want?
    name = name or (persister_cls.__name__ + "PWrapped")
    qname = name or (persister_cls.__qualname__ + "PWrapped")

    cls.__qualname__ = qname
    cls.__name__ = name

    return cls

    # name = name or (persister_cls.__qualname__ + "PWrapped")
    #
    # cls = type(name, (Store,), {})
    #
    # # TODO: Investigate sanity and alternatives (cls = type(name, (Store, persister_cls), {}) leads to MRO problems)
    # for attr in set(dir(persister_cls)) - set(dir(Store)):
    #     persister_cls_attribute = getattr(persister_cls, attr)
    #     setattr(cls, attr, persister_cls_attribute)  # copy the attribute over to cls
    #
    # if hasattr(persister_cls, '__doc__'):
    #     cls.__doc__ = persister_cls.__doc__
    #
    # @wraps(persister_cls.__init__)
    # def __init__(self, *args, **kwargs):
    #     super(cls, self).__init__(persister_cls(*args, **kwargs))
    #
    # cls.__init__ = __init__
    #
    # return cls


def _wrap_outcoming(
    store_cls: type, wrapped_method: str, trans_func: Callable | None = None
):
    """Output-transforming wrapping of the wrapped_method of store_cls.
    The transformation is given by trans_func, which could be a one (trans_func(x)
    or two (trans_func(self, x)) argument function.

    Args:
        store_cls: The class that will be transformed
        wrapped_method: The method (name) that will be transformed.
        trans_func: The transformation function.
        wrap_arg_idx: The index of the

    Returns: Nothing. It transforms the class in-place

    >>> from dol.trans import store_wrap
    >>> S = store_wrap(dict)
    >>> _wrap_outcoming(S, '_key_of_id', lambda x: f'wrapped_{x}')
    >>> s = S({'a': 1, 'b': 2})
    >>> list(s)
    ['wrapped_a', 'wrapped_b']
    >>> _wrap_outcoming(S, '_key_of_id', lambda self, x: f'wrapped_{x}')
    >>> s = S({'a': 1, 'b': 2}); assert list(s) == ['wrapped_a', 'wrapped_b']
    >>> class A:
    ...     def __init__(self, prefix='wrapped_'):
    ...         self.prefix = prefix
    ...     def _key_of_id(self, x):
    ...         return self.prefix + x
    >>> _wrap_outcoming(S, '_key_of_id', A(prefix='wrapped_')._key_of_id)
    >>> s = S({'a': 1, 'b': 2}); assert list(s) == ['wrapped_a', 'wrapped_b']
    >>>
    >>> S = store_wrap(dict)
    >>> _wrap_outcoming(S, '_obj_of_data', lambda x: x * 7)
    >>> s = S({'a': 1, 'b': 2})
    >>> list(s.values())
    [7, 14]
    """
    if trans_func is not None:
        wrapped_func = getattr(store_cls, wrapped_method)

        if not _has_unbound_self(trans_func):
            # print(f"00000: {store_cls}: {wrapped_method}, {trans_func}, {wrapped_func}, {wrap_arg_idx}")
            @wraps(wrapped_func)
            def new_method(self, x):
                # # Long form (for explanation)
                # super_method = getattr(super(store_cls, self), wrapped_method)
                # output_of_super_method = super_method(x)
                # transformed_output_of_super_method = trans_func(output_of_super_method)
                # return transformed_output_of_super_method
                return trans_func(getattr(super(store_cls, self), wrapped_method)(x))

        else:
            # print(f"11111: {store_cls}: {wrapped_method}, {trans_func}, {wrapped_func}, {wrap_arg_idx}")
            @wraps(wrapped_func)
            def new_method(self, x):
                # # Long form (for explanation)
                # super_method = getattr(super(store_cls, self), wrapped_method)
                # output_of_super_method = super_method(x)
                # transformed_output_of_super_method = trans_func(self, output_of_super_method)
                # return transformed_output_of_super_method
                return trans_func(
                    self, getattr(super(store_cls, self), wrapped_method)(x)
                )

        setattr(store_cls, wrapped_method, new_method)


def _wrap_ingoing(store_cls, wrapped_method: str, trans_func: Callable | None = None):
    if trans_func is not None:
        wrapped_func = getattr(store_cls, wrapped_method)

        if not _has_unbound_self(trans_func):

            @wraps(wrapped_func)
            def new_method(self, x):
                return getattr(super(store_cls, self), wrapped_method)(trans_func(x))

        else:

            @wraps(wrapped_func)
            def new_method(self, x):
                return getattr(super(store_cls, self), wrapped_method)(
                    trans_func(self, x)
                )

        setattr(store_cls, wrapped_method, new_method)


@store_decorator
def wrap_kvs(
    store=None,
    *,
    wrapper=None,
    name=None,
    key_of_id=None,
    id_of_key=None,
    obj_of_data=None,
    data_of_obj=None,
    preset=None,
    postget=None,
    key_codec=None,
    value_codec=None,
    key_encoder=None,
    key_decoder=None,
    value_encoder=None,
    value_decoder=None,
    __module__=None,
    outcoming_key_methods=(),
    outcoming_value_methods=(),
    ingoing_key_methods=(),
    ingoing_value_methods=(),
):
    r"""Make a Store that is wrapped with the given key/val transformers.

    Naming convention:
        Morphemes:
            key: outer key
            _id: inner key
            obj: outer value
            data: inner value
        Grammar:
            Y_of_X: means that you get a Y output when giving an X input. Also known as X_to_Y.


    Args:
        store: Store class or instance
        name: Name to give the wrapper class
        key_of_id: The outcoming key transformation function.
            Forms are `k = key_of_id(_id)` or `k = key_of_id(self, _id)`
        id_of_key: The ingoing key transformation function.
            Forms are `_id = id_of_key(k)` or `_id = id_of_key(self, k)`
        obj_of_data: The outcoming val transformation function.
            Forms are `obj = obj_of_data(data)` or `obj = obj_of_data(self, data)`
        data_of_obj: The ingoing val transformation function.
            Forms are `data = data_of_obj(obj)` or `data = data_of_obj(self, obj)`
        preset: A function that is called before doing a `__setitem__`.
            The function is called with both `k` and `v` as inputs, and should output a transformed value.
            The intent use is to do ingoing value transformations conditioned on the key.
            For example, you may want to serialize an object depending on if you're writing to a
             '.csv', or '.json', or '.pickle' file.
            Forms are `preset(k, obj)` or `preset(self, k, obj)`
        postget: A function that is called after the value `v` for a key `k` is be `__getitem__`.
            The function is called with both `k` and `v` as inputs, and should output a transformed value.
            The intent use is to do outcoming value transformations conditioned on the key.
            We already have `obj_of_data` for outcoming value trans, but cannot condition it's behavior on k.
            For example, you may want to deserialize the bytes of a '.csv', or '.json', or '.pickle' in different ways.
            Forms are `obj = postget(k, data)` or `obj = postget(self, k, data)`

    Returns: A key and/or value transformed wrapped (or wrapper) class (or instance).

    >>> def key_of_id(_id):
    ...     return _id.upper()
    >>> def id_of_key(k):
    ...     return k.lower()
    >>> def obj_of_data(data):
    ...     return data - 100
    >>> def data_of_obj(obj):
    ...     return obj + 100
    >>>
    >>> A = wrap_kvs(dict, name='A',
    ...             key_of_id=key_of_id, id_of_key=id_of_key, obj_of_data=obj_of_data, data_of_obj=data_of_obj)
    >>> a = A()
    >>> a['KEY'] = 1
    >>> a  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
    {'key': 101}
    >>> a['key'] = 2
    >>> print(a)  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
    {'key': 102}
    >>> a['kEy'] = 3
    >>> a  # repr is just the base class (dict) repr, so shows "inside" the store (lower case keys and +100)
    {'key': 103}
    >>> list(a)  # but from the point of view of the interface the keys are all upper case
    ['KEY']
    >>> list(a.items())  # and the values are those we put there.
    [('KEY', 3)]
    >>>
    >>> # And now this: Showing how to condition the value transform (like obj_of_data), but conditioned on key.
    >>> B = wrap_kvs(dict, name='B', postget=lambda k, v: f'upper {v}' if k[0].isupper() else f'lower {v}')
    >>> b = B()
    >>> b['BIG'] = 'letters'
    >>> b['small'] = 'text'
    >>> list(b.items())
    [('BIG', 'upper letters'), ('small', 'lower text')]
    >>>
    >>>
    >>> # Let's try preset and postget. We'll wrap a dict and write the same list of lists object to
    >>> # keys ending with .csv, .json, and .pkl, specifying the obvious extension-dependent
    >>> # serialization/deserialization we want to associate with it.
    >>>
    >>> # First, some very simple csv transformation functions
    >>> to_csv = lambda LoL: '\\n'.join(map(','.join, map(lambda L: (x for x in L), LoL)))
    >>> from_csv = lambda csv: list(map(lambda x: x.split(','), csv.split('\\n')))
    >>> LoL = [['a','b','c'],['d','e','f']]
    >>> assert from_csv(to_csv(LoL)) == LoL
    >>>
    >>> import json, pickle
    >>>
    >>> def preset(k, v):
    ...     if k.endswith('.csv'):
    ...         return to_csv(v)
    ...     elif k.endswith('.json'):
    ...         return json.dumps(v)
    ...     elif k.endswith('.pkl'):
    ...         return pickle.dumps(v)
    ...     else:
    ...         return v  # as is
    ...
    ...
    >>> def postget(k, v):
    ...     if k.endswith('.csv'):
    ...         return from_csv(v)
    ...     elif k.endswith('.json'):
    ...         return json.loads(v)
    ...     elif k.endswith('.pkl'):
    ...         return pickle.loads(v)
    ...     else:
    ...         return v  # as is
    ...
    >>> mydict = wrap_kvs(dict, preset=preset, postget=postget)
    >>>
    >>> obj = [['a','b','c'],['d','e','f']]
    >>> d = mydict()
    >>> d['foo.csv'] = obj  # store the object as csv
    >>> d  # "printing" a dict by-passes the transformations, so we see the data in the "raw" format it is stored in.
    {'foo.csv': 'a,b,c\\nd,e,f'}
    >>> d['foo.csv']  # but if we actually ask for the data, it deserializes to our original object
    [['a', 'b', 'c'], ['d', 'e', 'f']]
    >>> d['bar.json'] = obj  # store the object as json
    >>> d
    {'foo.csv': 'a,b,c\\nd,e,f', 'bar.json': '[["a", "b", "c"], ["d", "e", "f"]]'}
    >>> d['bar.json']
    [['a', 'b', 'c'], ['d', 'e', 'f']]
    >>> d['bar.json'] = {'a': 1, 'b': [1, 2], 'c': 'normal json'}  # let's write a normal json instead.
    >>> d
    {'foo.csv': 'a,b,c\\nd,e,f', 'bar.json': '{"a": 1, "b": [1, 2], "c": "normal json"}'}
    >>> del d['foo.csv']
    >>> del d['bar.json']
    >>> d['foo.pkl'] = obj  # 'save' obj as pickle
    >>> d['foo.pkl']
    [['a', 'b', 'c'], ['d', 'e', 'f']]

    # TODO: Add tests for outcoming_key_methods etc.
    """
    # kwargs = dict(
    #     locals(), wrapper=wrapper or Store, name=store.__qualname__ + "Wrapped"
    # )
    # If name is not explicitly provided, use the store's qualname directly
    # without adding "Wrapped" to preserve the original class name
    if name is None and store is not None:
        name = store.__qualname__

    kwargs = dict(locals(), wrapper=wrapper or Store)
    _handle_codecs(kwargs)
    return _wrap_store(_wrap_kvs, kwargs)


def _handle_codecs(kwargs: dict):
    """Handle the key_codec and data_codec kwargs, converting them to key_of_id
    and obj_of_data.

    Warning: Mutates kwargs in place.

    >>> kwargs = {'value_decoder': int, 'value_encoder': str}
    >>> _handle_codecs(kwargs)
    >>> assert kwargs['obj_of_data'] == int
    >>> assert kwargs['data_of_obj'] == str
    >>> from types import SimpleNamespace
    >>> kwargs = {'key_codec': SimpleNamespace(decoder=int, encoder=str)}
    >>> _handle_codecs(kwargs)
    >>> assert kwargs['key_of_id'] == int
    >>> assert kwargs['id_of_key'] == str

    """
    if key_codec := kwargs.get("key_codec", None):
        if kwargs.get("key_of_id", None):
            raise ValueError("Cannot specify both key_codec and key_of_id")
        kwargs["key_of_id"] = key_codec.decoder
        if kwargs.get("id_of_key", None):
            raise ValueError("Cannot specify both key_codec and id_of_key")
        kwargs["id_of_key"] = key_codec.encoder
        del kwargs["key_codec"]
    if value_codec := kwargs.get("value_codec", None):
        if kwargs.get("obj_of_data", None):
            raise ValueError("Cannot specify both value_codec and obj_of_data")
        kwargs["obj_of_data"] = value_codec.decoder
        if kwargs.get("data_of_obj", None):
            raise ValueError("Cannot specify both value_codec and data_of_obj")
        kwargs["data_of_obj"] = value_codec.encoder
        del kwargs["value_codec"]

    if key_decoder := kwargs.get("key_decoder", None):
        if kwargs.get("key_of_id", None):
            raise ValueError("Cannot specify both key_decoder and key_of_id")
        kwargs["key_of_id"] = key_decoder
        del kwargs["key_decoder"]

    if key_encoder := kwargs.get("key_encoder", None):
        if kwargs.get("id_of_key", None):
            raise ValueError("Cannot specify both key_encoder and id_of_key")
        kwargs["id_of_key"] = key_encoder
        del kwargs["key_encoder"]

    if value_decoder := kwargs.get("value_decoder", None):
        if kwargs.get("obj_of_data", None):
            raise ValueError("Cannot specify both value_decoder and obj_of_data")
        kwargs["obj_of_data"] = value_decoder
        del kwargs["value_decoder"]

    if value_encoder := kwargs.get("value_encoder", None):
        if kwargs.get("data_of_obj", None):
            raise ValueError("Cannot specify both value_encoder and data_of_obj")
        kwargs["data_of_obj"] = value_encoder
        del kwargs["value_encoder"]


# TODO: Below is more general and clean, but breaks tests. Fix tests and use this.
# def _handle_codecs(kwargs: dict):
#     """
#     Handle the key_codec, data_codec, key_decoder, key_encoder, value_decoder, and value_encoder
#     kwargs, converting them to key_of_id, obj_of_data, id_of_key, and data_of_obj respectively.

#     Warning: Mutates kwargs in place.

#     >>> kwargs = {'value_decoder': int, 'value_encoder': str}
#     >>> _handle_codecs(kwargs)
#     >>> assert kwargs['obj_of_data'] == int
#     >>> assert kwargs['data_of_obj'] == str
#     >>> from types import SimpleNamespace
#     >>> kwargs = {'key_codec': SimpleNamespace(decoder=int, encoder=str)}
#     >>> _handle_codecs(kwargs)
#     >>> assert kwargs['key_of_id'] == int
#     >>> assert kwargs['id_of_key'] == str

#     """
#     from operator import attrgetter

#     def handle_replacements(source_key, target_keys_and_extractors):
#         for target_key, extractor in target_keys_and_extractors.items():
#             if target_key in kwargs:
#                 raise ValueError(f'Cannot specify both {source_key} and {target_key}')
#             kwargs[target_key] = extractor(kwargs[source_key])

#     replacements = {
#         'key_codec': {'key_of_id': attrgetter('decoder'), 'id_of_key': attrgetter('encoder')},
#         'value_codec': {'obj_of_data': attrgetter('decoder'), 'data_of_obj': attrgetter('encoder')},
#         'key_decoder': {'key_of_id': lambda x: x},
#         'key_encoder': {'id_of_key': lambda x: x},
#         'value_decoder': {'obj_of_data': lambda x: x},
#         'value_encoder': {'data_of_obj': lambda x: x},
#     }

#     for source_key, write_instructions in replacements.items():
#         if source_key in kwargs:
#             handle_replacements(source_key, write_instructions)
#             del kwargs[source_key]


@store_decorator
def add_decoder(store_cls=None, *, decoder: Callable = None, name=None):
    """Add a decoder layer to a store.

    Note: This is a convenience function for ``wrap_kvs(..., obj_of_data=decoder)``.

    >>> s = {'a': "42"}
    >>> ss = add_decoder(s, decoder=int)
    >>> ss['a']
    42

    If there's only one callable argument, it is assumed to be the decoder:

    >>> wrapper = add_decoder(int)
    >>> S = wrapper(dict)
    >>> sss = S({'a': "42"})
    >>> dict(sss) == {'a': 42}
    True
    """
    if decoder is None:
        if callable(store_cls):
            # assume the first argument is the decoder and return a wrapper using it
            return add_decoder(decoder=store_cls)
        else:
            raise ValueError(
                "If the decoder keyword is not given, the first argument must be the "
                "(callable) decoder"
            )
    return wrap_kvs(store_cls, obj_of_data=decoder, name=name)


class FirstArgIsMapping(LiteralVal):
    """A Literal class to mark a function as being one where the first argument is
    a mapping (store). This is intended to be used in wrappers such as ``wrap_kvs``
    to indicate when the first argument of a transformer function ``trans`` like
    ``key_of_id``, ``preset``, etc. is the store itself, therefore should be applied as
    ``trans(store, ...)`` instead of ``trans(...)``.
    """

    # TODO: Use this for it's intent!


def _wrap_kvs(
    store_cls: type,
    *,
    name=None,  # TODO: Remove when safe
    key_of_id=None,
    id_of_key=None,
    obj_of_data=None,
    data_of_obj=None,
    preset=None,
    postget=None,
    __module__=None,
    outcoming_key_methods=(),
    outcoming_value_methods=(),
    ingoing_key_methods=(),
    ingoing_value_methods=(),
    **kwargs,
):
    for method_name in {"_key_of_id"} | ensure_set(outcoming_key_methods):
        _wrap_outcoming(store_cls, method_name, key_of_id)

    for method_name in {"_obj_of_data"} | ensure_set(outcoming_value_methods):
        _wrap_outcoming(store_cls, method_name, obj_of_data)

    for method_name in {"_id_of_key"} | ensure_set(ingoing_key_methods):
        _wrap_ingoing(store_cls, method_name, id_of_key)

    for method_name in {"_data_of_obj"} | ensure_set(ingoing_value_methods):
        _wrap_ingoing(store_cls, method_name, data_of_obj)

    # TODO: postget and preset uses num_of_args. Not robust:
    #  Should only count args with no defaults or partial won't be able to be used to make postget/preset funcs
    # TODO: Extract postget and preset patterns?
    if postget is not None:
        if num_of_args(postget) < 2:
            raise ValueError(
                "A postget function needs to have (key, value) or (self, key, value) arguments"
            )

        if not _has_unbound_self(postget):

            def __getitem__(self, k):
                return postget(k, super(store_cls, self).__getitem__(k))

        else:

            def __getitem__(self, k):
                return postget(self, k, super(store_cls, self).__getitem__(k))

        store_cls.__getitem__ = __getitem__

    if preset is not None:
        if num_of_args(preset) < 2:
            raise ValueError(
                "A preset function needs to have (key, value) or (self, key, value) arguments"
            )

        if not _has_unbound_self(preset):

            def __setitem__(self, k, v):
                return super(store_cls, self).__setitem__(k, preset(k, v))

        else:

            def __setitem__(self, k, v):
                return super(store_cls, self).__setitem__(k, preset(self, k, v))

        store_cls.__setitem__ = __setitem__

    if __module__ is not None:
        store_cls.__module__ = __module__

    return store_cls


def _kv_wrap_outcoming_keys(trans_func):
    """Transform 'out-coming' keys, that is, the keys you see when you ask for them,
    say, through __iter__(), keys(), or first element of the items() pairs.

    Use this when you wouldn't use the keys in their original format,
    or when you want to extract information from it.

    Warning: If you haven't also wrapped incoming keys with a corresponding inverse transformation,
    you won't be able to use the outcoming keys to fetch data.

    >>> from collections import UserDict
    >>> S = kv_wrap.outcoming_keys(lambda x: x[5:])(UserDict)
    >>> s = S({'root/foo': 10, 'root/bar': 'xo'})
    >>> list(s)
    ['foo', 'bar']
    >>> list(s.keys())
    ['foo', 'bar']

    # TODO: Asymmetric key trans breaks getting items (therefore items()). Resolve (remove items() for asym keys?)
    # >>> list(s.items())
    # [('foo', 10), ('bar', 'xo')]
    """

    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_kr"
        )
        return wrap_kvs(o, name=name, key_of_id=trans_func)

    return wrapper


def _kv_wrap_ingoing_keys(trans_func):
    """Transform 'in-going' keys, that is, the keys you see when you ask for them,
    say, through __iter__(), keys(), or first element of the items() pairs.

    Use this when your context holds objects themselves holding key information, but you don't want to
    (because you shouldn't) 'manually' extract that information and construct the key manually every time you need
    to write something or fetch some existing data.

    Warning: If you haven't also wrapped outcoming keys with a corresponding inverse transformation,
    you won't be able to use the incoming keys to fetch data.

    >>> from collections import UserDict
    >>> S = kv_wrap.ingoing_keys(lambda x: 'root/' + x)(UserDict)
    >>> s = S()
    >>> s['foo'] = 10
    >>> s['bar'] = 'xo'
    >>> list(s)
    ['root/foo', 'root/bar']
    >>> list(s.keys())
    ['root/foo', 'root/bar']

    # TODO: Asymmetric key trans breaks getting items (therefore items()). Resolve (remove items() for asym keys?)
    # >>> list(s.items())
    # [('root/foo', 10), ('root/bar', 'xo')]
    """

    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_kw"
        )
        return wrap_kvs(o, name=name, id_of_key=trans_func)

    return wrapper


def _kv_wrap_outcoming_vals(trans_func):
    """Transform 'out-coming' values, that is, the values you see when you ask for them,
    say, through the values() or the second element of items() pairs.
    This can be seen as adding a de-serialization layer: trans_func being the de-serialization function.

    For example, say your store gives you values of the bytes type, but you want to use text, or gives you text,
    but you want it to be interpreted as a JSON formatted text and get a dict instead. Both of these are
    de-serialization layers, or out-coming value transformations.

    Warning: If it matters, make sure you also wrapped with a corresponding inverse serialization.

    >>> from collections import UserDict
    >>> S = kv_wrap.outcoming_vals(lambda x: x * 2)(UserDict)
    >>> s = S(foo=10, bar='xo')
    >>> list(s.values())
    [20, 'xoxo']
    >>> list(s.items())
    [('foo', 20), ('bar', 'xoxo')]
    """

    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_vr"
        )
        return wrap_kvs(o, name=name, obj_of_data=trans_func)

    return wrapper


def _kv_wrap_ingoing_vals(trans_func):
    """Transform 'in-going' values, that is, the values at the level of the store's interface are transformed
    to a different value before writing to the wrapped store.
    This can be seen as adding a serialization layer: trans_func being the serialization function.

    For example, say you have a list of audio samples, and you want to save these in a WAV format.

    Warning: If it matters, make sure you also wrapped with a corresponding inverse de-serialization.

    >>> from collections import UserDict
    >>> S = kv_wrap.ingoing_vals(lambda x: x * 2)(UserDict)
    >>> s = S()
    >>> s['foo'] = 10
    >>> s['bar'] = 'xo'
    >>> list(s.values())
    [20, 'xoxo']
    >>> list(s.items())
    [('foo', 20), ('bar', 'xoxo')]
    """

    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_vw"
        )
        return wrap_kvs(o, name=name, data_of_obj=trans_func)

    return wrapper


def _ingoing_vals_wrt_to_keys(trans_func):
    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_vwk"
        )
        return wrap_kvs(o, name=name, preset=trans_func)

    return wrapper


def _outcoming_vals_wrt_to_keys(trans_func):
    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_vrk"
        )
        return wrap_kvs(o, name=name, postget=trans_func)

    return wrapper


def mk_trans_obj(**kwargs):
    """Convenience method to quickly make a trans_obj (just an object holding some trans functions"""
    # TODO: Could make this more flexible (assuming here only staticmethods) and validate inputs...
    return type("TransObj", (), {k: staticmethod(v) for k, v in kwargs.items()})()


_kv_wrap_trans_names = {
    "_key_of_id",
    "_id_of_key",
    "_obj_of_data",
    "_data_of_obj",
    "_preset",
    "_postget",
}


class SimpleDelegator:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __call__(self, *args, **kwargs):
        if callable(self._obj):
            return self._obj(*args, **kwargs)
        else:
            raise TypeError(f"{self._obj=} is not callable")


def add_aliases(obj, **aliases):
    """A function that wraps the object instance and adds aliases.

    See also, and not to be confused with ``insert_aliases``, which adds aliases to
    dunder mapping methods (like ``__iter__``, ``__getitem__``) etc.

    """
    if not aliases:
        return obj
    else:
        new_obj = SimpleDelegator(obj)
        for attr, alias in aliases.items():
            setattr(new_obj, attr, getattr(new_obj, alias))
        return new_obj


def kv_wrap(trans_obj):
    """
    A function that makes a wrapper (a decorator) that will get the wrappers from
    methods of the input object.

    :param trans_obj: An object that contains (as attributes) the collection of
        transformation functions. The attribute names that are used, natively, to make
        the wrapper are ``_key_of_id``, ``_id_of_key``, ``_obj_of_data``,
        ``_data_of_obj``, ``_preset``, and ``_postget``.

    If your ``trans_obj`` uses different names for these functions, you can use the
    ``add_aliases`` function. We'll demo the use of ``add_aliases`` here:

    >>> from dol import kv_wrap, add_aliases, Pipe
    >>> from functools import partial
    >>>
    >>> class SeparatorTrans:
    ...     def __init__(self, sep: str):
    ...         self.sep = sep
    ...     def string_to_tuple(self, string: str):
    ...         return tuple(string.split(self.sep))
    ...     def tuple_to_string(self, tup: tuple):
    ...         return self.sep.join(tup)
    >>>
    >>> _add_aliases = partial(
    ...     add_aliases, _key_of_id='string_to_tuple', _id_of_key='tuple_to_string'
    ... )
    >>> mk_sep_trans = Pipe(SeparatorTrans, _add_aliases, kv_wrap)
    >>> sep_trans = mk_sep_trans('/')
    >>> d = sep_trans({'a/b/c': 1, 'd/e': 2})
    >>> list(d)
    [('a', 'b', 'c'), ('d', 'e')]
    >>> d['d', 'e']
    2

    ``kv_wrap`` also has convenience attributes:
        ``outcoming_keys``, ``ingoing_keys``, ``outcoming_vals``, ``ingoing_vals``,
        and ``val_reads_wrt_to_keys``
    which will only add a single specific wrapper (specified as a function),
    when that's what you need.

    """

    key_of_id = getattr(trans_obj, "_key_of_id", None)
    id_of_key = getattr(trans_obj, "_id_of_key", None)
    obj_of_data = getattr(trans_obj, "_obj_of_data", None)
    data_of_obj = getattr(trans_obj, "_data_of_obj", None)
    preset = getattr(trans_obj, "_preset", None)
    postget = getattr(trans_obj, "_postget", None)

    def wrapper(o, name=None):
        name = (
            name
            or getattr(o, "__qualname__", getattr(o.__class__, "__qualname__")) + "_kr"
        )
        return wrap_kvs(
            o,
            name=name,
            key_of_id=key_of_id,
            id_of_key=id_of_key,
            obj_of_data=obj_of_data,
            data_of_obj=data_of_obj,
            preset=preset,
            postget=postget,
        )

    return wrapper


kv_wrap.mk_trans_obj = mk_trans_obj  # to have a trans_obj maker handy
kv_wrap.outcoming_keys = _kv_wrap_outcoming_keys
kv_wrap.ingoing_keys = _kv_wrap_ingoing_keys
kv_wrap.outcoming_vals = _kv_wrap_outcoming_vals
kv_wrap.ingoing_vals = _kv_wrap_ingoing_vals
kv_wrap.ingoing_vals_wrt_to_keys = _ingoing_vals_wrt_to_keys
kv_wrap.outcoming_vals_wrt_to_keys = _outcoming_vals_wrt_to_keys


def mk_wrapper(wrap_cls):
    """

    You have a wrapper class and you want to make a wrapper out of it,
    that is, a decorator factory with which you can make wrappers, like this:
    ```
    wrapper = mk_wrapper(wrap_cls)
    ```
    that you can then use to transform stores like thiis:
    ```
    MyStore = wrapper(**wrapper_kwargs)(StoreYouWantToTransform)
    ```

    :param wrap_cls:
    :return:

    >>> class RelPath:
    ...     def __init__(self, root):
    ...         self.root = root
    ...         self._root_length = len(root)
    ...     def _key_of_id(self, _id):
    ...         return _id[self._root_length:]
    ...     def _id_of_key(self, k):
    ...         return self.root + k
    >>> relpath_wrap = mk_wrapper(RelPath)
    >>> RelDict = relpath_wrap(root='foo/')(dict)
    >>> s = RelDict()
    >>> s['bar'] = 42
    >>> assert list(s) == ['bar']
    >>> assert s['bar'] == 42
    >>> assert str(s) == "{'foo/bar': 42}"  # reveals that actually, behind the scenes, there's a "foo/" prefix
    """

    @wraps(wrap_cls)
    def wrapper(*args, **kwargs):
        return kv_wrap(wrap_cls(*args, **kwargs))

    return wrapper


@double_up_as_factory
def add_wrapper_method(wrap_cls=None, *, method_name="wrapper"):
    """Decorator that adds a wrapper method (itself a decorator) to a wrapping class
    Clear?
    See `mk_wrapper` function and doctest example if not.

    What `add_wrapper_method` does is just to add a `"wrapper"` method
    (or another name if you ask for it) to `wrap_cls`, so that you can use that
    class for it's purpose of transforming stores more conveniently.

    :param wrap_cls: The wrapper class (the definitioin of the transformation.
        If None, the functiion will make a decorator to decorate wrap_cls later
    :param method_name: The method name you want to use (default is 'wrapper')

    >>>
    >>> @add_wrapper_method
    ... class RelPath:
    ...     def __init__(self, root):
    ...         self.root = root
    ...         self._root_length = len(root)
    ...     def _key_of_id(self, _id):
    ...         return _id[self._root_length:]
    ...     def _id_of_key(self, k):
    ...         return self.root + k
    ...
    >>> RelDict = RelPath.wrapper(root='foo/')(dict)
    >>> s = RelDict()
    >>> s['bar'] = 42
    >>> assert list(s) == ['bar']
    >>> assert s['bar'] == 42
    >>> assert str(s) == "{'foo/bar': 42}"  # reveals that actually, behind the scenes, there's a "foo/" prefix
    """
    setattr(wrap_cls, method_name, mk_wrapper(wrap_cls))
    return wrap_cls


########################################################################################################################
# Aliasing

_method_name_for = {
    "write": "__setitem__",
    "read": "__getitem__",
    "delete": "__delitem__",
    "list": "__iter__",
    "count": "__len__",
}


def _conditional_data_trans(v, condition, data_trans):
    if condition(v):
        return data_trans(v)
    return v


@store_decorator
def conditional_data_trans(store=None, *, condition, data_trans):
    _data_trans = partial(
        _conditional_data_trans, condition=condition, data_trans=data_trans
    )
    return Pipe(data_trans, wrap_kvs(store, obj_of_data=_data_trans))


@store_decorator
def add_path_get(store=None, *, name=None, path_type: type = tuple):
    """
    Make nested stores accessible through key paths.
    In a way "flatten the nested keys access".
    By default, the path object will be a tuple (e.g. ``('a', 'b', 'c')``, but you can
    make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
    forms like ``'a.b.c'``, ``'a/b/c'``, etc.

    (Warning: ``path_type`` only effects the first level.
    That is, it doesn't work recursively.
    See issue: https://github.com/i2mint/dol/issues/10.)

    Say you have some nested stores.
    You know... like a `ZipFileReader` store whose values are `ZipReader`s,
    whose values are bytes of the zipped files
    (and you can go on... whose (json) values are...).

    For our example, let's take a nested dict instead:

    >>> s = {'a': {'b': {'c': 42}}}

    Well, you can access any node of this nested tree of stores like this:

    >>> s['a']['b']['c']
    42

    And that's fine. But maybe you'd like to do it this way instead:

    >>> s = add_path_get(s)
    >>> s['a', 'b', 'c']
    42

    You might also want to access 42 with `a.b.c` or `a/b/c` etc.
    To do that you can use `dol.paths.KeyPath` in combination with

    Args:
        store: The store (class or instance) you're wrapping.
            If not specified, the function will return a decorator.
        name: The name to give the class (not applicable to instance wrapping)
        path_type: The type that paths are expressed as. Needs to be an Iterable type.
            By default, a tuple.
            This is used to decide whether the key should be taken as a "normal"
            key of the store,
            or should be used to iterate through, recursively getting values.

    Returns:
        A wrapped store (class or instance), or a store wrapping decorator
        (if store is not specified)

    .. seealso::

        ``KeyPath`` in :doc:`paths`

    Wrapping an instance

    >>> s = add_path_get({'a': {'b': {'c': 42}}})
    >>> s['a']
    {'b': {'c': 42}}
    >>> s['a', 'b']
    {'c': 42}
    >>> s['a', 'b', 'c']
    42

    Wrapping a class

    >>> S = add_path_get(dict)
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == 42

    Using add_path_get as a decorator

    >>> @add_path_get
    ... class S(dict):
    ...    pass
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42

    A different kind of path?
    You can choose a different path_type, but sometimes (say both keys and key paths are strings)
    You need to involve more tools. Like dol.paths.KeyPath...

    >>> from dol.paths import KeyPath
    >>> from dol.trans import kv_wrap
    >>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
    >>> s = SS({'a': {'b': {'c': 42}}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a.b'] == s['a']['b'];
    >>> assert s['a.b.c'] == s['a']['b']['c']
    """
    name = name or store.__qualname__ + "WithPathGet"

    # TODO: This is not the best way to handle this. Investigate another way. ######################
    global_names = set(globals()).union(locals())
    if name in global_names:
        raise NameError("That name is already in use")
    # TODO: ########################################################################################

    store_cls = kv_wrap_persister_cls(store, name=name)
    store_cls._path_type = path_type

    def __getitem__(self, k):
        if isinstance(k, self._path_type):
            return reduce(lambda store, key: store[key], k, self)
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__getitem__(k)

    store_cls.__getitem__ = __getitem__

    return store_cls


# TODO: Should we keep add_path_get, or add "read_only" flag to add_path_access?
# TODO: See https://github.com/i2mint/dol/issues/10
@store_decorator
def add_path_access(store=None, *, name=None, path_type: type = tuple):
    """Make nested stores (read/write) accessible through key paths (iterable of keys).

    Like ``add_path_get``, but with write and delete accessible through key paths.

    In a way "flatten the nested keys access".
    (Warning: ``path_type`` only effects the first level.
    That is, it doesn't work recursively.
    See issue: https://github.com/i2mint/dol/issues/10.)

    By default, the path object will be a tuple (e.g. ``('a', 'b', 'c')``, but you can
    make it whatever you want, and/or use `dol.paths.KeyPath` to map to and from
    forms like ``'a.b.c'``, ``'a/b/c'``, etc.

    Say you have some nested stores.
    You know... like a `ZipFileReader` store whose values are `ZipReader`s,
    whose values are bytes of the zipped files
    (and you can go on... whose (json) values are...).

    For our example, let's take a nested dict instead:

    >>> s = {'a': {'b': {'c': 42}}}

    Well, you can access any node of this nested tree of stores like this:

    >>> s['a']['b']['c']
    42

    And that's fine. But maybe you'd like to do it this way instead:

    >>> s = add_path_access(s)
    >>> s['a', 'b', 'c']
    42

    So far, this is what ``add_path_get`` does. With ``add_path_access`` though you
    can also write and delete that way too:

    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    You might also want to access 42 with `a.b.c` or `a/b/c` etc.
    To do that you can use `dol.paths.KeyPath` in combination with

    Args:
        store: The store (class or instance) you're wrapping.
            If not specified, the function will return a decorator.
        name: The name to give the class (not applicable to instance wrapping)
        path_type: The type that paths are expressed as. Needs to be an Iterable type.
            By default, a tuple.
            This is used to decide whether the key should be taken as a "normal"
            key of the store,
            or should be used to iterate through, recursively getting values.

    Returns:
        A wrapped store (class or instance), or a store wrapping decorator
        (if store is not specified)

    .. seealso::

        ``KeyPath`` in :doc:`paths`

    Wrapping a class

    >>> S = add_path_access(dict)
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == 42
    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    Using add_path_get as a decorator

    >>> @add_path_access
    ... class S(dict):
    ...    pass
    >>> s = S(a={'b': {'c': 42}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a', 'b'] == s['a']['b'] == {'c': 42};
    >>> assert s['a', 'b', 'c'] == s['a']['b']['c'] == 42
    >>> s['a', 'b', 'c'] = 3.14
    >>> s['a', 'b', 'c']
    3.14
    >>> del s['a', 'b', 'c']
    >>> s
    {'a': {'b': {}}}

    A different kind of path?
    You can choose a different path_type, but sometimes (say both keys and key paths are strings)
    You need to involve more tools. Like dol.paths.KeyPath...

    >>> from dol.paths import KeyPath
    >>> from dol.trans import kv_wrap
    >>> SS = kv_wrap(KeyPath(path_sep='.'))(S)
    >>> s = SS({'a': {'b': {'c': 42}}})
    >>> assert s['a'] == {'b': {'c': 42}};
    >>> assert s['a.b'] == s['a']['b'];
    >>> assert s['a.b.c'] == s['a']['b']['c']
    >>> s['a.b.c'] = 3.14
    >>> s
    {'a': {'b': {'c': 3.14}}}
    >>> del s['a.b.c']
    >>> s
    {'a': {'b': {}}}

    Note: The add_path_access doesn't carry on to values.

    >>> s = add_path_access({'a': {'b': {'c': 42}}})
    >>> s['a', 'b', 'c']
    42
    >>> # but
    >>> s['a']['b', 'c']
    Traceback (most recent call last):
      ...
    KeyError: ('b', 'c')

    That said,

    >>> add_path_access(s['a'])['b', 'c']
    42

    The reason why we don't do this automatically is that it may not always be desirable.
    If one wanted to though, one could use ``wrap_kvs(obj_of_data=...)`` to wrap
    specific values with ``add_path_access``.
    For example, if you wanted to wrap all mappings recursively, you could:

    >>> from typing import Mapping
    >>> from dol.util import instance_checker
    >>> add_path_access_if_mapping = conditional_data_trans(
    ...     condition=instance_checker(Mapping), data_trans=add_path_access
    ... )
    >>> s = add_path_access_if_mapping({'a': {'b': {'c': 42}}})
    >>> s['a', 'b', 'c']
    42
    >>> # But now this works:
    >>> s['a']['b', 'c']
    42

    """
    store_cls = kv_wrap_persister_cls(store, name=name)
    store_cls = add_path_get(store_cls, name=name, path_type=path_type)

    def __setitem__(self, k, v):
        if isinstance(k, self._path_type):
            *path_head, last_key = k
            penultimate_level = reduce(lambda s, key: s[key], path_head, self)
            penultimate_level[last_key] = v
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__setitem__(k, v)

    def __delitem__(self, k):
        if isinstance(k, self._path_type):
            *path_head, last_key = k
            penultimate_level = reduce(lambda s, key: s[key], path_head, self)
            del penultimate_level[last_key]
        else:  # do things normally if the key is not a _path_type
            return super(store_cls, self).__delitem__(k)

    store_cls.__setitem__ = __setitem__
    store_cls.__delitem__ = __delitem__

    return store_cls


@store_decorator
def flatten(store=None, *, levels=None, cache_keys=False):
    """
    Flatten a nested store.

    Say you have a store that has three levels (or more), that is, that you can always
    ask for the value ``store[a][b][c]`` if ``a`` is a valid key of ``store``,
    ``b`` is a valid key of ``store[a]`` and ``c`` is a valid key of ``store[a][b]``.

    What ``flattened_store = flatten(store, levels=3)`` will give you is the ability
    to access the ``store[a][b][c]`` as ``store[a, b, c]``, while still being able
    to access these stores "normally".

    If that's all you need, you can just use the ``add_get_path`` wrapper for this.

    Why would you use ``flatten``? Because ``add_get_path(store)`` would still only
    give you the ``KvReader`` point of view of the root ``store``.
    If you ``list(store)``, you'd only get the first level keys,
    or if you ask if ``(a, b, c)`` is in the store, it will tell you it's not
    (though you can access data with such a key.

    Instead, a flattened store will consider that the keys are those ``(a, b, c)``
    key paths.

    Further, when flattening a store, you can ask for the view to cache the keys,
    specifying ``cache_keys=True`` or give it an explicit place to cache or
    factory to make a cache (see ``cached_keys`` wrapper for more details).
    Though caching keys is not the default it's highly recommended to do so in most
    cases. The only reason it is not the default is because if you have millions of
    keys, but little memory, that's not what you might want.

    Note: Flattening just provides a wrapper giving you a "flattened view". It doesn't
    change the store itself, or it's contents.

    :param store: The store instance or class to be wrapped
    :param levels: The number of nested levels to flatten
    :param cache_keys: Whether to cache the keys, or a cache factory or instance.

    >>> from dol import flatten
    >>> d = {
    ...     'a': {'b': {'c': 42}},
    ...     'aa': {'bb': {'cc': 'dragon_con'}}
    ... }

    You can get a flattened view of an instance:

    >>> m = flatten(d, levels=3, cache_keys=True)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a', 'b', 'c'), 42),
    ...             (('aa', 'bb', 'cc'), 'dragon_con')
    ...         ]
    ... )

    You can make a flattener and apply it to an instance (or a class):

    >>> my_flattener = flatten(levels=2)
    >>> m = my_flattener(d)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a', 'b'), {'c': 42}),
    ...             (('aa', 'bb'), {'cc': 'dragon_con'})
    ...         ]
    ... )

    Finally, you can wrap a class itself.

    >>> @flatten(levels=1)
    ... class MyFlatDict(dict):
    ...     pass
    >>> m = MyFlatDict(d)
    >>> assert (
    ...         list(m.items())
    ...         == [
    ...             (('a',), {'b': {'c': 42}}),
    ...             (('aa',), {'bb': {'cc': 'dragon_con'}})
    ...         ]
    ... )
    """
    return _wrap_store(_flatten, locals())
    #
    # arguments = {k: v for k, v in locals().items() if k != "arguments"}
    # store = arguments.pop("store")
    #
    # class_trans = partial(_flatten, **arguments)
    # return Store.wrap(store, class_trans=class_trans)


def _flatten(store, *, levels, cache_keys):
    store._levels = levels

    def __iter__(self):
        yield from leveled_paths_walk(self.store, self._levels)

    store.__iter__ = __iter__

    if cache_keys:
        if cache_keys is True:
            cache_keys = list
        return add_path_get(cached_keys(store, keys_cache=cache_keys))
    else:

        def __len__(self):
            i = 0
            for i, _ in enumerate(self, 1):
                pass
            return i

        def __contains__(self, k):
            if isinstance(k, tuple):
                assert len(k) < self._levels
                return super(store, self).__contains__(k[0]) and all(
                    k[i] in self[k[i - 1]] for i in range(1, self._levels)
                )
            else:
                return super(store, self).__contains__(k)

        store.__len__ = __len__
        store.__contains__ = __contains__

        # TODO: This adds read access to all levels, not limited to levels
        return add_path_get(store)


def mk_level_walk_filt(levels):
    """Makes a ``walk_filt`` function for ``kv_walk`` based on some level logic.
    If ``levels`` is an integer, will consider it as the max path length,
    if not it will just assert that ``levels`` is callable, and return it
    """
    if isinstance(levels, int):
        return lambda p, k, v: len(p) < levels - 1
    else:
        assert callable(levels), f"levels must be a callable or an integer: {levels=}"
        return levels


def leveled_paths_walk(m, levels):
    yield from kv_walk(
        m, leaf_yield=lambda p, k, v: p, walk_filt=mk_level_walk_filt(levels)
    )


def _insert_alias(store, method_name, alias=None):
    if isinstance(alias, str) and hasattr(store, method_name):
        setattr(store, alias, getattr(store, method_name))


@store_decorator
def insert_aliases(
    store=None, *, write=None, read=None, delete=None, list=None, count=None
):
    """Insert method aliases of CRUD operations of a store (class or instance).
    If store is a class, you'll get a copy of the class with those methods added.
    If store is an instance, the methods will be added in place (no copy will be made).

    Note: If an operation (write, read, delete, list, count) is not specified, no alias will be created for
    that operation.

    IMPORTANT NOTE: The signatures of the methods the aliases will point to will not change.
    We say this because, you can call the write method "dump", but you'll have to use it as
    `store.dump(key, val)`, not `store.dump(val, key)`, which is the signature you're probably used to
    (it's the one used by json.dump or pickle.dump for example). If you want that familiar interface,
    using the insert_load_dump_aliases function.

    See also (and not to be confused with): ``add_aliases``

    Args:
        store: The store to extend with aliases.
        write: Desired method name for __setitem__
        read: Desired method name for __getitem__
        delete: Desired method name for __delitem__
        list: Desired method name for __iter__
        count: Desired method name for __len__

    Returns: A store with the desired aliases.

    >>> # Example of extending a class
    >>> mydict = insert_aliases(dict, write='dump', read='load', delete='rm', list='peek', count='size')
    >>> s = mydict(true='love')
    >>> s.dump('friends', 'forever')
    >>> s
    {'true': 'love', 'friends': 'forever'}
    >>> s.load('true')
    'love'
    >>> list(s.peek())
    ['true', 'friends']
    >>> s.size()
    2
    >>> s.rm('true')
    >>> s
    {'friends': 'forever'}
    >>>
    >>> # Example of extending an instance
    >>> from collections import UserDict
    >>> s = UserDict(true='love')  # make (and instance) of a UserDict (can't modify a dict instance)
    >>> # make aliases of note that you don't need
    >>> s = insert_aliases(s, write='put', read='retrieve', count='num_of_items')
    >>> s.put('friends', 'forever')
    >>> s
    {'true': 'love', 'friends': 'forever'}
    >>> s.retrieve('true')
    'love'
    >>> s.num_of_items()
    2
    """

    if isinstance(store, type):
        store = type(store.__qualname__, (store,), {})
    for alias, method_name in _method_name_for.items():
        _insert_alias(store, method_name, alias=locals().get(alias))
    return store


@store_decorator
def insert_load_dump_aliases(store=None, *, delete=None, list=None, count=None):
    """Insert load and dump methods, with familiar dump(obj, location) signature.

    Args:
        store: The store to extend with aliases.
        delete: Desired method name for __delitem__
        list: Desired method name for __iter__
        count: Desired method name for __len__

    Returns: A store with the desired aliases.

    >>> mydict = insert_load_dump_aliases(dict)
    >>> s = mydict()
    >>> s.dump(obj='love', key='true')
    >>> s
    {'true': 'love'}
    """
    store = insert_aliases(store, read="load", delete=delete, list=list, count=count)

    def dump(self, obj, key):
        return self.__setitem__(key, obj)

    if isinstance(store, type):
        store.dump = dump
    else:
        store.dump = types.MethodType(dump, store)

    return store


from typing import TypeVar, Any
from collections.abc import Callable

FuncInput = TypeVar("FuncInput")
FuncOutput = TypeVar("FuncOutput")


def constant_output(return_val=None, *args, **kwargs):
    """Function that returns a constant value no matter what the inputs are.
    Is meant to be used with functools.partial to create custom versions.

    >>> from functools import partial
    >>> always_true = partial(constant_output, True)
    >>> always_true('regardless', 'of', the='input', will='return True')
    True

    """
    return return_val


@double_up_as_factory
def condition_function_call(
    func: Callable[[FuncInput], FuncOutput] = None,
    *,
    condition: Callable[[FuncInput], bool] = partial(constant_output, True),
    callback_if_condition_not_met: Callable[[FuncInput], Any] = partial(
        constant_output, None
    ),
):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if condition(*args, **kwargs):
            return func(*args, **kwargs)
        else:
            return callback_if_condition_not_met(*args, **kwargs)

    return wrapped_func


from typing import Any
from collections.abc import Callable, MutableMapping

Key = Any
Val = Any
SetitemCondition = Callable[[MutableMapping, Key, Val], bool]

# @store_decorator
# def only_allow_writes_that_obey_condition(*,
#                                           write_condition: SetitemCondition,
#                                           msg='Write arguments did not match condition.'):
#     def _only_allow_writes_that_obey_condition(store: MutableMapping):
#         pass


from dol.util import (
    has_enabled_clear_method,
    inject_method,
    _delete_keys_one_by_one,
)

InjectionValidator = Callable[[type, Callable], bool]


@double_up_as_factory
def ensure_clear_method(store=None, *, clear_method=_delete_keys_one_by_one):
    """If obj doesn't have an enabled clear method, will add one (a slow one that runs through keys and deletes them"""
    if not has_enabled_clear_method(store):
        inject_method(store, clear_method, "clear")
    return store


@store_decorator
def add_store_method(
    store: type,
    *,
    method_func,
    method_name=None,
    validator: InjectionValidator | None = None,
):
    """Add methods to store classes or instances

    :param store: A store type or instance
    :param method_func: The function of the method to be added
    :param method_name: The name of the store attribute this function should be written to
    :param validator: An optional validator. If not None, ``validator(store, method_func)`` will be called.
        If it doesn't return True, a ``SetattrNotAllowed`` will be raised.
        Note that ``validator`` can also raise its own exception.
    :return: A store with the added (or modified) method
    """
    method_name = method_name or method_func.__name__
    if validator is not None:
        if not validator(store, method_func):
            raise SetattrNotAllowed(
                f"Method is not allowed to be set (according to {validator}): {method_func}"
            )

    @wraps(store, updated=())
    class StoreWithAddedMethods(store):
        pass

    setattr(StoreWithAddedMethods, method_name, method_func)
    return StoreWithAddedMethods


class MapInvertabilityError(ValueError):
    """To be used to indicate that a mapping isn't, or wouldn't be, invertible"""


class CachedInvertibleTrans:
    """
    >>> t = CachedInvertibleTrans(lambda x: x[1])
    >>> t.ingress('ab')
    'b'
    >>> t.ingress((1, 2))
    2
    >>> t.egress('b')
    'ab'
    >>> t.egress(2)
    (1, 2)
    """

    def __init__(self, trans_func: Callable):
        self.trans_func = trans_func
        self.ingress_map = dict()
        self.egress_map = dict()

    def ingress(self, x):
        if x not in self.ingress_map:
            y = self.trans_func(x)
            self.ingress_map[x] = y
            if y in self.egress_map:
                raise MapInvertabilityError(
                    f"egress_map (the inverse map) already had key: {y}"
                )
            self.egress_map[y] = x
            return y
        else:
            return self.ingress_map[x]

    def egress(self, y):
        return self.egress_map[y]


def assert_min_num_of_args(func: Callable, num_of_args: int):
    """Assert that a function can be a store method.
    That is, it should have a signature that takes the store as the first argument
    """
    try:
        assert (
            len(Sig(func).parameters) >= num_of_args
        ), f"Function {func} doesn't have at least {num_of_args} arguments"
    except Exception as e:
        warn(
            f"Encountered error checking if {func} can be a store method. "
            "Will return True, to not disrupt process, but you may want to check on "
            f"this : {e}"
        )


@store_decorator
def add_missing_key_handling(store=None, *, missing_key_callback: Callable):
    """Overrides the ``__missing__`` method of a store with a custom callback.

    The callback must have two arguments: the store and the key.

    In the following example, we endow a store to return a sub-store when a key is
    missing. This substore will contain only keys that start with that missing key.
    This is useful, for example, to get "subfolder filtering" on a store.

    >>> def prefix_filter(store, prefix: str):
    ...     '''Filter the store to have only keys that start with prefix'''
    ...     from dol import filt_iter
    ...     return filt_iter(store, filt=lambda x: x.startswith(prefix))
    ...
    >>> @add_missing_key_handling(missing_key_callback=prefix_filter)
    ... class D(dict):
    ...     pass
    >>>
    >>> s = D({'a/b': 1, 'a/c': 2, 'd/e': 3, 'f': 4})
    >>> sorted(s)
    ['a/b', 'a/c', 'd/e', 'f']
    >>> 'a/' not in s
    True
    >>> # yet
    >>> v = s['a/']
    >>> assert dict(v) == {'a/b': 1, 'a/c': 2}
    """

    assert_min_num_of_args(missing_key_callback, 2)

    @wraps(store, updated=())
    class StoreWithMissingKeyCallaback(store):
        pass

    StoreWithMissingKeyCallaback.__missing__ = missing_key_callback
    return StoreWithMissingKeyCallaback


EncodedType = TypeVar("EncodedType")
DecodedType = TypeVar("DecodedType")


# TODO: Want a way to specify Encoded type and Decoded type
@dataclass
class Codec(Generic[DecodedType, EncodedType]):
    encoder: Callable[[DecodedType], EncodedType]
    decoder: Callable[[EncodedType], DecodedType]

    def __iter__(self):
        return iter((self.encoder, self.decoder))

    def compose_with(self, other):
        cls = type(self)
        return cls(
            encoder=Pipe(self.encoder, other.encoder),
            decoder=Pipe(other.decoder, self.decoder),
        )

    def invert(self):
        """Return a codec that is the inverse of this one.
        That is, encoder and decoder will be swapped."""
        cls = type(self)
        return cls(encoder=self.decoder, decoder=self.encoder)

    # operators
    __add__ = compose_with
    __invert__ = invert


_CodecT = (Generic[DecodedType, EncodedType], Codec[DecodedType, EncodedType])


class ValueCodec(*_CodecT):
    def __call__(self, obj):
        return wrap_kvs(obj, data_of_obj=self.encoder, obj_of_data=self.decoder)


class KeyCodec(*_CodecT):
    def __call__(self, obj):
        return wrap_kvs(obj, id_of_key=self.encoder, key_of_id=self.decoder)


class KeyValueCodec(*_CodecT):
    def __call__(self, obj):
        return wrap_kvs(obj, preset=self.encoder, postget=self.decoder)


# Note: An affix is a morpheme that is attached to a word stem to form a new word or
# word form. Affixes include prefixes, suffixes, infixes, and circumfixes.


def _affix_encoder(string: str, prefix: str = "", suffix: str = ""):
    """Affix a prefix and suffix to a string
    >>> _affix_encoder('name', prefix='/folder/', suffix='.txt')
    '/folder/name.txt'
    """
    return f"{prefix}{string}{suffix}"


def _affix_decoder(string: str, prefix: str = "", suffix: str = ""):
    """Remove prefix and suffix from string
    >>> _affix_decoder('/folder/name.txt', prefix='/folder/', suffix='.txt')
    'name'
    """
    end_idx = -len(suffix) or None  # if suffix is empty, end_idx should be None
    return string[len(prefix) : end_idx]


def affix_key_codec(prefix: str = "", suffix: str = ""):
    """A factory that creates a key codec that affixes a prefix and suffix to the key

    >>> codec = affix_key_codec(prefix='/folder/', suffix='.txt')
    >>> codec.encoder('name')
    '/folder/name.txt'
    >>> codec.decoder('/folder/name.txt')
    'name'
    """
    return KeyCodec(
        encoder=partial(_affix_encoder, prefix=prefix, suffix=suffix),
        decoder=partial(_affix_decoder, prefix=prefix, suffix=suffix),
    )


@store_decorator
def redirect_getattr_to_getitem(cls=None, *, keys_have_priority_over_attributes=False):
    """A mapping decorator that redirects attribute access to __getitem__.

    Warning: This decorator will make your class un-pickleable.

    :param keys_have_priority_over_attributes: If True, keys will have priority over existing attributes.

    >>> @redirect_getattr_to_getitem
    ... class MyDict(dict):
    ...     pass
    >>> d = MyDict(a=1, b=2)
    >>> d.a
    1
    >>> d.b
    2
    >>> list(d)
    ['a', 'b']

    """

    class RidirectGetattrToGetitem(cls):
        """A class that redirects attribute access to __getitem__"""

        _keys_have_priority_over_attributes = keys_have_priority_over_attributes

        def __getattr__(self, attr):
            if attr in self:
                if self._keys_have_priority_over_attributes or attr not in dir(
                    type(self)
                ):
                    return self[attr]
            # if attr not in self, or if it is in the class, then do normal getattr
            return super().__getattr__(attr)

        def __dir__(self) -> Iterable[str]:
            return list(self)

    return RidirectGetattrToGetitem
```

## util.py

```python
"""General util objects"""

import os
import shutil
import re
import platform
from collections import deque, namedtuple, defaultdict
from warnings import warn

from typing import (
    Any,
    Optional,
    Union,
    T,
    KT,
    NewType,
    Tuple,
    TypeVar,
)
from collections.abc import Hashable, Callable, Iterable, Mapping, Sequence, Container
from functools import update_wrapper as _update_wrapper
from functools import wraps as _wraps
from functools import partialmethod, partial, WRAPPER_ASSIGNMENTS
from types import MethodType, FunctionType
from inspect import Signature, signature, Parameter, getsource, ismethod


Key = TypeVar("Key")
Key.__doc__ = "The type of the keys used in the interface (outer keys)"
Id = TypeVar("Id")
Id.__doc__ = "The type of the keys used in the backend (inner keys)"
Val = TypeVar("Val")
Val.__doc__ = "The type of the values used in the interface (outer values)"
Data = TypeVar("Data")
Data.__doc__ = "The type of the values used in the backend (inner values)"
Item = tuple[Key, Val]
KeyIter = Iterable[Key]
ValIter = Iterable[Val]
ItemIter = Iterable[Item]

# monkey patching WRAPPER_ASSIGNMENTS to get "proper" wrapping (adding defaults and kwdefaults
wrapper_assignments = (*WRAPPER_ASSIGNMENTS, "__defaults__", "__kwdefaults__")

update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)

exhaust = partial(deque, maxlen=0)


def non_colliding_key(
    key: KT,
    exclude: Container[KT],
    *,
    collision_handler: Callable[[KT, int], KT] = None,
    max_attempts: int = 10000,
) -> KT:
    """
    Return a key not present in the exclude container.

    If the input key is already unique, it's returned as-is.
    Otherwise, applies a collision_handler until a unique key is found.

    Args:
        key: The candidate key to check/modify
        exclude: Container of keys to avoid
        collision_handler: Function taking (key, attempt_number) and returning a modified key.
                          For strings, defaults to appending " (N)" suffix before extension.
                          For other types, must be provided.
        max_attempts: Maximum number of transformation attempts

    Returns:
        A key not present in the exclude container

    Raises:
        ValueError: If no unique key found within max_attempts, or if collision_handler
                   is None for non-string keys

    >>> non_colliding_key("file.txt", set())
    'file.txt'
    >>> non_colliding_key("file.txt", {"file.txt"})
    'file (1).txt'
    >>> non_colliding_key("file.txt", {"file.txt", "file (1).txt"})
    'file (2).txt'
    >>> non_colliding_key(42, {42}, collision_handler=lambda k, n: k + n)
    43

    """
    if key not in exclude:
        return key

    if collision_handler is None:
        if isinstance(key, str):
            collision_handler = _default_string_collision_handler
        else:
            raise ValueError(
                f"collision_handler must be provided for non-string keys (got {type(key).__name__})"
            )

    for attempt in range(1, max_attempts + 1):
        candidate = collision_handler(key, attempt)
        if candidate not in exclude:
            return candidate

    raise ValueError(f"Could not find unique key after {max_attempts} attempts")


def _default_string_collision_handler(string: str, attempt: int) -> str:
    """
    Default collision handler for strings: insert " (N)" before the file extension.

    >>> _default_string_collision_handler("file.txt", 1)
    'file (1).txt'
    >>> _default_string_collision_handler("no_extension", 2)
    'no_extension (2)'
    """
    if "." in string:
        parts = string.rsplit(".", 1)
        return f"{parts[0]} ({attempt}).{parts[1]}"
    return f"{string} ({attempt})"


def safe_compile(path, normalize_path=True):
    r"""
    Safely compiles a file path into a regex pattern, ensuring compatibility
    across different operating systems (Windows, macOS, Linux).

    This function normalizes the input path to use the correct separators
    for the current platform and escapes any special characters to avoid
    invalid regex patterns.

    Args:
        path (str): The file path to be compiled into a regex pattern.

    Returns:
        re.Pattern: A compiled regular expression object for the given path.

    Examples:
        >>> regex = safe_compile(r"C:\\what\\happens\\if\\you\\escape")
        >>> regex.pattern  # Windows path is escaped properly
        'C:\\\\what\\\\happens\\\\if\\\\you\\\\escape'

        >>> regex = safe_compile("/fun/paths/are/awesome")
        >>> regex.pattern  # Unix path is unmodified
        '/fun/paths/are/awesome'
    """
    if normalize_path:
        # Normalize the path to handle cross-platform differences
        path = os.path.normpath(path)
    if platform.system() == "Windows":
        # Escape backslashes for Windows paths
        path = re.escape(path)
    return re.compile(path)


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.
def identity_func(x: T) -> T:
    return x


static_identity_method = staticmethod(identity_func)


def named_partial(func, *args, __name__=None, **keywords):
    """functools.partial, but with a __name__

    >>> f = named_partial(print, sep='\\n')
    >>> f.__name__
    'print'

    >>> f = named_partial(print, sep='\\n', __name__='now_partial_has_a_name')
    >>> f.__name__
    'now_partial_has_a_name'
    """
    f = partial(func, *args, **keywords)
    f.__name__ = __name__ or func.__name__
    return f


def is_classmethod(obj):
    """Checks if an object is a classmethod.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a classmethod, False otherwise.

    Example usage:

    >>> class MyClass:
    ...     @classmethod
    ...     def class_method(cls):
    ...         pass
    ...
    ...     def instance_method(self):
    ...         pass
    >>> obj1 = MyClass.class_method
    >>> obj2 = MyClass().instance_method
    >>> is_classmethod(obj1)
    True
    >>> is_classmethod(obj2)
    False
    """

    return ismethod(obj) and isinstance(obj.__self__, type)


def is_unbound_method(obj):
    """
    Determines if the given object is an unbound method.

    Args:
        obj: The object to check.

    Returns:
        True if obj is an unbound method, False otherwise.

    Examples:
        >>> import sys
        >>> import types
        >>> def function():
        ...     pass
        >>> class MyClass:
        ...     def method(self):
        ...         pass
        >>> is_unbound_method(MyClass.method)
        True
        >>> is_unbound_method(MyClass().method)
        False
        >>> is_unbound_method(function)
        False
    """
    if not isinstance(obj, FunctionType):
        return False
    qualname = getattr(obj, "__qualname__", "")
    # if '<locals>' in qualname:
    #     return False
    return "." in qualname


class staticproperty:
    """A decorator for defining static properties in classes.

    >>> class A:
    ...     @staticproperty
    ...     def foo():
    ...         return 2
    >>> A.foo
    2
    >>> A().foo
    2
    """

    def __init__(self, function):
        self.function = function

    def __get__(self, obj, owner=None):
        return self.function()


def add_as_attribute_of(obj, name=None):
    """Decorator that adds a function as an attribute of a container object ``obj``.

    If no ``name`` is given, the ``__name__`` of the function will be used, with a
    leading underscore removed. This is useful for adding helper functions to main
    "container" functions without polluting the namespace of the module, at least
    from the point of view of imports and tab completion.

    >>> def foo():
    ...    pass
    >>>
    >>> @add_as_attribute_of(foo)
    ... def _helper():
    ...    pass
    >>> hasattr(foo, 'helper')
    True
    >>> callable(foo.helper)
    True

    In reality, any object that has a ``__name__`` can be added to the attribute of
    ``obj``, but the intention is to add helper functions to main "container" functions.

    """

    def _decorator(f):
        attrname = name or f.__name__
        if attrname.startswith("_"):
            attrname = attrname[1:]  # remove leading underscore
        setattr(obj, attrname, f)
        return f

    return _decorator


def chain_get(d: Mapping, keys, default=None):
    """
    Returns the ``d[key]`` value for the first ``key`` in ``keys`` that is in ``d``, and default if none are found

    Note: Think of ``collections.ChainMap`` where you can look for a single key in a sequence of maps until we find it.
    Here we look for a sequence of keys in a single map, stopping as soon as we find a key that the map has.

    >>> d = {'here': '&', 'there': 'and', 'every': 'where'}
    >>> chain_get(d, ['not there', 'not there either', 'there', 'every'])
    'and'

    Notice how ``'not there'`` and ``'not there either'`` are skipped, ``'there'`` is found and used to retrieve
    the value, and ``'every'`` is not even checked (because ``'there'`` was found).
    If non of the keys are found, ``None`` is returned by default.

    >>> assert chain_get(d, ('none', 'of', 'these')) is None

    You can change this default though:

    >>> chain_get(d, ('none', 'of', 'these'), default='Not Found')
    'Not Found'

    """
    for key in keys:
        if key in d:
            return d[key]
    return default


class LiteralVal:
    """An object to indicate that the value should be considered literally.

    >>> t = LiteralVal(42)
    >>> t.get_val()
    42
    >>> t()
    42

    """

    def __init__(self, val):
        self.val = val

    def get_val(self):
        """Get the value wrapped by LiteralVal instance.

        One might want to use ``literal.get_val()`` instead ``literal()`` to get the
        value a ``LiteralVal`` is wrapping because ``.get_val`` is more explicit.

        That said, with a bit of hesitation, we allow the ``literal()`` form as well
        since it is useful in situations where we need to use a callback function to
        get a value.
        """
        return self.val

    __call__ = get_val

    # def __get__(self, instance, owner):
    #     return self.val


# TODO: The a.big() test is skipped because fails in doctest. It should be fixed.
def decorate_callables(decorator, cls=None):
    """Decorate all (non-underscored) callables in a class with a decorator.

    >>> from dol.util import LiteralVal
    >>> @decorate_callables(property)
    ... class A:
    ...     def wet(self):
    ...         return 'dry'
    ...     @LiteralVal
    ...     def big(self):
    ...         return 'small'
    >>> a = A()
    >>> a.wet
    'dry'
    >>> a.big()  # doctest: +SKIP
    'small'

    """
    if cls is None:
        return partial(decorate_callables, decorator)
    for name, attr in vars(cls).items():
        if isinstance(attr, LiteralVal):
            setattr(cls, name, attr.get_val())
        elif not name.startswith("_") and callable(attr):
            setattr(cls, name, decorator(attr))
    return cls


# class LiteralVal:
#     """
#     An object to indicate that the value should be considered literally.

#     >>> t = LiteralVal(42)
#     >>> t.get_val()
#     42
#     >>> t()
#     42

#     >>> class A:
#     ...     @LiteralVal
#     ...     def value(self):
#     ...         return 42
#     >>> a = A()
#     >>> a.value
#     42
#     """

#     def __init__(self, val):
#         if callable(val):
#             self.val = val()
#         else:
#             self.val = val

#     def get_val(self):
#         """Get the value wrapped by LiteralVal instance."""
#         return self.val

#     def __call__(self):
#         return self.get_val()

#     def __get__(self, instance, owner):
#         return self.val

# def decorate_callables(decorator, cls=None):
#     """
#     Decorate all (non-underscored) callables in a class with a decorator.

#     >>> @decorate_callables(property)
#     ... class A:
#     ...     def wet(self):
#     ...         return 'dry'
#     ...     @LiteralVal
#     ...     def big(self):
#     ...         return 'small'
#     >>> a = A()
#     >>> a.wet
#     'dry'
#     >>> a.big
#     'small'
#     """
#     if cls is None:
#         return partial(decorate_callables, decorator)
#     for name, attr in vars(cls).items():
#         if isinstance(attr, LiteralVal):
#             setattr(cls, name, property(attr.get_val))
#         elif not name.startswith('_') and callable(attr):
#             setattr(cls, name, decorator(attr))
#     return cls


def _isinstance(obj, class_or_tuple):
    """The same as the builtin isinstance, but without the position only restriction,
    allowing us to use partial to define filter functions for specific types
    """
    return isinstance(obj, class_or_tuple)


def instance_checker(*types):
    """Makes a filter function that checks the type of an object.

    >>> f = instance_checker(int, float)
    >>> f(1)
    True
    >>> f(1.0)
    True
    >>> f('1.0')
    False
    """
    return partial(_isinstance, class_or_tuple=types)


def not_a_mac_junk_path(path: str):
    """A function that will tell you if the path is not a mac junk path/
    More precisely, doesn't end with '.DS_Store' or have a `__MACOSX` folder somewhere
    on it's way.

    This is usually meant to be used with `filter` or `filt_iter` to "filter in" only
    those actually wanted files (not the junk that mac writes to your filesystem).

    These files annoyingly show up often in zip files, and are usually unwanted.

    See https://apple.stackexchange.com/questions/239578/compress-without-ds-store-and-macosx

    >>> paths = ['A/normal/path', 'A/__MACOSX/path', 'path/ending/in/.DS_Store', 'foo/b']
    >>> list(filter(not_a_mac_junk_path, paths))
    ['A/normal/path', 'foo/b']
    """
    if path.endswith(".DS_Store") or "__MACOSX" in path.split(os.path.sep):
        return False  # This is indeed math junk (so filter out)
    return True  # this is not mac junk (you can keep it)


def inject_method(obj, method_function, method_name=None):
    """
    method_function could be:
        * a function
        * a {method_name: function, ...} dict (for multiple injections)
        * a list of functions or (function, method_name) pairs
    """
    if method_name is None:
        method_name = method_function.__name__
    assert callable(
        method_function
    ), f"method_function (the second argument) is supposed to be a callable!"
    assert isinstance(
        method_name, str
    ), f"method_name (the third argument) is supposed to be a string!"
    if not isinstance(obj, type):
        method_function = MethodType(method_function, obj)
    setattr(obj, method_name, method_function)
    return obj


def _disabled_clear_method(self):
    """The clear method is disabled to make dangerous difficult.
    You don't want to delete your whole DB
    If you really want to delete all your data, you can do so by doing something like this:

    .. code-block:: python

        for k in self:
            del self[k]


    or (in some cases)

    .. code-block:: python

        for k in self:
            try:
                del self[k]
            except KeyError:
                pass

    """
    raise NotImplementedError(f"Instance of {type(self)}: {self.clear.__doc__}")


# to be able to check if clear is disabled (see ensure_clear_method function for example):
_disabled_clear_method.disabled = True


def has_enabled_clear_method(store):
    """Returns True iff obj has a clear method that is enabled (i.e. not disabled)"""
    return hasattr(store, "clear") and (  # has a clear method...
        not hasattr(store.clear, "disabled")  # that doesn't have a disabled attribute
        or not store.clear.disabled
    )  # ... or if it does, than it must not be == True


def _delete_keys_one_by_one(self):
    """clear the entire store (delete all keys)"""
    for k in self:
        del self[k]


def _delete_keys_one_by_one_with_keyerror_supressed(self):
    """clear the entire store (delete all keys), ignoring KeyErrors"""
    for k in self:
        try:
            del self[k]
        except KeyError:
            pass


_delete_keys_one_by_one.disabled = False
_delete_keys_one_by_one_with_keyerror_supressed.disabled = False


# Note: Vendored in i2.multi_objects and lkj.strings
def truncate_string_with_marker(
    s, *, left_limit=15, right_limit=15, middle_marker="..."
):
    """
    Return a string with a limited length.

    If the string is longer than the sum of the left_limit and right_limit,
    the string is truncated and the middle_marker is inserted in the middle.

    If the string is shorter than the sum of the left_limit and right_limit,
    the string is returned as is.

    >>> truncate_string_with_marker('1234567890')
    '1234567890'

    But if the string is longer than the sum of the limits, it is truncated:

    >>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=3)
    '123...890'
    >>> truncate_string_with_marker('1234567890', left_limit=3, right_limit=0)
    '123...'
    >>> truncate_string_with_marker('1234567890', left_limit=0, right_limit=3)
    '...890'

    If you're using a specific parametrization of the function often, you can
    create a partial function with the desired parameters:

    >>> from functools import partial
    >>> truncate_string = partial(truncate_string_with_marker, left_limit=2, right_limit=2, middle_marker='---')
    >>> truncate_string('1234567890')
    '12---90'
    >>> truncate_string('supercalifragilisticexpialidocious')
    'su---us'

    """
    middle_marker_len = len(middle_marker)
    if len(s) <= left_limit + right_limit:
        return s
    elif right_limit == 0:
        return s[:left_limit] + middle_marker
    elif left_limit == 0:
        return middle_marker + s[-right_limit:]
    else:
        return s[:left_limit] + middle_marker + s[-right_limit:]


def signature_string_or_default(func, default="(-no signature-)"):
    try:
        return str(signature(func))
    except ValueError:
        return default


def function_info_string(func: Callable):
    func_name = getattr(func, "__name__", str(func))
    if func_name == "<lambda>":
        return f"a lambda function on {signature(func)}"
    return f"{func_name}{signature_string_or_default(func)}"


# Note: Pipe code is completely independent (with inspect imports signature & Signature)
#  If you only need simple pipelines, use this, or even copy/paste it where needed.
# TODO: Public interface mis-aligned with i2. funcs list here, in i2 it's dict. Align?
#  If we do so, it would be a breaking change since any dependents that expect funcs
#  to be a list of funcs will iterate over a iterable of names instead.
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5
    >>> len(f)
    2

    You can name functions, but this would just be for documentation purposes.
    The names are completely ignored.

    >>> g = Pipe(
    ...     add_numbers = lambda x, y: x + y,
    ...     multiply_by_2 = lambda x: x * 2,
    ...     stringify = str
    ... )
    >>> g(2, 3)
    '10'
    >>> len(g)
    3

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)

    You can specify a single functions:

    >>> Pipe(lambda x: x + 1)(2)
    3

    but

    >>> Pipe()
    Traceback (most recent call last):
      ...
    ValueError: You need to specify at least one function!

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as function names):

    >>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
    >>> f(lambda x: x * 10, [1, 2, 3])
    60
    >>> f.__name__
    'map_and_sum'
    >>> f.__doc__
    'Apply func and add'

    """

    funcs = ()

    def __init__(self, *funcs, **named_funcs):
        named_funcs = self._process_reserved_names(named_funcs)
        funcs = list(funcs) + list(named_funcs.values())
        self.funcs = funcs
        n_funcs = len(funcs)
        if n_funcs == 0:
            raise ValueError("You need to specify at least one function!")

        elif n_funcs == 1:
            other_funcs = ()
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs = funcs
            *_, last_func = other_funcs

        self.__signature__ = Pipe._signature_from_first_and_last_func(
            first_func, last_func
        )
        self.first_func, self.other_funcs = first_func, other_funcs

    _reserved_names = ("__name__", "__doc__")

    def _process_reserved_names(self, named_funcs):
        for name in self._reserved_names:
            if (value := named_funcs.pop(name, None)) is not None:
                setattr(self, name, value)
        return named_funcs

    def __call__(self, *args, **kwargs):
        try:
            out = self.first_func(*args, **kwargs)
        except Exception as e:
            raise self._mk_pipe_call_error(e, 0, None, args, kwargs) from e
        try:  # first call has no exeption handling, but subsequent calls do
            for i, func in enumerate(self.other_funcs, 1):
                out = func(out)
        except Exception as e:
            raise self._mk_pipe_call_error(e, i, out, args, kwargs) from e
        return out

    def _mk_pipe_call_error(self, error_obj, i, out, args, kwargs):
        msg = f"Error calling function {self._func_info_str(i)}\n"
        out_str = f"{out}"
        msg += f"on input {truncate_string_with_marker(out_str)}\n"
        msg += "which was the output of previous function "
        msg += f"\t{self._func_info_str(i - 1)}\n"
        args_str = ", ".join(map(str, args))
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        msg += f"The error was cause by calling {self} on ({args_str}, {kwargs_str})\n"
        msg += f"Error was: {error_obj}"
        new_error_obj = type(error_obj)(msg)
        new_error_obj.error_context = {
            "Pipe": self,
            "args": args,
            "kwargs": kwargs,
            "func_index": i,
            "func": self.funcs[i],
            "func_input": out,
        }
        return new_error_obj

    def _func_info_str(self, i):
        func = self.funcs[i]
        func_info = function_info_string(func)
        return f"{func_info} (index={i})"

    def __len__(self):
        return len(self.funcs)

    _dflt_signature = Signature.from_callable(lambda *args, **kwargs: None)

    @staticmethod
    def _signature_from_first_and_last_func(first_func, last_func):
        try:
            input_params = signature(first_func).parameters.values()
        except ValueError:  # function doesn't have a signature, so take default
            input_params = Pipe._dflt_signature.parameters.values()
        try:
            return_annotation = signature(last_func).return_annotation
        except ValueError:  # function doesn't have a signature, so take default
            return_annotation = Pipe._dflt_signature.return_annotation
        return Signature(tuple(input_params), return_annotation=return_annotation)


def _flatten_pipe(pipe):
    for func in pipe.funcs:
        if isinstance(func, Pipe):
            yield from _flatten_pipe(func)
        else:
            yield func


def flatten_pipe(pipe):
    """
    Unravel nested Pipes to get a flat 'sequence of functions' version of input.

    >>> def f(x): return x + 1
    >>> def g(x): return x * 2
    >>> def h(x): return x - 3
    >>> a = Pipe(f, g, h)
    >>> b = Pipe(f, Pipe(g, h))
    >>> len(a)
    3
    >>> len(b)
    2
    >>> c = flatten_pipe(b)
    >>> len(c)
    3
    >>> assert a(10) == b(10) == c(10) == 19
    """
    return Pipe(*_flatten_pipe(pipe))


def partialclass(cls, *args, **kwargs):
    """What partial(cls, *args, **kwargs) does, but returning a class instead of an object.

    :param cls: Class to get the partial of
    :param kwargs: The kwargs to fix

    The raison d'être of partialclass is that it returns a type, so let's have a look at that with
    a useless class.

    >>> from inspect import signature
    >>> class A:
    ...     pass
    >>> assert isinstance(A, type) == isinstance(partialclass(A), type) == True

    >>> class A:
    ...     def __init__(self, a=0, b=1):
    ...         self.a, self.b = a, b
    ...     def mysum(self):
    ...         return self.a + self.b
    ...     def __repr__(self):
    ...         return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
    >>>
    >>> assert isinstance(A, type) == isinstance(partialclass(A), type) == True
    >>>
    >>> assert str(signature(A)) == '(a=0, b=1)'
    >>>
    >>> a = A()
    >>> assert a.mysum() == 1
    >>> assert str(a) == 'A(a=0, b=1)'
    >>>
    >>> assert A(a=10).mysum() == 11
    >>> assert str(A()) == 'A(a=0, b=1)'
    >>>
    >>>
    >>> AA = partialclass(A, b=2)
    >>> assert str(signature(AA)) == '(a=0, *, b=2)'
    >>> aa = AA()
    >>> assert aa.mysum() == 2
    >>> assert str(aa) == 'A(a=0, b=2)'
    >>> assert AA(a=1, b=3).mysum() == 4
    >>> assert str(AA(3)) == 'A(a=3, b=2)'
    >>>
    >>> AA = partialclass(A, a=7)
    >>> assert str(signature(AA)) == '(*, a=7, b=1)'
    >>> assert AA().mysum() == 8
    >>> assert str(AA(a=3)) == 'A(a=3, b=1)'

    Note in the last partial that since ``a`` was fixed, you need to specify the keyword ``AA(a=3)``.
    ``AA(3)`` won't work:

    >>> AA(3)  # doctest: +SKIP
    Traceback (most recent call last):
      ...
    TypeError: __init__() got multiple values for argument 'a'

    On the other hand, you can use *args to specify the fixtures:

    >>> AA = partialclass(A, 22)
    >>> assert str(AA()) == 'A(a=22, b=1)'
    >>> assert str(signature(AA)) == '(b=1)'
    >>> assert str(AA(3)) == 'A(a=22, b=3)'


    """
    assert isinstance(cls, type), f"cls should be a type, was a {type(cls)}: {cls}"

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    copy_attrs(
        PartialClass,
        cls,
        attrs=("__name__", "__qualname__", "__module__", "__doc__"),
    )

    return PartialClass


def copy_attrs(target, source, attrs, raise_error_if_an_attr_is_missing=True):
    """Copy attributes from one object to another.

    >>> class A:
    ...     x = 0
    >>> class B:
    ...     x = 1
    ...     yy = 2
    ...     zzz = 3
    >>> dict_of = lambda o: {a: getattr(o, a) for a in dir(A) if not a.startswith('_')}
    >>> dict_of(A)
    {'x': 0}
    >>> copy_attrs(A, B, 'yy')
    >>> dict_of(A)
    {'x': 0, 'yy': 2}
    >>> copy_attrs(A, B, ['x', 'zzz'])
    >>> dict_of(A)
    {'x': 1, 'yy': 2, 'zzz': 3}

    But if you try to copy something that `B` (the source) doesn't have, copy_attrs will complain:

    >>> copy_attrs(A, B, 'this_is_not_an_attr')
    Traceback (most recent call last):
        ...
    AttributeError: type object 'B' has no attribute 'this_is_not_an_attr'

    If you tell it not to complain, it'll just ignore attributes that are not in source.

    >>> copy_attrs(A, B, ['nothing', 'here', 'exists'], raise_error_if_an_attr_is_missing=False)
    >>> dict_of(A)
    {'x': 1, 'yy': 2, 'zzz': 3}
    """
    if isinstance(attrs, str):
        attrs = (attrs,)
    if raise_error_if_an_attr_is_missing:
        filt = lambda a: True
    else:
        filt = lambda a: hasattr(source, a)
    for a in filter(filt, attrs):
        setattr(target, a, getattr(source, a))


def copy_attrs_from(from_obj, to_obj, attrs):
    from warnings import warn

    warn(f"Deprecated. Use copy_attrs instead.", DeprecationWarning)
    copy_attrs(to_obj, from_obj, attrs)
    return to_obj


def norm_kv_filt(kv_filt: Callable[[Any], bool]):
    """Prepare a boolean function to be used with `filter` when fed an iterable of (k, v) pairs.

    So you have a mapping. Say a dict `d`. Now you want to go through d.items(),
    filtering based on the keys, or the values, or both.

    It's not hard to do, really. If you're using a dict you might use a dict comprehension,
    or in the general case you might do a `filter(lambda kv: my_filt(kv[0], kv[1]), d.items())`
    if you have a `my_filt` that works wiith k and v, etc.

    But thought simple, it can become a bit muddled.
    `norm_kv_filt` simplifies this by allowing you to bring your own filtering boolean function,
    whether it's a key-based, value-based, or key-value-based one, and it will make a
    ready-to-use with `filter` function for you.

    Only thing: Your function needs to call a key `k` and a value `v`.
    But hey, it's alright, if you have a function that calls things differently, just do
    something like

    .. code-block:: python

        new_filt_func = lambda k, v: your_filt_func(..., key=k, ..., value=v, ...)

    and all will be fine.

    :param kv_filt: callable (starting with signature (k), (v), or (k, v)), and returning  a boolean
    :return: A normalized callable.

    >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> list(filter(norm_kv_filt(lambda k: k in {'b', 'd'}), d.items()))
    [('b', 2), ('d', 4)]
    >>> list(filter(norm_kv_filt(lambda v: v > 2), d.items()))
    [('c', 3), ('d', 4)]
    >>> list(filter(norm_kv_filt(lambda k, v: (v > 1) & (k != 'c')), d.items()))
    [('b', 2), ('d', 4)]
    """
    if kv_filt is None:
        return None  # because `filter` works with a callable, or None, so we align

    raise_msg = (
        f"kv_filt should be callable (starting with signature (k), (v), or (k, v)),"
        "and returning  a boolean. What you gave me was {fv_filt}"
    )
    assert callable(kv_filt), raise_msg

    params = list(signature(kv_filt).parameters.values())
    assert len(params), raise_msg
    _kv_filt = kv_filt
    if params[0].name == "v":

        def kv_filt(k, v):
            return _kv_filt(v)

    elif params[0].name == "k":
        if len(params) > 1:
            if params[1].name != "v":
                raise ValueError(raise_msg)
        else:

            def kv_filt(k, v):
                return _kv_filt(k)

    else:
        raise ValueError(raise_msg)

    def __kv_filt(kv_item):
        return kv_filt(*kv_item)

    __kv_filt.__name__ = kv_filt.__name__

    return __kv_filt


var_str_p = re.compile(r"\W|^(?=\d)")

Item = Any


def add_attrs(remember_added_attrs=True, if_attr_exists="raise", **attrs):
    """Make a function that will add attributes to an obj.
    Originally meant to be used as a decorator of a function, to inject

    >>> from dol.util import add_attrs
    >>> @add_attrs(bar='bituate', hello='world')
    ... def foo():
    ...     pass
    >>> [x for x in dir(foo) if not x.startswith('_')]
    ['bar', 'hello']
    >>> foo.bar
    'bituate'
    >>> foo.hello
    'world'
    >>> foo._added_attrs  # Another attr was added to hold the list of attributes added (in case we need to remove them
    ['bar', 'hello']
    """

    def add_attrs_to_func(obj):
        attrs_added = []
        for attr_name, attr_val in attrs.items():
            if hasattr(obj, attr_name):
                if if_attr_exists == "raise":
                    raise AttributeError(
                        f"Attribute {attr_name} already exists in {obj}"
                    )
                elif if_attr_exists == "warn":
                    warn(f"Attribute {attr_name} already exists in {obj}")
                elif if_attr_exists == "skip":
                    continue
                else:
                    raise ValueError(
                        f"Unknown value for if_attr_exists: {if_attr_exists}"
                    )
            setattr(obj, attr_name, attr_val)
            attrs_added.append(attr_name)

        if remember_added_attrs:
            obj._added_attrs = attrs_added

        return obj

    return add_attrs_to_func


def fullpath(path):
    if path.startswith("~"):
        path = os.path.expanduser(path)
    return os.path.abspath(path)


def attrs_of(obj):
    return set(dir(obj))


def format_invocation(name="", args=(), kwargs=None):
    """Given a name, positional arguments, and keyword arguments, format
    a basic Python-style function call.

    >>> print(format_invocation('func', args=(1, 2), kwargs={'c': 3}))
    func(1, 2, c=3)
    >>> print(format_invocation('a_func', args=(1,)))
    a_func(1)
    >>> print(format_invocation('kw_func', kwargs=[('a', 1), ('b', 2)]))
    kw_func(a=1, b=2)

    """
    kwargs = kwargs or {}
    a_text = ", ".join([repr(a) for a in args])
    if isinstance(kwargs, dict):
        kwarg_items = [(k, kwargs[k]) for k in sorted(kwargs)]
    else:
        kwarg_items = kwargs
    kw_text = ", ".join(["{}={!r}".format(k, v) for k, v in kwarg_items])

    all_args_text = a_text
    if all_args_text and kw_text:
        all_args_text += ", "
    all_args_text += kw_text

    return "{}({})".format(name, all_args_text)


def groupby(
    items: Iterable[Item],
    key: Callable[[Item], Hashable],
    val: Callable[[Item], Any] | None = None,
    group_factory=list,
) -> dict:
    """Groups items according to group keys updated from those items through the given
    (item_to_)key function.

    Args:
        items: iterable of items
        key: The function that computes a key from an item. Needs to return a hashable.
        val: An optional function that computes a val from an item. If not given, the item itself will be taken.
        group_factory: The function to make new (empty) group objects and accumulate group items.
            group_items = group_factory() will be called to make a new empty group collection
            group_items.append(x) will be called to add x to that collection
            The default is `list`

    Returns: A dict of {group_key: items_in_that_group, ...}

    See Also: regroupby, itertools.groupby, and dol.source.SequenceKvReader

    >>> groupby(range(11), key=lambda x: x % 3)
    {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8]}
    >>>
    >>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
    >>> groupby(tokens, len)
    {3: ['the', 'fox', 'box'], 2: ['is', 'in'], 1: ['a']}
    >>> key_map = {1: 'one', 2: 'two'}
    >>> groupby(tokens, lambda x: key_map.get(len(x), 'more'))
    {'more': ['the', 'fox', 'box'], 'two': ['is', 'in'], 'one': ['a']}
    >>> stopwords = {'the', 'in', 'a', 'on'}
    >>> groupby(tokens, lambda w: w in stopwords)
    {True: ['the', 'in', 'a'], False: ['fox', 'is', 'box']}
    >>> groupby(tokens, lambda w: ['words', 'stopwords'][int(w in stopwords)])
    {'stopwords': ['the', 'in', 'a'], 'words': ['fox', 'is', 'box']}
    """
    groups = defaultdict(group_factory)
    if val is None:
        for item in items:
            groups[key(item)].append(item)
    else:
        for item in items:
            groups[key(item)].append(val(item))
    return dict(groups)


def regroupby(items, *key_funcs, **named_key_funcs):
    """Recursive groupby. Applies the groupby function recursively, using a sequence of key functions.

    Note: The named_key_funcs argument names don't have any external effect.
        They just give a name to the key function, for code reading clarity purposes.

    See Also: groupby, itertools.groupby, and dol.source.SequenceKvReader

    >>> # group by how big the number is, then by it's mod 3 value
    >>> # note that named_key_funcs argument names doesn't have any external effect (but give a name to the function)
    >>> regroupby([1, 2, 3, 4, 5, 6, 7], lambda x: 'big' if x > 5 else 'small', mod3=lambda x: x % 3)
    {'small': {1: [1, 4], 2: [2, 5], 0: [3]}, 'big': {0: [6], 1: [7]}}
    >>>
    >>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
    >>> stopwords = {'the', 'in', 'a', 'on'}
    >>> word_category = lambda x: 'stopwords' if x in stopwords else 'words'
    >>> regroupby(tokens, word_category, len)
    {'stopwords': {3: ['the'], 2: ['in'], 1: ['a']}, 'words': {3: ['fox', 'box'], 2: ['is']}}
    >>> regroupby(tokens, len, word_category)
    {3: {'stopwords': ['the'], 'words': ['fox', 'box']}, 2: {'words': ['is'], 'stopwords': ['in']}, 1: {'stopwords': ['a']}}
    """
    key_funcs = list(key_funcs) + list(named_key_funcs.values())
    assert len(key_funcs) > 0, "You need to have at least one key_func"
    if len(key_funcs) == 1:
        return groupby(items, key=key_funcs[0])
    else:
        key_func, *key_funcs = key_funcs
        groups = groupby(items, key=key_func)
        return {
            group_key: regroupby(group_items, *key_funcs)
            for group_key, group_items in groups.items()
        }


Groups = dict
GroupKey = Hashable
GroupItems = Iterable[Item]
GroupReleaseCond = Union[
    Callable[[GroupKey, GroupItems], bool],
    Callable[[Groups, GroupKey, GroupItems], bool],
]


def igroupby(
    items: Iterable[Item],
    key: Callable[[Item], GroupKey],
    val: Callable[[Item], Any] | None = None,
    group_factory: Callable[[], GroupItems] = list,
    group_release_cond: GroupReleaseCond = lambda k, v: False,
    release_remainding=True,
    append_to_group_items: Callable[[GroupItems, Item], Any] = list.append,
    grouper_mapping=defaultdict,
):
    """The generator version of dol groupby.
    Groups items according to group keys updated from those items through the given (item_to_)key function,
    yielding the groups according to a logic defined by ``group_release_cond``

    Args:
        items: iterable of items
        key: The function that computes a key from an item. Needs to return a hashable.
        val: An optional function that computes a val from an item. If not given, the item itself will be taken.
        group_factory: The function to make new (empty) group objects and accumulate group items.
            group_items = group_collector() will be called to make a new empty group collection
            group_items.append(x) will be called to add x to that collection
            The default is `list`
        group_release_cond: A boolean function that will be applied, at every iteration,
            to the accumulated items of the group that was just updated,
            and determines (if True) if the (group_key, group_items) should be yielded.
            The default is False, which results in
            ``lambda group_key, group_items: False`` being used.
        release_remainding: Once the input items have been consumed, there may still be some
            items in the grouping "cache". ``release_remainding`` is a boolean that indicates whether
            the contents of this cache should be released or not.

    Yields: ``(group_key, items_in_that_group)`` pairs


    The following will group numbers according to their parity (0 for even, 1 for odd),
    releasing a list of numbers collected when that list reaches length 3:

    >>> g = igroupby(items=range(11),
    ...             key=lambda x: x % 2,
    ...             group_release_cond=lambda k, v: len(v) == 3)
    >>> list(g)
    [(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10]), (1, [7, 9])]

    If we specify ``release_remainding=False`` though, we won't get

    >>> g = igroupby(items=range(11),
    ...             key=lambda x: x % 2,
    ...             group_release_cond=lambda k, v: len(v) == 3,
    ...             release_remainding=False)
    >>> list(g)
    [(0, [0, 2, 4]), (1, [1, 3, 5]), (0, [6, 8, 10])]

    # >>> grps = partial(igroupby, group_release_cond=False, release_remainding=True)


    Below we show that, with the default ``group_release_cond = lambda k, v: False``
    and release_remainding=True`` we have ``dict(igroupby(...)) == groupby(...)``

    >>> from functools import partial
    >>> from dol import groupby
    >>>
    >>> kws = dict(items=range(11), key=lambda x: x % 3)
    >>> assert (dict(igroupby(**kws)) == groupby(**kws)
    ...         == {0: [0, 3, 6, 9], 1: [1, 4, 7, 10], 2: [2, 5, 8]})
    >>>
    >>> tokens = ['the', 'fox', 'is', 'in', 'a', 'box']
    >>> kws = dict(items=tokens, key=len)
    >>> assert (dict(igroupby(**kws)) == groupby(**kws)
    ...         == {3: ['the', 'fox', 'box'], 2: ['is', 'in'], 1: ['a']})
    >>>
    >>> key_map = {1: 'one', 2: 'two'}
    >>> kws.update(key=lambda x: key_map.get(len(x), 'more'))
    >>> assert (dict(igroupby(**kws)) == groupby(**kws)
    ...         == {'more': ['the', 'fox', 'box'], 'two': ['is', 'in'], 'one': ['a']})
    >>>
    >>> stopwords = {'the', 'in', 'a', 'on'}
    >>> kws.update(key=lambda w: w in stopwords)
    >>> assert (dict(igroupby(**kws)) == groupby(**kws)
    ...         == {True: ['the', 'in', 'a'], False: ['fox', 'is', 'box']})
    >>> kws.update(key=lambda w: ['words', 'stopwords'][int(w in stopwords)])
    >>> assert (dict(igroupby(**kws)) == groupby(**kws)
    ...         == {'stopwords': ['the', 'in', 'a'], 'words': ['fox', 'is', 'box']})

    """
    groups = grouper_mapping(group_factory)

    assert callable(group_release_cond), (
        "group_release_cond should be callable (filter boolean function) or False. "
        f"Was {group_release_cond}"
    )
    n_group_release_cond_args = len(signature(group_release_cond).parameters)
    assert n_group_release_cond_args in {2, 3}, (
        "group_release_cond should take two or three inputs:\n"
        " - (group_key, group_items), or\n"
        " - (groups, group_key, group_items)"
        f"The arguments of the function you gave me are: {signature(group_release_cond)}"
    )

    if val is None:
        _append_to_group_items = append_to_group_items
    else:
        _append_to_group_items = lambda group_items, item: (
            group_items,
            val(item),
        )

    for item in items:
        group_key = key(item)
        group_items = groups[group_key]
        _append_to_group_items(group_items, item)

        if group_release_cond(group_key, group_items):
            yield group_key, group_items
            del groups[group_key]

    if release_remainding:
        for group_key, group_items in groups.items():
            yield group_key, group_items


def ntup(**kwargs):
    return namedtuple("NamedTuple", list(kwargs))(**kwargs)


def str_to_var_str(s: str) -> str:
    """Make a valid python variable string from the input string.
    Left untouched if already valid.

    >>> str_to_var_str('this_is_a_valid_var_name')
    'this_is_a_valid_var_name'
    >>> str_to_var_str('not valid  #)*(&434')
    'not_valid_______434'
    >>> str_to_var_str('99_ballons')
    '_99_ballons'
    """
    return var_str_p.sub("_", s)


def fill_with_dflts(d, dflt_dict=None):
    """
    Fed up with multiline handling of dict arguments?
    Fed up of repeating the if d is None: d = {} lines ad nauseam (because defaults can't be dicts as a default
    because dicts are mutable blah blah, and the python kings don't seem to think a mutable dict is useful enough)?
    Well, my favorite solution would be a built-in handling of the problem of complex/smart defaults,
    that is visible in the code and in the docs. But for now, here's one of the tricks I use.

    Main use is to handle defaults of function arguments. Say you have a function `func(d=None)` and you want
    `d` to be a dict that has at least the keys `foo` and `bar` with default values 7 and 42 respectively.
    Then, in the beginning of your function code you'll say:

        d = fill_with_dflts(d, {'a': 7, 'b': 42})

    See examples to know how to use it.

    ATTENTION: A shallow copy of the dict is made. Know how that affects you (or not).
    ATTENTION: This is not recursive: It won't be filling any nested fields with defaults.

    Args:
        d: The dict you want to "fill"
        dflt_dict: What to fill it with (a {k: v, ...} dict where if k is missing in d, you'll get a new field k, with
            value v.

    Returns:
        a dict with the new key:val entries (if the key was missing in d).

    >>> fill_with_dflts(None)
    {}
    >>> fill_with_dflts(None, {'a': 7, 'b': 42})
    {'a': 7, 'b': 42}
    >>> fill_with_dflts({}, {'a': 7, 'b': 42})
    {'a': 7, 'b': 42}
    >>> fill_with_dflts({'b': 1000}, {'a': 7, 'b': 42})
    {'a': 7, 'b': 1000}
    """
    if d is None:
        d = {}
    if dflt_dict is None:
        dflt_dict = {}
    return dict(dflt_dict, **d)


# Note: Had replaced with cached_property (new in 3.8)
# if not sys.version_info >= (3, 8):
#     from functools import cached_property
# # etc...
# But then I realized that the way cached_property is implemented, pycharm does not see the properties (lint)
# So I'm reverting to lazyprop
# TODO: Keep track of the evolution of functools.cached_property and compare performance.
class lazyprop:
    """
    A descriptor implementation of lazyprop (cached property).
    Made based on David Beazley's "Python Cookbook" book and enhanced with boltons.cacheutils ideas.

    >>> class Test:
    ...     def __init__(self, a):
    ...         self.a = a
    ...     @lazyprop
    ...     def len(self):
    ...         print('generating "len"')
    ...         return len(self.a)
    >>> t = Test([0, 1, 2, 3, 4])
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4]}
    >>> t.len
    generating "len"
    5
    >>> t.__dict__
    {'a': [0, 1, 2, 3, 4], 'len': 5}
    >>> t.len
    5
    >>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
    >>> t.a = [0, 1, 2]  # if we change a...
    >>> t.len  # ... we still get the old cached value of len
    5
    >>> del t.len  # if we delete the len prop
    >>> t.len  # ... then len being recomputed again
    generating "len"
    3
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.__isabstractmethod__ = getattr(func, "__isabstractmethod__", False)
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = instance.__dict__[self.func.__name__] = self.func(instance)
            return value

    def __repr__(self):
        cn = self.__class__.__name__
        return "<{} func={}>".format(cn, self.func)


from functools import lru_cache, wraps
import weakref


@wraps(lru_cache)
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # Storing the wrapped method inside the instance since a strong reference to self would not allow it to die.
            self_weak = weakref.ref(self)

            @wraps(func)
            @lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


class lazyprop_w_sentinel(lazyprop):
    """
    A descriptor implementation of lazyprop (cached property).
    Inserts a `self.func.__name__ + '__cache_active'` attribute

    >>> class Test:
    ...     def __init__(self, a):
    ...         self.a = a
    ...     @lazyprop_w_sentinel
    ...     def len(self):
    ...         print('generating "len"')
    ...         return len(self.a)
    >>> t = Test([0, 1, 2, 3, 4])
    >>> lazyprop_w_sentinel.cache_is_active(t, 'len')
    False
    >>> t.__dict__  # let's look under the hood
    {'a': [0, 1, 2, 3, 4]}
    >>> t.len
    generating "len"
    5
    >>> lazyprop_w_sentinel.cache_is_active(t, 'len')
    True
    >>> t.len  # notice there's no 'generating "len"' print this time!
    5
    >>> t.__dict__  # let's look under the hood
    {'a': [0, 1, 2, 3, 4], 'len': 5, 'sentinel_of__len': True}
    >>> # But careful when using lazyprop that no one will change the value of a without deleting the property first
    >>> t.a = [0, 1, 2]  # if we change a...
    >>> t.len  # ... we still get the old cached value of len
    5
    >>> del t.len  # if we delete the len prop
    >>> t.len  # ... then len being recomputed again
    generating "len"
    3
    """

    sentinel_prefix = "sentinel_of__"

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = instance.__dict__[self.func.__name__] = self.func(instance)
            setattr(
                instance, self.sentinel_prefix + self.func.__name__, True
            )  # my hack
            return value

    @classmethod
    def cache_is_active(cls, instance, attr):
        return getattr(instance, cls.sentinel_prefix + attr, False)


class Struct:
    def __init__(self, **attr_val_dict):
        for attr, val in attr_val_dict.items():
            setattr(self, attr, val)


class MutableStruct(Struct):
    def extend(self, **attr_val_dict):
        for attr in attr_val_dict.keys():
            if hasattr(self, attr):
                raise AttributeError(
                    f"The attribute {attr} already exists. Delete it if you want to reuse it!"
                )
        for attr, val in attr_val_dict.items():
            setattr(self, attr, val)


def max_common_prefix(a: Sequence, *, default=""):
    """
    Given a list of strings (or other sliceable seq), returns the longest common prefix

    :param a: list-like of strings
    :return: the smallest common prefix of all strings in a

    >>> max_common_prefix(['absolutely', 'abc', 'abba'])
    'ab'
    >>> max_common_prefix(['absolutely', 'not', 'abc', 'abba'])
    ''
    >>> max_common_prefix([[3,2,1], [3,2,0]])
    [3, 2]
    >>> max_common_prefix([[3,2,1], [3,2,0], [1,2,3]])
    []

    If the input is empty, will return default (which defaults to '').

    >>> max_common_prefix([])
    ''

    If you want a different default, you can specify it with the default
    keyword argument.

    >>> from functools import partial
    >>> my_max_common_prefix = partial(max_common_prefix, default=[])
    >>> my_max_common_prefix([])
    []
    """
    if not a:
        return default
    # Note: Try to optimize by using a min_max function to give me both in one pass.
    # The current version is still faster
    s1 = min(a)  # lexicographically minimal
    s2 = max(a)  # lexicographically maximal
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


class SimpleProperty:
    def __get__(self, obj, objtype=None):
        return obj.d

    def __set__(self, obj, value):
        obj.d = value

    def __delete__(self, obj):
        del obj.d


class DelegatedAttribute:
    def __init__(self, delegate_name, attr_name):
        self.attr_name = attr_name
        self.delegate_name = delegate_name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            # return instance.delegate.attr
            return getattr(self.delegate(instance), self.attr_name)

    def __set__(self, instance, value):
        # instance.delegate.attr = value
        setattr(self.delegate(instance), self.attr_name, value)

    def __delete__(self, instance):
        delattr(self.delegate(instance), self.attr_name)

    def delegate(self, instance):
        return getattr(instance, self.delegate_name)

    def __str__(self):
        return ""

    # def __call__(self, instance, *args, **kwargs):
    #     return self.delegate(instance)(*args, **kwargs)


def delegate_as(delegate_cls, to="delegate", include=frozenset(), exclude=frozenset()):
    raise NotImplementedError("Didn't manage to make this work fully")
    # turn include and ignore into sets, if they aren't already
    include = set(include)
    exclude = set(exclude)
    delegate_attrs = set(delegate_cls.__dict__.keys())
    attributes = include | delegate_attrs - exclude

    def inner(cls):
        # create property for storing the delegate
        setattr(cls, to, property())
        # don't bother adding attributes that the class already has
        attrs = attributes - set(cls.__dict__.keys())
        # set all the attributes
        for attr in attrs:
            setattr(cls, attr, DelegatedAttribute(to, attr))
        return cls

    return inner


class HashableMixin:
    def __hash__(self):
        return id(self)


class ImmutableMixin:
    def _immutable(self, *args, **kws):
        raise TypeError("object is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


# TODO: Lint still considers instances of imdict to be mutable.
#  Probably because it still sees the mutator methods in the class definition.
#  Maybe I should just remove them from the class definition?
# TODO: Generalize to a function that makes any class immutable.
class imdict(ImmutableMixin, dict, HashableMixin):
    """A frozen hashable dict"""


def move_files_of_folder_to_trash(folder):
    trash_dir = os.path.join(
        os.getenv("HOME"), ".Trash"
    )  # works with mac (perhaps linux too?)
    assert os.path.isdir(trash_dir), f"{trash_dir} directory not found"

    for f in os.listdir(folder):
        src = os.path.join(folder, f)
        if os.path.isfile(src):
            dst = os.path.join(trash_dir, f)
            print(f"Moving to trash: {src}")
            shutil.move(src, dst)


class ModuleNotFoundErrorNiceMessage:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            if self.msg is not None:
                warn(self.msg)
            else:
                raise ModuleNotFoundError(
                    f"""
It seems you don't have required `{exc_val.name}` package for this Store.
Try installing it by running:

    pip install {exc_val.name}
    
in your terminal.
For more information: https://pypi.org/project/{exc_val.name}
            """
                )


class ModuleNotFoundWarning:
    def __init__(self, msg="It seems you don't have a required package."):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            warn(self.msg)
            #             if exc_val is not None and getattr(exc_val, 'name', None) is not None:
            #                 warn(f"""
            # It seems you don't have required `{exc_val.name}` package for this Store.
            # This is just a warning: The process goes on...
            # (But, hey, if you really need that package, try installing it by running:
            #
            #     pip install {exc_val.name}
            #
            # in your terminal.
            # For more information: https://pypi.org/project/{exc_val.name}, or google around...
            #                 """)
            #             else:
            #                 print("It seems you don't have a required package")
            return True


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def num_of_required_args(func):
    """Number or REQUIRED arguments of a function.

    Contrast the behavior below with that of ``num_of_args``, which counts all
    parameters, including the variadics and defaulted ones.

    >>> num_of_required_args(lambda a, b, c: None)
    3
    >>> num_of_required_args(lambda a, b, c=3: None)
    2
    >>> num_of_required_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
    2
    """
    var_param_kinds = {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
    sig = signature(func)
    return sum(
        1
        for p in sig.parameters.values()
        if p.default is Parameter.empty and p.kind not in var_param_kinds
    )


def num_of_args(func):
    """Number of arguments (parameters) of the function.

    Contrast the behavior below with that of ``num_of_required_args``.

    >>> num_of_args(lambda a, b, c: None)
    3
    >>> num_of_args(lambda a, b, c=3: None)
    3
    >>> num_of_args(lambda a, *args, b, c=1, d=2, **kwargs: None)
    6
    """
    return len(signature(func).parameters)


def single_nest_in_dict(key, value):
    return {key: value}


def nest_in_dict(keys, values):
    return {k: v for k, v in zip(keys, values)}


import io
from typing import Any, VT, Union, KT
from collections.abc import Callable
from functools import partial
import tempfile

Buffer = Union[io.BytesIO, io.StringIO]
FileWriter = Callable[[VT, Buffer], Any]
Writer = Union[Callable[[VT, KT], Any], Callable[[KT, VT], Any]]


def _call_writer(
    writer: Writer,
    obj: VT,
    destination: KT | Buffer,
    obj_arg_position_in_writer: int = 0,
):
    """
    Helper function to handle writing to the buffer based on obj_arg_position_in_writer.

    :param writer: A function that writes an object to a file-like object.
    :param obj: The object to write.
    :param destination: The key (e.g. filepath) or file-like object.
    :param obj_arg_position_in_writer: Position of the object argument in writer function (0 or 1).

    :raises ValueError: If obj_arg_position_in_writer is not 0 or 1.
    """
    if obj_arg_position_in_writer == 0:
        writer(obj, destination)
    elif obj_arg_position_in_writer == 1:
        writer(destination, obj)
    else:
        raise ValueError("obj_arg_position_in_writer must be 0 or 1")


def written_bytes(
    file_writer: FileWriter,
    obj: VT = None,
    *,
    obj_arg_position_in_writer: int = 0,
    io_buffer_cls: Buffer = io.BytesIO,
):
    """
    Takes a file writing function that expects an object and a file-like object,
    and returns a function that instead of writing to a file, returns the bytes that
    would have been written.

    This is the write version of the `read_from_bytes` function of the same module.

    Note: If obj is not given, `write_bytes` will return a "bytes writer" function that
    takes obj as the first argument, and uses the file_writer to write the bytes.

    :param file_writer: A function that writes an object to a file-like object.
    :param obj: The object to write.
    :return: The bytes that would have been written to a file.

    Use case: When you have a function that writes to files, and you want to get an
    equivalent function but that gives you what bytes or string WOULD have been written
    to a file, so you can better reuse (to write elsewhere, for example, or because
    you need to pipe those bytes to another function).

    Example usage: Yes, we have json.dumps to get the JSON string, but what if
    (like is often the case) you just have a function that writes to a file-like object,
    like the `json.dump(obj, fp)` function? You can use `written_bytes` to get a
    function that will act as `json.dumps` like so:

    >>> import json
    >>> get_json_bytes = written_bytes(json.dump, io_buffer_cls=io.StringIO)
    >>> get_json_bytes({'a': 1, 'b': 2})
    '{"a": 1, "b": 2}'

    Here's another example with pandas DataFrame.to_parquet:

    >>> import pandas as pd  # doctest: +SKIP
    >>> df = pd.DataFrame({  # doctest: +SKIP
    ...     'column1': [1, 2, 3],
    ...     'column2': ['A', 'B', 'C']
    ... })

    Get a function that converts DataFrame to Parquet bytes

    df_to_parquet_bytes = written_bytes(pd.DataFrame.to_parquet)

    # Get the bytes of the DataFrame in Parquet format
    parquet_bytes = df_to_parquet_bytes(df)
    all(pd.read_parquet(io.BytesIO(parquet_bytes)) == df)


    """
    if obj is None:
        return partial(
            written_bytes,
            file_writer,
            obj_arg_position_in_writer=obj_arg_position_in_writer,
            io_buffer_cls=io_buffer_cls,
        )

    # Create a BytesIO object to act as an in-memory file
    buffer = io_buffer_cls()

    # Use the provided file_writer function to write to the buffer
    _call_writer(file_writer, obj, buffer, obj_arg_position_in_writer)

    # Retrieve the bytes from the buffer
    buffer.seek(0)
    return buffer.read()


def _call_reader(
    reader: Callable,
    buffer: Buffer,
    buffer_arg_position: int = 0,
    buffer_arg_name: str = None,
    *args,
    **kwargs,
):
    """
    Helper function to handle reading from the buffer based on buffer_arg_position or buffer_arg_name.

    :param reader: A function that reads from a file-like object.
    :param buffer: The file-like object to read from.
    :param buffer_arg_position: Position of the file-like object argument in reader function.
    :param buffer_arg_name: Name of the file-like object argument in reader function.
    :raises ValueError: If buffer_arg_position is not valid.
    """
    if buffer_arg_name is not None:
        kwargs[buffer_arg_name] = buffer
        return reader(*args, **kwargs)
    else:
        args = list(args)
        # Ensure the args list is long enough
        while len(args) < buffer_arg_position:
            args.append(None)
        args.insert(buffer_arg_position, buffer)
        return reader(*args, **kwargs)


def read_from_bytes(
    file_reader: Callable,
    obj: bytes = None,
    *,
    buffer_arg_position: int = 0,
    buffer_arg_name: str = None,
    io_buffer_cls: Buffer = io.BytesIO,
    **kwargs,
):
    """
    Takes a file reading function that expects a file-like object,
    and returns a function that instead of reading from a file, reads from bytes.

    This is the read version of the `written_bytes` function of the same module.

    Note: If obj is not given, read_from_bytes will return a "bytes reader" function that
    takes obj as the first argument, and uses the file_reader to read the bytes.

    :param file_reader: A function that reads from a file-like object.
    :param obj: The bytes to read.
    :param buffer_arg_position: The position of the file-like object in file_reader's arguments.
    :param buffer_arg_name: The name of the file-like object argument in file_reader.
    :return: The result of reading from the bytes.

    Example usage:

    Using `json.load` to read a JSON object from bytes:

    >>> import json
    >>> data = {'a': 1, 'b': 2}
    >>> json_bytes = json.dumps(data).encode('utf-8')
    >>> read_json_from_bytes = read_from_bytes(json.load)
    >>> data_loaded = read_json_from_bytes(json_bytes)
    >>> data_loaded == data
    True

    Using `pickle.load` to read an object from bytes:

    >>> import pickle
    >>> obj = {'x': [1, 2, 3], 'y': ('a', 'b')}
    >>> pickle_bytes = pickle.dumps(obj)
    >>> read_pickle_from_bytes = read_from_bytes(pickle.load)
    >>> obj_loaded = read_pickle_from_bytes(pickle_bytes)
    >>> obj_loaded == obj
    True
    """
    if obj is None:
        return partial(
            read_from_bytes,
            file_reader,
            buffer_arg_position=buffer_arg_position,
            buffer_arg_name=buffer_arg_name,
            io_buffer_cls=io_buffer_cls,
            **kwargs,
        )

    buffer = io_buffer_cls(obj)

    return _call_reader(
        file_reader, buffer, buffer_arg_position, buffer_arg_name, **kwargs
    )


def write_to_file(obj: VT, key: KT):
    if isinstance(obj, bytes):
        with open(key, "wb") as f:
            f.write(obj)
    elif isinstance(obj, str):
        with open(key, "w") as f:
            f.write(obj)
    else:
        raise ValueError(
            f"Object of type {type(obj)} cannot be written to a file. "
            "Use an encoder to encode it to bytes or string."
        )


def written_key(
    obj: VT = None,
    writer: Writer = write_to_file,
    *,
    key: KT | Callable | None = None,
    obj_arg_position_in_writer: int = 0,
    encoder: Callable = identity_func,
):
    """
    Writes an object to a key and returns the key.
    If key is not given, a temporary file is created and its path is returned.

    :param obj: The object to write.
    :param writer: A function that writes an object to a file.
    :param key: The key (by default, filepath) to write to.
        If None, a temporary file is created.
        If a string starting with '*', the '*' is replaced with a unique temporary filename.
        If a string that has a '*' somewhere in the middle, what's on the left of if is used as a directory
        and the '*' is replaced with a unique temporary filename. For example
        '/tmp/*_file.ext' would be replaced with '/tmp/oiu8fj9873_file.ext'.
        If a callable, it will be called with obj as input to get the key. One use case
        is to use a function that generates a key based on the object.
    :param obj_arg_position_in_writer: Position of the object argument in writer function (0 or 1).
    :param encoder: A function that encodes the object before writing it.

    :return: The file path where the object was written.

    Example usage:

    Let's make a store and a writer for that store.

    >>> store = dict()
    >>> writer = writer=lambda obj, key: store.__setitem__(key, obj)

    Note the order a writer expects is (obj, key), or we'd just be able to use
    `store.__setitem__` as our writer.

    If we specify a key, the object will be written to that key in the store
    and the key is output.

    >>> written_key(42, writer=writer, key='my_key')
    'my_key'
    >>> store
    {'my_key': 42}

    Often, you'll want to fix your writer (and possibly your key).
    You can do so with `functools.partial`, but for convenience, you can also
    just specify a writer, without an input object, and get a function that
    will write an object to a key.

    >>> write_to_store = written_key(writer=writer, key='another_key')
    >>> write_to_store(99)
    'another_key'
    >>> store
    {'my_key': 42, 'another_key': 99}

    If you don't specify a key, a temporary file is created and the key is output.

    >>> write_to_store = written_key(writer=writer)
    >>> key = write_to_store(43)
    >>> key  # doctest: +SKIP
    '/var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tmp8yaczd8b'
    >>> store[key]
    43

    If the key you specify is a string with a '*', the '*' is replaced with a
    unique temporary filename, or the full path of the temporary file if the *
    is at the start.

    >>> write_to_store = written_key(writer=writer, key='*.ext')
    >>> key = write_to_store(44)
    >>> key  # doctest: +ELLIPSIS
    '....ext'
    >>> store[key]
    44

    One useful use case is when you want to pipe the output of one function into
    another function that expects a file path.
    What you need to do then is just pipe your written_key function into that
    function that expects to work with a file path, and it'll be like piping the
    value of your input object into that function (just via a temp file).

    >>> from dol.util import Pipe
    >>> store.clear()
    >>> key_func = lambda key: store.get(key) * 10
    >>> pipe_obj_to_reader = Pipe(written_key(writer=writer), key_func)
    >>> pipe_obj_to_reader(45)
    450
    >>> store  # doctest: +ELLIPSIS
    {...: 45}

    The default writer is `write_to_file`, which can write bytes or strings to a file.
    If your object is not a bytes or string, you can specify an encoder to encode it
    before calling the writer.

    >>> import json, pathlib
    >>> json_written_temp_filepath = written_key(key='*.json', encoder=json.dumps)
    >>> filepath = json_written_temp_filepath({'a': 1, 'b': 2})
    >>> filepath  # doctest: +SKIP
    '/var/folders/mc/c070wfh51kxd9lft8dl74q1r0000gn/T/tmp8yaczd8b.json'
    >>> json.loads(open(filepath).read())
    {'a': 1, 'b': 2}

    """
    if obj is None:
        return partial(
            written_key,
            writer=writer,
            key=key,
            obj_arg_position_in_writer=obj_arg_position_in_writer,
            encoder=encoder,
        )

    if key is None:
        # Create a temporary file
        fd, temp_filepath = tempfile.mkstemp()
        os.close(fd)
        key = temp_filepath
    elif callable(key):
        key_func = key
        key = key_func(obj)
    elif isinstance(key, str) and "*" in key:
        temp_filepath = tempfile.mktemp()
        if key.startswith("*"):
            # Replace * of key with a unique temporary filename
            key = key.replace("*", temp_filepath)
        else:  # only use the name part of the temp_filepath
            # separate directory and filename
            dir_name, base_name = os.path.split(temp_filepath)
            # Replace * of key with a unique temporary filename
            key = key.replace("*", base_name)

    bytes_or_text = encoder(obj)
    # Write the object to the specified filepath
    _call_writer(writer, bytes_or_text, key, obj_arg_position_in_writer)

    return key


# Often, a user might want to encode an object before writing it to a file.
# Therefore, they'll want to do a writer=Pipe(encoder, write_to_file), so we
# put write_to_file as an attribute of written_key to have it handy.
written_key.write_to_file = write_to_file


# TODO: This function should be symmetric, and if so, the code should use recursion
def invertible_maps(
    mapping: Mapping = None, inv_mapping: Mapping = None
) -> tuple[Mapping, Mapping]:
    """Returns two maps that are inverse of each other.
    Raises an AssertionError iif both maps are None, or if the maps are not inverse of
    each other.

    Get a pair of invertible maps

    >>> invertible_maps({1: 11, 2: 22})
    ({1: 11, 2: 22}, {11: 1, 22: 2})
    >>> invertible_maps(None, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    You can specify one argument as an iterable (of keys for the mapping) and the
    other as a function (to be applied to the keys to get the inverse mapping).
    The function acts similarly to a `Mapping.__getitem__`, transforming each key to
    its associated value. The iterable defines the keys for the mapping, while the
    function is applied to each key to produce the values.

    >>> invertible_maps([1,2,3], lambda x: x * 10)
    ({10: 1, 20: 2, 30: 3}, {1: 10, 2: 20, 3: 30})
    >>> invertible_maps(lambda x: x * 10, [1,2,3])
    ({1: 10, 2: 20, 3: 30}, {10: 1, 20: 2, 30: 3})

    If two maps are given and invertible, you just get them back

    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 2})
    ({1: 11, 2: 22}, {11: 1, 22: 2})

    Or if they're not invertible

    >>> invertible_maps({1: 11, 2: 22}, {11: 1, 22: 'ha, not what you expected!'})
    Traceback (most recent call last):
      ...
    AssertionError: mapping and inv_mapping are not inverse of each other!

    >>> invertible_maps(None, None)
    Traceback (most recent call last):
      ...
    ValueError: You need to specify one or both maps
    """
    if inv_mapping is None and mapping is None:
        raise ValueError("You need to specify one or both maps")

    # Take care of the case where one is a function and the other is a list
    # Here, we apply the function to the list items to get the mappings
    if callable(mapping):
        assert isinstance(
            inv_mapping, Iterable
        ), f"If one argument is callable, the other one must be an iterable of keys"
        mapping = {k: mapping(k) for k in inv_mapping}
        inv_mapping = {v: k for k, v in mapping.items()}
    elif callable(inv_mapping):
        assert isinstance(
            mapping, Iterable
        ), f"If one argument is callable, the other one must be an iterable of keys"
        inv_mapping = {k: inv_mapping(k) for k in mapping}
        mapping = {v: k for k, v in inv_mapping.items()}

    if inv_mapping is None:
        assert hasattr(mapping, "items")
        inv_mapping = {v: k for k, v in mapping.items()}
        assert len(inv_mapping) == len(
            mapping
        ), "The values of mapping are not unique, so the mapping is not invertible"
    elif mapping is None:
        assert hasattr(inv_mapping, "items")
        mapping = {v: k for k, v in inv_mapping.items()}
        assert len(mapping) == len(
            inv_mapping
        ), "The values of inv_mapping are not unique, so the mapping is not invertible"
    else:
        assert (len(mapping) == len(inv_mapping)) and (
            mapping == {v: k for k, v in inv_mapping.items()}
        ), "mapping and inv_mapping are not inverse of each other!"

    return mapping, inv_mapping


# -------------------------------------------------------------------------------------
# Attribute Mapping Classes
# (Vendored in dol)

from types import SimpleNamespace
from collections.abc import MutableMapping, Iterator


class AttributeMapping(SimpleNamespace, Mapping[str, Any]):
    """
    A read-only mapping with attribute access.

    Useful when you want mapping interface but don't need mutation.

    Examples:

    >>> ns = AttributeMapping(x=10, y=20)
    >>> ns.x
    10
    >>> ns['y']
    20
    >>> list(ns)
    ['x', 'y']
    """

    @classmethod
    def from_mapping(self, mapping: Mapping[str, Any]) -> "AttributeMapping":
        """
        Create an AttributeMapping from a regular mapping.

        This is useful when you want to convert a dictionary or other mapping
        into an AttributeMapping for attribute-style access.
        """
        return self(**mapping)

    def __getitem__(self, key: str) -> Any:
        """Get item with proper KeyError on missing keys."""
        return _get_attr_or_key_error(self, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over attribute names."""
        return iter(self.__dict__)

    def __len__(self) -> int:
        """Return number of attributes."""
        return len(self.__dict__)


class AttributeMutableMapping(AttributeMapping, MutableMapping[str, Any]):
    """
    A mutable mapping that provides both attribute and dictionary-style access.

    Extends AttributeMapping with mutation capabilities,
    ensuring proper error handling and protocol compliance.

    Examples:

    >>> ns = AttributeMutableMapping(apple=1, banana=2)
    >>> ns.apple
    1
    >>> ns['banana']
    2
    >>> ns['cherry'] = 3
    >>> ns.cherry
    3
    >>> list(ns)
    ['apple', 'banana', 'cherry']
    >>> len(ns)
    3
    >>> 'apple' in ns
    True
    >>> del ns['banana']
    >>> 'banana' in ns
    False
    """

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item via attribute assignment."""
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        """Delete item with proper KeyError on missing keys."""
        try:
            delattr(self, key)
        except AttributeError:
            raise KeyError(key)


def _get_attr_or_key_error(obj: object, key: str) -> Any:
    """
    Get attribute or raise KeyError if not found.

    Helper function to maintain consistent error handling across
    mapping implementations.
    """
    try:
        return getattr(obj, key)
    except AttributeError:
        raise KeyError(key)


# ----------------------------------------------------------------------------------------------------------------------
# More or less vendored from config2py

from typing import Literal, Optional, Tuple
from collections.abc import Sequence
from collections import namedtuple

FolderSpec = namedtuple("FolderSpec", ["env_var", "default_path"])

if os.name == "nt":
    APP_FOLDER_STANDARDS = dict(
        config=FolderSpec("APPDATA", os.getenv("APPDATA", "")),
        data=FolderSpec("LOCALAPPDATA", os.getenv("LOCALAPPDATA", "")),
        cache=FolderSpec(
            "LOCALAPPDATA", os.path.join(os.getenv("LOCALAPPDATA", ""), "Temp")
        ),
        state=FolderSpec("LOCALAPPDATA", os.getenv("LOCALAPPDATA", "")),
        runtime=FolderSpec("TEMP", os.getenv("TEMP", "")),
    )
else:
    APP_FOLDER_STANDARDS = dict(
        config=FolderSpec("XDG_CONFIG_HOME", "~/.config"),
        data=FolderSpec("XDG_DATA_HOME", "~/.local/share"),
        cache=FolderSpec("XDG_CACHE_HOME", "~/.cache"),
        state=FolderSpec("XDG_STATE_HOME", "~/.local/state"),
        runtime=FolderSpec("XDG_RUNTIME_DIR", "/tmp"),
    )

AppFolderKind = Literal["config", "data", "cache", "state", "runtime"]
DFLT_APP_FOLDER_KIND = "config"


def get_app_folder(folder_kind: AppFolderKind = DFLT_APP_FOLDER_KIND):
    """
    Get the full path of a directory suitable for storing application-specific configs,
    (or data, or cache, or state or runtime)

    On Windows, this is typically %APPDATA%.
    On macOS, this is typically ~/.config.
    On Linux, this is typically ~/.config.

    Parameters:
        folder_kind (str): The kind of folder to get. One of 'config', 'data', 'cache', 'state', 'runtime'.
            Defaults to 'config'.
            Here are concise explanations for each folder kind:
            **config**: User preferences and settings files (e.g., API keys, theme preferences, editor settings). Files users might edit manually or that define how the app behaves.
            **data**: Essential user-created content and application state (e.g., databases, saved games, user documents, session files). Data that should be backed up and persists across updates.
            **cache**: Temporary, regeneratable files (e.g., downloaded images, compiled assets, web cache). Can be safely deleted to free space without losing user work.
            **state**: Application state and logs that persist between sessions but aren't critical user data (e.g., command history, undo history, recently opened files, log files). Unlike cache, shouldn't be auto-deleted.
            **runtime**: Temporary runtime files that only exist while the app runs (e.g., PID files, Unix sockets, lock files, named pipes). Typically cleared on logout/reboot.
            **TL;DR**: config = settings, data = user files, cache = disposable, state = logs/history, runtime = process files.

    Returns:
        str: The full path of the app data folder.

    See https://github.com/i2mint/i2mint/issues/1.
    """
    env_var, default = APP_FOLDER_STANDARDS[folder_kind]
    return os.path.expanduser(os.getenv(env_var, default))


get_app_config_folder = partial(get_app_folder, folder_kind="config")
get_app_data_folder = partial(get_app_folder, folder_kind="data")
```

## zipfiledol.py

```python
"""
Data object layers and other utils to work with zip files.
"""

import os
from pathlib import Path
from io import BytesIO
from functools import partial, wraps
from typing import Union, Literal
from collections.abc import Callable, Iterable, Mapping
import zipfile
from zipfile import (
    ZipFile,
    BadZipFile,
    ZIP_STORED,
    ZIP_DEFLATED,
    ZIP_BZIP2,
    ZIP_LZMA,
)
from dol.base import KvReader, KvPersister
from dol.trans import filt_iter
from dol.filesys import FileCollection, Files
from dol.util import lazyprop, fullpath
from dol.sources import FlatReader

__all__ = [
    "COMPRESSION",
    "DFLT_COMPRESSION",
    "compression_methods",
    "zip_compress",
    "zip_decompress",
    "to_zip_file",
    "file_or_folder_to_zip_file",
    "if_i_zipped_stats",
    "ZipReader",
    "ZipInfoReader",
    "ZipFilesReader",
    "ZipFilesReaderAndBytesWriter",
    "FlatZipFilesReader",
    "mk_flatzips_store",
    "FilesOfZip",
    "FileStreamsOfZip",
    "ZipFileStreamsReader",
    "OverwriteNotAllowed",
    "EmptyZipError",
    "ZipStore",
    "ZipFiles",
    "remove_some_entries_from_zip",
    "remove_mac_junk_from_zip",
]

# TODO: Do all systems have this? If not, need to choose dflt carefully
#  (choose dynamically?)
DFLT_COMPRESSION = zipfile.ZIP_DEFLATED
DFLT_ENCODING = "utf-8"


class COMPRESSION:
    # The numeric constant for an uncompressed archive member.
    ZIP_STORED = ZIP_STORED
    # The numeric constant for the usual ZIP compression method. Requires zlib module.
    ZIP_DEFLATED = ZIP_DEFLATED
    # The numeric constant for the BZIP2 compression method. Requires the bz2 module:
    ZIP_BZIP2 = ZIP_BZIP2
    # The numeric constant for the LZMA compression method. Requires the lzma module:
    ZIP_LZMA = ZIP_LZMA


compression_methods = {
    "stored": zipfile.ZIP_STORED,  # doesn't even compress
    "deflated": zipfile.ZIP_DEFLATED,  # usual zip compression method
    "bzip2": zipfile.ZIP_BZIP2,  # BZIP2 compression method.
    "lzma": zipfile.ZIP_LZMA,  # LZMA compression method
}


def take_everything(fileinfo):
    return True


def zip_compress(
    b: bytes | str,
    filename="some_bytes",
    *,
    compression=DFLT_COMPRESSION,
    allowZip64=True,
    compresslevel=None,
    strict_timestamps=True,
    encoding=DFLT_ENCODING,
) -> bytes:
    """Compress input bytes, returning the compressed bytes

    >>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
    >>> len(b)
    2000
    >>>
    >>> zipped_bytes = zip_compress(b)
    >>> # Note: Compression details will be system dependent
    >>> len(zipped_bytes)  # doctest: +SKIP
    137
    >>> unzipped_bytes = zip_decompress(zipped_bytes)
    >>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
    True
    >>>
    >>> from dol.zipfiledol import compression_methods
    >>>
    >>> zipped_bytes = zip_compress(b, compression=compression_methods['bzip2'])
    >>> # Note: Compression details will be system dependent
    >>> len(zipped_bytes)  # doctest: +SKIP
    221
    >>> unzipped_bytes = zip_decompress(zipped_bytes)
    >>> unzipped_bytes == b  # verify that unzipped bytes are the same as the original
    True
    """
    kwargs = dict(
        compression=compression,
        allowZip64=allowZip64,
        compresslevel=compresslevel,
        strict_timestamps=strict_timestamps,
    )
    bytes_buffer = BytesIO()
    if isinstance(b, str):  # if b is a string, need to convert to bytes
        b = b.encode(encoding)
    with ZipFile(bytes_buffer, "w", **kwargs) as fp:
        fp.writestr(filename, b)
    return bytes_buffer.getvalue()


def zip_decompress(
    b: bytes,
    *,
    allowZip64=True,
    compresslevel=None,
    strict_timestamps=True,
) -> bytes:
    """Decompress input bytes of a single file zip, returning the uncompressed bytes

    See ``zip_compress`` for usage examples.
    """
    kwargs = dict(
        allowZip64=allowZip64,
        compresslevel=compresslevel,
        strict_timestamps=strict_timestamps,
    )
    bytes_buffer = BytesIO(b)
    with ZipFile(bytes_buffer, "r", **kwargs) as zip_file:
        file_list = zip_file.namelist()
        if len(file_list) != 1:
            raise RuntimeError("zip_decompress only works with single file zips")
        filename = file_list[0]
        with zip_file.open(filename, "r") as fp:
            file_bytes = fp.read()
    return file_bytes


def _filename_from_zip_path(path):
    filename = path  # default
    if path.endswith(".zip"):
        filename, _ = os.path.splitext(os.path.basename(path))
    return filename


# TODO: Look into pwd: Should we use it for setting pwd when pwd doesn't exist?
def to_zip_file(
    b: bytes | str,
    zip_filepath,
    filename=None,
    *,
    compression=DFLT_COMPRESSION,
    allow_overwrites=True,
    pwd=None,
    encoding=DFLT_ENCODING,
):
    """Zip input bytes and save to a single-file zip file.

    :param b: Input bytes or string
    :param zip_filepath: zip filepath to save the zipped input to
    :param filename: The name/path of the zip entry we want to save to
    :param encoding: In case the input is str, the encoding to use to convert to bytes

    """
    z = ZipFiles(
        zip_filepath,
        compression=compression,
        allow_overwrites=allow_overwrites,
        pwd=pwd,
    )
    filename = filename or _filename_from_zip_path(zip_filepath)
    if isinstance(b, str):  # if b is a string, need to convert to bytes
        b = b.encode(encoding)
    z[filename] = b


def file_or_folder_to_zip_file(
    src_path: str,
    zip_filepath=None,
    filename=None,
    *,
    compression=DFLT_COMPRESSION,
    allow_overwrites=True,
    pwd=None,
):
    """Zip input bytes and save to a single-file zip file."""

    if zip_filepath is None:
        zip_filepath = os.path.basename(src_path) + ".zip"

    z = ZipFiles(
        zip_filepath,
        compression=compression,
        allow_overwrites=allow_overwrites,
        pwd=pwd,
    )

    if os.path.isfile(src_path):
        filename = filename or os.path.basename(src_path)
        z[filename] = Path(src_path).read_bytes()
    elif os.path.isdir(src_path):
        src = Files(src_path)
        for k, v in src.items():
            z[k] = v
    else:
        raise FileNotFoundError(f"{src_path}")


def if_i_zipped_stats(b: bytes):
    """Compress and decompress bytes with four different methods and return a dictionary
    of (size and time) stats.

    >>> b = b'x' * 1000 + b'y' * 1000  # 2000 (quite compressible) bytes
    >>> if_i_zipped_stats(b)  # doctest: +SKIP
    {'uncompressed': {'bytes': 2000,
      'comp_time': 0,
      'uncomp_time': 0},
     'deflated': {'bytes': 137,
      'comp_time': 0.00015592575073242188,
      'uncomp_time': 0.00012612342834472656},
     'bzip2': {'bytes': 221,
      'comp_time': 0.0013129711151123047,
      'uncomp_time': 0.0011119842529296875},
     'lzma': {'bytes': 206,
      'comp_time': 0.0058901309967041016,
      'uncomp_time': 0.0005228519439697266}}
    """
    import time

    stats = dict()
    stats["uncompressed"] = {"bytes": len(b), "comp_time": 0, "uncomp_time": 0}
    for name, compression in compression_methods.items():
        if name != "stored":
            try:
                stats[name] = dict.fromkeys(stats["uncompressed"])
                tic = time.time()
                compressed = zip_compress(b, compression=compression)
                elapsed = time.time() - tic
                stats[name]["bytes"] = len(compressed)
                stats[name]["comp_time"] = elapsed
                tic = time.time()
                uncompressed = zip_decompress(compressed)
                elapsed = time.time() - tic
                assert (
                    uncompressed == b
                ), "the uncompressed bytes were different than the original"
                stats[name]["uncomp_time"] = elapsed
            except Exception:
                raise
                pass
    return stats


class ZipReader(KvReader):
    r"""A KvReader to read the contents of a zip file.
    Provides a KV perspective of https://docs.python.org/3/library/zipfile.html

    ``ZipReader`` has two value categories: Directories and Files.
    Both categories are distinguishable by the keys, through the "ends with slash" convention.

    When a file, the value return is bytes, as usual.

    When a directory, the value returned is a ``ZipReader`` itself, with all params the same,
    except for the ``prefix``
     which serves `to specify the subfolder (that is, ``prefix`` acts as a filter).

    Note: If you get data zipped by a mac, you might get some junk along with it.
    Namely `__MACOSX` folders `.DS_Store` files. I won't rant about it, since others have.
    But you might find it useful to remove them from view. One choice is to use
    `dol.trans.filt_iter`
    to get a filtered view of the zips contents. In most cases, this should do the job:

    .. code-block::

        # applied to store instance or class:
        store = filt_iter(filt=lambda x: not x.startswith('__MACOSX') and '.DS_Store' not in x)(store)


    Another option is just to remove these from the zip file once and for all. In unix-like systems:

    .. code-block::

        zip -d filename.zip __MACOSX/\*
        zip -d filename.zip \*/.DS_Store


    Examples:

    .. code-block::

        # >>> s = ZipReader('/path/to/some_zip_file.zip')
        # >>> len(s)
        # 53432
        # >>> list(s)[:3]  # the first 3 elements (well... their keys)
        # ['odir/', 'odir/app/', 'odir/app/data/']
        # >>> list(s)[-3:]  # the last 3 elements (well... their keys)
        # ['odir/app/data/audio/d/1574287049078391/m/Ctor.json',
        #  'odir/app/data/audio/d/1574287049078391/m/intensity.json',
        #  'odir/app/data/run/status.json']
        # >>> # getting a file (note that by default, you get bytes, so need to decode)
        # >>> s['odir/app/data/run/status.json'].decode()
        # b'{"test_phase_number": 9, "test_phase": "TestActions.IGNORE_TEST", "session_id": 0}'
        # >>> # when you ask for the contents for a key that's a directory,
        # >>> # you get a ZipReader filtered for that prefix:
        # >>> s['odir/app/data/audio/']
        # ZipReader('/path/to/some_zip_file.zip', 'odir/app/data/audio/', {}, <function
        take_everything at 0x1538999e0>)
        # >>> # Often, you only want files (not directories)
        # >>> # You can filter directories out using the file_info_filt argument
        # >>> s = ZipReader('/path/to/some_zip_file.zip', file_info_filt=ZipReader.FILES_ONLY)
        # >>> len(s)  # compare to the 53432 above, that contained dirs too
        # 53280
        # >>> list(s)[:3]  # first 3 keys are all files now
        # ['odir/app/data/plc/d/1574304926795633/d/1574305026895702',
        #  'odir/app/data/plc/d/1574304926795633/d/1574305276853053',
        #  'odir/app/data/plc/d/1574304926795633/d/1574305159343326']
        # >>>
        # >>> # ZipReader.FILES_ONLY and ZipReader.DIRS_ONLY are just convenience filt functions
        # >>> # Really, you can provide any custom one yourself.
        # >>> # This filter function should take a ZipInfo object, and return True or False.
        # >>> # (https://docs.python.org/3/library/zipfile.html#zipfile.ZipInfo)
        # >>>
        # >>> import re
        # >>> p = re.compile('audio.*\.json$')
        # >>> my_filt_func = lambda fileinfo: bool(p.search(fileinfo.filename))
        # >>> s = ZipReader('/Users/twhalen/Downloads/2019_11_21.zip', file_info_filt=my_filt_func)
        # >>> len(s)
        # 48
        # >>> list(s)[:3]
        # ['odir/app/data/audio/d/1574333557263758/m/Ctor.json',
        #  'odir/app/data/audio/d/1574333557263758/m/intensity.json',
        #  'odir/app/data/audio/d/1574288084739961/m/Ctor.json']
    """

    def __init__(
        self,
        zip_file,
        prefix="",
        *,
        open_kws=None,
        file_info_filt=None,
    ):
        """

        Args:
            zip_file: A path to make ZipFile(zip_file)
            prefix: A prefix to filter by.
            open_kws:  To be used when doing a ZipFile(...).open
            file_info_filt: Filter for the FileInfo objects (see
            https://docs.python.org/3/library/zipfile.html)
                of the paths listed in the zip file
        """
        self.open_kws = open_kws or {}
        self.file_info_filt = file_info_filt or ZipReader.EVERYTHING
        self.prefix = prefix
        if not isinstance(zip_file, ZipFile):
            if isinstance(zip_file, str):
                zip_file = fullpath(zip_file)
            if isinstance(zip_file, dict):
                zip_file = ZipFile(**zip_file)
            elif isinstance(zip_file, (tuple, list)):
                zip_file = ZipFile(*zip_file)
            elif isinstance(zip_file, bytes):
                zip_file = ZipFile(BytesIO(zip_file))
            else:
                zip_file = ZipFile(zip_file)
        self.zip_file = zip_file

    @classmethod
    def for_files_only(cls, zip_file, prefix="", open_kws=None, file_info_filt=None):
        if file_info_filt is None:
            file_info_filt = ZipReader.FILES_ONLY
        else:
            _file_info_filt = file_info_filt

            def file_info_filt(x):
                return ZipReader.FILES_ONLY(x) and _file_info_filt(x)

        return cls(zip_file, prefix, open_kws, file_info_filt)

    # TODO: Unaware of trans (filters, key trans, etc.)
    @lazyprop
    def info_for_key(self):
        return {
            x.filename: x
            for x in self.zip_file.infolist()
            if x.filename.startswith(self.prefix) and self.file_info_filt(x)
        }

    def __iter__(self):
        # using zip_file.infolist(), we could also filter for info (like directory/file)
        yield from self.info_for_key.keys()

    def __getitem__(self, k):
        if not self.info_for_key[k].is_dir():
            with self.zip_file.open(k, **self.open_kws) as fp:
                return fp.read()
        else:  # is a directory
            return self.__class__(self.zip_file, k, self.open_kws, self.file_info_filt)

    def __len__(self):
        return len(self.info_for_key)

    @staticmethod
    def FILES_ONLY(fileinfo):
        return not fileinfo.is_dir()

    @staticmethod
    def DIRS_ONLY(fileinfo):
        return fileinfo.is_dir()

    @staticmethod
    def EVERYTHING(fileinfo):
        return True

    def __repr__(self):
        args_str = ", ".join(
            (
                f"'{self.zip_file.filename}'",
                f"'{self.prefix}'",
                f"{self.open_kws}",
                f"{self.file_info_filt}",
            )
        )
        return f"{self.__class__.__name__}({args_str})"

    # TODO: Unaware of trans (filters, key trans, etc.)
    def get_info_reader(self):
        return ZipInfoReader(
            zip_file=self.zip_file,
            prefix=self.prefix,
            open_kws=self.open_kws,
            file_info_filt=self.file_info_filt,
        )


class ZipInfoReader(ZipReader):
    def __getitem__(self, k):
        return self.zip_file.getinfo(k)


class FilesOfZip(ZipReader):
    def __init__(self, zip_file, prefix="", open_kws=None):
        super().__init__(
            zip_file,
            prefix=prefix,
            open_kws=open_kws,
            file_info_filt=ZipReader.FILES_ONLY,
        )


# TODO: This file object item is more fundemental than file contents.
#  Should it be at the base?
class FileStreamsOfZip(FilesOfZip):
    """Like FilesOfZip, but object returns are file streams instead.
    So you use it like this:

    .. code-block::

        z = FileStreamsOfZip(rootdir)
        with z[relpath] as fp:
            ...  # do stuff with fp, like fp.readlines() or such...

    """

    def __getitem__(self, k):
        return self.zip_file.open(k, **self.open_kws)


class ZipFilesReader(FileCollection, KvReader):
    """A local file reader whose keys are the zip filepaths of the rootdir and values are
    corresponding ZipReaders.
    """

    def __init__(
        self,
        rootdir,
        subpath=r".+\.zip",
        pattern_for_field=None,
        max_levels=0,
        zip_reader=ZipReader,
        **zip_reader_kwargs,
    ):
        super().__init__(rootdir, subpath, pattern_for_field, max_levels)
        self.zip_reader = zip_reader
        self.zip_reader_kwargs = zip_reader_kwargs
        if self.zip_reader is ZipReader:
            self.zip_reader_kwargs = dict(
                dict(
                    prefix="",
                    open_kws=None,
                    file_info_filt=ZipReader.FILES_ONLY,
                ),
                **self.zip_reader_kwargs,
            )

    def __getitem__(self, k):
        try:
            return self.zip_reader(k, **self.zip_reader_kwargs)
        except FileNotFoundError as e:
            raise KeyError(f"FileNotFoundError: {e}")


class ZipFilesReaderAndBytesWriter(ZipFilesReader):
    """Like ZipFilesReader, but the ability to write bytes (assumed to be valid bytes of
    the zip format) to a key
    """

    def __setitem__(self, k, v):
        with open(k, "wb") as fp:
            fp.write(v)


ZipFileReader = ZipFilesReader  # back-compatibility alias


# TODO: Add easy connection to ExplicitKeymapReader and other path trans and cache useful
#  for the folder of zips context
# TODO: The "injection" of _readers to be able to use FlatReader stinks.
class FlatZipFilesReader(FlatReader, ZipFilesReader):
    """Read the union of the contents of multiple zip files.
    A local file reader whose keys are the zip filepaths of the rootdir and values are
    corresponding ZipReaders.

    Example use case:

    A remote data provider creates snapshots of whatever changed (modified files and new
    ones...) since the last snapshot, dumping snapshot zip files in a specic
    accessible location.

    You make `remote` and `local` stores and can update your local. Then you can perform
    syncing actions such as:

    .. code-block:: python

        missing_keys = remote.keys() - local.keys()
        local.update({k: remote[k] for k in missing_keys})  # downloads missing snapshots


    The data will look something like this:

    .. code-block:: python

        dump_folder/
           2021_09_11.zip
           2021_09_12.zip
           2021_09_13.zip
           etc.

    both on remote and local.

    What should then local do to use this data?
    Unzip and merge?

    Well, one solution, provided through FlatZipFilesReader, is to not unzip at all,
    but instead, give you a store that provides you a view "as if you unzipped and
    merged".

    """

    __init__ = ZipFilesReader.__init__

    @lazyprop
    def _readers(self):
        rootdir_len = len(self.rootdir)
        return {
            path[rootdir_len:]: ZipFilesReader.__getitem__(self, path)
            for path in ZipFilesReader.__iter__(self)
        }

    _zip_readers = _readers  # back-compatibility alias


# TODO: Refactor zipfiledol to make it possible to design FlatZipFilesReaderFromBytes
#  better than the following.
#  * init doesn't use super, but super is locked to rootdir specification
#  * perhaps better making _readers a lazy mapping (not precompute all FilesOfZip(v))?
#  * Should ZipFilesReader be generalized to take bytes instead of rootdir?
#  * Using .zips to delegate the what in is
class FlatZipFilesReaderFromBytes(FlatReader, FilesOfZip):
    """Like FlatZipFilesReader but instead of sourcing with folder of zips, we source
    with the bytes of a zipped folder of zips"""

    @wraps(FilesOfZip.__init__)
    def __init__(self, *args, **kwargs):
        self.zips = FilesOfZip(*args, **kwargs)

    @lazyprop
    def _readers(self):
        return {k: FilesOfZip(v) for k, v in self.zips.items()}


def mk_flatzips_store(
    dir_of_zips,
    zip_pair_path_preproc=sorted,
    mk_store=FlatZipFilesReader,
    **extra_mk_store_kwargs,
):
    """A store so that you can work with a folder that has a bunch of zip files,
    as if they've all been extracted in the same folder.
    Note that `zip_pair_path_preproc` can be used to control how to resolve key conflicts
    (i.e. when you get two different zip files that have a same path in their contents).
    The last path encountered by `zip_pair_path_preproc(zip_path_pairs)` is the one that
    will be used, so one should make `zip_pair_path_preproc` act accordingly.
    """
    from dol.explicit import ExplicitKeymapReader

    z = mk_store(dir_of_zips, **extra_mk_store_kwargs)
    path_to_pair = {pair[1]: pair for pair in zip_pair_path_preproc(z)}
    return ExplicitKeymapReader(z, id_of_key=path_to_pair)


from dol.paths import mk_relative_path_store
from dol.util import partialclass

ZipFileStreamsReader = mk_relative_path_store(
    partialclass(ZipFilesReader, zip_reader=FileStreamsOfZip),
    prefix_attr="rootdir",
)
ZipFileStreamsReader.__name__ = "ZipFileStreamsReader"
ZipFileStreamsReader.__qualname__ = "ZipFileStreamsReader"
ZipFileStreamsReader.__doc__ = (
    """Like ZipFilesReader, but objects returned are file streams instead."""
)

from dol.errors import OverWritesNotAllowedError


class OverwriteNotAllowed(FileExistsError, OverWritesNotAllowedError): ...


class EmptyZipError(KeyError, FileNotFoundError): ...


class _EmptyZipReader(KvReader):
    def __init__(self, zip_filepath):
        self.zip_filepath = zip_filepath

    def __iter__(self):
        yield from ()

    def infolist(self):
        return []

    def __getitem__(self, k):
        raise EmptyZipError(
            "The store is empty: ZipFiles(zip_filepath={self.zip_filepath})"
        )

    def open(self, *args, **kwargs):
        raise EmptyZipError(
            f"The zip file doesn't exist yet! Nothing was written in it: {self.zip_filepath}"
        )
        #
        # class OpenedNotExistingFile:
        #     zip_filepath = self.zip_filepath
        #
        #     def read(self):
        #         raise EmptyZipError(
        #             f"The zip file doesn't exist yet! Nothing was written in it: {
        #             self.zip_filepath}")
        #
        #     def __enter__(self, ):
        #         return self
        #
        #     def __exit__(self, *exc):
        #         return False
        #
        # return OpenedNotExistingFile()


# TODO: Revise ZipReader and ZipFilesReader architecture and make ZipFiles be a subclass of
#  Reader if poss
# TODO: What if I just want to zip a (single) file. What does dol offer for that?
# TODO: How about set_obj (in misc.py)? Make it recognize the .zip extension and subextension (
#  e.g. .txt.zip) serialize
class ZipFiles(KvPersister):
    """Zip read and writing.
    When you want to read zips, there's the `FilesOfZip`, `ZipReader`, or `ZipFilesReader` we
    know and love.

    Sometimes though, you want to write to zips too. For this, we have `ZipFiles`.

    Since ZipFiles can write to a zip, it's read functionality is not going to assume static data,
    and cache things, as your favorite zip readers did.
    This, and the acrobatics need to disguise the weird zipfile into something more... key-value
    natural,
    makes for a not so efficient store, out of the box.

    I advise using one of the zip readers if all you need to do is read, or subclassing or
     wrapping ZipFiles with caching layers if it is appropriate to you.

    Let's verify that a ZipFiles can indeed write data. First, we'll set things up!

    >>> from tempfile import gettempdir
    >>> import os
    >>>
    >>> rootdir = gettempdir()
    >>>
    >>> # preparation
    >>> test_zipfile = os.path.join(rootdir, 'zipstore_test_file.zip')
    >>> if os.path.isfile(test_zipfile):
    ...     os.remove(test_zipfile)
    >>> assert not os.path.isfile(test_zipfile)

    Okay, test_zipfile doesn't exist (but will soon...)

    >>> z = ZipFiles(test_zipfile)

    See that the file still doesn't exist (it will only be created when we start writing)

    >>> assert not os.path.isfile(test_zipfile)
    >>> list(z)  # z "is" empty (which makes sense?)
    []

    Now let's write something interesting (notice, it has to be in bytes):

    >>> z['foo'] = b'bar'
    >>> list(z)  # now we have something in z
    ['foo']
    >>> z['foo']  # and that thing is what we put there
    b'bar'

    And indeed we have a zip file now:

    >>> assert os.path.isfile(test_zipfile)

    """

    _zipfile_init_kw = dict(
        compression=DFLT_COMPRESSION,
        allowZip64=True,
        compresslevel=None,
        strict_timestamps=True,
    )
    _open_kw = dict(pwd=None, force_zip64=False)
    _writestr_kw = dict(compress_type=None, compresslevel=None)
    zip_writer = None

    # @wraps(ZipReader.__init__)
    def __init__(
        self,
        zip_filepath,
        compression=DFLT_COMPRESSION,
        allow_overwrites=True,
        pwd=None,
    ):
        self.zip_filepath = fullpath(zip_filepath)
        self.zip_filepath = zip_filepath
        self.zip_writer_opened = False
        self.allow_overwrites = allow_overwrites
        self._zipfile_init_kw = dict(self._zipfile_init_kw, compression=compression)
        self._open_kw = dict(self._open_kw, pwd=pwd)

    @staticmethod
    def files_only_filt(fileinfo):
        return not fileinfo.is_dir()

    @property
    def zip_reader(self):
        if os.path.isfile(self.zip_filepath):
            return ZipFile(self.zip_filepath, mode="r", **self._zipfile_init_kw)
        else:
            return _EmptyZipReader(self.zip_filepath)

    def __iter__(self):
        # using zip_file.infolist(), we could also filter for info (like directory/file)
        yield from (
            fi.filename for fi in self.zip_reader.infolist() if self.files_only_filt(fi)
        )

    def __getitem__(self, k):
        with self.zip_reader.open(k, **dict(self._open_kw, mode="r")) as fp:
            return fp.read()

    def __repr__(self):
        args_str = ", ".join(
            (
                f"'{self.zip_filepath}'",
                f"'allow_overwrites={self.allow_overwrites}'",
            )
        )
        return f"{self.__class__.__name__}({args_str})"

    def __contains__(self, k):
        try:
            with self.zip_reader.open(k, **dict(self._open_kw, mode="r")) as fp:
                pass
            return True
        except (
            KeyError,
            BadZipFile,
        ):  # BadZipFile is to catch when zip file exists, but is empty.
            return False

    # # TODO: Find better way to avoid duplicate keys!
    # # TODO: What's the right Error to raise
    # def _assert_non_existing_key(self, k):
    #     # if self.zip_writer is not None:
    #     if not self.zip_writer_opened:
    #         try:
    #             self.zip_reader.open(k)
    #             raise OverwriteNotAllowed(f"You're not allowed to overwrite an existing key: {k}")
    #         except KeyError as e:
    #             if isinstance(e, EmptyZipError) or e.args[-1].endswith('archive'):
    #                 pass  #
    #             else:
    #                 raise OverwriteNotAllowed(f"You're not allowed to overwrite an existing
    #                 key: {k}")

    # TODO: Repeated with zip_writer logic. Consider DRY possibilities.
    def __setitem__(self, k, v):
        if k in self:
            if self.allow_overwrites and not self.zip_writer_opened:
                del self[k]  # remove key so it can be overwritten
            else:
                if self.zip_writer_opened:
                    raise OverwriteNotAllowed(
                        f"When using the context mode, you're not allowed to overwrite an "
                        f"existing key: {k}"
                    )
                else:
                    raise OverwriteNotAllowed(
                        f"You're not allowed to overwrite an existing key: {k}"
                    )

        if self.zip_writer_opened:
            with self.zip_writer.open(k, **dict(self._open_kw, mode="w")) as fp:
                return fp.write(v)
        else:
            with ZipFile(
                self.zip_filepath, mode="a", **self._zipfile_init_kw
            ) as zip_writer:
                with zip_writer.open(k, **dict(self._open_kw, mode="w")) as fp:
                    return fp.write(v)

    def __delitem__(self, k):
        try:
            os.system(f"zip -d {self.zip_filepath} {k}")
        except Exception as e:
            raise KeyError(f"{e.__class__}: {e.args}")
        # raise NotImplementedError("zipfile, the backend of ZipFiles, doesn't support deletion,
        # so neither will we.")

    def open(self):
        self.zip_writer = ZipFile(self.zip_filepath, mode="a", **self._zipfile_init_kw)
        self.zip_writer_opened = True
        return self

    def close(self):
        if self.zip_writer is not None:
            self.zip_writer.close()
        self.zip_writer_opened = False

    __enter__ = open

    def __exit__(self, *exc):
        self.close()
        return False


ZipStore = ZipFiles  # back-compatibility alias
PathString = str
PathFilterFunc = Callable[[PathString], bool]


def _not_in(excluded, obj=None):
    if obj is None:
        return partial(_not_in, excluded)
    return obj not in excluded


def remove_some_entries_from_zip(
    zip_source,
    keys_to_be_removed: PathFilterFunc | Iterable[PathString],
    ask_before_before_deleting=True,
    *,
    remove_action: Literal["delete", "filter"] = "filter",
):
    """Removes specific keys from a zip file.

    :param zip_source: zip filepath, bytes, or whatever a ``ZipFiles`` can take
    :param keys_to_be_removed: An iterable of keys or a boolean filter function
    :param ask_before_before_deleting: True (default) if the user should be
        presented with the keys first, and asked permission to delete.
    :return: The ZipFiles (in case you want to do further work with it)

    Tip: If you want to delete with no questions asked, use currying:

    >>> from functools import partial
    >>> rm_keys_without_asking = partial(
    ...     remove_some_entries_from_zip,
    ...     ask_before_before_deleting=False
    ... )

    """
    z = zip_source
    if not isinstance(z, Mapping):
        z = ZipFiles(z)
    if not isinstance(keys_to_be_removed, Callable):
        if isinstance(keys_to_be_removed, str):
            keys_to_be_removed = [keys_to_be_removed]
        assert isinstance(keys_to_be_removed, Iterable)
        keys_to_be_removed = lambda x: x in set(keys_to_be_removed)
    keys_that_will_be_deleted = list(filter(keys_to_be_removed, z))
    if keys_that_will_be_deleted:
        if remove_action == "delete":
            if ask_before_before_deleting:
                print("These keys will be removed:\n\r")
                print(*keys_that_will_be_deleted, sep="\n")
                n = len(keys_that_will_be_deleted)
                answer = input(f"\nShould I go ahead and delete these {n} keys? (y/N)")
                if not answer == "y":
                    print("Okay, I will NOT delete these.")
                    return
            for k in keys_that_will_be_deleted:
                del z[k]
        else:  # remove_action == 'filter'
            z = filt_iter(z, filt=_not_in(keys_that_will_be_deleted))

    return z


from dol.util import not_a_mac_junk_path


def is_a_mac_junk_path(path):
    return not not_a_mac_junk_path(path)


remove_mac_junk_from_zip = partial(
    remove_some_entries_from_zip,
    keys_to_be_removed=is_a_mac_junk_path,
    ask_before_before_deleting=False,
)
remove_mac_junk_from_zip.__doc__ = "Removes mac junk keys from zip"

# TODO: The way prefix and file_info_filt is handled is not efficient
# TODO: prefix is silly: less general than filename_filt would be, and not even producing
#  relative paths
#  (especially when getitem returns subdirs)


# trans alternative:
# from dol.trans import mk_kv_reader_from_kv_collection, wrap_kvs
#
# ZipFileReader = wrap_kvs(mk_kv_reader_from_kv_collection(FileCollection, name='_ZipFileReader'),
#                          name='ZipFileReader',
#                          obj_of_data=ZipReader)


# ----------------------------- Extras -------------------------------------------------


def tar_compress(data_bytes, file_name="data.bin"):
    import tarfile
    import io

    with io.BytesIO() as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            data_file = io.BytesIO(data_bytes)
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data_bytes)
            tar.addfile(tarinfo, fileobj=data_file)
        return tar_buffer.getvalue()


def tar_decompress(tar_bytes):
    import tarfile
    import io

    with io.BytesIO(tar_bytes) as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode="r:") as tar:
            for member in tar.getmembers():
                extracted_file = tar.extractfile(member)
                if extracted_file:
                    return extracted_file.read()
    return None
```

## README.md

```python
# dol

Base builtin tools make and transform data object layers (dols).

The main idea comes in many names such as 
[Data Access Object (DAO)](https://en.wikipedia.org/wiki/Data_access_object),
[Repository Pattern](https://www.cosmicpython.com/book/chapter_02_repository.html),
[Hexagonal architecture, or ports and adapters architecture](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
for data. 
But simply put, what `dol` provides is tools to make your interface with data be domain-oriented, simple, and isolated from the underlying data infrastucture. This makes the business logic code simple and stable, enables you to develop and test it without the need of any data infrastructure, and allows you to change this infrastructure independently.

The package is light-weight: Pure python; no third-party dependencies.

To install:	```pip install dol```

[Documentation here](https://i2mint.github.io/dol/)

## Example use

Say you have a source backend that has pickles of some lists-of-lists-of-strings, 
using the `.pkl` extension, and you want to copy this data to a target backend, 
but saving them as gzipped csvs with the `csv.gz` extension. 

We'll first work with dictionaries instead of files here, so we can test more easily, 
and safely.

```python
import pickle

src_backend = {
    'file_1.pkl': pickle.dumps([['A', 'B', 'C'], ['one', 'two', 'three']]),
    'file_2.pkl': pickle.dumps([['apple', 'pie'], ['one', 'two'], ['hot', 'cold']]),
}
targ_backend = dict()
```

Here's how you can do it with `dol` tools

```python
from dol import ValueCodecs, KeyCodecs, Pipe

# decoder here will unpickle data and remove remove the .pkl extension from the key
src_wrap = Pipe(KeyCodecs.suffixed('.pkl'), ValueCodecs.pickle())

# encoder here will convert the lists to csv string, the string into bytes, 
# and the bytes will be gzipped. 
# ... also, we'll add .csv.gz on write.
targ_wrap = Pipe(
    KeyCodecs.suffixed('.csv.gz'), 
    ValueCodecs.csv() + ValueCodecs.str_to_bytes() + ValueCodecs.gzip()
)

# Let's wrap our backends:
src = src_wrap(src_backend)
targ = targ_wrap(targ_backend)

# and copy src over to targ
print(f"Before: {list(targ_backend)=}")
targ.update(src)
print(f"After: {list(targ_backend)=}")
```

From the point of view of src and targ, you see the same thing.

```python
assert list(src) == list(targ) == ['file_1', 'file_2']
assert (
    src['file_1'] 
    == targ['file_1']
    == [['A', 'B', 'C'], ['one', 'two', 'three']]
)
```

But the backend of targ is different:

```python
src_backend['file_1.pkl']
# b'\x80\x04\x95\x19\x00\x00\x00\x00\x00\x00\x00]\x94(]\x94(K\x01K\x02K\x03e]\x94(K\x04K\x05K\x06ee.'
targ_backend['file_1.csv.gz']
# b'\x1f\x8b\x08\x00*YWe\x02\xff3\xd41\xd21\xe6\xe52\xd11\xd51\xe3\xe5\x02\x00)4\x83\x83\x0e\x00\x00\x00'
```

Now that you've tested your setup with dictionaries, you're ready to move on to real, 
persisted storage. If you wanted to do this with local files, you'd:

```python
from dol import Files
src = Files('PATH_TO_LOCAL_SOURCE_FOLDER')
targ = Files('PATH_TO_LOCAL_TARGET_FOLDER)
```

But you could do this with AWS S3 using tools from 
[s3dol](https://github.com/i2mint/s3dol), or Azure using tools from 
[azuredol](https://github.com/i2mint/azuredol), or mongoDB with 
[mongodol](https://github.com/i2mint/mongodol), 
github with [hubcap](https://github.com/thorwhalen/hubcap), and so on...

All of these extensions provide adapters from various data sources/targets to the 
dict-like interface (called "Mapping" in python typing).
What `dol` provides are base tools to make a path from these to the interface 
that makes sense for the domain, or business logic in front of you, 
so that you can purify your code from implementation details, and therefore be
create more robust and flexible code as far as data operations are concerned. 


## A list various packages that use dol

`py2store` provides tools to create the dict-like interface to data you need. 
If you want to just use existing interfaces, build on it, or find examples of how to make such 
interfaces, check out the ever-growing list of `py2store`-using projects:

- [mongodol](https://github.com/i2mint/mongodol): For MongoDB
- [tabled](https://github.com/i2mint/tabled): Data as `pandas.DataFrame` from various sources
- [msword](https://pypi.org/project/msword/): Simple mapping view to docx (Word Doc) elements
- [sshdol](https://github.com/i2mint/sshdol): Remote (ssh) files access
- [haggle](https://github.com/otosense/haggle): Easily search, download, and use kaggle datasets.
- [pyckup](https://github.com/i2mint/pyckup): Grab data simply and define protocols for others to do the same.
- [hubcap](https://pypi.org/project/hubcap/): Dict-like interface to github.
- [graze](https://github.com/thorwhalen/graze): Cache the internet.
- [grub](https://github.com/thorwhalen/grub): A ridiculously simple search engine maker. 
- [hear](https://github.com/otosense/hear): Read/write audio data flexibly. 

Just for fun projects:
- [cult](https://github.com/thorwhalen/cult): Religious texts search engine. 18mn application of `grub`.
- [laugh](https://github.com/thorwhalen/laugh): A (py2store-based) joke finder.


# Caching




# Use cases

## Interfacing reads

How many times did someone share some data with you in the form of a zip of some nested folders 
whose structure and naming choices are fascinatingly obscure? And how much time do you then spend to write code 
to interface with that freak of nature? Well, one of the intents of py2store is to make that easier to do. 
You still need to understand the structure of the data store and how to deserialize these datas into python 
objects you can manipulate. But with the proper tool, you shouldn't have to do much more than that.

## Changing where and how things are stored

Ever have to switch where you persist things (say from file system to S3), or change the way key into your data, 
or the way that data is serialized? If you use py2store tools to separate the different storage concerns, 
it'll be quite easy to change, since change will be localized. And if you're dealing with code that was already 
written, with concerns all mixed up, py2store should still be able to help since you'll be able to
more easily give the new system a facade that makes it look like the old one. 

All of this can also be applied to data bases as well, in-so-far as the CRUD operations you're using 
are covered by the base methods.

## Adapters: When the learning curve is in the way of learning

Shinny new storage mechanisms (DBs etc.) are born constantly, and some folks start using them, and we are eventually lead to use them 
as well if we need to work with those folks' systems. And though we'd love to learn the wonderful new 
capabilities the new kid on the block has, sometimes we just don't have time for that. 

Wouldn't it be nice if someone wrote an adapter to the new system that had an interface we were familiar with? 
Talking to SQL as if it were mongo (or visa versa). Talking to S3 as if it were a file system. 
Now it's not a long term solution: If we're really going to be using the new system intensively, we 
should learn it. But when you just got to get stuff done, having a familiar facade to something new 
is a life saver. 

py2store would like to make it easier for you roll out an adapter to be able to talk 
to the new system in the way **you** are familiar with.
 
## Thinking about storage later, if ever

You have a new project or need to write a new app. You'll need to store stuff and read stuff back. 
Stuff: Different kinds of resources that your app will need to function. Some people enjoy thinking 
of how to optimize that aspect. I don't. I'll leave it to the experts to do so when the time comes. 
Often though, the time is later, if ever. Few proof of concepts and MVPs ever make it to prod. 

So instead, I'd like to just get on with the business logic and write my program. 
So what I need is an easy way to get some minimal storage functionality. 
But when the time comes to optimize, I shouldn't have to change my code, but instead just change the way my 
DAO does things. What I need is py2store.


## Remove data access entropy

Data comes from many different sources, organization, and formats. 

Data is needed in many different contexts, which comes with its own natural data organization and formats. 

In between both: A entropic mess of ad-hoc connections and annoying time-consuming and error prone boilerplate. 

`py2store` (and it's now many extensions) is there to mitigate this. 

The design gods say SOC, DRY, SOLID* and such. That's good design, yes. But it can take more work to achieve these principles. 
We'd like to make it _easier_ to do it right than do it wrong.

_(*) Separation (Of) Concerns, Don't Repeat Yourself, https://en.wikipedia.org/wiki/SOLID))_

We need to determine what are the most common operations we want to do on data, and decide on a common way to express these operations, no matter what the implementation details are. 
- get/read some data
- set/write some data
- list/see what data we have
- filter
- cache
...

Looking at this, we see that the base operations for complex data systems such as data bases and file systems overlap significantly with the base operations on python (or any programming language) objects. 

So we'll reflect this in our choice of a common "language" for these operations. For examples, once projected to a `py2store` object, iterating over the contents of a data base, or over files, or over the elements of a python (iterable) object should look the same, in code. Achieving this, we achieve SOC, but also set ourselves up for tooling that can assume this consistency, therefore be DRY, and many of the SOLID principles of design.

Also mentionable: So far, `py2store` core tools are all pure python -- no dependencies on anything else. 

Now, when you want to specialize a store (say talk to data bases, web services, acquire special formats (audio, etc.)), then you'll need to pull in a few helpful packages. But the core tooling is pure.


# A few words about design

By store we mean key-value store. This could be files in a filesystem, objects in s3, or a database. Where and
how the content is stored should be specified, but StoreInterface offers a dict-like interface to this.

    __getitem__ calls: _id_of_key			                    _obj_of_data
    __setitem__ calls: _id_of_key		        _data_of_obj
    __delitem__ calls: _id_of_key
    __iter__    calls:	            _key_of_id

```pydocstring
>>> from dol import Store
```

A Store can be instantiated with no arguments. By default it will make a dict and wrap that.

```pydocstring
>>> # Default store: no key or value conversion ################################################
>>> s = Store()
>>> s['foo'] = 33
>>> s['bar'] = 65
>>> assert list(s.items()) == [('foo', 33), ('bar', 65)]
>>> assert list(s.store.items()) == [('foo', 33), ('bar', 65)]  # see that the store contains the same thing
```

Now let's make stores that have a key and value conversion layer 
input keys will be upper cased, and output keys lower cased 
input values (assumed int) will be converted to ascii string, and visa versa 

```pydocstring
>>>
>>> def test_store(s):
...     s['foo'] = 33  # write 33 to 'foo'
...     assert 'foo' in s  # __contains__ works
...     assert 'no_such_key' not in s  # __nin__ works
...     s['bar'] = 65  # write 65 to 'bar'
...     assert len(s) == 2  # there are indeed two elements
...     assert list(s) == ['foo', 'bar']  # these are the keys
...     assert list(s.keys()) == ['foo', 'bar']  # the keys() method works!
...     assert list(s.values()) == [33, 65]  # the values() method works!
...     assert list(s.items()) == [('foo', 33), ('bar', 65)]  # these are the items
...     assert list(s.store.items()) == [('FOO', '!'), ('BAR', 'A')]  # but note the internal representation
...     assert s.get('foo') == 33  # the get method works
...     assert s.get('no_such_key', 'something') == 'something'  # return a default value
...     del(s['foo'])  # you can delete an item given its key
...     assert len(s) == 1  # see, only one item left!
...     assert list(s.items()) == [('bar', 65)]  # here it is
>>>
```

We can introduce this conversion layer in several ways. 

Here are few... 

## by subclassing
```pydocstring
>>> # by subclassing ###############################################################################
>>> class MyStore(Store):
...     def _id_of_key(self, k):
...         return k.upper()
...     def _key_of_id(self, _id):
...         return _id.lower()
...     def _data_of_obj(self, obj):
...         return chr(obj)
...     def _obj_of_data(self, data):
...         return ord(data)
>>> s = MyStore(store=dict())  # note that you don't need to specify dict(), since it's the default
>>> test_store(s)
>>>
```

## by assigning functions to converters

```pydocstring
>>> # by assigning functions to converters ##########################################################
>>> class MyStore(Store):
...     def __init__(self, store, _id_of_key, _key_of_id, _data_of_obj, _obj_of_data):
...         super().__init__(store)
...         self._id_of_key = _id_of_key
...         self._key_of_id = _key_of_id
...         self._data_of_obj = _data_of_obj
...         self._obj_of_data = _obj_of_data
...
>>> s = MyStore(dict(),
...             _id_of_key=lambda k: k.upper(),
...             _key_of_id=lambda _id: _id.lower(),
...             _data_of_obj=lambda obj: chr(obj),
...             _obj_of_data=lambda data: ord(data))
>>> test_store(s)
>>>
```

## using a Mixin class

```pydocstring
>>> # using a Mixin class #############################################################################
>>> class Mixin:
...     def _id_of_key(self, k):
...         return k.upper()
...     def _key_of_id(self, _id):
...         return _id.lower()
...     def _data_of_obj(self, obj):
...         return chr(obj)
...     def _obj_of_data(self, data):
...         return ord(data)
...
>>> class MyStore(Mixin, Store):  # note that the Mixin must come before Store in the mro
...     pass
...
>>> s = MyStore()  # no dict()? No, because default anyway
>>> test_store(s)
```

## adding wrapper methods to an already made Store instance

```pydocstring
>>> # adding wrapper methods to an already made Store instance #########################################
>>> s = Store(dict())
>>> s._id_of_key=lambda k: k.upper()
>>> s._key_of_id=lambda _id: _id.lower()
>>> s._data_of_obj=lambda obj: chr(obj)
>>> s._obj_of_data=lambda data: ord(data)
>>> test_store(s)
```


# And more...

## Why the name?

- because it's short
- because it's cute
- because it reminds one of "russian dolls" (one way to think of wrappers)
- because we can come up with an acronym the contains "Data Object" in it. 


## Historical note

Note: This project started as [`py2store`](https://github.com/i2mint/py2store). 
`dol` is the core of py2store has now been factored out 
and many of the specialized data object layers moved to separate packages. 
`py2store` is acting more as an aggregator package -- a shoping mall where you can quickly access many (but not all)
functionalities that use `dol`. 

It's advised to use `dol` (and/or its specialized spin-off packages) directly when the core functionality is all you need.
```