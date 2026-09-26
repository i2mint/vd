# vd.config

Configuration management for vd.

Provides support for configuration files (YAML, TOML), environment variables,
and configuration profiles for managing backend connections.

### Functions

| [`apply_env_overrides`](#vd.config.apply_env_overrides)(config)               | Apply environment variable overrides to configuration.   |
|--------------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`connect_from_config`](#vd.config.connect_from_config)([path, profile, ...]) | Connect to a backend using configuration from a file.    |
| [`create_example_config`](#vd.config.create_example_config)([format])           | Generate an example configuration file content.          |
| [`get_profile`](#vd.config.get_profile)(config[, profile])            | Get a specific profile from configuration.               |
| [`load_config`](#vd.config.load_config)([path, format])               | Load configuration from a file.                          |
| [`load_toml_config`](#vd.config.load_toml_config)(path)                    | Load configuration from a TOML file.                     |
| [`load_yaml_config`](#vd.config.load_yaml_config)(path)                    | Load configuration from a YAML file.                     |
| [`save_config`](#vd.config.save_config)(config, path, \*[, format])   | Save configuration to a file.                            |

### vd.config.apply_env_overrides(config)

Apply environment variable overrides to configuration.

Looks for environment variables with the 

```
VD_
```

 prefix:

- VD_BACKEND: Override backend name
- VD_EMBEDDING_MODEL: Override embedding model

* **Parameters:**
  **config** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Configuration dictionary
* **Returns:**
  Configuration with environment overrides applied
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> import os
>>> os.environ['VD_BACKEND'] = 'chroma'
>>> config = apply_env_overrides({'backend': 'memory'})
>>> config['backend']
'chroma'
```

### vd.config.connect_from_config(path=None, , profile=None, apply_env=True, embedder=None, \*\*overrides)

Connect to a backend using configuration from a file.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Path to configuration file. If not provided, searches for default
    config files.
  * **profile** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Profile name to use from configuration. Defaults to ‘default’ or
    the VD_PROFILE environment variable.
  * **apply_env** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to apply environment variable overrides
  * **embedder** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Optional `text -> vector` convenience embedder, passed to
    [`vd.connect()`](vd.md#vd.connect). A vd config file describes the \*backend
    connection\*, not embedding — embedding stays the caller’s concern.
  * **\*\*overrides** – Additional keyword arguments to override configuration values
* **Returns:**
  Connected client instance
* **Return type:**
  [`Client`](vd.base.md#vd.base.Client)

### Examples

```pycon
>>> # With a config file
>>> client = connect_from_config('vd.yaml')
```

```pycon
>>> # With a specific profile
>>> client = connect_from_config('vd.yaml', profile='production')
```

```pycon
>>> # With environment variable VD_PROFILE=dev
>>> client = connect_from_config()
```

```pycon
>>> # With overrides
>>> client = connect_from_config('vd.yaml', persist_directory='./data')
```

### vd.config.create_example_config(format='yaml')

Generate an example configuration file content.

* **Parameters:**
  **format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Format of configuration: ‘yaml’ or ‘toml’
* **Returns:**
  Example configuration as a string
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> yaml_config = create_example_config('yaml')
>>> print(yaml_config)
>>> toml_config = create_example_config('toml')
```

### vd.config.get_profile(config, profile=None)

Get a specific profile from configuration.

* **Parameters:**
  * **config** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Full configuration dictionary
  * **profile** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Profile name. If not provided, uses ‘default’ or the profile
    specified by the VD_PROFILE environment variable.
* **Returns:**
  Profile configuration
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> config = {'profiles': {'dev': {'backend': 'memory'}, 'prod': {'backend': 'chroma'}}}
>>> dev_config = get_profile(config, 'dev')
>>> prod_config = get_profile(config, 'prod')
```

### vd.config.load_config(path=None, , format=None)

Load configuration from a file.

Automatically detects format from file extension if not specified.

* **Parameters:**
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – Path to configuration file. If not provided, looks for default
    config files in: ./vd.yaml, ./vd.yml, ./vd.toml, ~/.vd/config.yaml, etc.
  * **format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Configuration format: ‘yaml’ or ‘toml’. Auto-detected from extension
    if not provided.
* **Returns:**
  Configuration dictionary
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### Examples

```pycon
>>> config = load_config('vd.yaml')
>>> config = load_config('vd.toml')
>>> config = load_config()  # Looks for default config files
```

### vd.config.load_toml_config(path)

Load configuration from a TOML file.

* **Parameters:**
  **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to TOML configuration file
* **Returns:**
  Configuration dictionary
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If tomli/tomllib is not available
  * [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – If configuration file doesn’t exist

### vd.config.load_yaml_config(path)

Load configuration from a YAML file.

* **Parameters:**
  **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to YAML configuration file
* **Returns:**
  Configuration dictionary
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)
* **Raises:**
  * [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) – If PyYAML is not installed
  * [**FileNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#FileNotFoundError) – If configuration file doesn’t exist

### vd.config.save_config(config, path, , format=None)

Save configuration to a file.

* **Parameters:**
  * **config** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)) – Configuration dictionary to save
  * **path** (`Union`[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)]) – Path to save configuration file
  * **format** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Format to save: ‘yaml’ or ‘toml’. Auto-detected from extension
    if not provided.
* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### Examples

```pycon
>>> config = {
...     'profiles': {
...         'dev': {'backend': 'memory'},
...         'prod': {'backend': 'chroma', 'persist_directory': './data'}
...     }
... }
>>> save_config(config, 'vd.yaml')
```
