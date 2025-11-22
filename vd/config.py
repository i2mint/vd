"""
Configuration management for vd.

Provides support for configuration files (YAML, TOML), environment variables,
and configuration profiles for managing backend connections.
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

from vd.base import Client, Vector


def load_yaml_config(path: Union[str, Path]) -> dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to YAML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    ImportError
        If PyYAML is not installed
    FileNotFoundError
        If configuration file doesn't exist
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config support. "
            "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_toml_config(path: Union[str, Path]) -> dict:
    """
    Load configuration from a TOML file.

    Parameters
    ----------
    path : str or Path
        Path to TOML configuration file

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    ImportError
        If tomli/tomllib is not available
    FileNotFoundError
        If configuration file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Try tomllib (Python 3.11+) first, fall back to tomli
    try:
        import tomllib

        with open(path, 'rb') as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli

            with open(path, 'rb') as f:
                return tomli.load(f)
        except ImportError:
            raise ImportError(
                "tomli is required for TOML config support on Python < 3.11. "
                "Install it with: pip install tomli"
            )


def load_config(
    path: Optional[Union[str, Path]] = None,
    *,
    format: Optional[str] = None,
) -> dict:
    """
    Load configuration from a file.

    Automatically detects format from file extension if not specified.

    Parameters
    ----------
    path : str or Path, optional
        Path to configuration file. If not provided, looks for default
        config files in: ./vd.yaml, ./vd.yml, ./vd.toml, ~/.vd/config.yaml, etc.
    format : str, optional
        Configuration format: 'yaml' or 'toml'. Auto-detected from extension
        if not provided.

    Returns
    -------
    dict
        Configuration dictionary

    Examples
    --------
    >>> config = load_config('vd.yaml')  # doctest: +SKIP
    >>> config = load_config('vd.toml')  # doctest: +SKIP
    >>> config = load_config()  # Looks for default config files  # doctest: +SKIP
    """
    # If no path provided, search for default config files
    if path is None:
        default_paths = [
            Path('vd.yaml'),
            Path('vd.yml'),
            Path('vd.toml'),
            Path('.vd.yaml'),
            Path('.vd.yml'),
            Path('.vd.toml'),
            Path.home() / '.vd' / 'config.yaml',
            Path.home() / '.vd' / 'config.yml',
            Path.home() / '.vd' / 'config.toml',
        ]

        for default_path in default_paths:
            if default_path.exists():
                path = default_path
                break

        if path is None:
            # No config file found, return empty config
            return {}

    path = Path(path)

    # Auto-detect format from extension
    if format is None:
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            format = 'yaml'
        elif suffix == '.toml':
            format = 'toml'
        else:
            raise ValueError(
                f"Cannot detect format from extension '{suffix}'. "
                "Specify format explicitly with format='yaml' or format='toml'"
            )

    # Load config
    if format == 'yaml':
        return load_yaml_config(path)
    elif format == 'toml':
        return load_toml_config(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_profile(
    config: dict,
    profile: Optional[str] = None,
) -> dict:
    """
    Get a specific profile from configuration.

    Parameters
    ----------
    config : dict
        Full configuration dictionary
    profile : str, optional
        Profile name. If not provided, uses 'default' or the profile
        specified by the VD_PROFILE environment variable.

    Returns
    -------
    dict
        Profile configuration

    Examples
    --------
    >>> config = {'profiles': {'dev': {'backend': 'memory'}, 'prod': {'backend': 'chroma'}}}  # doctest: +SKIP
    >>> dev_config = get_profile(config, 'dev')  # doctest: +SKIP
    >>> prod_config = get_profile(config, 'prod')  # doctest: +SKIP
    """
    # Check for environment variable
    if profile is None:
        profile = os.environ.get('VD_PROFILE', 'default')

    # Look for profile in config
    if 'profiles' in config:
        if profile not in config['profiles']:
            available = ', '.join(config['profiles'].keys())
            raise ValueError(
                f"Profile '{profile}' not found in configuration. "
                f"Available profiles: {available}"
            )
        return config['profiles'][profile]

    # If no profiles section, treat entire config as default profile
    if profile != 'default':
        raise ValueError(
            f"Profile '{profile}' not found. Configuration has no profiles section."
        )

    return config


def apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to configuration.

    Looks for environment variables with the VD_ prefix:
    - VD_BACKEND: Override backend name
    - VD_EMBEDDING_MODEL: Override embedding model

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Configuration with environment overrides applied

    Examples
    --------
    >>> import os  # doctest: +SKIP
    >>> os.environ['VD_BACKEND'] = 'chroma'  # doctest: +SKIP
    >>> config = apply_env_overrides({'backend': 'memory'})  # doctest: +SKIP
    >>> config['backend']  # doctest: +SKIP
    'chroma'
    """
    config = config.copy()

    # Override backend
    if 'VD_BACKEND' in os.environ:
        config['backend'] = os.environ['VD_BACKEND']

    # Override embedding model
    if 'VD_EMBEDDING_MODEL' in os.environ:
        config['embedding_model'] = os.environ['VD_EMBEDDING_MODEL']

    return config


def connect_from_config(
    path: Optional[Union[str, Path]] = None,
    *,
    profile: Optional[str] = None,
    apply_env: bool = True,
    embedding_model: Optional[Callable[[str], Vector]] = None,
    **overrides,
) -> Client:
    """
    Connect to a backend using configuration from a file.

    Parameters
    ----------
    path : str or Path, optional
        Path to configuration file. If not provided, searches for default
        config files.
    profile : str, optional
        Profile name to use from configuration. Defaults to 'default' or
        the VD_PROFILE environment variable.
    apply_env : bool, default True
        Whether to apply environment variable overrides
    embedding_model : callable, optional
        Embedding model to use. Overrides config file setting.
    **overrides
        Additional keyword arguments to override configuration values

    Returns
    -------
    Client
        Connected client instance

    Examples
    --------
    >>> # With a config file
    >>> client = connect_from_config('vd.yaml')  # doctest: +SKIP

    >>> # With a specific profile
    >>> client = connect_from_config('vd.yaml', profile='production')  # doctest: +SKIP

    >>> # With environment variable VD_PROFILE=dev
    >>> client = connect_from_config()  # doctest: +SKIP

    >>> # With overrides
    >>> client = connect_from_config('vd.yaml', persist_directory='./data')  # doctest: +SKIP
    """
    import vd

    # Load config
    config = load_config(path)

    # Get profile
    config = get_profile(config, profile)

    # Apply environment variable overrides
    if apply_env:
        config = apply_env_overrides(config)

    # Apply explicit overrides
    config.update(overrides)

    # Use provided embedding model if given
    if embedding_model is not None:
        config['embedding_model'] = embedding_model

    # Extract backend name
    backend_name = config.pop('backend', 'memory')

    # Connect
    return vd.connect(backend_name, **config)


def save_config(
    config: dict,
    path: Union[str, Path],
    *,
    format: Optional[str] = None,
) -> None:
    """
    Save configuration to a file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save
    path : str or Path
        Path to save configuration file
    format : str, optional
        Format to save: 'yaml' or 'toml'. Auto-detected from extension
        if not provided.

    Examples
    --------
    >>> config = {
    ...     'profiles': {
    ...         'dev': {'backend': 'memory'},
    ...         'prod': {'backend': 'chroma', 'persist_directory': './data'}
    ...     }
    ... }
    >>> save_config(config, 'vd.yaml')  # doctest: +SKIP
    """
    path = Path(path)

    # Auto-detect format
    if format is None:
        suffix = path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            format = 'yaml'
        elif suffix == '.toml':
            format = 'toml'
        else:
            raise ValueError(
                f"Cannot detect format from extension '{suffix}'. "
                "Specify format explicitly with format='yaml' or format='toml'"
            )

    # Save config
    if format == 'yaml':
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    elif format == 'toml':
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "tomli-w is required for writing TOML config files. "
                "Install it with: pip install tomli-w"
            )

        with open(path, 'wb') as f:
            tomli_w.dump(config, f)

    else:
        raise ValueError(f"Unsupported format: {format}")


def create_example_config(format: str = 'yaml') -> str:
    """
    Generate an example configuration file content.

    Parameters
    ----------
    format : str, default 'yaml'
        Format of configuration: 'yaml' or 'toml'

    Returns
    -------
    str
        Example configuration as a string

    Examples
    --------
    >>> yaml_config = create_example_config('yaml')  # doctest: +SKIP
    >>> print(yaml_config)  # doctest: +SKIP
    >>> toml_config = create_example_config('toml')  # doctest: +SKIP
    """
    example_config = {
        'profiles': {
            'default': {
                'backend': 'memory',
                'embedding_model': 'text-embedding-3-small',
            },
            'dev': {
                'backend': 'memory',
                'embedding_model': 'text-embedding-3-small',
            },
            'prod': {
                'backend': 'chroma',
                'persist_directory': './vector_db',
                'embedding_model': 'text-embedding-3-large',
            },
            'local_chroma': {
                'backend': 'chroma',
                'persist_directory': './chroma_data',
                'embedding_model': 'text-embedding-3-small',
            },
        }
    }

    if format == 'yaml':
        try:
            import yaml

            return yaml.dump(example_config, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML config support. "
                "Install it with: pip install pyyaml"
            )

    elif format == 'toml':
        try:
            import tomli_w

            import io

            buf = io.BytesIO()
            tomli_w.dump(example_config, buf)
            return buf.getvalue().decode('utf-8')
        except ImportError:
            raise ImportError(
                "tomli-w is required for writing TOML config files. "
                "Install it with: pip install tomli-w"
            )

    else:
        raise ValueError(f"Unsupported format: {format}")
