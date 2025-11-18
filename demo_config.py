"""
Demo script showing the configuration management features of vd.

This demonstrates how users can:
- Create configuration files (YAML/TOML)
- Use multiple profiles for different environments
- Connect using configuration
- Override settings with environment variables
"""

import hashlib
import tempfile
from pathlib import Path

import vd


def mock_embedding_function(text: str) -> list[float]:
    """Simple mock embedding for demo."""
    text_hash = hashlib.md5(text.encode()).digest()
    embedding = [(b / 128.0) - 1.0 for b in text_hash]
    while len(embedding) < 16:
        embedding.extend(embedding)
    return embedding[:16]


def main():
    print("=" * 80)
    print("VD Configuration Management Demo")
    print("=" * 80)
    print()

    # 1. Create an example configuration
    print("1. Creating example configuration:")
    print("-" * 80)

    config = {
        'profiles': {
            'default': {
                'backend': 'memory',
            },
            'dev': {
                'backend': 'memory',
            },
            'prod': {
                'backend': 'chroma',
                'persist_directory': './production_data',
            },
        }
    }

    print("Example configuration:")
    print(config)
    print()

    # 2. Save configuration to YAML
    print("2. Saving configuration to YAML file:")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'vd.yaml'

        try:
            vd.save_config(config, config_path)
            print(f"✓ Configuration saved to {config_path}")
            print()

            # 3. Load configuration
            print("3. Loading configuration from file:")
            print("-" * 80)
            loaded_config = vd.load_config(config_path)
            print(f"Loaded config profiles: {list(loaded_config['profiles'].keys())}")
            print()

            # 4. Connect using configuration (default profile)
            print("4. Connecting using default profile:")
            print("-" * 80)
            client = vd.connect_from_config(
                config_path, profile='default', embedding_model=mock_embedding_function
            )
            print(f"✓ Connected using 'default' profile")
            print()

            # 5. Connect using different profile
            print("5. Connecting using dev profile:")
            print("-" * 80)
            dev_client = vd.connect_from_config(
                config_path, profile='dev', embedding_model=mock_embedding_function
            )
            print(f"✓ Connected using 'dev' profile")
            print()

            # 6. Use the connection
            print("6. Using the connection to create a collection:")
            print("-" * 80)
            collection = dev_client.create_collection('test_docs')
            collection['doc1'] = "This is a test document"
            collection['doc2'] = "Another example document"
            print(f"✓ Created collection with {len(collection)} documents")
            print()

            # 7. Search
            print("7. Searching the collection:")
            print("-" * 80)
            results = list(collection.search("test", limit=2))
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['id']}: {result['text'][:50]}...")
            print()

        except ImportError as e:
            print(f"⚠ Could not complete config demo: {e}")
            print(
                "  Install PyYAML for YAML config support: pip install vd[config] or pip install pyyaml"
            )
            print()

    # 8. Show example YAML config
    print("8. Example YAML configuration format:")
    print("-" * 80)
    try:
        example_yaml = vd.create_example_config('yaml')
        print(example_yaml)
    except ImportError:
        print("  (PyYAML not installed - install with: pip install pyyaml)")
    print()

    # 9. Show example TOML config
    print("9. Example TOML configuration format:")
    print("-" * 80)
    try:
        example_toml = vd.create_example_config('toml')
        print(example_toml)
    except ImportError:
        print("  (tomli-w not installed - install with: pip install tomli-w)")
    print()

    print("=" * 80)
    print("Configuration Features:")
    print("  • YAML and TOML support")
    print("  • Multiple profiles (dev, prod, etc.)")
    print("  • Environment variable overrides (VD_PROFILE, VD_BACKEND)")
    print("  • Auto-detection of config files")
    print("  • Profile-specific settings")
    print("=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
