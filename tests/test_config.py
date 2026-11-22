"""
Tests for :mod:`vd.config`.

Every file lives under ``tmp_path``; the default-path search runs with the
working directory and ``HOME`` both pointed at temporary folders, and
environment variables are set through ``monkeypatch``, so nothing reads or
writes the real user configuration.
"""

import sys

import pytest

import vd
from vd.backends.memory import MemoryClient
from vd.config import (
    apply_env_overrides,
    connect_from_config,
    create_example_config,
    get_profile,
    load_config,
    load_toml_config,
    load_yaml_config,
    save_config,
)

CONFIG = {"profiles": {"default": {"backend": "memory"}, "dev": {"backend": "memory", "x": 1}}}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Run in an empty cwd with a fake HOME and no VD_* variables."""
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    for var in ("VD_PROFILE", "VD_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    return work


@pytest.fixture
def tomli_w():
    return pytest.importorskip("tomli_w")


# ----- loading ------------------------------------------------------------- #


def test_load_yaml_config(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("backend: memory\nprofiles:\n  dev:\n    backend: chroma\n")
    assert load_yaml_config(path) == {
        "backend": "memory",
        "profiles": {"dev": {"backend": "chroma"}},
    }


def test_load_yaml_config_empty_file_is_empty_dict(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("")
    assert load_yaml_config(path) == {}


@pytest.mark.parametrize("loader", [load_yaml_config, load_toml_config])
def test_loaders_raise_on_missing_file(tmp_path, loader):
    with pytest.raises(FileNotFoundError, match="not found"):
        loader(tmp_path / "missing")


def test_load_yaml_config_without_pyyaml(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("a: 1\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError, match="PyYAML"):
        load_yaml_config(path)


def test_load_toml_config(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[profiles.default]\nbackend = "memory"\n')
    assert load_toml_config(path) == {"profiles": {"default": {"backend": "memory"}}}


@pytest.mark.parametrize("name", ["c.yaml", "c.yml", "c.toml", "C.YAML"])
def test_load_config_detects_format_from_extension(tmp_path, name, tomli_w):
    path = tmp_path / name
    save_config(CONFIG, path)
    assert load_config(path) == CONFIG


def test_load_config_explicit_format(tmp_path):
    path = tmp_path / "settings.conf"
    path.write_text("backend: memory\n")
    assert load_config(path, format="yaml") == {"backend": "memory"}


def test_load_config_rejects_unknown_extension_and_format(tmp_path):
    path = tmp_path / "settings.conf"
    path.write_text("")
    with pytest.raises(ValueError, match="Cannot detect format"):
        load_config(path)
    with pytest.raises(ValueError, match="Unsupported format"):
        load_config(path, format="ini")


def test_load_config_without_file_is_empty():
    assert load_config() == {}


def test_load_config_finds_file_in_cwd(isolated_env):
    (isolated_env / ".vd.yml").write_text("backend: memory\n")
    assert load_config() == {"backend": "memory"}


def test_load_config_cwd_wins_over_home(isolated_env, tmp_path):
    (tmp_path / "home" / ".vd").mkdir()
    (tmp_path / "home" / ".vd" / "config.yaml").write_text("backend: home\n")
    assert load_config() == {"backend": "home"}
    (isolated_env / "vd.yaml").write_text("backend: cwd\n")
    assert load_config() == {"backend": "cwd"}


# ----- profiles and env overrides ----------------------------------------- #


def test_get_profile_by_name_and_default():
    assert get_profile(CONFIG, "dev") == {"backend": "memory", "x": 1}
    assert get_profile(CONFIG) == {"backend": "memory"}


def test_get_profile_from_env(monkeypatch):
    monkeypatch.setenv("VD_PROFILE", "dev")
    assert get_profile(CONFIG) == {"backend": "memory", "x": 1}


def test_get_profile_unknown_lists_available():
    with pytest.raises(ValueError, match="Available profiles: default, dev"):
        get_profile(CONFIG, "prod")


def test_get_profile_without_profiles_section():
    flat = {"backend": "memory"}
    assert get_profile(flat) is flat
    with pytest.raises(ValueError, match="no profiles section"):
        get_profile(flat, "dev")


def test_apply_env_overrides(monkeypatch):
    original = {"backend": "memory", "k": 1}
    assert apply_env_overrides(original) == original
    monkeypatch.setenv("VD_BACKEND", "chroma")
    overridden = apply_env_overrides(original)
    assert overridden == {"backend": "chroma", "k": 1}
    assert original["backend"] == "memory"  # input not mutated


# ----- connect_from_config ------------------------------------------------- #


def test_connect_from_config_memory(tmp_path, embedder):
    path = tmp_path / "vd.yaml"
    save_config(
        {"profiles": {"default": {"backend": "memory", "embedding_model": "legacy"}}}, path
    )
    client = connect_from_config(path, embedder=embedder)
    assert isinstance(client, MemoryClient)
    col = client.create_collection("t")
    col["a"] = "hello"  # the embedder reached the client
    assert col["a"].vector == embedder("hello")


def test_connect_from_config_defaults_to_memory_without_file():
    assert isinstance(connect_from_config(), MemoryClient)


def test_connect_from_config_env_and_overrides_reach_connect(tmp_path, monkeypatch):
    path = tmp_path / "vd.yaml"
    save_config(CONFIG, path)
    seen = {}

    def fake_connect(backend, *, embedder=None, **config):
        seen.update(backend=backend, embedder=embedder, config=config)
        return "client"

    monkeypatch.setattr(vd, "connect", fake_connect)
    monkeypatch.setenv("VD_BACKEND", "qdrant")
    assert connect_from_config(path, profile="dev", url="http://h") == "client"
    assert seen == {"backend": "qdrant", "embedder": None, "config": {"x": 1, "url": "http://h"}}

    connect_from_config(path, profile="dev", apply_env=False)
    assert seen["backend"] == "memory"


# ----- saving and the example --------------------------------------------- #


def test_save_config_yaml_keeps_key_order(tmp_path):
    path = tmp_path / "c.yaml"
    save_config({"z": 1, "a": 2}, path)
    assert path.read_text().splitlines() == ["z: 1", "a: 2"]


def test_save_config_toml_round_trip(tmp_path, tomli_w):
    path = tmp_path / "c.toml"
    save_config(CONFIG, path)
    assert load_toml_config(path) == CONFIG


def test_save_config_explicit_format(tmp_path):
    path = tmp_path / "c.conf"
    save_config(CONFIG, path, format="yaml")
    assert load_config(path, format="yaml") == CONFIG


def test_save_config_errors(tmp_path):
    with pytest.raises(ValueError, match="Cannot detect format"):
        save_config(CONFIG, tmp_path / "c.conf")
    with pytest.raises(ValueError, match="Unsupported format"):
        save_config(CONFIG, tmp_path / "c.conf", format="ini")


@pytest.mark.parametrize("module, suffix", [("yaml", "c.yaml"), ("tomli_w", "c.toml")])
def test_save_config_missing_writer_library(tmp_path, monkeypatch, module, suffix):
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match="pip install"):
        save_config(CONFIG, tmp_path / suffix)


def test_create_example_config_yaml_parses_back(tmp_path):
    text = create_example_config()
    path = tmp_path / "example.yaml"
    path.write_text(text)
    config = load_config(path)
    assert config["profiles"]["default"] == {"backend": "memory"}
    assert set(config["profiles"]) == {"default", "dev", "prod", "qdrant_cloud"}


def test_create_example_config_toml_parses_back(tmp_path, tomli_w):
    path = tmp_path / "example.toml"
    path.write_text(create_example_config("toml"))
    assert load_config(path)["profiles"]["prod"]["backend"] == "chroma"


def test_create_example_config_errors(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported format"):
        create_example_config("ini")
    monkeypatch.setitem(sys.modules, "yaml", None)
    monkeypatch.setitem(sys.modules, "tomli_w", None)
    with pytest.raises(ImportError, match="PyYAML"):
        create_example_config("yaml")
    with pytest.raises(ImportError, match="tomli-w"):
        create_example_config("toml")
