"""
Tests for :mod:`vd.io`: export and import round trips through tmp files.

Every format (``jsonl``, ``json``, ``directory``) is exported from one memory
collection and re-imported into a fresh one, which must hold the same
documents, vectors and metadata.
"""

import json

import pytest

import vd
from vd import Document
from vd.io import (
    export_collection,
    export_to_directory,
    export_to_json,
    export_to_jsonl,
    import_collection,
    import_from_directory,
    import_from_json,
    import_from_jsonl,
)

FORMATS = ["jsonl", "json", "directory"]


def _new_collection(name="t"):
    return vd.connect("memory").create_collection(name, dimension=2)


@pytest.fixture
def col():
    """A memory collection of three documents, one with unicode text."""
    c = _new_collection("src")
    c["a"] = Document(id="a", text="alpha", vector=[1.0, 0.0], metadata={"k": 1})
    c["b"] = Document(id="b", text="béta ✓", vector=[0.0, 1.0], metadata={})
    c["c"] = Document(id="c", text="gamma", vector=[0.5, 0.5], metadata={"tags": ["x"]})
    return c


def _snapshot(collection):
    return {
        k: (collection[k].text, collection[k].vector, collection[k].metadata)
        for k in collection
    }


def _target_path(tmp_path, fmt):
    return tmp_path / ("out" if fmt == "directory" else f"out.{fmt}")


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip_preserves_documents(col, tmp_path, fmt):
    path = _target_path(tmp_path, fmt)
    assert export_collection(col, path, format=fmt) == 3
    target = _new_collection()
    assert import_collection(target, path) == 3  # format inferred from the path
    assert _snapshot(target) == _snapshot(col)


@pytest.mark.parametrize("fmt", FORMATS)
def test_export_without_vectors_omits_them(col, tmp_path, fmt):
    path = _target_path(tmp_path, fmt)
    export_collection(col, path, format=fmt, include_vectors=False)
    if fmt == "jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines()]
    elif fmt == "json":
        records = json.loads(path.read_text())
    else:
        records = [
            json.loads(p.read_text()) for p in path.glob("*.json") if p.name != "_metadata.json"
        ]
    assert len(records) == 3
    assert all("vector" not in r for r in records)
    assert all(set(r) == {"id", "text", "metadata"} for r in records)


def test_jsonl_is_one_utf8_object_per_line(col, tmp_path):
    path = tmp_path / "out.jsonl"
    export_to_jsonl(col, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "béta ✓" in lines[1]  # ensure_ascii=False: text stays readable


def test_json_indent_none_is_compact(col, tmp_path):
    path = tmp_path / "out.json"
    export_to_json(col, path, indent=None)
    assert "\n" not in path.read_text()


def test_directory_export_writes_metadata_file(col, tmp_path):
    out = tmp_path / "nested" / "dir"
    assert export_to_directory(col, out) == 3
    meta = json.loads((out / "_metadata.json").read_text())
    assert meta == {"collection_name": "src", "total_documents": 3}
    assert sorted(p.name for p in out.glob("*.json")) == [
        "_metadata.json",
        "a.json",
        "b.json",
        "c.json",
    ]


def test_directory_export_sanitizes_slashes_but_keeps_the_id(tmp_path):
    c = _new_collection()
    c["x/y"] = Document(id="x/y", text="t", vector=[1.0, 0.0])
    export_to_directory(c, tmp_path / "d")
    assert (tmp_path / "d" / "x_y.json").exists()
    target = _new_collection()
    import_from_directory(target, tmp_path / "d")
    assert list(target) == ["x/y"]


def test_directory_round_trip_keeps_ids_that_sanitize_alike(tmp_path):
    c = _new_collection()
    c["a/b"] = Document(id="a/b", text="slash", vector=[1.0, 0.0])
    c["a_b"] = Document(id="a_b", text="underscore", vector=[0.0, 1.0])
    export_to_directory(c, tmp_path / "d")
    target = _new_collection()
    import_from_directory(target, tmp_path / "d")
    assert sorted(target) == ["a/b", "a_b"]


@pytest.mark.parametrize(
    "exporter, importer, name",
    [
        (export_to_jsonl, import_from_jsonl, "f.jsonl"),
        (export_to_json, import_from_json, "f.json"),
        (export_to_directory, import_from_directory, "d"),
    ],
)
def test_import_skip_existing(col, tmp_path, exporter, importer, name):
    path = tmp_path / name
    exporter(col, path)
    target = _new_collection()
    target["a"] = Document(id="a", text="kept", vector=[1.0, 0.0])
    assert importer(target, path, skip_existing=True) == 2
    assert target["a"].text == "kept"
    assert sorted(target) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "exporter, importer, name",
    [
        (export_to_jsonl, import_from_jsonl, "f.jsonl"),
        (export_to_json, import_from_json, "f.json"),
        (export_to_directory, import_from_directory, "d"),
    ],
)
def test_import_flushes_in_batches(col, tmp_path, exporter, importer, name):
    path = tmp_path / name
    exporter(col, path)
    target = _new_collection()
    batches = []
    original = target.add_documents

    def spy(docs, **kwargs):
        batches.append(len(docs))
        return original(docs, **kwargs)

    target.add_documents = spy
    assert importer(target, path, batch_size=2) == 3
    assert batches == [2, 1]
    assert len(target) == 3


def test_import_without_vectors_needs_an_embedder(col, tmp_path, embedder):
    path = tmp_path / "f.jsonl"
    export_to_jsonl(col, path, include_vectors=False)
    with pytest.raises(vd.EmbeddingRequiredError):
        import_from_jsonl(_new_collection(), path)
    target = vd.connect("memory", embedder=embedder).create_collection("t")
    assert import_from_jsonl(target, path) == 3
    assert target["a"].vector == embedder("alpha")


def test_import_from_directory_pattern(col, tmp_path):
    out = tmp_path / "d"
    export_to_directory(col, out)
    target = _new_collection()
    assert import_from_directory(target, out, pattern="a*.json") == 1
    assert list(target) == ["a"]


def test_import_collection_explicit_format_overrides_suffix(col, tmp_path):
    path = tmp_path / "data.txt"
    export_collection(col, path, format="jsonl")
    target = _new_collection()
    assert import_collection(target, path, format="jsonl") == 3


def test_import_collection_cannot_infer_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("")
    with pytest.raises(ValueError, match="Cannot infer format"):
        import_collection(_new_collection(), path)


def test_unknown_formats_raise(col, tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        export_collection(col, tmp_path / "x", format="csv")
    with pytest.raises(ValueError, match="Unknown format"):
        import_collection(_new_collection(), tmp_path / "x.jsonl", format="csv")


def test_export_empty_collection(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert export_to_jsonl(_new_collection(), path) == 0
    assert path.read_text() == ""
    assert import_from_jsonl(_new_collection(), path) == 0
