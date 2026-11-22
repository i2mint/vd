"""
Tests for :mod:`vd.cli`, driven through ``main()`` with a patched ``sys.argv``.

Each CLI command calls ``vd.connect(backend)`` itself, and a fresh memory
client is empty, so commands that read an existing collection run against a
``shared`` memory client that ``vd.connect`` is patched to return.
"""

import json
import sys

import pytest

import vd
from vd import Document
from vd import cli


def run(monkeypatch, *argv):
    """Run ``vd <argv>`` and return its exit code (0 when it does not exit)."""
    monkeypatch.setattr(sys, "argv", ["vd", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.fixture
def shared(monkeypatch):
    """A memory client holding collection ``docs``, returned by every ``vd.connect``."""
    client = vd.connect("memory")
    col = client.create_collection("docs")
    col["a"] = Document(id="a", text="alpha", vector=[1.0, 0.0], metadata={"cat": "x"})
    col["b"] = Document(id="b", text="beta", vector=[0.0, 1.0], metadata={"lang": "en"})
    monkeypatch.setattr(vd, "connect", lambda backend, **kw: client)
    return client


def test_no_command_prints_help_and_exits_1(monkeypatch, capsys):
    assert run(monkeypatch) == 1
    assert "usage: vd" in capsys.readouterr().out


def test_backends_lists_memory(monkeypatch, capsys):
    assert run(monkeypatch, "backends") == 0
    assert "memory" in capsys.readouterr().out


def test_install_info(monkeypatch, capsys):
    assert run(monkeypatch, "install", "memory") == 0
    assert "Status: client library is installed." in capsys.readouterr().out


def test_install_info_unknown_backend_exits_1(monkeypatch, capsys):
    assert run(monkeypatch, "install", "no_such_backend") == 1
    assert "Error: Unknown provider 'no_such_backend'" in capsys.readouterr().err


def test_health(monkeypatch, capsys):
    assert run(monkeypatch, "health", "memory") == 0
    out = capsys.readouterr().out
    assert "Status: healthy" in out
    assert "collections_count: 0" in out


def test_health_unavailable_has_no_details(monkeypatch, capsys):
    assert run(monkeypatch, "health", "no_such_backend") == 0
    out = capsys.readouterr().out
    assert "Status: unavailable" in out
    assert "Details:" not in out


@pytest.mark.parametrize("extra, has_vectors", [((), True), (("--no-vectors",), False)])
def test_export(shared, monkeypatch, capsys, tmp_path, extra, has_vectors):
    out_file = tmp_path / "out.jsonl"
    assert run(monkeypatch, "export", "memory", "docs", "-o", str(out_file), *extra) == 0
    assert f"Exported 2 documents to {out_file}" in capsys.readouterr().out
    records = [json.loads(line) for line in out_file.read_text().splitlines()]
    assert [r["id"] for r in records] == ["a", "b"]
    assert all(("vector" in r) is has_vectors for r in records)


def test_export_directory_format(shared, monkeypatch, tmp_path):
    out_dir = tmp_path / "d"
    assert run(monkeypatch, "export", "memory", "docs", "-o", str(out_dir), "-f", "directory") == 0
    assert (out_dir / "a.json").exists()


def test_import_into_a_new_collection(monkeypatch, capsys, tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": i, "text": i, "vector": [1.0, 0.0]}) for i in ("x", "y")
        )
    )
    assert run(monkeypatch, "import", "memory", "fresh", "-i", str(path)) == 0
    out = capsys.readouterr().out
    assert "Created new collection: fresh" in out
    assert "Imported 2 documents" in out


def test_import_into_an_existing_collection_skipping_existing(
    shared, monkeypatch, capsys, tmp_path
):
    path = tmp_path / "in.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "text": "replacement", "vector": [1.0, 1.0]},
                {"id": "c", "text": "gamma", "vector": [1.0, 1.0]},
            ]
        )
    )
    assert run(monkeypatch, "import", "memory", "docs", "-i", str(path), "--skip-existing") == 0
    out = capsys.readouterr().out
    assert "Using existing collection: docs" in out
    assert "Imported 1 documents" in out
    assert shared["docs"]["a"].text == "alpha"
    assert shared["docs"]["c"].text == "gamma"


def test_stats_verbose(shared, monkeypatch, capsys):
    assert run(monkeypatch, "stats", "memory", "docs", "-v") == 0
    out = capsys.readouterr().out
    assert "Total documents: 2" in out
    assert "Embedding dimension: 2" in out
    assert "Metadata fields: cat, lang" in out
    assert "  cat: 1" in out


def test_stats_empty_collection_omits_optional_sections(shared, monkeypatch, capsys):
    shared.create_collection("empty")
    assert run(monkeypatch, "stats", "memory", "empty") == 0
    out = capsys.readouterr().out
    assert "Total documents: 0" in out
    assert "Embedding dimension" not in out
    assert "Metadata fields" not in out


def test_stats_missing_collection_errors(monkeypatch, capsys):
    # A real fresh memory client: the collection does not exist.
    assert run(monkeypatch, "stats", "memory", "ghost", "-v") == 1
    err = capsys.readouterr().err
    assert "Error: " in err and "ghost" in err
    assert "Traceback" in err  # -v prints the traceback


def test_validate_healthy(shared, monkeypatch, capsys):
    assert run(monkeypatch, "validate", "memory", "docs") == 0
    out = capsys.readouterr().out
    assert "Valid: True" in out
    assert "Issues" not in out and "Warnings" not in out
    assert "total_documents: 2" in out


def test_validate_reports_issues_and_warnings(shared, monkeypatch, capsys):
    native = shared["docs"].native
    native["blank"] = Document(id="blank", text=" ", vector=[1.0, 0.0])
    native["odd"] = Document(id="odd", text="t", vector=[1.0, 0.0, 0.0])
    assert run(monkeypatch, "validate", "memory", "docs") == 0
    out = capsys.readouterr().out
    assert "Valid: False" in out
    assert "Issues (1):" in out
    assert "Warnings (1):" in out
    assert "Document blank has empty text" in out


def test_migrate_to_new_and_existing_collection(shared, monkeypatch, capsys):
    assert run(monkeypatch, "migrate", "memory", "docs", "memory", "copy", "--batch-size", "1") == 0
    out = capsys.readouterr().out
    assert "Created target collection: copy" in out
    assert "Progress: 2/2 (100.0%)" in out
    assert "Migrated: 2" in out
    assert sorted(shared["copy"]) == ["a", "b"]

    assert run(monkeypatch, "migrate", "memory", "docs", "memory", "copy") == 0
    assert "Using existing target collection: copy" in capsys.readouterr().out


def test_migrate_prints_errors(shared, monkeypatch, capsys):
    # Recomputing vectors needs an embedder; the CLI's client has none.
    assert run(monkeypatch, "migrate", "memory", "docs", "memory", "copy", "--recompute-vectors") == 0
    out = capsys.readouterr().out
    assert "Failed: 2" in out
    assert "Errors:" in out
    assert "Error in final batch" in out


def test_benchmark_with_an_embedding_client(monkeypatch, capsys, embedder):
    client = vd.connect("memory", embedder=embedder)
    client.create_collection("docs")["a"] = "alpha"
    monkeypatch.setattr(vd, "connect", lambda backend, **kw: client)
    assert run(monkeypatch, "benchmark", "memory", "docs", "--queries", "3", "--limit", "1") == 0
    out = capsys.readouterr().out
    assert "Running 3 queries..." in out
    assert "Throughput:" in out


@pytest.mark.xfail(
    strict=True,
    reason="cmd_benchmark searches with a text query but connects without an embedder, so it always fails",
)
def test_benchmark_against_a_vector_collection(shared, monkeypatch, capsys):
    assert run(monkeypatch, "benchmark", "memory", "docs", "--queries", "2") == 0


def test_keyboard_interrupt_exits_130(monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_backends", interrupted)
    assert run(monkeypatch, "backends") == 130
    assert "Interrupted" in capsys.readouterr().err


def test_cmd_functions_are_callable_directly(shared, capsys):
    from argparse import Namespace

    cli.cmd_stats(Namespace(backend="memory", collection="docs", verbose=False))
    assert "Total documents: 2" in capsys.readouterr().out
