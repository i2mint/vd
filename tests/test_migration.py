"""
Tests for :mod:`vd.migration`, memory backend to memory backend.

Source and target live on separate memory clients, as they would on two real
backends; ``copy_collection``'s tuple form is checked against the registry by
name.
"""

import pytest

import vd
from vd import Document
from vd.migration import copy_collection, migrate_client, migrate_collection


def _fill(collection, n):
    for i in range(n):
        collection[f"d{i}"] = Document(
            id=f"d{i}", text=f"text {i}", vector=[float(i), 1.0], metadata={"i": i}
        )
    return collection


@pytest.fixture
def source():
    return _fill(vd.connect("memory").create_collection("src"), 5)


@pytest.fixture
def target():
    return vd.connect("memory").create_collection("tgt")


def test_migrate_collection_copies_text_vectors_and_metadata(source, target):
    stats = migrate_collection(source, target)
    assert stats == {"total": 5, "migrated": 5, "skipped": 0, "failed": 0, "errors": []}
    assert sorted(target) == sorted(source)
    for k in source:
        assert target[k].text == source[k].text
        assert target[k].vector == source[k].vector
        assert target[k].metadata == source[k].metadata


def test_migrate_collection_batches_and_reports_progress(source, target):
    calls = []
    batches = []
    original = target.add_documents

    def spy(docs, **kwargs):
        batches.append(len(docs))
        return original(docs, **kwargs)

    target.add_documents = spy
    stats = migrate_collection(
        source, target, batch_size=2, progress_callback=lambda c, t: calls.append((c, t))
    )
    assert stats["migrated"] == 5
    assert batches == [2, 2, 1]
    assert calls[:2] == [(2, 5), (4, 5)]


@pytest.mark.xfail(
    strict=True,
    reason="progress_callback is not called after the final partial batch, so it never reaches total",
)
def test_migrate_collection_progress_reaches_total(source, target):
    calls = []
    migrate_collection(source, target, batch_size=2, progress_callback=lambda c, t: calls.append((c, t)))
    assert calls[-1] == (5, 5)


def test_migrate_collection_skip_existing(source, target):
    target["d0"] = Document(id="d0", text="already here", vector=[9.0, 9.0])
    calls = []
    stats = migrate_collection(
        source, target, skip_existing=True, progress_callback=lambda c, t: calls.append((c, t))
    )
    assert stats["skipped"] == 1
    assert stats["migrated"] == 4
    assert target["d0"].text == "already here"
    assert calls[0] == (1, 5)


def test_migrate_collection_without_skip_overwrites(source, target):
    target["d0"] = Document(id="d0", text="old", vector=[9.0, 9.0])
    migrate_collection(source, target)
    assert target["d0"].text == "text 0"


def test_migrate_collection_recomputes_vectors_with_target_embedder(source, embedder):
    target = vd.connect("memory", embedder=embedder).create_collection("tgt")
    stats = migrate_collection(source, target, preserve_vectors=False)
    assert stats["migrated"] == 5
    assert target["d1"].vector == embedder("text 1")


@pytest.mark.xfail(
    strict=True,
    reason="preserve_vectors=False sets doc.vector=None on the object read from the source, mutating the source",
)
def test_migrate_collection_recompute_leaves_source_untouched(source, embedder):
    target = vd.connect("memory", embedder=embedder).create_collection("tgt")
    migrate_collection(source, target, preserve_vectors=False)
    assert source["d1"].vector == [1.0, 1.0]


def test_migrate_collection_records_a_failed_final_batch(source, target):
    def boom(docs, **kwargs):
        raise RuntimeError("target down")

    target.add_documents = boom
    stats = migrate_collection(source, target)
    assert stats["migrated"] == 0
    assert stats["failed"] == 5
    assert stats["errors"] == ["Error in final batch: target down"]


def test_migrate_collection_records_per_document_failures(target):
    class Flaky(dict):
        def __getitem__(self, key):
            if key == "bad":
                raise RuntimeError("unreadable")
            return super().__getitem__(key)

    src = Flaky(
        good=Document(id="good", text="g", vector=[1.0, 0.0]),
        bad=Document(id="bad", text="b", vector=[0.0, 1.0]),
    )
    calls = []
    stats = migrate_collection(src, target, progress_callback=lambda c, t: calls.append((c, t)))
    assert stats["migrated"] == 1
    assert stats["failed"] == 1
    assert stats["errors"] == ["Error migrating bad: unreadable"]
    assert (1, 2) in calls
    assert list(target) == ["good"]


def test_migrate_client_moves_every_collection():
    src_client = vd.connect("memory")
    _fill(src_client.create_collection("one"), 2)
    _fill(src_client.create_collection("two"), 3)
    dst_client = vd.connect("memory")
    calls = []
    stats = migrate_client(
        src_client, dst_client, batch_size=2, progress_callback=lambda *a: calls.append(a)
    )
    assert stats["collections_total"] == 2
    assert stats["collections_migrated"] == 2
    assert stats["total_documents"] == 5
    assert stats["migrated_documents"] == 5
    assert stats["errors"] == []
    assert sorted(dst_client.list_collections()) == ["one", "two"]
    assert len(dst_client.get_collection("two")) == 3
    assert ("one", 2, 2) in calls


def test_migrate_client_selected_and_existing_collections():
    src_client = vd.connect("memory")
    _fill(src_client.create_collection("one"), 2)
    _fill(src_client.create_collection("two"), 2)
    dst_client = vd.connect("memory")
    dst_client.create_collection("one")  # exists already: reused, not an error
    stats = migrate_client(src_client, dst_client, collection_names=["one"])
    assert stats["collections_migrated"] == 1
    assert list(dst_client.list_collections()) == ["one"]
    assert len(dst_client.get_collection("one")) == 2


def test_migrate_client_reports_missing_source_collection():
    stats = migrate_client(vd.connect("memory"), vd.connect("memory"), collection_names=["ghost"])
    assert stats["collections_failed"] == 1
    assert stats["collections_migrated"] == 0
    assert stats["errors"][0].startswith("Failed to migrate collection ghost")


def test_copy_collection_between_collection_objects(source, target):
    stats = copy_collection(source, target, batch_size=2)
    assert stats["migrated"] == 5
    assert sorted(target) == sorted(source)


@pytest.mark.parametrize("target_spec", [("memory", "copy"), ("memory", "copy", {})])
def test_copy_collection_to_a_backend_by_name(source, target_spec):
    stats = copy_collection(source, target_spec)
    assert stats["total"] == 5
    assert stats["migrated"] == 5


def test_copy_collection_by_name_resolves_through_connect(monkeypatch):
    shared = vd.connect("memory")
    _fill(shared.create_collection("src"), 3)
    shared.create_collection("copy")  # exists already: reused, not an error
    seen = []

    def fake_connect(backend, **config):
        seen.append((backend, config))
        return shared

    monkeypatch.setattr(vd, "connect", fake_connect)
    stats = copy_collection(("memory", "src"), ("memory", "copy", {"opt": 1}))
    assert stats["migrated"] == 3
    assert len(shared.get_collection("copy")) == 3
    assert seen == [("memory", {}), ("memory", {"opt": 1})]
