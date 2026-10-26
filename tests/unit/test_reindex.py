"""Tests for repository re-index change detection."""

import hashlib
from pathlib import Path

import pytest

from app.ingestion.file_filter import (
    PythonFile,
    discover_python_files,
)
from app.ingestion.reindex import detect_reindex_changes
from app.storage.metadata_store import (
    MetadataStore,
    MetadataStoreError,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"

    (root / "src").mkdir(parents=True)

    (root / "main.py").write_text(
        "print('hello')\n",
        encoding="utf-8",
    )

    (root / "src" / "utils.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def store(tmp_path: Path) -> MetadataStore:
    return MetadataStore(tmp_path / "metadata.sqlite3")


@pytest.fixture
def repository(store: MetadataStore, workspace: Path):
    record = store.register_repository(
        source_type="local",
        source_reference="test-repo",
        workspace_path=workspace,
    )

    scan = discover_python_files(workspace)

    store.save_discovered_files(
        record.repository_id,
        scan.python_files,
    )

    return record


def scan(root: Path) -> list[PythonFile]:
    return discover_python_files(root).python_files


def test_everything_unchanged_returns_no_processing_work(
    store,
    repository,
    workspace,
):
    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    assert {item.kind for item in report.changes} == {"unchanged"}
    assert len(report.unchanged) == 2
    assert report.needs_processing == ()
    assert report.removed == ()


def test_detects_added_file(store, repository, workspace):
    (workspace / "new_module.py").write_text(
        "ANSWER = 42\n",
        encoding="utf-8",
    )

    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    assert [
        item.relative_path for item in report.added
    ] == ["new_module.py"]

    assert len(report.unchanged) == 2
    assert len(report.needs_processing) == 1


def test_detects_modified_file_using_content_hash(
    store,
    repository,
    workspace,
):
    target = workspace / "main.py"

    target.write_text(
        "print('changed')\n",
        encoding="utf-8",
    )

    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    change = next(
        item
        for item in report.modified
        if item.relative_path == "main.py"
    )

    assert change.previous_hash != change.current_hash

    assert change.current_hash == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()

    assert change.previous_size_bytes != change.current_size_bytes


def test_detects_removed_file_without_mutating_metadata(
    store,
    repository,
    workspace,
):
    (workspace / "src" / "utils.py").unlink()

    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    assert [
        item.relative_path for item in report.removed
    ] == ["src/utils.py"]

    # Detection must not delete stored metadata.
    assert len(store.list_files(repository.repository_id)) == 2


def test_detects_mixed_changes_in_deterministic_order(
    store,
    repository,
    workspace,
):
    (workspace / "main.py").write_text(
        "print('changed')\n",
        encoding="utf-8",
    )

    (workspace / "src" / "utils.py").unlink()

    (workspace / "z_new.py").write_text(
        "Z = 1\n",
        encoding="utf-8",
    )

    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    assert [
        (item.relative_path, item.kind)
        for item in report.changes
    ] == [
        ("main.py", "modified"),
        ("z_new.py", "added"),
        ("src/utils.py", "removed"),
    ]


def test_empty_scan_marks_all_stored_files_removed(
    store,
    repository,
):
    report = detect_reindex_changes(
        store,
        repository.repository_id,
        [],
    )

    assert len(report.removed) == 2
    assert report.added == ()
    assert report.modified == ()
    assert report.unchanged == ()


def test_unknown_repository_raises_clear_error(store):
    with pytest.raises(
        MetadataStoreError,
        match="Unknown repository",
    ):
        detect_reindex_changes(
            store,
            "not-a-repository",
            [],
        )


def test_duplicate_discovery_paths_are_rejected(
    store,
    repository,
    workspace,
):
    first = scan(workspace)[0]

    with pytest.raises(
        MetadataStoreError,
        match="Duplicate file path",
    ):
        detect_reindex_changes(
            store,
            repository.repository_id,
            [first, first],
        )


def test_path_traversal_is_rejected(
    store,
    repository,
    workspace,
):
    fake = PythonFile(
        "../outside.py",
        workspace / "main.py",
        0,
    )

    with pytest.raises(
        MetadataStoreError,
        match="Invalid repository-relative",
    ):
        detect_reindex_changes(
            store,
            repository.repository_id,
            [fake],
        )


def test_absolute_path_mismatch_is_rejected(
    store,
    repository,
    workspace,
):
    fake = PythonFile(
        "main.py",
        workspace / "src" / "utils.py",
        0,
    )

    with pytest.raises(
        MetadataStoreError,
        match="Absolute path does not match",
    ):
        detect_reindex_changes(
            store,
            repository.repository_id,
            [fake],
        )


def test_missing_file_is_not_silently_skipped(
    store,
    repository,
    workspace,
):
    fake = PythonFile(
        "missing.py",
        workspace / "missing.py",
        0,
    )

    with pytest.raises(
        MetadataStoreError,
        match="missing or outside",
    ):
        detect_reindex_changes(
            store,
            repository.repository_id,
            [fake],
        )


def test_changed_file_with_same_size_is_detected_by_hash(
    store,
    repository,
    workspace,
):
    target = workspace / "main.py"

    original = target.read_text(encoding="utf-8")

    target.write_text(
        "print('world')\n",
        encoding="utf-8",
    )

    assert len(target.read_text(encoding="utf-8")) == len(original)

    report = detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    assert any(
        item.relative_path == "main.py"
        for item in report.modified
    )


def test_detection_does_not_change_metadata(
    store,
    repository,
    workspace,
):
    before = store.list_files(repository.repository_id)

    (workspace / "main.py").write_text(
        "print('changed')\n",
        encoding="utf-8",
    )

    detect_reindex_changes(
        store,
        repository.repository_id,
        scan(workspace),
    )

    after = store.list_files(repository.repository_id)

    assert before == after