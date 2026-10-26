"""Detect file-level changes between a repository scan and stored metadata.

This module is read-only. It creates a re-index plan without changing the
SQLite metadata. Applying the plan is a separate operation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from app.ingestion.file_filter import PythonFile
from app.storage.metadata_store import (
    FileRecord,
    MetadataStore,
    MetadataStoreError,
)

ChangeKind = Literal["added", "modified", "unchanged", "removed"]


@dataclass(frozen=True, slots=True)
class FileChange:
    """Describe one file's status in the current scan."""

    relative_path: str
    kind: ChangeKind
    previous_hash: str | None = None
    current_hash: str | None = None
    previous_size_bytes: int | None = None
    current_size_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ReindexReport:
    """Summary of changes found during a repository scan."""

    repository_id: str
    changes: tuple[FileChange, ...]

    @property
    def added(self) -> tuple[FileChange, ...]:
        return tuple(
            change for change in self.changes
            if change.kind == "added"
        )

    @property
    def modified(self) -> tuple[FileChange, ...]:
        return tuple(
            change for change in self.changes
            if change.kind == "modified"
        )

    @property
    def unchanged(self) -> tuple[FileChange, ...]:
        return tuple(
            change for change in self.changes
            if change.kind == "unchanged"
        )

    @property
    def removed(self) -> tuple[FileChange, ...]:
        return tuple(
            change for change in self.changes
            if change.kind == "removed"
        )

    @property
    def needs_processing(self) -> tuple[FileChange, ...]:
        """Files that need downstream parsing or indexing."""
        return self.added + self.modified


def _hash_file(path: Path) -> str:
    """Calculate a SHA-256 hash without loading the whole file into memory."""
    digest = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)

                if not block:
                    break

                digest.update(block)

    except OSError as exc:
        raise MetadataStoreError(
            f"Cannot hash source file '{path}': {exc}"
        ) from exc

    return digest.hexdigest()


def _validate_and_hash_file(
    root: Path,
    item: PythonFile,
) -> tuple[str, int, str]:
    """Validate a discovered file and return its path, size, and hash."""

    if not isinstance(item, PythonFile):
        raise MetadataStoreError(
            "Each discovered item must be a PythonFile."
        )

    relative = item.relative_path

    # Require a safe, normalized, repository-relative path.
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or relative.startswith("/")
        or any(
            part in {"", ".", ".."}
            for part in relative.split("/")
        )
    ):
        raise MetadataStoreError(
            f"Invalid repository-relative path: {relative!r}"
        )

    relative_path = Path(*relative.split("/"))

    if relative_path.suffix.casefold() != ".py":
        raise MetadataStoreError(
            f"Only Python files can be re-indexed: {relative}"
        )

    candidate = root.joinpath(*relative.split("/"))

    try:
        # Reject symlinks in every path component.
        current = root

        for part in relative.split("/"):
            current = current / part

            if current.is_symlink():
                raise MetadataStoreError(
                    f"Symbolic links cannot be indexed: {relative}"
                )

        source = candidate.resolve(strict=True)
        source.relative_to(root)

        supplied = Path(item.absolute_path).resolve(strict=True)

    except MetadataStoreError:
        raise

    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise MetadataStoreError(
            f"Discovered file is missing or outside the workspace: "
            f"{relative}"
        ) from exc

    if source != supplied:
        raise MetadataStoreError(
            f"Absolute path does not match relative path: {relative}"
        )

    if not source.is_file():
        raise MetadataStoreError(
            f"Expected a regular Python file: {relative}"
        )

    try:
        size = source.stat().st_size

    except OSError as exc:
        raise MetadataStoreError(
            f"Cannot inspect source file '{source}': {exc}"
        ) from exc

    content_hash = _hash_file(source)

    return relative, size, content_hash


def detect_reindex_changes(
    store: MetadataStore,
    repository_id: str,
    discovered_files: Iterable[PythonFile],
) -> ReindexReport:
    """Compare the current file scan against stored SQLite metadata.

    Returns:
        ReindexReport containing added, modified, unchanged, and removed
        files.

    Raises:
        MetadataStoreError: If the repository is unknown, the workspace
        cannot be accessed, or any discovered file fails validation.

    Important:
        This function does not update or delete stored metadata.
    """

    repository = store.get_repository(repository_id)

    if repository is None:
        raise MetadataStoreError(
            f"Unknown repository ID: {repository_id}"
        )

    try:
        root = Path(repository.workspace_path).resolve(strict=True)

    except (OSError, RuntimeError, ValueError) as exc:
        raise MetadataStoreError(
            f"Cannot resolve repository workspace: {exc}"
        ) from exc

    if not root.is_dir():
        raise MetadataStoreError(
            "Registered workspace is no longer a directory."
        )

    # relative path -> (size in bytes, SHA-256 hash)
    current: dict[str, tuple[int, str]] = {}

    for item in discovered_files:
        relative, size, digest = _validate_and_hash_file(root, item)

        if relative in current:
            raise MetadataStoreError(
                f"Duplicate file path in scan: {relative}"
            )

        current[relative] = (size, digest)

    previous_records: list[FileRecord] = store.list_files(
        repository_id
    )

    previous = {
        record.relative_path: record
        for record in previous_records
    }

    changes: list[FileChange] = []

    # Classify files found in the current scan.
    for relative in sorted(
        current,
        key=lambda value: (value.casefold(), value),
    ):
        size, digest = current[relative]
        old = previous.get(relative)

        if old is None:
            kind: ChangeKind = "added"

        elif old.content_hash != digest:
            kind = "modified"

        else:
            kind = "unchanged"

        changes.append(
            FileChange(
                relative_path=relative,
                kind=kind,
                previous_hash=(
                    old.content_hash if old is not None else None
                ),
                current_hash=digest,
                previous_size_bytes=(
                    old.size_bytes if old is not None else None
                ),
                current_size_bytes=size,
            )
        )

    # Previously indexed files absent from this scan are removed.
    removed_paths = set(previous) - set(current)

    for relative in sorted(
        removed_paths,
        key=lambda value: (value.casefold(), value),
    ):
        old = previous[relative]

        changes.append(
            FileChange(
                relative_path=relative,
                kind="removed",
                previous_hash=old.content_hash,
                current_hash=None,
                previous_size_bytes=old.size_bytes,
                current_size_bytes=None,
            )
        )

    return ReindexReport(
        repository_id=repository_id,
        changes=tuple(changes),
    )