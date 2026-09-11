"""Raft project discovery and on-disk paths.

A project is marked by ``raft.json`` and owns one persona dataset.  Legacy
datasets remain supported through their historical ``data/{name}*`` layout;
the rest of Raft consumes :class:`DatasetPaths` so the pipeline does not need
to care which layout it is using.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

MANIFEST_NAME = "raft.json"
PROJECT_FORMAT = "raft.project.v1"
PROJECT_DIRS = ("fetch", "blobs", "metadata", "conversations", "corpus")


class ProjectError(ValueError):
    """A project cannot be found, opened, or initialized safely."""


def _collection_slug(value: str) -> str:
    """Return a stable Chroma-safe collection name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("._-").lower()
    if len(slug) < 3:
        slug = f"raft-{slug or 'project'}"
    # Chroma accepts substantially longer names, but a short portable slug is
    # kinder to filenames, provider job suffixes, and terminal output.
    slug = slug[:63].rstrip("._-")
    return slug or "raft-project"


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectError(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != PROJECT_FORMAT:
        raise ProjectError(
            f"{path} is not a supported Raft project manifest "
            f"(expected format {PROJECT_FORMAT!r})"
        )
    name = data.get("name")
    collection = data.get("collection")
    if not isinstance(name, str) or not name.strip():
        raise ProjectError(f"{path} needs a non-empty string 'name'")
    if not isinstance(collection, str) or collection != _collection_slug(collection):
        raise ProjectError(f"{path} has an invalid Chroma collection name")
    return data


@dataclass(frozen=True)
class DatasetPaths:
    """Every path belonging to one project or legacy named dataset."""

    root: Path
    name: str
    collection: str
    project: bool = False
    manifest: Mapping[str, Any] | None = None

    @classmethod
    def legacy(cls, name: str, root: str | os.PathLike[str] | None = None) -> "DatasetPaths":
        if not name:
            raise ProjectError("a legacy dataset name is required outside a Raft project")
        return cls(
            root=Path(root or Path.cwd()).expanduser().resolve(),
            name=name,
            collection=name,
        )

    @classmethod
    def from_project(cls, root: str | os.PathLike[str]) -> "DatasetPaths":
        project_root = Path(root).expanduser().resolve()
        data = _read_manifest(project_root)
        return cls(
            root=project_root,
            name=str(data["name"]),
            collection=str(data["collection"]),
            project=True,
            manifest=data,
        )

    @property
    def corpus_path(self) -> Path:
        return self.root / "corpus" / "documents.jsonl" if self.project else self.root / "data" / f"{self.name}.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.root / "corpus" / "chunks.jsonl" if self.project else self.root / "data" / f"{self.name}_chunked.jsonl"

    @property
    def chroma_path(self) -> Path:
        return self.root / "corpus" / "chroma" if self.project else self.root / "data" / self.name

    @property
    def meta_path(self) -> Path:
        return self.root / "metadata" / "state.json" if self.project else self.root / "data" / f"{self.name}_meta.json"

    def transcript_path(self, index: int | str) -> Path:
        if str(index) == "benchmark":
            return (
                self.root / "conversations" / "benchmark.json"
                if self.project
                else self.root / "data" / f"{self.name}_transcript_benchmark.json"
            )
        number = int(index)
        return (
            self.root / "conversations" / f"transcript-{number:04d}.json"
            if self.project
            else self.root / "data" / f"{self.name}_transcript_{number}.json"
        )

    @property
    def finetune_path(self) -> Path:
        return self._conversation_artifact("finetune.json", "finetune.json")

    @property
    def finetune_openai_path(self) -> Path:
        return self._conversation_artifact("finetune.openai.jsonl", "finetune_openai.jsonl")

    @property
    def benchmark_generated_path(self) -> Path:
        return self._conversation_artifact("benchmark.generated.json", "benchmark.json")

    @property
    def benchmark_openai_path(self) -> Path:
        return self._conversation_artifact("benchmark.openai.jsonl", "benchmark_openai.jsonl")

    @property
    def hf_run_path(self) -> Path:
        return (
            self.root / "conversations" / "hf-run"
            if self.project
            else self.root / "data" / f"{self.name}_hf_run"
        )

    @property
    def ariadne_cache_path(self) -> Path:
        return (
            self.root / "fetch" / "ariadne-cache.json"
            if self.project
            else self.root / ".ariadne-cache.json"
        )

    def _conversation_artifact(self, project_name: str, legacy_suffix: str) -> Path:
        return (
            self.root / "conversations" / project_name
            if self.project
            else self.root / "data" / f"{self.name}_{legacy_suffix}"
        )

    def command(self, action: str) -> str:
        """Render a follow-up command for this layout."""
        return f"raft {action}" if self.project else f"raft {action} {self.name}"


DatasetLike: TypeAlias = str | DatasetPaths


def dataset_paths(dataset: DatasetLike) -> DatasetPaths:
    """Coerce a historical dataset name to its legacy layout."""
    return dataset if isinstance(dataset, DatasetPaths) else DatasetPaths.legacy(dataset)


def find_project(start: str | os.PathLike[str] | None = None) -> DatasetPaths | None:
    """Find the nearest enclosing Raft project, starting at ``start``/cwd."""
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        marker = candidate / MANIFEST_NAME
        if marker.exists():
            if not marker.is_file() or marker.is_symlink():
                raise ProjectError(f"{marker} must be a regular file")
            return DatasetPaths.from_project(candidate)
    return None


def initialize_project(
    directory: str | os.PathLike[str] | None = None,
) -> tuple[DatasetPaths, bool]:
    """Initialize ``directory`` (cwd by default) without overwriting data.

    Returns ``(paths, created)``. Re-running against a valid project is
    idempotent and creates any missing empty scaffold directories.
    """
    root = Path(directory or ".").expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ProjectError(f"project path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    marker = root / MANIFEST_NAME
    if marker.exists():
        project = DatasetPaths.from_project(root)
        _ensure_project_dirs(root, adopting=True)
        return project, False

    _ensure_project_dirs(root, adopting=False)
    name = root.name or "raft-project"
    manifest = {
        "format": PROJECT_FORMAT,
        "name": name,
        "collection": _collection_slug(name),
        "target": "",
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ProjectError(f"project appeared while initializing: {marker}") from exc
    return DatasetPaths.from_project(root), True


def _ensure_project_dirs(root: Path, *, adopting: bool) -> None:
    """Preflight and create the five reserved project directories."""
    for name in PROJECT_DIRS:
        path = root / name
        if path.is_symlink():
            raise ProjectError(f"reserved project path may not be a symlink: {path}")
        if path.exists() and not path.is_dir():
            raise ProjectError(f"reserved project path is not a directory: {path}")
        if not adopting and path.is_dir() and any(path.iterdir()):
            raise ProjectError(
                f"refusing to adopt populated reserved directory without a manifest: {path}"
            )
    for name in PROJECT_DIRS:
        (root / name).mkdir(exist_ok=True)
