"""Filesystem helpers used by dataset adapters."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(root: Path, extensions: set[str] | None = None) -> list[Path]:
    exts = extensions or IMAGE_EXTENSIONS
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts)


def direct_image_files(root: Path, extensions: set[str] | None = None) -> list[Path]:
    exts = extensions or IMAGE_EXTENSIONS
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in exts)


def child_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def find_first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None

