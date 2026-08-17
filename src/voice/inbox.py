from __future__ import annotations

from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def consume_inbox(inbox_dir: str | Path) -> list[str]:
    """Read pending text notes from an inbox and move them to ``processed/``.

    Image files are moved aside and reported as unsupported (no vision model).
    """
    root = Path(inbox_dir)
    if not root.is_dir():
        return []
    processed = root / "processed"
    notes: list[str] = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        suffix = path.suffix.lower()
        if suffix in _IMAGE_SUFFIXES:
            _relocate(path, processed)
            notes.append(f"(Unsupported image {path.name} was ignored.)")
            continue
        if suffix not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        _relocate(path, processed)
        if text:
            notes.append(text)
    return notes


def _relocate(path: Path, processed: Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    destination = processed / path.name
    if destination.exists():
        destination = processed / f"{path.stem}-{len(list(processed.iterdir()))}{path.suffix}"
    path.rename(destination)
