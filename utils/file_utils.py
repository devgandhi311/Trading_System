"""Safe file and tabular output helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openpyxl import Workbook


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its resolved path."""

    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the file path."""

    file_path = Path(path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def safe_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Create parent folders and write text to a file."""

    file_path = ensure_parent_dir(path)
    file_path.write_text(content, encoding=encoding)
    return file_path


def safe_touch(path: str | Path) -> Path:
    """Create parent folders and touch a file."""

    file_path = ensure_parent_dir(path)
    file_path.touch(exist_ok=True)
    return file_path


def rows_to_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize mappings, dataclasses, and objects into dictionaries."""

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if is_dataclass(row) and not isinstance(row, type):
            normalized.append(asdict(row))
        elif isinstance(row, Mapping):
            normalized.append(dict(row))
        elif hasattr(row, "__dict__"):
            normalized.append(dict(vars(row)))
        else:
            raise TypeError(f"Unsupported row type: {type(row).__name__}")
    return normalized


def save_csv(rows: Iterable[Any], path: str | Path, fieldnames: Sequence[str] | None = None) -> Path:
    """Save rows to CSV, creating parent directories safely."""

    normalized_rows = rows_to_dicts(rows)
    file_path = ensure_parent_dir(path)
    headers = list(fieldnames or _collect_fieldnames(normalized_rows))
    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)
    return file_path


def save_excel(
    rows: Iterable[Any],
    path: str | Path,
    *,
    sheet_name: str = "Sheet1",
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Save rows to an Excel workbook, creating parent directories safely."""

    normalized_rows = rows_to_dicts(rows)
    file_path = ensure_parent_dir(path)
    headers = list(fieldnames or _collect_fieldnames(normalized_rows))
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name[:31] or "Sheet1"
    worksheet.append(headers)
    for row in normalized_rows:
        worksheet.append([row.get(header) for header in headers])
    workbook.save(file_path)
    return file_path


def _collect_fieldnames(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    return tuple(fieldnames)
