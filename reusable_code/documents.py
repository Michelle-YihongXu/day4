from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

DOCUMENT_TYPES = ("emails", "contracts", "board_papers")
EXPECTED_COUNTS = {"emails": 600, "contracts": 100, "board_papers": 50}
KNOWN_ENTITIES = (
    "Jane Wu",
    "PayWise",
    "Paywise",
    "Alphabear",
    "Bravocat",
    "Charlemont",
    "Deltaforce",
    "Echona",
    "Canvassian",
)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _header_value(text: str, names: tuple[str, ...]) -> str:
    for line in text.splitlines()[:25]:
        match = re.match(r"^\s*([^:]{1,40})\s*:\s*(.+?)\s*$", line)
        if match and match.group(1).strip().lower() in names:
            return match.group(2).strip()
    return ""


def _mentioned_entities(text: str) -> list[str]:
    found: list[str] = []
    lowered = text.casefold()
    for entity in KNOWN_ENTITIES:
        canonical = "PayWise" if entity.casefold() == "paywise" else entity
        if entity.casefold() in lowered and canonical not in found:
            found.append(canonical)
    return found


def scan_documents(data_root: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Scan only the three source folders and return a stable, auditable manifest."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evidence root does not exist: {root}")

    manifest: list[dict[str, Any]] = []
    diagnostics: list[str] = []

    for doc_type in DOCUMENT_TYPES:
        source_dir = (root / doc_type).resolve()
        if not _is_within(source_dir, root) or not source_dir.is_dir():
            diagnostics.append(f"Missing source directory: {doc_type}")
            continue

        for candidate in sorted(source_dir.rglob("*.txt"), key=lambda p: p.as_posix().casefold()):
            if not candidate.is_file():
                continue
            path = candidate.resolve()
            if not _is_within(path, source_dir):
                diagnostics.append(f"Skipped path outside source directory: {candidate}")
                continue

            relative_path = path.relative_to(root).as_posix()
            doc_id = "doc_" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
            entry: dict[str, Any] = {
                "doc_id": doc_id,
                "source_path": relative_path,
                "filename": path.name,
                "doc_type": doc_type,
                "content_hash": "",
                "title": "",
                "document_date": "",
                "date_basis": "",
                "mentioned_entities": [],
                "contract_parties": [],
                "document_status": "unknown",
                "line_count": 0,
                "index_status": "not_indexed",
                "read_error": "",
            }
            try:
                raw = path.read_bytes()
                text = raw.decode("utf-8")
                entry["content_hash"] = hashlib.sha256(raw).hexdigest()
                entry["line_count"] = len(text.splitlines())
                entry["title"] = _header_value(text, ("subject", "title"))
                entry["document_date"] = _header_value(text, ("date", "document date"))
                entry["date_basis"] = "explicit_header" if entry["document_date"] else ""
                entry["mentioned_entities"] = _mentioned_entities(text)
            except (OSError, UnicodeError) as exc:
                entry["read_error"] = f"{type(exc).__name__}: {exc}"
                diagnostics.append(f"Could not read {relative_path}: {entry['read_error']}")
            manifest.append(entry)

    manifest.sort(key=lambda item: (item["doc_type"], item["source_path"].casefold()))
    return manifest, diagnostics


def manifest_counts(manifest: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(item["doc_type"] for item in manifest)
    return {doc_type: counts.get(doc_type, 0) for doc_type in DOCUMENT_TYPES}


def manifest_fingerprint(manifest: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{item['doc_id']}:{item['content_hash']}:{item['read_error']}" for item in manifest
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validated_path(data_root: str | Path, entry: dict[str, Any]) -> Path:
    root = Path(data_root).expanduser().resolve()
    relative = PurePosixPath(str(entry.get("source_path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid source path")
    if relative.parts[0] not in DOCUMENT_TYPES or relative.suffix.lower() != ".txt":
        raise ValueError("Source path is outside the evidence folders")
    path = (root / Path(*relative.parts)).resolve()
    expected_parent = (root / relative.parts[0]).resolve()
    if not _is_within(path, expected_parent) or not path.is_file():
        raise ValueError("Source document is unavailable")
    return path


def read_document(
    data_root: str | Path,
    manifest: list[dict[str, Any]],
    doc_id: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    matches = [item for item in manifest if item["doc_id"] == doc_id]
    if len(matches) != 1:
        raise ValueError("Unknown or non-unique doc_id")
    entry = matches[0]
    if entry.get("read_error"):
        raise ValueError(f"Document has a recorded read error: {entry['read_error']}")
    path = _validated_path(data_root, entry)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    first = 1 if start_line is None else max(1, int(start_line))
    last = len(lines) if end_line is None or int(end_line) <= 0 else min(len(lines), int(end_line))
    if last < first:
        raise ValueError("End line must not precede start line")
    selected = lines[first - 1 : last]
    return {
        "doc_id": doc_id,
        "filename": entry["filename"],
        "source_path": entry["source_path"],
        "doc_type": entry["doc_type"],
        "start_line": first,
        "end_line": last,
        "text": "\n".join(selected),
        "numbered_text": "\n".join(
            f"{line_no:>5} | {line}" for line_no, line in enumerate(selected, start=first)
        ),
    }


def chunk_document(
    data_root: str | Path,
    manifest: list[dict[str, Any]],
    doc_id: str,
    max_chars: int = 2600,
    overlap_lines: int = 2,
) -> list[dict[str, Any]]:
    source = read_document(data_root, manifest, doc_id)
    lines = source["text"].splitlines()
    if not lines:
        return []

    chunks: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        start = cursor
        length = 0
        end = cursor
        while end < len(lines):
            proposed = length + len(lines[end]) + 1
            if end > start and proposed > max_chars:
                break
            length = proposed
            end += 1
            if end > start + 2 and not lines[end - 1].strip() and length >= max_chars * 0.55:
                break
        if end == start:
            end += 1
        text = "\n".join(lines[start:end]).strip()
        if text:
            first_line = start + 1
            last_line = end
            chunk_id = "chk_" + hashlib.sha256(
                f"{doc_id}:{first_line}:{last_line}:{text}".encode("utf-8")
            ).hexdigest()[:20]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_type": source["doc_type"],
                    "filename": source["filename"],
                    "source_path": source["source_path"],
                    "start_line": first_line,
                    "end_line": last_line,
                    "text": text,
                }
            )
        if end >= len(lines):
            break
        cursor = max(end - overlap_lines, start + 1)
    return chunks


def chunk_manifest(
    data_root: str | Path,
    manifest: list[dict[str, Any]],
    max_chars: int = 2600,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for entry in manifest:
        if not entry.get("read_error"):
            chunks.extend(chunk_document(data_root, manifest, entry["doc_id"], max_chars=max_chars))
    return chunks


def candidate_documents(
    manifest: list[dict[str, Any]], entity: str, doc_type: str | None = None
) -> list[dict[str, Any]]:
    canonical = entity.casefold()
    results = []
    for item in manifest:
        if doc_type and item["doc_type"] != doc_type:
            continue
        names = [str(name).casefold() for name in item.get("mentioned_entities", [])]
        if canonical in names or (canonical == "paywise" and "paywise" in names):
            results.append(item)
    return results
