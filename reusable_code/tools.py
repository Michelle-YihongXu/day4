from __future__ import annotations

import hashlib
import re
from typing import Any

from reusable_code.documents import candidate_documents, read_document
from reusable_code.retrieval import EvidenceIndex

ALLOWED_DOC_TYPES = {"emails", "contracts", "board_papers"}
ALLOWED_STATEMENT_TYPES = {"document_statement", "allegation", "inference", "unknown"}
ALLOWED_EVIDENCE_STATUS = {"supported", "conflicted", "insufficient"}


def list_documents(
    manifest: list[dict[str, Any]],
    doc_type: str | None = None,
    entity: str | None = None,
    offset: int = 0,
    limit: int = 25,
) -> dict[str, Any]:
    if doc_type and doc_type not in ALLOWED_DOC_TYPES:
        raise ValueError("Invalid document type")
    items = manifest
    if entity:
        items = candidate_documents(items, entity, doc_type=doc_type)
    elif doc_type:
        items = [item for item in items if item["doc_type"] == doc_type]
    start = max(0, int(offset))
    page = items[start : start + min(max(int(limit), 1), 100)]
    return {
        "total": len(items),
        "offset": start,
        "items": [
            {
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "doc_type": item["doc_type"],
                "source_path": item["source_path"],
                "line_count": item["line_count"],
                "mentioned_entities": item.get("mentioned_entities", []),
                "read_error": item.get("read_error", ""),
            }
            for item in page
        ],
    }


def search_documents(
    index: EvidenceIndex,
    query: str,
    embedding_model: str,
    doc_type: str | None = None,
    entity: str | None = None,
    n_results: int = 8,
) -> dict[str, Any]:
    results = index.search(
        query=query,
        embedding_model=embedding_model,
        n_results=min(max(int(n_results), 1), 20),
        doc_type=doc_type,
        entity=entity,
    )
    return {
        "query": query,
        "doc_type": doc_type or "all",
        "entity": entity or "",
        "result_count": len(results),
        "results": results,
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_evidence(
    data_root: str,
    manifest: list[dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    doc_id = str(evidence.get("doc_id", ""))
    start_line = int(evidence.get("start_line", 0))
    end_line = int(evidence.get("end_line", 0))
    quote = str(evidence.get("quote", "")).strip()
    if not doc_id or start_line < 1 or end_line < start_line or not quote:
        raise ValueError("Evidence needs a valid doc_id, line range and non-empty quote")
    source = read_document(data_root, manifest, doc_id, start_line, end_line)
    if _normalise(quote) not in _normalise(source["text"]):
        raise ValueError(f"Quote was not found in {doc_id} lines {start_line}-{end_line}")
    return {
        "doc_id": doc_id,
        "filename": source["filename"],
        "source_path": source["source_path"],
        "start_line": start_line,
        "end_line": end_line,
        "quote": quote,
    }


def save_finding(
    state: dict[str, Any],
    data_root: str,
    manifest: list[dict[str, Any]],
    finding: dict[str, Any],
) -> dict[str, Any]:
    issue_id = str(finding.get("issue_id", ""))
    if issue_id not in state.get("issues", {}):
        raise ValueError("Finding references an unknown issue")
    statement_type = finding.get("statement_type")
    evidence_status = finding.get("evidence_status")
    if statement_type not in ALLOWED_STATEMENT_TYPES:
        raise ValueError("Invalid statement type")
    if evidence_status not in ALLOWED_EVIDENCE_STATUS:
        raise ValueError("Invalid evidence status")
    claim = str(finding.get("claim", "")).strip()
    if not claim:
        raise ValueError("Finding claim is required")

    supporting = [
        validate_evidence(data_root, manifest, item)
        for item in finding.get("supporting_evidence", [])
    ]
    contradictory = [
        validate_evidence(data_root, manifest, item)
        for item in finding.get("contradictory_evidence", [])
    ]
    if evidence_status == "supported" and not supporting:
        raise ValueError("A supported finding needs at least one validated source quote")

    digest_source = claim + "|" + "|".join(
        f"{item['doc_id']}:{item['start_line']}:{item['end_line']}" for item in supporting
    )
    finding_id = "finding_" + hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    saved = {
        "finding_id": finding_id,
        "issue_id": issue_id,
        "claim": claim,
        "statement_type": statement_type,
        "supporting_evidence": supporting,
        "contradictory_evidence": contradictory,
        "assessment": str(finding.get("assessment", "")).strip(),
        "potential_impact": str(finding.get("potential_impact", "")).strip(),
        "priority": str(finding.get("priority", "unknown")),
        "evidence_status": evidence_status,
        "open_questions": [str(item) for item in finding.get("open_questions", []) if str(item).strip()],
        "next_action": str(finding.get("next_action", "")).strip(),
        "reviewed_by_human": False,
    }
    existing_ids = {item["finding_id"] for item in state.get("findings", [])}
    if finding_id not in existing_ids:
        state.setdefault("findings", []).append(saved)
    state["issues"][issue_id]["status"] = "in_progress"
    return saved


def read_findings(state: dict[str, Any], issue_id: str | None = None) -> dict[str, Any]:
    findings = state.get("findings", [])
    if issue_id:
        findings = [item for item in findings if item.get("issue_id") == issue_id]
    return {"count": len(findings), "findings": findings}


def update_plan(
    state: dict[str, Any], issue_id: str, status: str, note: str
) -> dict[str, Any]:
    if issue_id not in state.get("issues", {}):
        raise ValueError("Unknown issue")
    if status not in {"not_started", "in_progress", "complete", "unresolved"}:
        raise ValueError("Invalid issue status")
    issue = state["issues"][issue_id]
    if issue_id.startswith("contract_") and status == "complete":
        candidates = set(issue.get("candidate_doc_ids", []))
        reviewed = set(issue.get("reviewed_doc_ids", []))
        if not candidates.issubset(reviewed):
            status = "unresolved"
            note = (note + " Candidate contract review is incomplete.").strip()
    issue["status"] = status
    issue["note"] = note.strip()
    return issue
