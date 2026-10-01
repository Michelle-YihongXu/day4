from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from reusable_code.documents import candidate_documents, manifest_fingerprint

CLIENTS = ("PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_case_state(
    manifest: list[dict[str, Any]],
    scope_note: str,
    analysis_date: str,
    max_rounds: int,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    issues: dict[str, dict[str, Any]] = {
        "jane_wu": {
            "label": "Jane Wu — continued involvement and motivation",
            "status": "not_started",
            "note": "",
            "candidate_doc_ids": [],
            "reviewed_doc_ids": [],
        },
        "paywise_financial_risk": {
            "label": "PayWise — financial difficulty rumour and potential impact",
            "status": "not_started",
            "note": "",
            "candidate_doc_ids": [],
            "reviewed_doc_ids": [],
        },
        "other_material_risks": {
            "label": "Other material risks",
            "status": "not_started",
            "note": "",
            "candidate_doc_ids": [],
            "reviewed_doc_ids": [],
        },
    }
    for client in CLIENTS:
        candidates = candidate_documents(manifest, client, doc_type="contracts")
        issues[f"contract_{client.casefold()}"] = {
            "label": f"{client} — change in control contract review",
            "status": "not_started",
            "note": "",
            "candidate_doc_ids": [item["doc_id"] for item in candidates],
            "reviewed_doc_ids": [],
        }

    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "ready",
        "stop_reason": "",
        "scope_note": scope_note,
        "analysis_date": analysis_date,
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "manifest_count": len(manifest),
        "max_rounds": int(max_rounds),
        "round": 0,
        "ai_calls": 0,
        "consecutive_failures": 0,
        "no_progress_count": 0,
        "seen_actions": [],
        "issues": issues,
        "findings": [],
        "activity": [],
        "last_tool_result": {},
        "incomplete_items": [],
    }


def run_directory(runtime_root: str | Path, run_id: str) -> Path:
    path = Path(runtime_root).resolve() / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_case_state(runtime_root: str | Path, state: dict[str, Any]) -> Path:
    state["updated_at"] = _now()
    directory = run_directory(runtime_root, state["run_id"])
    target = directory / "case_state.json"
    temporary = directory / "case_state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_case_state(runtime_root: str | Path, run_id: str) -> dict[str, Any]:
    target = Path(runtime_root).resolve() / "runs" / run_id / "case_state.json"
    return json.loads(target.read_text(encoding="utf-8"))


def list_runs(runtime_root: str | Path) -> list[dict[str, Any]]:
    runs_root = Path(runtime_root).resolve() / "runs"
    if not runs_root.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in runs_root.glob("*/case_state.json"):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "run_id": state.get("run_id", path.parent.name),
                    "status": state.get("status", "unknown"),
                    "updated_at": state.get("updated_at", ""),
                    "round": state.get("round", 0),
                    "findings": len(state.get("findings", [])),
                }
            )
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)


def record_activity(
    state: dict[str, Any],
    action: str,
    issue_id: str,
    purpose: str,
    result_summary: str,
    status: str = "ok",
) -> None:
    state.setdefault("activity", []).append(
        {
            "timestamp": _now(),
            "round": state.get("round", 0),
            "action": action,
            "issue_id": issue_id,
            "purpose": purpose,
            "result_summary": result_summary,
            "status": status,
        }
    )
    state["activity"] = state["activity"][-250:]
