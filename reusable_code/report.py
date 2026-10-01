from __future__ import annotations

from pathlib import Path
from typing import Any

from reusable_code.documents import manifest_counts
from reusable_code.memory import CLIENTS, run_directory


def _finding_block(finding: dict[str, Any]) -> str:
    lines = [
        f"### {finding['finding_id']} — {finding['priority'].upper()}",
        "",
        f"**Finding:** {finding['claim']}",
        "",
        f"**Characterisation:** {finding['statement_type']} · Evidence: {finding['evidence_status']}",
        "",
        f"**Assessment:** {finding.get('assessment') or 'Not yet assessed.'}",
        "",
        f"**Potential impact:** {finding.get('potential_impact') or 'Not yet determined.'}",
    ]
    if finding.get("contradictory_evidence"):
        lines.extend(["", "**Contrary or inconsistent material:** recorded in the evidence appendix."])
    if finding.get("open_questions"):
        lines.extend(["", "**Open questions:**"])
        lines.extend(f"- {item}" for item in finding["open_questions"])
    if finding.get("next_action"):
        lines.extend(["", f"**Next action:** {finding['next_action']}"])
    return "\n".join(lines)


def generate_report(
    state: dict[str, Any],
    manifest: list[dict[str, Any]],
    index_status: dict[str, Any],
) -> str:
    findings = state.get("findings", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        grouped.setdefault(finding["issue_id"], []).append(finding)

    critical = [item for item in findings if item.get("priority") in {"critical", "high"}]
    counts = manifest_counts(manifest)
    reviewed_ids = {
        doc_id
        for issue in state.get("issues", {}).values()
        for doc_id in issue.get("reviewed_doc_ids", [])
    }
    lines = [
        "# Preliminary Red-Flag Due Diligence Report",
        "",
        "**Transaction:** Proposed acquisition of Canvassian Pty Ltd  ",
        "**Perspective:** Purchaser  ",
        f"**Run ID:** {state.get('run_id', '')}  ",
        f"**Analysis basis/date:** {state.get('analysis_date') or 'Not specified'}  ",
        "",
        "> This is an initial evidence-led review for human legal review. It is not a final legal opinion or a recommendation to approve the transaction.",
        "",
        "## 1. Executive summary",
        "",
    ]
    if critical:
        lines.extend(f"- **{item['priority'].upper()}:** {item['claim']} ({item['finding_id']})" for item in critical)
    else:
        lines.append("- No finding has yet been saved as critical or high priority. This does not mean that no such risk exists.")
    if state.get("incomplete_items"):
        lines.append(f"- {len(state['incomplete_items'])} investigation limitation(s) remain; see section 7.")

    sections = [
        ("2. Jane Wu — continued involvement and motivation", "jane_wu"),
        ("3. PayWise — financial difficulty rumour and potential impact", "paywise_financial_risk"),
    ]
    for title, issue_id in sections:
        lines.extend(["", f"## {title}", ""])
        issue_findings = grouped.get(issue_id, [])
        if issue_findings:
            lines.extend(_finding_block(item) + "\n" for item in issue_findings)
        else:
            issue = state.get("issues", {}).get(issue_id, {})
            lines.append(f"No supported finding has been saved. Status: **{issue.get('status', 'not_started')}**. {issue.get('note', '')}")

    lines.extend([
        "",
        "## 4. Six-client contract matrix",
        "",
        "| Client | Candidate contracts | Reviewed | Status | Current conclusion / gap |",
        "|---|---:|---:|---|---|",
    ])
    for client in CLIENTS:
        issue = state.get("issues", {}).get(f"contract_{client.casefold()}", {})
        candidates = len(issue.get("candidate_doc_ids", []))
        reviewed = len(set(issue.get("reviewed_doc_ids", [])))
        note = (issue.get("note") or "No conclusion saved.").replace("|", "\|")
        lines.append(f"| {client} | {candidates} | {reviewed} | {issue.get('status', 'not_started')} | {note} |")

    lines.extend(["", "### Contract findings", ""])
    for client in CLIENTS:
        for finding in grouped.get(f"contract_{client.casefold()}", []):
            lines.extend([_finding_block(finding), ""])

    lines.extend(["", "## 5. Other material risks", ""])
    other = grouped.get("other_material_risks", [])
    if other:
        lines.extend(_finding_block(item) + "\n" for item in other)
    else:
        issue = state.get("issues", {}).get("other_material_risks", {})
        lines.append(f"No other-risk finding has been saved. Status: **{issue.get('status', 'not_started')}**. {issue.get('note', '')}")

    open_questions = []
    for finding in findings:
        open_questions.extend(finding.get("open_questions", []))
        if finding.get("next_action"):
            open_questions.append(finding["next_action"])
    lines.extend(["", "## 6. Matters for management / transaction team", ""])
    if open_questions:
        lines.extend(f"- {item}" for item in dict.fromkeys(open_questions))
    else:
        lines.append("- No management questions have yet been saved; confirm that the investigation has actually completed before relying on this report.")

    lines.extend([
        "",
        "## 7. Scope and limitations",
        "",
        f"- Source archive scanned: **{len(manifest)}** text documents — {counts['emails']} emails, {counts['contracts']} contracts and {counts['board_papers']} board papers.",
        f"- ChromaDB: **{index_status.get('indexed_chunks', 0)}** chunks from **{index_status.get('indexed_documents', 0)}** documents using {index_status.get('embedding_model') or 'an unrecorded embedding model'}.",
        f"- Documents substantively opened during this run: **{len(reviewed_ids)}**. Indexing a document is not the same as reviewing it.",
        f"- Investigation stopped because: {state.get('stop_reason') or 'the run remains open'}",
        f"- User scope note: {state.get('scope_note') or 'None supplied.'}",
        f"- The supplied archive contains {len(manifest)} documents although the original README refers to 1,000; {max(1000 - len(manifest), 0)} referenced document(s) were not available for review.",
        "- Revenue exposure percentages (PayWise 20%; five other named clients together 40%) come from the task instructions, not from independent financial verification.",
        "- Absence of a retrieved result is not evidence that a risk or clause does not exist. Contract applicability may depend on transaction structure, execution status, definitions, exceptions and external legal analysis.",
    ])
    for item in state.get("incomplete_items", []):
        lines.append(f"- Outstanding: {item}")

    lines.extend(["", "## 8. Evidence appendix", ""])
    if not findings:
        lines.append("No validated findings have been saved.")
    for finding in findings:
        lines.append(f"### {finding['finding_id']} — {finding['claim']}")
        for label, evidence_items in (
            ("Supporting", finding.get("supporting_evidence", [])),
            ("Contradictory", finding.get("contradictory_evidence", [])),
        ):
            for evidence in evidence_items:
                quote = evidence["quote"].replace("\n", " ")
                lines.append(
                    f"- **{label}:** {evidence['doc_id']} — {evidence['filename']}, lines {evidence['start_line']}–{evidence['end_line']}: “{quote}”"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_report(runtime_root: str | Path, state: dict[str, Any], report: str) -> Path:
    path = run_directory(runtime_root, state["run_id"]) / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
