from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from reusable_code.agent import auto_run, run_agent_step
from reusable_code.ai_client import api_key_available
from reusable_code.documents import (
    DOCUMENT_TYPES,
    EXPECTED_COUNTS,
    candidate_documents,
    manifest_counts,
    read_document,
    scan_documents,
)
from reusable_code.memory import list_runs, load_case_state, new_case_state, save_case_state
from reusable_code.report import generate_report, save_report
from reusable_code.retrieval import EvidenceIndex

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "lab2-starter-main"
RUNTIME_ROOT = ROOT / "runtime"
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="LAWS90286 Lab 2", page_icon="⚖️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 4rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    .small-note {color: #5b6472; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Scanning source documents…")
def load_manifest(data_root: str):
    return scan_documents(data_root)


@st.cache_resource
def load_index(runtime_root: str):
    return EvidenceIndex(runtime_root)


st.title("LAWS90286 Lab 2")
st.caption("Bounded single-agent M&A due diligence · purchaser perspective · evidence first")

try:
    manifest, scan_diagnostics = load_manifest(str(DATA_ROOT))
except Exception as exc:
    st.error(f"Evidence scan failed: {type(exc).__name__}: {exc}")
    st.stop()

counts = manifest_counts(manifest)
index = load_index(str(RUNTIME_ROOT))
index_status = index.status()
has_key = api_key_available()

with st.sidebar:
    st.header("Run settings")
    st.write("OpenAI API", "Ready" if has_key else "Not configured")
    model = st.text_input("Reasoning model", value=os.getenv("OPENAI_MODEL", "gpt-5"))
    embedding_model = st.text_input(
        "Embedding model", value=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    max_rounds = st.number_input("Maximum investigation rounds", 10, 120, 40, 5)
    search_limit = st.slider("Search results per action", 3, 15, 8)
    st.caption("Model names are configurable because account access can differ.")
    if st.button("Refresh source manifest"):
        load_manifest.clear()
        st.rerun()

metrics = st.columns(5)
metrics[0].metric("Documents", len(manifest), delta=f"README says 1,000; actual {len(manifest)}")
metrics[1].metric("Emails", counts["emails"])
metrics[2].metric("Contracts", counts["contracts"])
metrics[3].metric("Board papers", counts["board_papers"])
metrics[4].metric("Indexed chunks", index_status["indexed_chunks"])

if counts == EXPECTED_COUNTS:
    st.success("Archive verified: 600 emails, 100 contracts and 50 board papers (750 documents).")
else:
    st.warning(f"Archive counts differ from the expected 600/100/50 split: {counts}")
if scan_diagnostics:
    with st.expander(f"Scan diagnostics ({len(scan_diagnostics)})"):
        st.code("\n".join(scan_diagnostics), language="text")

overview_tab, evidence_tab, investigation_tab, findings_tab, report_tab, testing_tab = st.tabs(
    ["Overview & index", "Evidence library", "Investigation", "Findings", "Report", "Checks"]
)

with overview_tab:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Evidence index")
        st.write(
            {
                "indexed_documents": index_status["indexed_documents"],
                "indexed_chunks": index_status["indexed_chunks"],
                "collection_count": index_status["collection_count"],
                "embedding_model": index_status["embedding_model"] or "Not built",
            }
        )
        if not has_key:
            st.info("Add OPENAI_API_KEY to .env, then rerun the app to build embeddings and use the agent.")
        if st.button("Build or update ChromaDB index", disabled=not has_key, type="primary"):
            progress_bar = st.progress(0, text="Preparing index…")

            def report_progress(done: int, total: int, message: str) -> None:
                progress_bar.progress(min(done / max(total, 1), 1.0), text=message)

            try:
                with st.spinner("Indexing changed source documents only…"):
                    result = index.build(
                        DATA_ROOT,
                        manifest,
                        embedding_model=embedding_model,
                        progress=report_progress,
                    )
                progress_bar.progress(1.0, text="Index complete")
                st.success(result)
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"Index build failed: {type(exc).__name__}: {exc}")
    with right:
        st.subheader("Required investigation lines")
        st.markdown(
            """
            1. Jane Wu's continued involvement and motivation.
            2. PayWise financial difficulty rumour and potential impact (task-stated revenue exposure: 20%).
            3. Change-in-control arrangements for PayWise, Alphabear, Bravocat, Charlemont, Deltaforce and Echona.
            4. Open-ended exploration for other material risks.
            """
        )
        st.info(
            "Index coverage, retrieval hits and substantive review are reported separately. A missing hit is never treated as proof of no risk."
        )

with evidence_tab:
    st.subheader("Source document browser")
    filters = st.columns([1, 1, 2])
    selected_type = filters[0].selectbox("Document type", ["all", *DOCUMENT_TYPES])
    selected_entity = filters[1].selectbox(
        "Named entity",
        ["all", "Jane Wu", "PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona"],
    )
    name_filter = filters[2].text_input("Filename contains")
    filtered = manifest
    if selected_type != "all":
        filtered = [item for item in filtered if item["doc_type"] == selected_type]
    if selected_entity != "all":
        ids = {item["doc_id"] for item in candidate_documents(filtered, selected_entity)}
        filtered = [item for item in filtered if item["doc_id"] in ids]
    if name_filter:
        filtered = [item for item in filtered if name_filter.casefold() in item["filename"].casefold()]
    st.caption(f"{len(filtered)} document(s) match the current filters.")
    if filtered:
        labels = {
            item["doc_id"]: f"{item['filename']} — {item['doc_type']} — {item['doc_id']}"
            for item in filtered
        }
        selected_id = st.selectbox("Document", list(labels), format_func=labels.get)
        entry = next(item for item in filtered if item["doc_id"] == selected_id)
        st.json({key: value for key, value in entry.items() if key != "content_hash"})
        try:
            source = read_document(DATA_ROOT, manifest, selected_id)
            st.code(source["numbered_text"], language="text", line_numbers=False, wrap_lines=True)
        except Exception as exc:
            st.error(f"Could not open document: {exc}")
    else:
        st.info("No documents match these filters.")

    st.divider()
    st.subheader("Hybrid search playground")
    query = st.text_input("Search query", placeholder="PayWise overdue invoices payment delay")
    search_type = st.selectbox("Search within", ["all", *DOCUMENT_TYPES], key="playground_type")
    if st.button(
        "Run search",
        disabled=not has_key or index_status["indexed_chunks"] == 0 or not query.strip(),
    ):
        try:
            results = index.search(
                query,
                embedding_model,
                n_results=search_limit,
                doc_type=None if search_type == "all" else search_type,
            )
            for result in results:
                with st.expander(
                    f"#{result['rank']} · {result['filename']} · lines {result['start_line']}–{result['end_line']} · score {result['score']:.3f}"
                ):
                    st.code(result["text"], language="text", wrap_lines=True)
                    st.caption(f"{result['doc_id']} / {result['chunk_id']}")
        except Exception as exc:
            st.error(f"Search failed: {type(exc).__name__}: {exc}")

with investigation_tab:
    st.subheader("Bounded investigation")
    run_summaries = list_runs(RUNTIME_ROOT)
    load_options = ["Current session"] + [item["run_id"] for item in run_summaries]
    selected_run = st.selectbox("Open saved run", load_options)
    if selected_run != "Current session" and st.button("Load selected run"):
        st.session_state.case_state = load_case_state(RUNTIME_ROOT, selected_run)
        st.rerun()

    new_run_cols = st.columns([2, 2, 1])
    scope_note = new_run_cols[0].text_input(
        "Scope note", value="Preliminary purchaser-side red-flag review of supplied archive"
    )
    analysis_date = new_run_cols[1].text_input(
        "Analysis basis/date", placeholder="e.g. Teaching dataset; no assumed current-date expiry"
    )
    if new_run_cols[2].button("Start new run", type="primary"):
        st.session_state.case_state = new_case_state(
            manifest, scope_note, analysis_date, int(max_rounds)
        )
        save_case_state(RUNTIME_ROOT, st.session_state.case_state)
        st.rerun()

    state = st.session_state.get("case_state")
    if not state:
        st.info("Start a new run or load a saved run. Each run has isolated findings and checkpoints.")
    else:
        st.write(
            {
                "run_id": state["run_id"],
                "status": state["status"],
                "round": f"{state['round']} / {state['max_rounds']}",
                "ai_calls": state["ai_calls"],
                "findings": len(state["findings"]),
                "stop_reason": state.get("stop_reason", ""),
            }
        )
        issue_rows = []
        for issue_id, issue in state["issues"].items():
            issue_rows.append(
                {
                    "issue": issue_id,
                    "status": issue["status"],
                    "candidate contracts": len(issue.get("candidate_doc_ids", [])),
                    "reviewed": len(issue.get("reviewed_doc_ids", [])),
                    "note": issue.get("note", ""),
                }
            )
        st.dataframe(issue_rows, use_container_width=True, hide_index=True)

        can_run = has_key and index_status["indexed_chunks"] > 0 and state["status"] not in {"complete", "stopped"}
        action_cols = st.columns([1, 1, 3])
        if action_cols[0].button("Execute next step", disabled=not can_run):
            with st.spinner("Executing one bounded agent action…"):
                st.session_state.case_state = run_agent_step(
                    state,
                    runtime_root=RUNTIME_ROOT,
                    data_root=DATA_ROOT,
                    manifest=manifest,
                    index=index,
                    model=model,
                    embedding_model=embedding_model,
                    search_limit=search_limit,
                )
            st.rerun()
        auto_steps = action_cols[2].slider("Auto-run steps", 1, 10, 3)
        if action_cols[1].button("Auto run", disabled=not can_run):
            with st.spinner(f"Executing up to {auto_steps} bounded actions…"):
                st.session_state.case_state = auto_run(
                    state,
                    steps=auto_steps,
                    runtime_root=RUNTIME_ROOT,
                    data_root=DATA_ROOT,
                    manifest=manifest,
                    index=index,
                    model=model,
                    embedding_model=embedding_model,
                    search_limit=search_limit,
                )
            st.rerun()
        if not has_key:
            st.warning("Agent controls are locked until OPENAI_API_KEY is present in .env.")
        elif index_status["indexed_chunks"] == 0:
            st.warning("Build the evidence index before running the investigation.")

        st.subheader("Activity log")
        for item in reversed(state.get("activity", [])[-30:]):
            icon = "✅" if item["status"] == "ok" else "⚠️"
            st.markdown(
                f"{icon} **Round {item['round']} · {item['action']} · {item.get('issue_id') or 'system'}** — {item['purpose']}  \n{item['result_summary']}"
            )

with findings_tab:
    st.subheader("Validated findings")
    state = st.session_state.get("case_state")
    if not state or not state.get("findings"):
        st.info("No validated findings have been saved in the current run.")
    else:
        for finding in state["findings"]:
            with st.expander(
                f"{finding['priority'].upper()} · {finding['finding_id']} · {finding['claim']}",
                expanded=finding["priority"] in {"critical", "high"},
            ):
                st.write(
                    {
                        "issue": finding["issue_id"],
                        "statement_type": finding["statement_type"],
                        "evidence_status": finding["evidence_status"],
                        "assessment": finding["assessment"],
                        "potential_impact": finding["potential_impact"],
                        "open_questions": finding["open_questions"],
                        "next_action": finding["next_action"],
                    }
                )
                for evidence in finding["supporting_evidence"]:
                    st.code(
                        f"{evidence['filename']} lines {evidence['start_line']}–{evidence['end_line']}\n{evidence['quote']}",
                        language="text",
                        wrap_lines=True,
                    )

with report_tab:
    st.subheader("Board-ready preliminary report")
    state = st.session_state.get("case_state")
    if not state:
        st.info("Open an investigation run first.")
    else:
        report = generate_report(state, manifest, index.status())
        st.markdown(report)
        report_cols = st.columns(2)
        report_cols[0].download_button(
            "Download report.md",
            data=report,
            file_name=f"{state['run_id']}_report.md",
            mime="text/markdown",
        )
        if report_cols[1].button("Save report checkpoint"):
            path = save_report(RUNTIME_ROOT, state, report)
            st.success(f"Saved to {path.relative_to(ROOT)}")

with testing_tab:
    st.subheader("Acceptance checks")
    checks = [
        ("Actual archive total is reported", len(manifest) == 750, f"Observed {len(manifest)}"),
        ("All doc_ids are unique", len({item['doc_id'] for item in manifest}) == len(manifest), "Stable path-derived IDs"),
        ("Only allowed source folders are scanned", all(item["doc_type"] in DOCUMENT_TYPES for item in manifest), str(DOCUMENT_TYPES)),
        ("UTF-8 read failures are visible", not scan_diagnostics, f"{len(scan_diagnostics)} diagnostic(s)"),
        ("Chroma registry agrees with collection", index_status["indexed_chunks"] == index_status["collection_count"], str(index_status)),
    ]
    for label, passed, detail in checks:
        st.write(("✅" if passed else "⚠️") + f" **{label}** — {detail}")
    st.caption(
        "Further tests cover path traversal rejection, duplicate IDs, exact quote validation, incomplete contract coverage and deterministic report limitations."
    )
