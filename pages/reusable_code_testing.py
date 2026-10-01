from pathlib import Path

import streamlit as st

from reusable_code.agent import coverage_check
from reusable_code.documents import (
    DOCUMENT_TYPES,
    candidate_documents,
    chunk_document,
    manifest_counts,
    read_document,
    scan_documents,
)
from reusable_code.memory import new_case_state
from reusable_code.retrieval import EvidenceIndex

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "lab2-starter-main"
RUNTIME_ROOT = ROOT / "runtime"

st.set_page_config(page_title="Lab 2 component checks", page_icon="🧪", layout="wide")
st.title("Reusable code testing")
st.caption("Deterministic checks that do not call the model API or alter source evidence.")

manifest, diagnostics = scan_documents(DATA_ROOT)
counts = manifest_counts(manifest)
index = EvidenceIndex(RUNTIME_ROOT)

if st.button("Run deterministic checks", type="primary"):
    checks = []
    checks.append(("Expected archive distribution", counts == {"emails": 600, "contracts": 100, "board_papers": 50}, str(counts)))
    checks.append(("Unique stable document IDs", len({item["doc_id"] for item in manifest}) == len(manifest), f"{len(manifest)} IDs"))
    checks.append(("Allowed source folders only", all(item["doc_type"] in DOCUMENT_TYPES for item in manifest), str(DOCUMENT_TYPES)))
    checks.append(("UTF-8 reads", not diagnostics, f"{len(diagnostics)} error(s)"))

    sample = next(item for item in manifest if not item.get("read_error"))
    source = read_document(DATA_ROOT, manifest, sample["doc_id"], 1, min(5, sample["line_count"]))
    checks.append(("Line-addressable source read", source["start_line"] == 1 and bool(source["text"]), sample["filename"]))
    chunks = chunk_document(DATA_ROOT, manifest, sample["doc_id"])
    checks.append(("Chunks return to source lines", bool(chunks) and chunks[0]["start_line"] >= 1, f"{len(chunks)} chunk(s)"))

    paywise_contracts = candidate_documents(manifest, "PayWise", doc_type="contracts")
    checks.append(("PayWise contract candidate enumeration", bool(paywise_contracts), f"{len(paywise_contracts)} candidate(s)"))

    state = new_case_state(manifest, "test", "", 10)
    complete, problems = coverage_check(state)
    checks.append(("Premature finish is rejected", not complete and bool(problems), f"{len(problems)} blocking item(s)"))

    status = index.status()
    checks.append(("Chroma count is internally consistent", status["collection_count"] == status["indexed_chunks"], str(status)))

    for label, passed, detail in checks:
        st.write(("✅" if passed else "❌") + f" **{label}** — {detail}")

st.subheader("Six-client candidate contract coverage")
rows = []
for client in ("PayWise", "Alphabear", "Bravocat", "Charlemont", "Deltaforce", "Echona"):
    matches = candidate_documents(manifest, client, doc_type="contracts")
    rows.append({"client": client, "candidate contracts": len(matches), "doc_ids": ", ".join(item["doc_id"] for item in matches)})
st.dataframe(rows, use_container_width=True, hide_index=True)
