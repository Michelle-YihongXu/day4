from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reusable_code.agent import coverage_check
from reusable_code.documents import (
    candidate_documents,
    chunk_document,
    manifest_fingerprint,
    read_document,
    scan_documents,
)
from reusable_code.memory import new_case_state
from reusable_code.tools import save_finding


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for folder in ("emails", "contracts", "board_papers"):
            (self.root / folder).mkdir()
        (self.root / "emails" / "message.txt").write_text(
            "Subject: Payment update\nDate: 2024-01-10\n\nPayWise asked for more time.\n",
            encoding="utf-8",
        )
        (self.root / "contracts" / "customer.txt").write_text(
            "Agreement between Canvassian and PayWise\n\n12 Change in control\nConsent is required.\n",
            encoding="utf-8",
        )
        (self.root / "board_papers" / "paper.txt").write_text(
            "Board paper\nJane Wu discussed succession planning.\n",
            encoding="utf-8",
        )
        (self.root / "ignored.txt").write_text("must not be scanned", encoding="utf-8")
        self.manifest, self.diagnostics = scan_documents(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_scan_scope_and_stable_ids(self):
        self.assertEqual(len(self.manifest), 3)
        first = manifest_fingerprint(self.manifest)
        second_manifest, _ = scan_documents(self.root)
        self.assertEqual(first, manifest_fingerprint(second_manifest))
        self.assertEqual(len({item["doc_id"] for item in self.manifest}), 3)

    def test_line_read_and_chunk_provenance(self):
        email = next(item for item in self.manifest if item["doc_type"] == "emails")
        source = read_document(self.root, self.manifest, email["doc_id"], 1, 2)
        self.assertEqual(source["start_line"], 1)
        self.assertIn("Subject", source["numbered_text"])
        chunks = chunk_document(self.root, self.manifest, email["doc_id"], max_chars=40)
        self.assertTrue(chunks)
        self.assertGreaterEqual(chunks[0]["start_line"], 1)

    def test_candidate_contracts_use_content_entities(self):
        matches = candidate_documents(self.manifest, "PayWise", doc_type="contracts")
        self.assertEqual(len(matches), 1)

    def test_path_escape_is_rejected(self):
        tampered = [dict(item) for item in self.manifest]
        tampered[0]["source_path"] = "../secret.txt"
        with self.assertRaises(ValueError):
            read_document(self.root, tampered, tampered[0]["doc_id"])

    def test_exact_quote_validation(self):
        state = new_case_state(self.manifest, "test", "", 10)
        email = next(item for item in self.manifest if item["doc_type"] == "emails")
        finding = {
            "issue_id": "paywise_financial_risk",
            "claim": "PayWise asked for more time.",
            "statement_type": "document_statement",
            "assessment": "The request is a warning sign but not proof of insolvency.",
            "potential_impact": "Potential collection and concentration risk.",
            "priority": "high",
            "evidence_status": "supported",
            "supporting_evidence": [{"doc_id": email["doc_id"], "start_line": 4, "end_line": 4, "quote": "PayWise asked for more time."}],
            "contradictory_evidence": [],
            "open_questions": ["Was payment ultimately made?"],
            "next_action": "Check later correspondence.",
        }
        saved = save_finding(state, str(self.root), self.manifest, finding)
        self.assertEqual(saved["evidence_status"], "supported")
        finding["supporting_evidence"][0]["quote"] = "Invented quote"
        with self.assertRaises(ValueError):
            save_finding(state, str(self.root), self.manifest, finding)

    def test_premature_finish_is_rejected(self):
        state = new_case_state(self.manifest, "test", "", 10)
        complete, problems = coverage_check(state)
        self.assertFalse(complete)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
