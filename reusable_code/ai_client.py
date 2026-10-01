from __future__ import annotations

import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel


class EvidenceInput(BaseModel):
    doc_id: str
    start_line: int
    end_line: int
    quote: str


class AgentDecision(BaseModel):
    action: Literal[
        "list_documents",
        "search_documents",
        "read_document",
        "save_finding",
        "read_findings",
        "update_plan",
        "request_finish",
    ]
    issue_id: str
    purpose: str
    query: str
    doc_type: Literal["emails", "contracts", "board_papers", "all"]
    entity: str
    doc_id: str
    start_line: int
    end_line: int
    claim: str
    statement_type: Literal["document_statement", "allegation", "inference", "unknown"]
    assessment: str
    potential_impact: str
    priority: Literal["critical", "high", "medium", "low", "unknown"]
    evidence_status: Literal["supported", "conflicted", "insufficient"]
    supporting_evidence: list[EvidenceInput]
    contradictory_evidence: list[EvidenceInput]
    open_questions: list[str]
    next_action: str
    task_status: Literal["not_started", "in_progress", "complete", "unresolved"]
    update_note: str
    finish_reason: str
    incomplete_items: list[str]


def api_key_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def openai_client() -> OpenAI:
    if not api_key_available():
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(timeout=75.0, max_retries=1)


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    client: OpenAI | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    active_client = client or openai_client()
    response = active_client.embeddings.create(
        model=model,
        input=[text.replace("\n", " ") for text in texts],
        encoding_format="float",
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [list(item.embedding) for item in ordered]


def request_agent_decision(
    system_prompt: str,
    context: str,
    model: str,
    client: OpenAI | None = None,
) -> AgentDecision:
    active_client = client or openai_client()
    response = active_client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ],
        text_format=AgentDecision,
        reasoning={"effort": "minimal"},
    )
    if response.output_parsed is None:
        raise RuntimeError("The model did not return a parsed agent decision")
    return response.output_parsed
