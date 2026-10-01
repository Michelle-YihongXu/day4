# LAWS90286 Lab 2 — Evidence-led M&A due diligence

This Streamlit application performs a bounded, purchaser-side preliminary red-flag review of the supplied Canvassian evidence archive.

## What it does

- Scans only lab2-starter-main/emails, contracts and board_papers.
- Reports the actual archive count (750 documents: 600 emails, 100 contracts and 50 board papers) instead of repeating the README's 1,000-document estimate.
- Assigns stable document and chunk IDs, content hashes and original line ranges.
- Builds an incremental persistent ChromaDB index using OpenAI embeddings.
- Combines semantic retrieval with exact keyword/entity matching.
- Runs a single bounded agent whose requested actions are validated and executed by Python.
- Keeps every run isolated in runtime/runs and checkpoints after each action.
- Validates that quoted evidence exists in the claimed source lines.
- Refuses premature completion when required issues or candidate contract reviews remain open.
- Generates a preliminary board report from saved findings only, with scope limits and an evidence appendix.

## Required issues

1. Jane Wu's continued involvement and motivation.
2. The PayWise financial-difficulty rumour and potential impact. The 20% revenue exposure is task-provided, not independently verified.
3. Change-in-control arrangements for PayWise, Alphabear, Bravocat, Charlemont, Deltaforce and Echona.
4. Open-ended exploration for other material risks.

## Setup

Use the existing virtual environment in the Codespace, then install dependencies:

    source .venv/bin/activate
    python -m pip install -r requirements.txt

Add your own API key to the existing ignored .env file. Do not commit the key:

    OPENAI_API_KEY=your_key_here

Optional model settings:

    OPENAI_MODEL=gpt-5
    OPENAI_EMBEDDING_MODEL=text-embedding-3-large

Model names are configurable because access differs by account. The code does not assume that a particular current model is available.

## Run

    source .venv/bin/activate
    streamlit run home.py

In the app:

1. Confirm that the Overview shows 750 source documents.
2. Build or update the ChromaDB index. Unchanged documents are not re-embedded.
3. Start a new investigation run and provide an analysis basis/date if known.
4. Execute one step at a time first; then use the bounded auto-run control.
5. Review every saved finding and its source quote.
6. Generate and download the Markdown report.

## Tests

    source .venv/bin/activate
    python -m unittest discover -s tests -v

The Streamlit Checks tab and pages/reusable_code_testing.py provide additional acceptance checks without calling the model API.

## Project structure

- home.py — main Streamlit workflow.
- reusable_code/documents.py — manifest, safe source reads, metadata and chunking.
- reusable_code/retrieval.py — incremental ChromaDB indexing and hybrid search.
- reusable_code/ai_client.py — OpenAI embeddings and structured Responses API calls.
- reusable_code/tools.py — whitelisted tools and evidence validation.
- reusable_code/memory.py — isolated run state and checkpoints.
- reusable_code/agent.py — one-action-at-a-time bounded loop and stop rules.
- reusable_code/report.py — deterministic evidence-based report.
- pages/reusable_code_testing.py — standalone component checks.
- tests/test_core.py — offline unit tests.
- runtime/ — generated index, checkpoints and reports; ignored by Git.

## Safety and limitations

The app does not approve the transaction, send communications, use open-web research, execute model-generated code, or modify the original evidence. Retrieval is not the same as substantive review. A missing result does not prove absence of a clause or risk. Contract conclusions remain conditional on the transaction structure, signed version, definitions, exceptions and human legal review.
