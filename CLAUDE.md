# capstone_backend — CeduSync API
Workspace rules, purpose and the plan loop are in `../CLAUDE.md`. Shared memory: `../capstone_docs/`.

## Stack
Python, FastAPI + Uvicorn, SQLAlchemy 2 on Supabase Postgres (session pooler), Supabase Storage,
OpenAI Whisper (`base`; `tiny` for yes/no), DistilBERT NER, ReportLab (PDF), qrcode/pyzbar, pyttsx3, JWT auth.

## Commands (Windows, from this folder)
- Activate: `venv\Scripts\activate`
- Run: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (or `python app/main.py`; `--host 0.0.0.0` lets
  other devices on the Wi-Fi reach it; docs at /docs only when `DEBUG=True`). Testing laptop: `../capstone_docs/TESTING-LAPTOP.md`
- Test: `pytest tests/ -v` — uses a throwaway SQLite DB, NER and Whisper preload off; never touches Supabase
- Lint: `ruff check app tests` (installed by the setup handoff)

## Map
- `app/main.py` app factory + routers · `app/core/` config, database, security
- `app/api/routes/` auth, citizen_auth, speech, extraction, qr, forms, settings, sus
- `app/services/` — `answer_service.py` (live interview parsing, biggest file), `speech_service.py` (Whisper),
  gazetteers + `name_lexicon.py`, `tax_service.py`, `pdf_service.py`, `storage_service.py`, `qr_service.py`
- `app/data/` gazetteer CSVs (see its README) · `supabase/migrations/` schema · `tests/test_backend.py`

## Rules for this repo
- Every new endpoint or service function gets a test in `tests/`. Run the full suite before `/result`.
- Schema change = new file in `supabase/migrations/` + matching SQLAlchemy model change. Never edit an applied migration.
- Settings come from `app/core/config.py`; add new ones there and to `.env.example` (never to `.env`).
- Never make a name "correct" into a different real or fictional name; the applicant confirms every field.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
