# CLAUDE.md — Staged, Grounded LLM Analysis Pipeline

> **Adapt before use:** replace `{INPUT}` (e.g. email thread / log file / documents),
> `{SOURCE_ID}` (e.g. EMAIL_1, LOG_3, DOC_2), and the `EXTRACT_*` stages with whatever
> the problem actually asks to extract. Everything else stays.

## Goal
Take messy `{INPUT}`, run it through a **strict multi-stage pipeline**, and produce
**validated, source-grounded, structured output**. Correctness and traceability over features.

## Constraints (timed ~60-min build)
- Favor a **working end-to-end skeleton over completeness**. Stub extra stages and label them `# TODO`.
- A stage may run on small sample data as long as the whole pipeline executes end-to-end.
- Language: **Python**. LLM reached through **one provider, key from `.env`** — never hardcoded.
- Propose a scoped plan that fits the time budget before writing code; don't over-build.

## Build order (do in this order; each layer must work before the next)
1. Runs end-to-end on sample input and prints clean per-stage output. **Non-negotiable.**
2. Stage state machine that **cannot skip or reorder** stages.
3. At least one real LLM stage returning **typed structured output**.
4. **Grounding**: LLM cites a `{SOURCE_ID}` for every claim; citations validated to exist.
5. `validate.py` + audit trail (`llm_calls.jsonl`).
6. README, `.env.example`, `requirements.txt`, Dockerfile, 1–2 pytest tests.
7. *(only if ahead)* a lightweight HTML/CLI report.

## Non-negotiables
- **Stage machine**: list of stages + `advance(expected, next)` that **raises** on any
  out-of-order transition. Log stage start/complete with timings.
- **Deterministic vs LLM split**: parse/structure with plain code; use the LLM **only for
  judgment** (classification, extraction, drafting). Keep deterministic work out of the LLM for
  reliability + cost. **All numeric computation, counting, and scoring is Python — never the LLM.**
  Deterministic stages must be reproducible: seed any randomness; same input → same output.
- **Mock LLM mode**: a `MOCK_LLM` flag makes LLM stages return canned **typed** output so the whole
  pipeline runs end-to-end with no live API. Protects the demo if the key/network fails, and lets
  `validate.py` run offline. Do not silently fall back to mock — switch only on the explicit flag.
- **Typed output**: every LLM response is parsed into a **Pydantic model**. Invalid output
  **fails loud**, never silently continues.
- **Grounding / anti-hallucination** (the headline feature):
  - Prompt rule: *"Use ONLY facts present in `{INPUT}`. Cite a `{SOURCE_ID}` for every claim.
    Do not invent, assume, or speculate. If evidence is missing, exclude it."*
  - Controlled vocabulary for confidence: `confirmed | implied | assumed`.
  - After each LLM stage, **validate every citation references a real `{SOURCE_ID}`**; reject if not.
- **Audit trail**: append every LLM call (prompt + response + model) to `llm_calls.jsonl`.
- **Validation suite** (`validate.py`): checks required files exist, JSON is valid,
  citations are real, vocabulary is compliant.
- **CLI logging**: colored, per-stage banners, ✓/✗, timings, final summary table. This is the "visual".

## Architecture
```
INIT → LOADED → PARSED (det.) → STRUCTURED (det.)
     → EXTRACT_A (LLM) → EXTRACT_B (LLM) → SUMMARY (det.)
     → VALIDATION → FINALISED
```
One stage per module under `STAGES/`. Orchestrator in `run_pipeline.py`.

## Style & safety
- Small, typed functions. Clear errors, no silent failures.
- No secrets in code — read keys from `.env` (`.env` is git-ignored; ship `.env.example`).
- Generate small, reviewable pieces one at a time — no giant unreviewable blocks.
- Prefer stdlib + minimal deps; pin them in `requirements.txt`.

## How I want you to work with me
- Propose the plan/stage list first; wait for my OK before large changes.
- When you hit a bug, read the actual error and state a hypothesis before editing.
- If you loop twice on the same fix, stop and ask me — don't thrash.
