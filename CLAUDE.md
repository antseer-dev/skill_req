# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Skill Req is an async Python pipeline that discovers unmet user needs from social media posts and generates "skill specs" — structured requirements for new Claude Code skills. It collects posts from a task API, clusters them with embeddings (UMAP + HDBSCAN), extracts structured needs via LLM, aggregates pain clusters, and produces ranked skill specifications.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (writes JSON + markdown report to output/)
python pipeline.py

# Run the FastAPI server
uvicorn api:app --reload

# Run all tests
pytest

# Run a single test file
pytest tests/unit/test_models.py

# Run a specific test
pytest tests/unit/test_models.py::test_post_valid -v
```

## Architecture

The pipeline is a 5-stage linear flow (`pipeline.py` orchestrates):

```
L0 Collector → L1 Clusterer → L2 Analyzer → L3 Aggregator → L4 Spec Generator
```

**L0 (`stages/l0_collector.py`)** — Fetches posts from the task API via `TaskClient`, deduplicates by `post_id`. Returns `list[Post]` + topic state dicts.

**L1 (`stages/l1_clusterer.py`)** — Embeds post text via OpenRouter (Qwen3-embedding-8b), reduces with UMAP, clusters with HDBSCAN. Uses c-TF-IDF for cluster keywords and LLM for cluster naming. Annotates each `Post` with `cluster_id`, `topic_label`, `topic_keywords`.

**L2 (`stages/l2_analyzer.py`)** — Sends each post to Gemini Flash via OpenRouter for structured need extraction. Produces `list[NeedRecord]` with entity, feature, pain point, sentiment, user type, etc. Filters out non-financial posts.

**L3 (`stages/l3_aggregator.py`)** — Re-embeds need records, clusters again (need-level UMAP + HDBSCAN), then builds `PainCluster` objects with personas, emotion profiles, existing solutions (via LLM), and ODI opportunity scores.

**L4 (`stages/l4_spec_generator.py`)** — Takes top-N pain clusters, calls Claude Sonnet via OpenRouter to generate full `SkillSpec` objects with refined opportunity scores.

### Key Data Flow

`Post` → (L1 annotates cluster) → `NeedRecord` → (L3 aggregates) → `PainCluster` → (L4 generates) → `SkillSpec`

All models are Pydantic v2 in `models.py`.

### Clients (`clients/`)

- **`task_client.py`** — Primary data source. Hits `http://54.169.201.50:8575/api/v1/mock/tasks` for posts + topic states. Filters to Latin-script posts >20 chars.
- **`openrouter.py`** — All LLM calls (chat + embeddings) go through OpenRouter. Has `chat_json()` with robust JSON parsing for LLM output quirks. Bulk model: Gemini Flash. Report model: Claude Sonnet.
- **`claude_client.py`** — Direct Anthropic SDK client (currently unused by the pipeline; pipeline routes Claude calls through OpenRouter).
- **`lunarcrush.py`** — Legacy LunarCrush API client with SQLite cache. Replaced by `task_client.py` as primary source.

### Prompt Templates (`prompts/`)

Templates use `{placeholder}` syntax rendered by `stages/prompt_utils.py:render()` (simple string replace, avoids `str.format()` conflicts with JSON braces).

### Storage

- **`storage/cache.py`** — SQLite-backed key-value cache with TTL expiry, used by `LunarCrushClient`.
- **`storage/tasks.db`** — Created by `api.py` for task tracking.

### API (`api.py`)

FastAPI app with three endpoints: `POST /create_task` (triggers pipeline as background task), `GET /tasks`, `GET /tasks/{id}`. Uses SQLite for task state.

## Environment Variables

Copy `.env.example` to `.env` and fill in:
- `LUNARCRUSH_KEY` — LunarCrush API key (for legacy client)
- `OPENROUTER_KEY` — OpenRouter API key (required for pipeline)
- `CLAUDE_API_KEY` — Anthropic API key (for direct Claude client)

## Testing

Tests use `pytest` with `asyncio_mode = auto`. Async tests run natively without explicit `@pytest.mark.asyncio`. External API calls are mocked with `respx` (for httpx) and `pytest-mock`. Test structure mirrors the pipeline stages: `tests/unit/`, `tests/integration/`, `tests/e2e/`.

## Config

All tunable parameters (model names, concurrency limits, clustering hyperparameters, batch sizes, paths) live in `config.py` as module-level constants.
