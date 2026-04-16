# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
cp .env.example .env
# Fill in credentials in .env
python3 main.py
```

## Running the Tool

```bash
python3 main.py
```

On startup, `main.py` reads from `langfuse_traces.csv` (local cache) by default, then presents an interactive menu (choices 1–5). To fetch fresh data from Langfuse instead of the cached CSV, set the `if False:` block on line 167 to `if True:`.

## Architecture

The project is an interactive CLI analytics tool for analyzing LLM chatbot usage data from Langfuse (an LLM observability platform). The target domain is student interactions with an educational chatbot at UZH.

**Data flow:**

1. **Data source** — either `langfuse_traces.csv` (fetched via `get_data_from_langfuse()`) or an Excel file. Controlled by `ANALYZE_EXCEL` / `ANALYZE_LANGFUSE` flags in `main.py`.
2. **Reduction** — `reduce_data_frame()` narrows the DataFrame to five columns: `id`, `timestamp`, `userId`, `sessionId`, `input.messages`.
3. **Analysis modules** (one imported per choice):
   - `ranking_by_conversations_and_messages_by_user.py` — ranks users by message/conversation count; exports a dated CSV and bar charts.
   - `conversations_and_messages_per_day.py` — aggregate or per-user daily time-series line charts.
   - `classifying_messages.py` — sends user messages to an LLM (OpenAI or LiteLLM proxy) and assigns them to one of 12 Swiss-German educational categories; results are saved as a `.pkl` file for incremental reuse.

**Key constants in `main.py`:**
- `TRACES_FILE` — local CSV cache path (`"langfuse_traces.csv"`)
- `pickle_file_name` — cached classification results (`.pkl`)
- `messages_log_file` — Excel export with pre-classified messages
- `user_id` — the user UUID targeted by choice 3 (change this line to analyze a different user)

## Environment Variables

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY` | Langfuse API public key |
| `LANGFUSE_SECRET_KEY` | Langfuse API secret key |
| `LANGFUSE_HOST` | Langfuse server URL |
| `OPENAI_API_KEY` | OpenAI key used for message classification |
| `LITELLM_API_BASE` | Optional local LiteLLM proxy base URL |
| `LITELLM_API_KEY` | Optional local LiteLLM proxy key |

To switch classification from OpenAI to LiteLLM, uncomment the alternative `client = OpenAI(...)` line in `classifying_messages.py:16`.
