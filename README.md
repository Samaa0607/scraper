# Saudi MOJ Laws — Scraper, RAG & Legal Agent (Prototype)
> **بوابة الأنظمة واللوائح - وزارة العدل السعودية**

An academic prototype that scrapes legislation from the Saudi Ministry of Justice
Laws & Regulations portal ([laws.moj.gov.sa](https://laws.moj.gov.sa/ar)), stores it
in SQLite, and answers legal questions with a RAG pipeline and a small tool-using
agent. Every answer cites the legislation and passages it came from.

> **Not legal advice.** Answers are generated from a limited local copy of the
> legislation texts for demonstration and education only.

---

## 1. Architecture

```text
MOJ portal (laws.moj.gov.sa)
      │  Playwright session + public gateway APIs     scraper.py, parser.py
      ▼
SQLite  data/saudi_laws.db  (+ JSON/CSV export)       database.py
      │  validation                                   verify_integrity.py
      ▼
Clean + chunk by article (with metadata)              rag/chunking.py
      ▼
Embeddings (multilingual-e5-small, local CPU)         rag/vector_store.py
      ▼
Local vector store  data/index/ (numpy + JSON)        build_index.py
      ▼
RAG: retrieve → context → Claude → grounded answer    rag/rag.py, rag/llm.py
      ▼
Agent: Claude + 3 tools (search / retrieve / details) rag/agent.py
      ▼
Flask API  /api/ask …                                 app.py
      ▼
Web page (Arabic, RTL)                                static/index.html
```

Everything runs locally. There is no database server or vector database, and
no cloud infrastructure; the only external service is the optional Claude API.

## 2. Folder structure

```text
scraper/
├── main.py              # Scraper CLI
├── scraper.py           # Browser session, search/detail API calls, retries, saving
├── parser.py            # API response → Legislation (HTML cleaning, article tree → text)
├── models.py            # Legislation dataclass, CSV columns
├── database.py          # SQLite schema, upsert, queries
├── config.py            # URLs, paths, delays, retry policy, RAG settings
├── verify_integrity.py  # Offline data-quality validation
├── build_index.py       # SQLite → chunks → embeddings → data/index/
├── ask.py               # Ask a question from the terminal
├── app.py               # Flask backend + serves the demo page
├── rag/
│   ├── chunking.py      # Text cleaning + article-aware chunking
│   ├── vector_store.py  # Embedder + numpy vector store
│   ├── rag.py           # Retrieval, context building, grounded answering, citations
│   ├── llm.py           # Claude API wrapper (optional)
│   └── agent.py         # Tool-using legal agent
├── static/index.html    # Demo frontend
├── tests/               # pytest suite (offline)
├── requirements.txt
└── data/
    ├── saudi_laws.db    # SQLite (cumulative across runs)
    ├── legislation.json # Last scraper run
    ├── legislation.csv  # Last scraper run (UTF-8 with BOM for Excel)
    └── index/           # Vector index: embeddings.npy, chunks.json, meta.json
```

## 3. Scraping

The portal is a Nuxt.js app that loads its data from a public gateway API. Plain
HTTP requests are blocked by the site's session protection. The scraper therefore
opens the portal in headless Chromium (Playwright) and calls the same public
endpoints the website uses, from inside that browser session:

| API | Purpose |
|---|---|
| `POST …/statute/section-search` | Search (title search, then keyword search), paginated |
| `GET  …/statute/get-Statute-gateway-Detail?Serial=…` | Full metadata + article tree (`statuteStructure`) |

How it works:
- **Two search passes:** first by title (`statuteName`), then by keyword. Results are deduplicated by `serial`.
- **Retries:** search pages and detail requests are retried 3 times with 3 s / 6 s / 12 s backoff. Network errors, HTTP errors, empty responses and `success: false` all count as failures.
- **Rate limiting:** a polite delay between requests (`--delay`, default 1.5 s).
- **Failure handling:** a failed record or page is logged and the run continues. If the first page of a pass fails, the pass stops, because the page count is unknown.
- **Content:** the article tree (chapters → articles) is turned into text with `##` chapter and `###` article headings.
- **Run summary:** reports pages processed and failed, records received, unique legislations, duplicates, detail failures, empty-content records and output locations.

```bash
python main.py --query "الإفلاس" --max-results 2
python main.py --query "الإفلاس" --max-results 20 --delay 2
```

| Flag | Description | Default |
|---|---|---|
| `--query`, `-q` | Search query (Arabic), **required** | – |
| `--max-results`, `-m` | Maximum legislations to scrape | all |
| `--delay`, `-d` | Seconds between requests | `1.5` |
| `--no-headless` | Show the browser | off |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

### The empty-content bug (fixed)
"اللائحة التنفيذية لنظام الإفلاس" used to be saved with empty content. The parser
was not the cause: it extracts 40,214 characters from the live response. The
cause was the detail request. It failed during that run, and the detail call had
**no retry** and swallowed the error, so the record was saved from search
metadata alone. In addition, `INSERT OR REPLACE` would let such a run overwrite
good content with `NULL`. Fixes:
- detail requests now go through the same retry logic as search pages
- failures and empty-content records are counted and listed in the summary
- the database upsert never replaces existing content with an empty value

## 4. SQLite database

`data/saudi_laws.db`, table `legislations`:

| Column | Notes |
|---|---|
| `id` | auto-increment, stable across re-scrapes |
| `serial` | portal ID, `UNIQUE`; this is the deduplication key |
| `title` | required |
| `legislation_type`, `status`, `issue_date`, `publication_date` | Hijri dates `YYYY-MM-DD` |
| `description` | official summary |
| `source_url` | `https://laws.moj.gov.sa/ar/legislation/<serial>` |
| `content` | full text, article by article |

Each scraper run **upserts** into the database, so the database accumulates
across runs. The JSON and CSV files contain only the last run. Records without a
serial or title are skipped and logged.

### Data-quality validation
```bash
python verify_integrity.py
```
Checks SQLite, JSON and CSV for duplicate serials, empty titles, empty content,
missing or malformed source URLs, missing serials, corrupted Arabic (mojibake or
`�`), and odd date formats. Missing optional fields are not errors. The script
exits with code 1 if it finds any error.

## 5. Preprocessing and chunking (`rag/chunking.py`)

- **Cleaning:** removes zero-width characters and collapses extra spaces and blank lines. The wording is not changed: Arabic letters, diacritics and Arabic-Indic digits are kept.
- **Chunking** uses the article structure:
  1. split on headings, remembering the current chapter (`##`)
  2. one article = one chunk if it is at most `CHUNK_MAX_CHARS` (1500) characters
  3. longer articles are split on line boundaries and labelled e.g. `المادة الخامسة (2/3)`
  4. very short neighbouring articles in the same chapter are merged, up to `CHUNK_MIN_CHARS` (300)
- Records with empty content are skipped.
- **Each chunk stores:** `chunk_id` (`<serial>-0007`), `serial`, `title`, `legislation_type`, `status`, `issue_date`, `publication_date`, `source_url`, `chapter`, `heading`, `text`.
- The embedded text is the title, chapter and article text together, so each chunk makes sense on its own.

## 6. Embeddings and vector store (`rag/vector_store.py`)

- **Model:** [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small). It is multilingual including Arabic, ~470 MB, and runs on CPU. It uses the `query:` / `passage:` prefixes e5 expects. You can change it with the `EMBEDDING_MODEL` environment variable.
- **Store:** normalized vectors in `data/index/embeddings.npy`, chunk texts and metadata in `chunks.json`, and build info in `meta.json` (model, counts, data fingerprint). Search is one matrix–vector product (cosine similarity). This is plenty for a prototype; FAISS would only matter at millions of chunks.
- **Offline:** after the first download, the model loads from the local Hugging Face cache without contacting the Hub, so the app works offline. `python app.py` loads it at startup, so the first question isn't slow; startup can take a minute or two on slower machines.
- **Rebuild:** `build_index.py` stores a hash of all `(serial, content)` pairs. Re-running it rebuilds only when the database changed; `--force` always rebuilds.

```bash
python build_index.py
python build_index.py --force
```

## 7. RAG (`rag/rag.py`)

```text
question → query embedding → cosine search (top 5)
         → keep chunks with score ≥ MIN_RELEVANCE_SCORE
         → numbered context [1] [2] … with title + source URL
         → Claude (answer only from context, cite [n]) → answer + sources
```

- **Grounding:** if no chunk passes the relevance threshold, Claude is not called and the answer is *"البيانات القانونية المتاحة غير كافية للإجابة عن هذا السؤال."* The system prompt (shared by RAG and the agent, `GROUNDING_RULES` in `rag/rag.py`) tells Claude to:
  - use only the numbered passages, with no general knowledge and no invented laws, articles, conditions, dates or procedures
  - cite each statement as legislation + article + `[n]`
  - put copied legal wording inside «», so legal text and explanation stay separate
  - start with the "insufficient" sentence when the passages don't cover the question
  - answer in the question's language, in plain text, without presenting it as legal advice
- **Citations:** each response contains:
  - `sources`: one entry per legislation (title, URL, type, status, and the chunk refs/IDs/scores used)
  - `passages`: the retrieved chunks with their text, numbered as cited
  - a "not legal advice" disclaimer
- **No API key:** the LLM is optional. Without one, the response mode is `retrieval_only`, and it returns the most relevant passages with full citations.
- **LLM failure:** if the Claude call fails (invalid key, network error, rate limit, refusal), the response still returns the retrieved passages as `retrieval_only`, with an `llm_error` field explaining why. The page shows that message.

LLM: Anthropic Claude via the official `anthropic` SDK (default `claude-opus-5-5`,
`LLM_MODEL` to change it). Requests enable the API's server-side refusal fallback
(`fallbacks: "default"`), use a 120 s timeout and 2 retries.

## 8. Agentic AI (`rag/agent.py`)

This is a single agent with a small, readable tool-use loop (at most 6 model calls). It has three read-only tools:

| Tool | What it does |
|---|---|
| `search_legislation(query)` | Finds legislations by name or keyword in SQLite |
| `retrieve_legal_passages(query, serial?)` | Semantic search over the chunks, optionally within one legislation |
| `get_legislation_details(serial)` | Returns metadata, summary, URL and article count |

Claude decides whether retrieval is needed: greetings get a direct answer, while
legal questions go through the tools. Passages are numbered across the whole run,
so the answer cites `[n]`. The response returns only the passages that were cited,
plus a `steps` trace of the tool calls for the demo. Without an LLM, the agent
falls back to plain retrieval.

## 9. Backend API (`app.py`, Flask)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/ask` | `{"question": "...", "mode": "agent" \| "rag"}` → `{answer, mode, sources, passages, steps?, disclaimer}` |
| `GET` | `/api/search?q=…&top_k=5` | Semantic passage search (no LLM) |
| `GET` | `/api/legislations` | List stored legislations (metadata only) |
| `GET` | `/api/legislations/<serial>` | One legislation with full content |
| `GET` | `/api/health` | Index and LLM status |

Errors come back as JSON: `400` invalid input, `404` unknown serial, `503` index
not built. LLM failures do not cause an error status; the endpoint falls back to
retrieved passages (see `llm_error` above).

`mode` in the response is one of:
- `agent`: the agent answered
- `llm`: plain RAG answered
- `retrieval_only`: no LLM configured, or the LLM call failed (`llm_error` is set)
- `insufficient_context`: not enough legal data to answer
- `no_retrieval`: the agent answered a non-legal message directly

## 10. Frontend (`static/index.html`)

A single Arabic (RTL) page served at `/`. You type a question, choose Agent or
plain RAG, and see the answer, the legislation sources (linked to the MOJ portal),
the retrieved passages (expandable) and the agent's tool steps.

## 11. Installation

Requires Python 3.10+.

```bash
cd scraper
pip install -r requirements.txt
python -m playwright install chromium
```

### Setting the Anthropic API key (optional)

The key is read only from the `ANTHROPIC_API_KEY` environment variable. Never put
it in source code, the README or any committed file; `.gitignore` already excludes
`.env` files.

**Windows PowerShell, current window only:**
```powershell
$env:ANTHROPIC_API_KEY = "<your-api-key>"
python app.py
```

**Windows PowerShell, permanently for your user (applies to new terminals):**
```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "<your-api-key>", "User")
# close and reopen PowerShell, then check (prints True/False, never the key):
[bool]$env:ANTHROPIC_API_KEY
```

macOS/Linux: `export ANTHROPIC_API_KEY="<your-api-key>"`

To remove it: `Remove-Item Env:ANTHROPIC_API_KEY` (current window) or
`[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $null, "User")`.

Check whether the app sees a key: open `http://127.0.0.1:5000/api/health` and
look at `"llm_available"`. It only reports true/false.

**Without a key**, everything still works in *retrieval-only* mode. Answers start
with "لم يتم إعداد نموذج اللغة (ANTHROPIC_API_KEY)…" followed by the most relevant
legal passages and their sources. Out-of-scope questions still get the
"insufficient data" reply.

## 12. How to run (end to end)

The commands below work the same in Windows PowerShell and in bash.

```bash
python main.py --query "الإفلاس" --max-results 20   # 1. scrape → SQLite/JSON/CSV
python verify_integrity.py                           # 2. validate data quality
python build_index.py                                # 3. chunk + embed → data/index/
python ask.py "ما هي شروط افتتاح إجراء التسوية الوقائية؟"   # 4a. ask from the terminal
python app.py                                        # 4b. web demo at http://127.0.0.1:5000
```

### Testing the agent

```powershell
python ask.py "ما هي شروط افتتاح إجراء التسوية الوقائية؟"              # agent (default)
python ask.py "ما هي شروط افتتاح إجراء التسوية الوقائية؟" --mode rag   # direct RAG
python ask.py "ما هي أحكام ضريبة القيمة المضافة؟"                     # out of scope → insufficient
```
With a key, agent mode prints `Mode: agent`, an Arabic answer citing `[n]`, the
sources, and the tool steps (e.g. `retrieve_legal_passages`). Without a key it
prints `Mode: retrieval_only` and the passages.

In the browser (`python app.py`, then http://127.0.0.1:5000): type the question,
choose **الوكيل (Agent)** or **RAG مباشر**, and press **اسأل**. The status line
under the box shows whether an LLM is configured.

### API from PowerShell

```powershell
$body = @{ question = "ما هي شروط افتتاح إجراء التسوية الوقائية؟" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/ask -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Example API call (bash/curl):
```bash
curl -X POST http://127.0.0.1:5000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "ما هي شروط افتتاح إجراء التسوية الوقائية؟"}'
```
On Windows, shells may not send Arabic command-line arguments as UTF-8. In that
case, save the JSON body to a UTF-8 file and send it with `--data-binary @question.json`.
The web page and `ask.py` are not affected.

## 13. Tests

```bash
python -m pytest -q        # 58 tests
```

The tests run offline. They use a temporary SQLite database, a mocked browser page
for the scraper, a deterministic fake embedder and a mocked LLM. Coverage:
- **Parser:** nested chapters, missing and malformed fields
- **Scraper:** retry recovery, deduplication, empty-content counting, `max_results`, fast abort when the first page fails, JSON/CSV/SQLite output with Arabic
- **Database:** no duplicates on re-insert, empty content never overwrites stored content, malformed records skipped
- **Validation:** duplicates, empty fields, bad URLs, mojibake
- **Chunking:** metadata, article boundaries, merge/split without losing text
- **Retrieval:** returns relevant chunks, filters by serial, handles empty or unrelated queries safely
- **RAG:** grounded prompt includes the sources; retrieval-only mode
- **Agent:** tool loop, direct answers, tool errors, step limit
- **LLM integration** (`tests/test_llm_integration.py`): uses the real `anthropic` SDK with a fake HTTP transport, so no key or network is needed. Checks:
  - the request Claude would receive: model, beta header, fallbacks, grounding prompt, the retrieved context, and tool results with the legal text
  - Arabic answer parsing and "insufficient" detection
  - out-of-scope questions never reach the LLM
  - 401/429/500 errors fall back to the retrieved passages
  - the agent's tool loop
  - whitespace-only keys are rejected
- **API:** valid JSON, input validation, 404/503 responses

## 14. Example

Question: `ما هي شروط افتتاح إجراء التسوية الوقائية؟`

Retrieval returns articles from **نظام الإفلاس** and **اللائحة التنفيذية لنظام
الإفلاس** about opening the protective-settlement procedure (التسوية الوقائية).
The response lists each source with its URL (for example
`https://laws.moj.gov.sa/ar/legislation/-FF20eK5iu1hvvKJe5GZVQ`) and the cited
passages.

## 15. Known limitations

- **Small corpus:** the database only holds what you scraped (currently 10 legislations from the "الإفلاس" query). Questions outside that scope correctly return "insufficient data".
- **Keyword search pagination:** the keyword search paginates over matching *articles* grouped by legislation, so most pages repeat the same legislation. A broad keyword means many requests for few new legislations; title search is more efficient.
- **Relevance threshold:** `MIN_RELEVANCE_SCORE` is 0.845 (cosine similarity for e5), calibrated on 28 Arabic questions:
  - in-scope questions scored 0.850–0.893, except one casually phrased question at 0.832, which now gets "insufficient"
  - out-of-scope questions (VAT, marriage, inheritance, traffic, …) scored 0.802–0.838 and are all rejected

  It's a heuristic, so new phrasings can land on the wrong side. When an unrelated passage does get through, the LLM prompt still requires an "insufficient" answer. Override it with the `MIN_RELEVANCE_SCORE` environment variable.
- **Live LLM test:** the Claude path is covered by tests that use the real SDK with a fake HTTP transport. It has not yet been run against the live API from this machine, because no `ANTHROPIC_API_KEY` was configured.
- **Ask in Arabic:** the corpus is Arabic, so cross-lingual similarity for English questions is lower (≈0.78) and plain RAG mode treats them as out of scope. Agent mode works around this because the agent writes its retrieval queries in Arabic.
- **Snapshot only:** amendments are captured only as they appear in the article text at scrape time; there is no version history.
- **Fragile portal:** the scraper depends on the portal's current public API and session behaviour, and may need updates if the site changes.
- **LLM quality:** answers are only as good as the retrieved passages. Always check the cited article on the official portal.

## 16. Ethics and compliance

- Uses only public, unauthenticated endpoints that the portal itself uses. No login, CAPTCHA or access control is bypassed.
- Polite request delays and bounded retries.
- Keeps legal text verbatim with a source URL for every record and every chunk. No LLM is used to create or alter stored content.

## 17. Deploying on Render

The repository includes a Render Blueprint (`render.yaml`). The repository root is
this `scraper/` folder.

1. Push this folder to a GitHub (or GitLab) repository.
2. In Render: **New → Blueprint**, select the repository, and confirm.
3. Optional: set `ANTHROPIC_API_KEY` in the service's **Environment** tab. Without it the site runs in retrieval-only mode.

The Blueprint uses these settings (use them if you create a Web Service manually instead):

| Setting | Value |
|---|---|
| Runtime | Python 3.13 (from `.python-version`) |
| Build command | `pip install -r requirements.txt && python -c "from rag.vector_store import Embedder; Embedder()._load()"` |
| Start command | `gunicorn "app:build_app()" --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:$PORT` |
| Health check path | `/api/health` |
| Environment | `HF_HOME=/opt/render/project/src/.hf_cache`, optional `ANTHROPIC_API_KEY` |
| Instance type | **Standard (2 GB)**. The app needs about 1–1.5 GB of RAM, so 512 MB instances run out of memory |

Notes:
- `data/saudi_laws.db` and `data/index/` are committed and read-only at runtime, so no build-time scraping or indexing is needed. After re-scraping, run `python build_index.py` locally and commit both.
- `requirements.txt` uses the CPU-only PyTorch index. Render has no GPU, and this cuts the torch download from ~530 MB to ~190 MB.
- The build command downloads the embedding model into the project folder, so the running service never needs huggingface.co.
- Locally, nothing changes: `python app.py`. gunicorn doesn't run on Windows.
