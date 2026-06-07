# RAG Support Chatbot

A conversational support bot that answers from ONE reference document
(~30 pages, .docx). LangGraph drives the dialogue, Gemini handles
generation and embeddings, Chroma is the in-container vector store, and
everything runs in Docker.

The bot scopes the question, asks diagnostic questions one at a time, and
then walks the user through the fix one step at a time — all from a
single shared persona so phase changes feel like one assistant shifting
focus, never like a new bot taking over.

### Two-container layout

```
┌──────────────────────────┐  HTTP   ┌──────────────────────────┐
│  chatbot                 │ ──────► │  llm-server              │
│  port 8000 (host)        │         │  port 8100 (internal)    │
│                          │         │                          │
│  • FastAPI + LangGraph   │         │  • FastAPI proxy         │
│  • Chroma + SQLite       │         │  • gemini-2.5-flash      │
│  • RemoteChatModel       │         │  • text-embedding-004    │
│  • RemoteEmbeddings      │         │  • holds GOOGLE_API_KEY  │
└──────────────────────────┘         └──────────────────────────┘
```

The chatbot container has **no Google credentials and no Google SDK** —
every Gemini call (chat generation AND embeddings) goes over HTTP to the
`llm-server` container. Swap the model provider by replacing one
service; the chatbot is unaware.

---

## 1. Plug in your document and key

1. Drop your reference `.docx` into `./data/` and rename it to
   `reference.docx` (or override `REFERENCE_DOC_PATH` in
   `docker-compose.yml`).
2. Copy `.env.example` to `.env` and put your Gemini key in it:

   ```bash
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY=<your key>
   ```

The key is read from the environment by Docker Compose and passed into
the container — it is never written into any source file or image layer.

## 2. Build and run (everything via Docker)

```bash
docker compose build
docker compose up -d
```

First boot will:

- start `llm-server` and wait for its `/health` to go green,
- start `chatbot`, which loads `./data/reference.docx`,
- chunk it (defaults: 800 / 120 overlap, see [app/config.py](app/config.py)),
- embed via `llm-server` using `text-embedding-004`,
- persist a Chroma store under `./storage/chroma/`.

The embedding step runs ONCE. Subsequent restarts reuse the persisted
store. Delete `./storage/chroma/` to force a re-ingest after changing the
document, chunking settings, or the embedding model (the vectors are
model-specific — mixing models will silently degrade retrieval quality).

Health check:

```bash
curl http://localhost:8000/health
```

## 3. Talk to the bot

**Easiest — open the built-in chat UI in a browser:**

http://localhost:8000

The page maintains a `session_id` in `localStorage` so refreshing keeps
the same conversation thread. Click "New conversation" to start a fresh
thread.

**Or hit the API directly:**

```bash
curl -s http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"abc-123","message":"Hi, I need help with X."}' | jq
```

Keep using the SAME `session_id` for every turn of the same
conversation. The bot's state (full message history + which solution
step you're on) lives under that ID in the SqliteSaver checkpoint DB at
`./storage/checkpoints.sqlite`. The checkpoint DB is on a mounted
volume, so conversations survive `docker compose down` / `up`.

## 4. Where to tweak things

Open [app/config.py](app/config.py) — every knob is there at the top:

| Setting | What it does |
|---|---|
| `REFERENCE_DOC_PATH` | Path inside the container (set via env in compose) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Splitter config |
| `TOP_K` | How many chunks `retrieve_docs` returns |
| `EMBEDDING_MODEL` | Default: `models/text-embedding-004` (requested from llm-server) |
| `GENERATION_MODEL` | Default: `gemini-2.5-flash` (requested from llm-server) |
| `GENERATION_TEMPERATURE` | Default `0.4` |
| `LLM_SERVER_URL` | Default: `http://llm-server:8100` (compose-internal) |

To force a re-ingest after changing `CHUNK_SIZE` or the document:

```bash
docker compose down
rm -rf ./storage/chroma
docker compose up -d
```

## 5. Reset a single conversation

Bad demo turn? Just send a new `session_id`. To reset everything:

```bash
docker compose down
rm -f ./storage/checkpoints.sqlite
docker compose up -d
```

## 6. Tail the logs

```bash
docker compose logs -f chatbot       # the bot
docker compose logs -f llm-server    # the Gemini proxy (every model call)
```

---

## Architecture notes (for the curious)

- **One conversational node.** [app/graph.py](app/graph.py) has a single
  `converse` node that handles scope, diagnose, and solve. There are no
  per-phase nodes and no node-to-node handoffs — handoffs are what make
  these bots feel robotic.
- **Shared persona.** [app/persona.py](app/persona.py) holds ONE
  identity. Each turn the focus hint at the bottom of the prompt nudges
  the model toward scoping vs. diagnosing vs. delivering the next step,
  but the voice never changes.
- **Tools fetch facts only.** [app/tools.py](app/tools.py) exposes
  `retrieve_docs(query)` — chunks from Chroma. There is deliberately no
  "give_solution" tool; the model composes diagnostic questions and
  solution steps live from the retrieved chunks.
- **`current_step` prevents jumping.** The model READS `current_step`
  from state and trusts it. The number only advances when the model
  emits the hidden `<<STEP_DONE>>` marker after delivering a new step —
  that's the one bookkeeping rule. The model never re-decides where it is.
- **`interrupt()` / `Command(resume=...)`.** [app/graph.py](app/graph.py)
  pauses at a dedicated `pause` node every time it asks the user
  anything. The next `/chat` call resumes the SAME paused thread via
  `Command(resume=<user message>)` — we never re-invoke from `START`
  with fresh input, so the conversation never restarts.
- **`add_messages` reducer.** The model sees the full history every
  turn. No node trims or overwrites `messages`.
- **SqliteSaver checkpointer.** State is persisted per `thread_id` at
  `/app/storage/checkpoints.sqlite`. The SQLite file lives on a mounted
  volume so threads survive container restarts.
