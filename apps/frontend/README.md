# RAG Harness Studio — Frontend

React + Vite + TypeScript control panel for the Agentic RAG Harness FastAPI
backend. No Node server — Vite is dev/build tooling only; at runtime this is
a static site that talks to your FastAPI service over HTTP.

## Quick start

```bash
cd rag-harness-studio
npm install
cp .env.example .env.local   # optional — see below
npm run dev
```

Open `http://localhost:5173`. By default, requests to `/api/*` are proxied
by Vite's dev server to `http://localhost:8000/v1/*` (see `vite.config.ts`),
so **start your FastAPI backend on port 8000 first**:

```bash
# in your backend project
uvicorn rag_harness.api.main:app --reload --port 8000
```

If your backend runs elsewhere, set `VITE_API_BASE_URL` in `.env.local` to
its full URL (e.g. `https://api.your-domain.com/v1`) and skip the proxy.

### Production build

```bash
npm run build      # outputs static files to dist/
npm run preview    # serve the production build locally to sanity-check it
```

Deploy `dist/` to any static host (Vercel, Netlify, S3+CloudFront, nginx,
etc.) — set `VITE_API_BASE_URL` as a build-time env var to your backend's
real, absolute URL before building.

---

## Project structure

```
src/
├── api/                  # Typed HTTP calls to the FastAPI backend — the
│   │                       ONLY place fetch() is called anywhere in the app.
│   ├── client.ts          Shared fetch wrapper: base URL resolution, BYOK
│   │                       header injection, error normalization.
│   ├── queryApi.ts        POST /v1/query, POST /v1/query/agentic
│   ├── healthApi.ts       GET /v1/health, GET /v1/health/llm
│   └── documentsApi.ts    POST /v1/documents (ingestion)
│
├── store/                 # Zustand stores — all app state lives here, not
│   │                       scattered across component useState calls.
│   ├── queryStore.ts       Active mode, in-flight step animation, latest
│   │                       result, run history. The core of the app.
│   ├── knowledgeStore.ts   Ingested documents (client-side tracked — see
│   │                       "Known gaps" below).
│   └── llmConfigStore.ts   Config tab state, persisted to localStorage.
│
├── components/
│   ├── layout/             Topbar (health pills), CollapsibleZone (generic
│   │                       collapsible side-panel wrapper).
│   ├── knowledge/          Zone A — DropZone (drag/drop or click-to-browse
│   │                       ingest), KnowledgePanel (stats + document list).
│   ├── query/               Zone B — ModeTabs, QueryInputBar, ResultCard,
│   │                       QueryWorkspace (composes all of Zone B).
│   ├── steps/               StepGraph (the animated circle+arrow pipeline
│   │                       visualization), StepTracker (linear trace list).
│   ├── config/               ConfigPanel — Zone C, the default/BYOK toggle.
│   └── history/              HistoryStrip — recent-run pills.
│
├── types/                  Shared TS types.
│   ├── domain.ts             Mirrors the FastAPI Pydantic response schemas.
│   └── llmConfig.ts           Config-tab-specific types.
│
├── utils/
│   └── parseActionHistory.ts  Turns the backend's flat `action_history`
│                                list into the 4-stage step pipeline the UI
│                                animates. This is the one piece of "reading
│                                between the lines" logic in the app — see
│                                its doc comment for the exact mapping rules.
│
├── styles/
│   ├── theme.css              Design tokens (colors/fonts), ported from the
│   │                          RAG Harness Studio HTML mockup.
│   └── app.css                 All component styling, incl. every animation
│                                 (step pulse, flowing arrows, expanding ring).
│
├── App.tsx                    Root: Topbar + 3-column workspace shell.
└── main.tsx                   React entry point.
```

## How a query actually flows through the app

1. `QueryInputBar` calls `useQueryStore().run()`.
2. `queryStore.run()` immediately renders an in-flight step pipeline via
   `buildInFlightSteps()` and starts a `setInterval` that advances one stage
   every ~900ms — **this part is a cosmetic "the agent is working"
   animation**, not real server-sent progress (the backend currently
   returns one JSON response at the end of the whole run, not a stream).
3. It calls `runAgenticQuery()` (or `runStandardQuery()` for Standard mode).
4. When the real response lands, `parseActionHistory()` converts the
   backend's actual `action_history` array into the real step list, and
   that — not the animation — is what gets rendered as the final state.
   So the animation never lies about what happened; it's just a
   placeholder while waiting.
5. `ResultCard` renders the answer, source chunk chips, and a degraded-run
   banner if `degraded: true` came back.

If you want **true** live step-by-step updates (not a placeholder
animation), the backend would need to stream progress — e.g. an SSE
endpoint that emits an event after each LangGraph node runs. That's a
moderate backend change (LangGraph supports `.stream()` instead of
`.invoke()`); ask if you want that built.

## Wiring up bring-your-own-key on the backend

The Config tab's "Custom" mode already sends `X-LLM-Provider`,
`X-LLM-Model`, `X-LLM-Api-Key` (and `X-LLM-Base-Url` for custom endpoints)
on every request (see `api/client.ts`). **The backend does not read these
yet** — it always uses its `.env`-configured chain. To make Custom mode
actually change which model answers, add something like this to your
FastAPI query router:

```python
from fastapi import Header

@router.post("/query/agentic")
def query_agentic(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    x_llm_provider: str | None = Header(default=None),
    x_llm_model: str | None = Header(default=None),
    x_llm_api_key: str | None = Header(default=None),
):
    if x_llm_api_key:
        # build a one-off LLMChain wrapping just that provider/model/key
        # instead of pipeline's default agent_chain/generate_chain
        ...
    ...
```

Until that's added, Custom mode is fully wired on the frontend and safe to
use (the key never leaves the browser except to your own backend), it just
won't change the answer yet.

## Known gaps (frontend limitations, tracked deliberately rather than faked)

- **Document upload**: `POST /v1/documents` ingests a file already on the
  *server's* filesystem (`file_path`), not a browser-uploaded file. See the
  doc comment in `api/documentsApi.ts` for the ~5-line multipart upload
  endpoint that would make browser drag-and-drop actually work end-to-end.
- **Document list persistence**: there's no `GET /v1/documents` endpoint, so
  `knowledgeStore` only knows about documents ingested through this UI in
  the current browser session — it resets on page reload. Add that
  endpoint if you want the Knowledge panel to reflect true server state.
- **Live step streaming**: see "How a query actually flows" above.

None of these break the app — they're documented instead of silently
papered over, so you know exactly what's real vs. simulated.
