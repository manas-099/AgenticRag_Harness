
# Final Plan — Page-Level Tagging + Contextual Preamble Pipeline

## 1. Goal

Replace the current design (1 whole-doc tag call + 1 preamble call *per chunk*) with a page-parallel design that:
- Generates tags **and** preambles for all of a page's chunks in a single LLM call per page.
- Merges tags across pages into one canonical, non-duplicated vocabulary, using hierarchical batching so no single merge call sees too many candidates at once.
- Lets a supervisor step add cross-page relationship context to preambles — but only pay for that LLM call when there's actually something worth comparing (similarity-gated, not automatic).
- Keeps everything auditable: original page-agent output is never overwritten, only appended to.

## 2. The Pipeline, in Plain English

**Phase 1 — Page Agents (parallel, worker pool of N)**
Each page, independently and concurrently:
- Reads its full page text (chunk boundaries already known from `StructureAwareChunker`, deterministic, done before this call).
- One LLM call returns, per chunk on that page: a preamble + tags (with one-sentence reasons per tag).
- Writes results into a shared `chunk_store` (one row per chunk), immediately — no waiting on other pages.

**Phase 2 — Tag Reconciliation (hierarchical, triggered after Phase 1 tag counts are known)**
- If any single page's tag count exceeds `WITHIN_PAGE_SPLIT_THRESHOLD`, that page's own tags get split and merged internally first.
- Remaining pages get dynamically batched by running tag count (not page count) so no merge call exceeds `PAGE_BATCH_THRESHOLD` tags.
- Each batch is merged in parallel; if more than one batch exists, one final call merges the batch results into the canonical tag list.
- Every chunk's tag (written in Phase 1) gets re-validated against this canonical list — an unvalidated tag is discarded, same principle as your existing `ContextualChunker`.

**Phase 3 — Supervisor Preamble Pass (sequential, page-order, gated)**
- Consumes Phase 1 results strictly in page order (1, then 2, then 3…), independent of the worker pool's actual completion order.
- Before each page: cheap, free similarity check (tag overlap or embedding cosine similarity, using the embedding model already in the pipeline) against a small running memory of themes/tags seen so far.
- Only if similarity crosses a threshold does an LLM call fire, to write a short relationship sentence.
- That sentence is **appended** to the relevant chunk(s)' `preamble_additions` — the original `preamble_base` from Phase 1 is never overwritten.
- Running memory stays small (condensed themes/tags, not full page text) so it doesn't grow unbounded with document length.

## 3. Data Model

```
chunk_store[chunk_id] = {
    "page": int,
    "raw_text": str,
    "preamble_base": str,          # from Phase 1 page agent
    "preamble_additions": [str],   # from Phase 3 supervisor, append-only
    "tags": [str],                 # validated against canonical list post-Phase 2
}
```

## 4. Pseudocode

```
# ---------- PHASE 1: Page agents, parallel ----------
with WorkerPool(size=N):
    for page in pages:                     # dispatched concurrently
        result = LLM_call(page.text, page.chunk_boundaries)
        # result = { chunks: [ {chunk_id, preamble, tags:[{tag, reason}]} ] }
        for chunk in result.chunks:
            chunk_store[chunk.chunk_id] = {
                "page": page.num, "raw_text": chunk.raw_text,
                "preamble_base": chunk.preamble, "preamble_additions": [],
                "tags": chunk.tags,
            }
        page_results[page.num] = result    # also feeds Phase 2 + Phase 3

# ---------- PHASE 2: Hierarchical tag merge ----------
for page_result in page_results:
    if len(page_result.tags) > WITHIN_PAGE_SPLIT_THRESHOLD:
        sub_batches = split_by_count(page_result.tags, WITHIN_PAGE_SPLIT_THRESHOLD)
        sub_merged  = parallel[ LLM_merge(b) for b in sub_batches ]
        page_result.tags = LLM_merge(sub_merged)

batches = dynamic_batch_by_tag_count(page_results, PAGE_BATCH_THRESHOLD)
batch_merged = parallel[ LLM_merge(b) for b in batches ]
canonical_tags = batch_merged[0] if len(batch_merged) == 1 else LLM_merge(batch_merged)

for chunk_id in chunk_store:
    chunk_store[chunk_id].tags = [t for t in chunk_store[chunk_id].tags if t in canonical_tags]

# ---------- PHASE 3: Supervisor preamble pass, sequential by page order ----------
running_memory = {}   # small, condensed themes/tags
for page_num in sorted(page_results.keys()):
    if cheap_similarity(page_results[page_num], running_memory) > THRESHOLD:
        edits = LLM_call_supervisor(page_results[page_num], running_memory)
        for chunk_id, sentence in edits.items():
            chunk_store[chunk_id]["preamble_additions"].append(sentence)
    running_memory = update_memory(running_memory, page_results[page_num])   # cheap, no LLM needed unless you choose to

# ---------- FINAL ----------
for chunk_id in chunk_store:
    row = chunk_store[chunk_id]
    contextual_text = " ".join([row.preamble_base] + row.preamble_additions) + "\n" + row.raw_text
    # → build Chunk(..., metadata=ChunkMetadata(..., topic_tag=row.tags[0] or similar))
```

## 5. Worst-Case LLM Call Count — 10 pages, 20 chunks

Worst case = every page produces enough tags to force max batching, *and* every page relates to a prior one (supervisor fires every time).

| Step | Calls | Notes |
|---|---|---|
| Phase 1 — page agents | **10** | 1 per page, regardless of chunk count per page (20 chunks handled *inside* these 10 calls, not separately) |
| Phase 2 — within-page splits | **0–3 per overflowing page** | Worst case: assume 1 page overflows badly → 2 sub-merges + 1 collapse = 3 |
| Phase 2 — cross-page batch merges | **~2–5** | Depends on tag density; worst case with `PAGE_BATCH_THRESHOLD=20` and 10 pages, assume 5 batches → 5 calls |
| Phase 2 — final merge-of-batches | **1** | Only if >1 batch (yes here) |
| Phase 3 — supervisor preamble | **9** (worst case) | Page 1 never triggers (nothing prior); pages 2–10 all trigger = 9 |
| **Total (worst case)** | **~27–31** |

**Compare to your original per-chunk design** on the same 10-page/20-chunk doc: 1 whole-doc tag call + 20 per-chunk preamble calls = **21 calls**, but with weaker cross-page awareness and higher tag-drift risk.

**Best case (no page overflow, no cross-page relationships, single tag batch):** 10 (page agents) + 1 (single merge) + 0 (supervisor never fires) = **11 calls** — roughly half of the original design, with *better* preamble quality since each page agent already sees full-page context instead of just neighboring chunks.

So the honest trade-off: **best case is cheaper and better; worst case is somewhat more expensive but produces meaningfully more coherent, cross-referenced output** — and the worst case only happens when the document genuinely has that much cross-page interrelation, which is exactly when the extra cost is buying you something real.

Ready to move to building this in LangGraph — want me to start with Phase 1 (page agent node + worker pool) first, or sketch the full graph structure (all phases, edges, `Send` fan-out) before writing any code?


```python

Here's the full implementation, split into cells you can add to your notebook. I'm building this as a **new class** (`PageLevelContextualChunker`) inserted after your existing `ContextualChunker` cell (cell 27), rather than overwriting it — so you can A/B test old vs new before committing. I used **plain Python state** (dicts/Pydantic objects) for memory, per what we locked in earlier — no LangChain memory class, since this isn't a chat-memory problem. For parallelism I used `ThreadPoolExecutor` (same pattern your codebase already uses everywhere) rather than a full LangGraph subgraph — it gets you identical behavior with far less new surface area, since LangGraph's `Send`/fan-in machinery isn't buying anything extra over a worker pool + ordered consumer loop here.

---

### 📍 Cell to ADD — insert as a new cell right after your `ContextualChunker` cell (currently cell 27)

**Part 1 of 4 — Data models + thresholds**

```python
# ============================================================
# CELL (new): Page-level tagging + contextual preamble — data models
# and tunable thresholds for hierarchical tag merging.
# ============================================================
from pydantic import BaseModel
from typing import Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Tunables (start here, adjust after testing on real docs) ----
WITHIN_PAGE_SPLIT_THRESHOLD = 40   # max tags from ONE page before splitting it internally
PAGE_BATCH_THRESHOLD = 20          # max tags per cross-page merge call
SUPERVISOR_SIMILARITY_THRESHOLD = 0.55  # cosine similarity gate for supervisor LLM call
PAGE_WORKER_COUNT = 4              # size of the page-agent worker pool


class TagWithReason(BaseModel):
    tag: str
    reason: str  # one sentence — why this tag applies


class ChunkPreambleAndTags(BaseModel):
    chunk_id: str          # explicit id, so we never rely on list position/order
    preamble: str
    tags: list[TagWithReason]


class PageAgentOutput(BaseModel):
    """One LLM call per page returns this — preamble + tags for EVERY chunk on that page."""
    page_num: int
    chunks: list[ChunkPreambleAndTags]


class TagMergeOutput(BaseModel):
    """Output of any merge call (within-page, batch, or final) — a deduped tag+reason list."""
    tags: list[TagWithReason]


class SupervisorEdit(BaseModel):
    chunk_id: str
    relationship_sentence: str


class SupervisorOutput(BaseModel):
    edits: list[SupervisorEdit]  # empty list if nothing worth linking
```

---

**Part 2 of 4 — Chunk store + Phase 1 (page agents, parallel)**

```python
# ============================================================
# CELL (new, continued): Chunk store + Phase 1 — page agents.
# One LLM call per page returns preamble+tags for every chunk on
# that page. Runs across a fixed worker pool, fully parallel.
# ============================================================

class ChunkStore:
    """One row per chunk. preamble_base is written once by the page agent
    and never overwritten. preamble_additions is append-only, written later
    by the supervisor (Phase 3) — so original output is always recoverable."""

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def write_base(self, chunk_id: str, page: int, raw_text: str, preamble: str, tags: list[str]):
        with self._lock:
            self._store[chunk_id] = {
                "page": page,
                "raw_text": raw_text,
                "preamble_base": preamble,
                "preamble_additions": [],
                "tags": tags,
            }

    def append_supervisor_sentence(self, chunk_id: str, sentence: str):
        with self._lock:
            if chunk_id in self._store:
                self._store[chunk_id]["preamble_additions"].append(sentence)

    def set_final_tags(self, chunk_id: str, tags: list[str]):
        with self._lock:
            if chunk_id in self._store:
                self._store[chunk_id]["tags"] = tags

    def get(self, chunk_id: str) -> Optional[dict]:
        return self._store.get(chunk_id)

    def all_rows(self) -> dict[str, dict]:
        return dict(self._store)


class PageAgent:
    """Reads one full page (with its already-known chunk boundaries) and
    returns preamble + tags for every chunk on that page, in ONE LLM call."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger("rag_harness.page_agent")
        self.grammar = XGrammar(PageAgentOutput, name="page_agent_output")

    def run(self, page_num: int, page_text: str, chunk_ids_and_texts: list[tuple[str, str]]) -> PageAgentOutput:
        chunk_list_str = "\n".join(f"- chunk_id={cid}: {text[:300]}..." for cid, text in chunk_ids_and_texts)

        system_prompt = (
            "You read a full page of a document and produce, for EVERY listed chunk, "
            "(1) a 1-2 sentence preamble situating that chunk within the page's content, and "
            "(2) topic tags (snake_case) that apply to that chunk, each with a one-sentence reason. "
            "Do NOT summarize the page as a whole — ground every preamble in the specific chunk. "
            "Return exactly one entry per chunk_id listed, using the SAME chunk_id given."
        )
        user_content = textwrap.dedent(f"""
            Full page text (page {page_num}):
            {page_text[:10000]}

            Chunks on this page (use these EXACT chunk_ids in your output):
            {chunk_list_str}
        """).strip()

        try:
            response = self.llm_client.generate(
                system_prompt, user_content,
                response_format=self.grammar.build_response_format(),
                max_tokens=1200, temperature=0.0,
            )
            parsed = PageAgentOutput.model_validate_json(response)

            # Validate: every chunk_id we asked for must be present — never trust silently
            expected_ids = {cid for cid, _ in chunk_ids_and_texts}
            returned_ids = {c.chunk_id for c in parsed.chunks}
            missing = expected_ids - returned_ids
            if missing:
                self.logger.warning(f"Page {page_num}: LLM omitted chunk_ids {missing} — filling with empty preamble")
                for cid in missing:
                    parsed.chunks.append(ChunkPreambleAndTags(chunk_id=cid, preamble="", tags=[]))

            return parsed
        except Exception as e:
            self.logger.error(f"Page {page_num} agent call failed: {e} — returning empty preambles")
            return PageAgentOutput(
                page_num=page_num,
                chunks=[ChunkPreambleAndTags(chunk_id=cid, preamble="", tags=[]) for cid, _ in chunk_ids_and_texts],
            )


def run_page_agents_parallel(
    page_agent: PageAgent,
    pages: list[dict],              # [{"page_num": int, "text": str, "chunks": [(chunk_id, chunk_text), ...]}]
    chunk_store: ChunkStore,
    max_workers: int = PAGE_WORKER_COUNT,
) -> dict[int, PageAgentOutput]:
    """Phase 1: fan-out across a worker pool. Each finished page writes
    its chunks' base preamble+tags into chunk_store immediately."""
    page_results: dict[int, PageAgentOutput] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(page_agent.run, p["page_num"], p["text"], p["chunks"]): p["page_num"]
            for p in pages
        }
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            result = future.result()
            page_results[page_num] = result
            for chunk in result.chunks:
                raw_text = dict(pages[0]["chunks"]).get(chunk.chunk_id, "")  # placeholder, fixed below
            # write into chunk_store — need raw_text, so look it up from the source page
            page_chunk_texts = dict(next(p for p in pages if p["page_num"] == page_num)["chunks"])
            for chunk in result.chunks:
                chunk_store.write_base(
                    chunk_id=chunk.chunk_id,
                    page=page_num,
                    raw_text=page_chunk_texts.get(chunk.chunk_id, ""),
                    preamble=chunk.preamble,
                    tags=[t.tag for t in chunk.tags],
                )

    return page_results
```

---

**Part 3 of 4 — Phase 2 (hierarchical tag merge)**

```python
# ============================================================
# CELL (new, continued): Phase 2 — hierarchical tag merging.
# Handles both a single page producing too many tags (within-page
# split) and too many pages' tags landing in one merge call
# (cross-page dynamic batching), so no merge call ever sees more
# than THRESHOLD tag candidates at once.
# ============================================================

class TagMerger:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger("rag_harness.tag_merger")
        self.grammar = XGrammar(TagMergeOutput, name="tag_merge_output")

    def merge(self, tag_groups: list[list[TagWithReason]]) -> list[TagWithReason]:
        """Merges N lists of {tag, reason} into one deduped canonical list.
        Works for within-page sub-batches, cross-page batches, or final merge."""
        flat_input = [t for group in tag_groups for t in group]
        if not flat_input:
            return []
        if len(flat_input) == 1:
            return flat_input  # nothing to merge

        tags_str = "\n".join(f"- {t.tag}: {t.reason}" for t in flat_input)
        system_prompt = (
            "You receive a list of topic tags with one-sentence reasons, possibly containing "
            "near-duplicates (different wording, same concept). Merge duplicates into ONE canonical "
            "snake_case tag per distinct concept, keeping the clearest reason (or a light merge of "
            "reasons). Do not invent new tags. Prefer fewer, broader tags over many narrow ones."
        )
        try:
            response = self.llm_client.generate(
                system_prompt, tags_str,
                response_format=self.grammar.build_response_format(),
                max_tokens=800, temperature=0.0,
            )
            parsed = TagMergeOutput.model_validate_json(response)
            return parsed.tags
        except Exception as e:
            self.logger.error(f"Tag merge call failed: {e} — falling back to naive dedup by tag string")
            seen, out = set(), []
            for t in flat_input:
                if t.tag not in seen:
                    seen.add(t.tag)
                    out.append(t)
            return out


def split_by_tag_count(tags: list[TagWithReason], limit: int) -> list[list[TagWithReason]]:
    """Greedily groups tags into sub-lists, each capped at `limit` tags."""
    batches, current, count = [], [], 0
    for t in tags:
        if count >= limit and current:
            batches.append(current)
            current, count = [], 0
        current.append(t)
        count += 1
    if current:
        batches.append(current)
    return batches


def dynamic_batch_pages_by_tag_count(
    page_results: dict[int, PageAgentOutput], threshold: int
) -> list[list[int]]:
    """Groups page_nums into batches, in page order, where each batch's
    TOTAL tag count stays under `threshold`. Returns lists of page_nums."""
    batches, current, count = [], [], 0
    for page_num in sorted(page_results.keys()):
        page_tag_count = sum(len(c.tags) for c in page_results[page_num].chunks)
        if count + page_tag_count > threshold and current:
            batches.append(current)
            current, count = [], 0
        current.append(page_num)
        count += page_tag_count
    if current:
        batches.append(current)
    return batches


def reconcile_tags(
    page_results: dict[int, PageAgentOutput],
    tag_merger: TagMerger,
    chunk_store: ChunkStore,
    within_page_threshold: int = WITHIN_PAGE_SPLIT_THRESHOLD,
    batch_threshold: int = PAGE_BATCH_THRESHOLD,
) -> list[str]:
    """Full Phase 2: within-page split (if needed) -> dynamic cross-page
    batching -> parallel batch merges -> final merge -> validate every
    chunk's tags against the canonical list."""

    # --- Step 1: within-page split for any page that alone overflows ---
    for page_num, result in page_results.items():
        all_page_tags = [t for c in result.chunks for t in c.tags]
        if len(all_page_tags) > within_page_threshold:
            sub_batches = split_by_tag_count(all_page_tags, within_page_threshold)
            with ThreadPoolExecutor(max_workers=len(sub_batches)) as executor:
                sub_merged = list(executor.map(lambda b: tag_merger.merge([b]), sub_batches))
            collapsed = tag_merger.merge(sub_merged)
            # Overwrite this page's effective tag pool with the deduped version
            # (used only for batching math below — per-chunk tags get re-validated at the end anyway)
            page_results[page_num]._collapsed_tags = collapsed  # stash for batching step

    # --- Step 2: dynamic cross-page batching ---
    page_batches = dynamic_batch_pages_by_tag_count(page_results, batch_threshold)

    def merge_one_batch(page_nums: list[int]) -> list[TagWithReason]:
        groups = []
        for pn in page_nums:
            result = page_results[pn]
            tags = getattr(result, "_collapsed_tags", None) or [t for c in result.chunks for t in c.tags]
            groups.append(tags)
        return tag_merger.merge(groups)

    with ThreadPoolExecutor(max_workers=max(1, len(page_batches))) as executor:
        batch_merged = list(executor.map(merge_one_batch, page_batches))

    # --- Step 3: final merge, only if >1 batch ---
    canonical = batch_merged[0] if len(batch_merged) == 1 else tag_merger.merge(batch_merged)
    canonical_tag_names = [t.tag for t in canonical]

    # --- Step 4: validate every chunk's tags against the canonical list ---
    for page_num, result in page_results.items():
        for chunk in result.chunks:
            validated = [t.tag for t in chunk.tags if t.tag in canonical_tag_names]
            chunk_store.set_final_tags(chunk.chunk_id, validated)

    return canonical_tag_names
```

---

**Part 4 of 4 — Phase 3 (supervisor, gated) + orchestrator**

```python
# ============================================================
# CELL (new, continued): Phase 3 — gated supervisor pass, and the
# top-level orchestrator tying Phases 1-3 together.
# ============================================================

class Supervisor:
    """Runs AFTER Phase 1, consuming page results in page order.
    Maintains a small running memory (condensed themes/tags — plain
    Python state, not a LangChain memory object). Only fires an LLM
    call when a cheap embedding-similarity check suggests genuine
    overlap with prior pages — most pages will skip the LLM call
    entirely."""

    def __init__(self, llm_client: BaseLLMClient, embedding_generator: "EmbeddingGenerator"):
        self.llm_client = llm_client
        self.embedding_generator = embedding_generator
        self.logger = logging.getLogger("rag_harness.supervisor")
        self.grammar = XGrammar(SupervisorOutput, name="supervisor_output")
        # running memory: condensed, NOT full page text
        self.memory_gists: list[str] = []   # one short line per page processed so far
        self.memory_embeddings: list[list[float]] = []

    def _page_gist(self, page_result: PageAgentOutput) -> str:
        tag_names = sorted({t.tag for c in page_result.chunks for t in c.tags})
        return f"page {page_result.page_num}: {', '.join(tag_names[:6])}"

    def _cheap_similarity(self, gist: str) -> float:
        if not self.memory_embeddings:
            return 0.0
        gist_emb = self.embedding_generator.model_registry.encode([gist])[0]
        import numpy as np
        sims = [
            float(np.dot(gist_emb, m) / (np.linalg.norm(gist_emb) * np.linalg.norm(m) + 1e-8))
            for m in self.memory_embeddings
        ]
        return max(sims)

    def process_page(self, page_result: PageAgentOutput, chunk_store: ChunkStore):
        gist = self._page_gist(page_result)
        similarity = self._cheap_similarity(gist)

        if similarity >= SUPERVISOR_SIMILARITY_THRESHOLD:
            prior_context = "\n".join(self.memory_gists)
            this_page_chunks = "\n".join(f"- {c.chunk_id}: {c.preamble}" for c in page_result.chunks)
            system_prompt = (
                "You check whether any chunk on the CURRENT page meaningfully relates to or builds "
                "on earlier pages (summarized below). If so, output ONE short sentence per such chunk "
                "to append to its preamble, naming the connection. If nothing genuinely relates, "
                "return an empty edits list — do not force a connection."
            )
            user_content = textwrap.dedent(f"""
                Earlier pages seen so far (condensed):
                {prior_context}

                Current page's chunks:
                {this_page_chunks}
            """).strip()
            try:
                response = self.llm_client.generate(
                    system_prompt, user_content,
                    response_format=self.grammar.build_response_format(),
                    max_tokens=500, temperature=0.0,
                )
                parsed = SupervisorOutput.model_validate_json(response)
                for edit in parsed.edits:
                    chunk_store.append_supervisor_sentence(edit.chunk_id, edit.relationship_sentence)
                self.logger.info(f"Supervisor: page {page_result.page_num} — {len(parsed.edits)} preamble edit(s)")
            except Exception as e:
                self.logger.error(f"Supervisor call failed on page {page_result.page_num}: {e}")
        else:
            self.logger.info(f"Supervisor: page {page_result.page_num} — similarity {similarity:.2f}, skipped LLM call")

        # update running memory (cheap, no LLM call needed for this part)
        gist_emb = self.embedding_generator.model_registry.encode([gist])[0]
        self.memory_gists.append(gist)
        self.memory_embeddings.append(gist_emb)


class PageLevelContextualChunker:
    """Top-level orchestrator: Phase 1 (parallel page agents) -> Phase 2
    (hierarchical tag merge) -> Phase 3 (sequential gated supervisor) ->
    final Chunk objects, same output shape your pipeline already expects."""

    def __init__(self, llm_client: BaseLLMClient, embedding_generator: "EmbeddingGenerator", settings: Settings):
        self.page_agent = PageAgent(llm_client)
        self.tag_merger = TagMerger(llm_client)
        self.supervisor = Supervisor(llm_client, embedding_generator)
        self.settings = settings
        self.logger = logging.getLogger("rag_harness.page_level_chunker")

    def chunk_document(self, doc_id: str, doc_version: str, source_type: str, page_texts: list[dict]) -> list[Chunk]:
        chunk_store = ChunkStore()

        # Build page structure with deterministic chunk boundaries (unchanged chunker)
        pages = []
        for page_entry in page_texts:
            base_chunks = structure_aware_chunker.split(page_entry["text"])
            chunk_tuples = [
                (f"{doc_id}_p{page_entry.get('page_num')}_c{i}", text)
                for i, text in enumerate(base_chunks)
            ]
            pages.append({"page_num": page_entry.get("page_num"), "text": page_entry["text"], "chunks": chunk_tuples})

        # Phase 1
        page_results = run_page_agents_parallel(self.page_agent, pages, chunk_store)

        # Phase 2
        canonical_tags = reconcile_tags(page_results, self.tag_merger, chunk_store)
        self.logger.info(f"doc={doc_id}: canonical tags = {canonical_tags}")

        # Phase 3 — sequential, page order
        for page_num in sorted(page_results.keys()):
            self.supervisor.process_page(page_results[page_num], chunk_store)

        # Build final Chunk objects
        result_chunks = []
        for chunk_id, row in chunk_store.all_rows().items():
            contextual_text = " ".join([row["preamble_base"]] + row["preamble_additions"]).strip()
            contextual_text = f"{contextual_text}\n{row['raw_text']}" if contextual_text else row["raw_text"]
            result_chunks.append(Chunk(
                raw_text=row["raw_text"],
                contextual_text=contextual_text,
                metadata=ChunkMetadata(
                    doc_id=doc_id, doc_version=doc_version, source_type=source_type,
                    page_num=row["page"], section_title=None,
                    topic_tag=(row["tags"][0] if row["tags"] else None),
                    ingestion_route=None,
                ),
            ))

        self.logger.info(f"doc={doc_id}: built {len(result_chunks)} chunks via page-level pipeline")
        return result_chunks


# Instantiate alongside your existing contextual_chunker (do NOT remove the old one yet —
# test this one in parallel first, per our earlier plan)
page_level_chunker = PageLevelContextualChunker(groq_fallback_client, embedding_generator, settings)
```



```

```markdown
Here's a plain walkthrough of every function/class I gave you, in the order they run, what each one actually does, and how they call each other.

## Part 1 — Data models (no logic, just shapes)

| Name | What it is |
|---|---|
| `TagWithReason` | One tag + the one-sentence reason for it. Used everywhere tags get passed around. |
| `ChunkPreambleAndTags` | One chunk's result from a page agent: its `chunk_id`, `preamble`, and list of `TagWithReason`. |
| `PageAgentOutput` | The full return value of one Phase-1 LLM call: `page_num` + list of `ChunkPreambleAndTags` (one per chunk on that page). |
| `TagMergeOutput` | The return value of any merge call — just a deduped `list[TagWithReason]`. |
| `SupervisorEdit` | One supervisor edit: `chunk_id` + the relationship sentence to append. |
| `SupervisorOutput` | The full return value of one Phase-3 LLM call — a list of `SupervisorEdit` (can be empty). |

These don't *do* anything — they're just the strict schemas `XGrammar` validates the LLM's JSON output against.

## Part 2 — Storage + Phase 1 (page agents)

**`ChunkStore`** — the shared "spreadsheet" we talked about. One row per chunk.
- `write_base(...)` — called once per chunk, right when its page agent finishes. Fills in `raw_text`, `preamble_base`, initial `tags`.
- `append_supervisor_sentence(...)` — called later by the supervisor, only adds to `preamble_additions`, never touches `preamble_base`.
- `set_final_tags(...)` — called after Phase 2, overwrites a chunk's `tags` with the *validated* (canonical-list-checked) version.
- `get(...)` / `all_rows()` — just reads.
- Has a `threading.Lock()` because multiple page agents write to it concurrently — without the lock, two threads writing at once could corrupt the dict.

**`PageAgent.run(page_num, page_text, chunk_ids_and_texts)`** — this is the actual **Phase-1 LLM call**.
- Builds one prompt containing the full page text + the list of known chunk IDs (chunk boundaries already computed beforehand by your existing `StructureAwareChunker` — the LLM doesn't decide where chunks start/end, it just writes preamble+tags for chunks it's told about).
- Sends one `llm_client.generate(...)` call with `response_format` forcing it to return `PageAgentOutput` JSON.
- **Validates the response**: checks every `chunk_id` we asked about actually came back. If the LLM dropped one, it logs a warning and fills that chunk in with an empty preamble rather than silently losing it.
- On total failure (exception), returns empty preambles for every chunk on that page rather than crashing the whole pipeline.

**`run_page_agents_parallel(page_agent, pages, chunk_store, max_workers)`** — this is the **worker pool**, functionally identical to `ThreadPoolExecutor` usage already in your `ContextualChunker.chunk_document`.
- Submits one `page_agent.run(...)` call per page into the pool (4 slots by default).
- As each page finishes (`as_completed`), it immediately writes that page's chunks into `chunk_store` via `write_base(...)` — this is why Phase 1 doesn't wait for all pages before storing results.
- Returns `page_results` — a dict of `{page_num: PageAgentOutput}`, used by Phase 2 and Phase 3 next.

## Part 3 — Phase 2 (hierarchical tag merging)

**`TagMerger.merge(tag_groups)`** — the actual **merge LLM call**, reused for every merge level (within-page, batch, final).
- Flattens however many tag lists you pass it into one list.
- If there's only 0 or 1 tags total, skips the LLM call entirely (nothing to merge).
- Otherwise sends one call asking the LLM to dedupe near-identical tags into canonical ones.
- On failure, falls back to a naive Python-only dedup (exact string match) rather than losing all tags.

**`split_by_tag_count(tags, limit)`** — pure Python, no LLM. Chops one tag list into smaller lists, each capped at `limit` tags. Used only when a single page overflows `WITHIN_PAGE_SPLIT_THRESHOLD`.

**`dynamic_batch_pages_by_tag_count(page_results, threshold)`** — pure Python, no LLM. Walks pages in order, grouping them into batches so each batch's **combined** tag count stays under `threshold`. This is the exact algorithm from our worked example (page1+page2 together, page3+page4 together, etc., depending on running totals).

**`reconcile_tags(page_results, tag_merger, chunk_store, ...)`** — the **Phase-2 orchestrator**, calls everything above in order:
1. For any page whose tag count alone exceeds `WITHIN_PAGE_SPLIT_THRESHOLD`: splits it, merges the pieces in parallel, merges those results together — collapses that one page down to a small deduped list.
2. Batches all pages by running tag count (`dynamic_batch_pages_by_tag_count`).
3. Merges each batch in parallel (`tag_merger.merge`, via `ThreadPoolExecutor`).
4. If more than one batch exists, merges the batch results together one final time → `canonical` tag list.
5. Goes back through every chunk in `page_results` and keeps only tags that survived into the canonical list, writing the validated result into `chunk_store` via `set_final_tags`.
6. Returns the canonical tag name list.

## Part 4 — Phase 3 (supervisor) + the orchestrator that ties it all together

**`Supervisor`** class — holds the running memory as plain instance attributes (`self.memory_gists`, `self.memory_embeddings`) — this is the "plain Python state" memory decision from earlier, no LangChain memory object.
- `_page_gist(page_result)` — pure Python, builds a short one-line summary of a page (its tag names) — this is what gets compared against, not full page text.
- `_cheap_similarity(gist)` — **no LLM call**. Embeds the new gist (via your existing `embedding_generator`), compares it against every gist already seen using cosine similarity, returns the highest score. This is the free pre-filter from our design.
- `process_page(page_result, chunk_store)` — the **Phase-3 per-page step**, called once per page, in page order:
  - Computes the gist + similarity.
  - **Only if similarity ≥ `SUPERVISOR_SIMILARITY_THRESHOLD`**, fires an actual LLM call asking "does anything on this page relate to earlier pages?" — this is the gating that avoids paying for a call on every page.
  - Writes any returned relationship sentences into `chunk_store` via `append_supervisor_sentence`.
  - Regardless of whether the LLM fired, updates `self.memory_gists`/`self.memory_embeddings` with this page's gist, so the *next* page's similarity check has something to compare against.

**`PageLevelContextualChunker`** — the **top-level class you actually call**, same role as your existing `ContextualChunker`.
- `chunk_document(doc_id, doc_version, source_type, page_texts)`:
  1. Builds each page's chunk boundaries using your existing `structure_aware_chunker.split(...)` (unchanged from your current pipeline) and assigns deterministic `chunk_id`s.
  2. Calls `run_page_agents_parallel(...)` → Phase 1.
  3. Calls `reconcile_tags(...)` → Phase 2.
  4. Loops through pages in order calling `self.supervisor.process_page(...)` → Phase 3.
  5. Reads everything back out of `chunk_store.all_rows()`, joins `preamble_base` + `preamble_additions` + `raw_text` into `contextual_text`, and builds actual `Chunk` objects — the same output shape your `VectorStore.upsert_chunks_batch` already expects.

**Last line**: `page_level_chunker = PageLevelContextualChunker(...)` — just instantiates it, doesn't run anything yet. You'd call `page_level_chunker.chunk_document(...)` yourself, same as you already do with `contextual_chunker`.

Want me to trace through what happens step-by-step for one concrete page (like we did earlier with the 5-page tag example) so you can see the actual data flowing through these functions instead of just their descriptions?
```

