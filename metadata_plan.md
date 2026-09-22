```metadata
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


```