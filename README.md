# travel-assist

A multi-turn RAG assistant over a Wikivoyage corpus, answering destination
questions with grounded, cited answers inside a managed context budget. A
deterministic pipeline: *we* wrote the order of the steps.

RootsAcademy 2026 · Dataroots · GenAI day, 24 September 2026.

Day 2 (Agentic AI) takes the same steps and lets the model decide their order
instead. It continues in its own repo, cloned from this one — nothing here
changes for that.

---

## Status

**This is a skeleton.** Every exercise under `src/` raises `NotImplementedError`
with a docstring describing the contract. Work through the **Milestones** table
below, top to bottom — `make test`'s own failure order is alphabetical by
filename, not exercise order, so don't chase it. Fell behind? `git checkout
checkpoint/d1-<name>` (`search`, `seam`, `grounded`) drops you in at a working
point.

## Quick start

```bash
make install   # sync the pinned environment (Python 3.12 via uv)
make lint      # ruff + mypy --strict
make test      # pytest
make help      # every target
```

## Repo map

```
src/travel_assist/
  retrieval/   dense + sparse search over the corpus, fused with RRF
  pipeline/    the deterministic chain: budget, citations, routing, memory
  cli/         chat, search, doctor — ships implemented, nothing to do here
  config.py, db.py, models.py   scaffolding — ships implemented, read and run against it

ingestion/     chunk → embed → load — the pipeline that built the index you query
tests/         one file per exercise below — your definition of done
data/          the corpus and its manifest, already built
```

Every exercise lives under `src/travel_assist/{retrieval,pipeline}/` as a
`NotImplementedError` with a contract docstring. Everything else in `src/` is
scaffolding: read it, run against it, don't write it.

## Milestones

Three milestones, worked top to bottom — one AM block, two PM blocks. Each
step's test file is green once it's done, and later steps depend on earlier
ones. A ✅ step has a checkpoint tag right after it: `git checkout
checkpoint/d1-<name>` drops you in at that point if you fall behind.

### 1. `search` — AM

| Step | Where | Tests | Build |
|---|---|---|---|
| vector-search | `retrieval/search.py` | `test_search.py` | Embed the query, rank chunks by cosine similarity. |
| keyword-search ✅ `d1-search` | `retrieval/search.py` | `test_search.py` | Rank chunks by Postgres full-text search. |
| rrf | `retrieval/search.py` | `test_search.py` | Fuse two rankings by reciprocal rank, not by score. |
| hybrid-search ✅ `d1-seam` | `retrieval/search.py` | `test_search.py` | Call both searches and fuse them — travel-assist's frozen retrieval entrypoint. |

### 2. `grounded-chat` — PM

| Step | Where | Tests | Build |
|---|---|---|---|
| context-budget | `pipeline/context.py` | `test_context.py` | Fit ranked chunks into a token budget; log what gets dropped and why. |
| structured-answer | `pipeline/citations.py`, `pipeline/steps.py` | `test_citations.py`, `test_steps.py` | Model returns cited claims, not free text; catch a claim citing a chunk it was never shown. |
| history | `pipeline/memory.py` | `test_memory.py` | Persist multi-turn conversation, threaded through the pipeline. |
| compaction ✅ `d1-grounded` | `pipeline/memory.py` | `test_memory.py` | Shrink an over-budget transcript: summarise, or drop the oldest. |

### 3. `routing-self-check` — PM

| Step | Where | Tests | Build |
|---|---|---|---|
| routing | `pipeline/routing.py`, `pipeline/chain.py` | `test_routing.py`, `test_chain.py` | Classify in/out-of-scope before retrieving; assemble the whole pipeline. |
| self-check | `pipeline/citations.py` | `test_citations.py` | Catch a claim with no citation at all. |

## Licence and attribution

The corpus is a subset of English [Wikivoyage](https://en.wikivoyage.org),
licensed **CC BY-SA** (dual-licensed 3.0 / 4.0). Every retrieved chunk carries
its source URL and every answer cites it — attribution is built into the
product. The exact articles included, with revision IDs and the dump date, are
published in `data/corpus_manifest.csv`.
