# Ingestion — you will not run this

Participants never run this — not a convenience, a curriculum choice.
Nineteen people each downloading a 125 MB dump and embedding tens of
thousands of passages would burn a morning and real money to produce
nineteen identical indexes. Instead, an instructor builds the index **once,
weeks before the course**, and hands it out read-only.

Read this code and argue with the decisions in it — that's the point — then
go use the index. It's also the realistic case: most people who work on a
RAG system inherit an index someone else already built.

## What the original corpus was 
The input is the **English Wikivoyage dump** — a bzip2-compressed XML export of
every article as raw wikitext, published by the Wikimedia Foundation. It lives at
`data/dumps/enwikivoyage-latest-pages-articles.xml.bz2` (125 MB compressed,
several GB expanded, gitignored), which is why `extract.py` streams it rather
than loading it.

## How the cleaned corpus gets built

```
Wikivoyage dump (bz2, ~125 MB)
        │
        ▼
scripts/corpus_build/     RUN ONCE EVER
  resolve the 10 named destinations
  clean wikitext -> prose, split into sections
        │
        ▼
data/corpus.jsonl          committed, ~3-6 MB
  one record per article — no chunks, no vectors
        │
        ▼
ingestion/                run whenever chunk size or embedding model changes
  chunk (pack sections to a token budget)
  embed (batched, cached, retried)
  load
        │
        ▼
Postgres: documents, chunks, index_metadata
```

`scripts/corpus_build/` touches the dump exactly once. `ingestion/` reads
`data/corpus.jsonl` and never looks at the dump again.

**Why split here:** chunk size and embedding model aren't properties of the
*articles*, only of how they get indexed. Freezing either into
`corpus.jsonl` would mean re-tuning chunk size needs the dump again, or pins
the corpus to one embedding model — the exact silent-corruption risk the
frozen-embedding guard below exists to catch. Both stay free parameters,
decided at *seed time* by `ingestion/run.py`.

| File | Does |
|---|---|
| `scripts/corpus_build/extract.py` | Parses the dump: status, type, hierarchy, coordinates; strips wikitext to prose; splits into sections |
| `scripts/corpus_build/select_corpus.py` | Resolves the named destinations; flags a hollow parent selected without its districts |
| `scripts/corpus_build/build_corpus.py` | Ties the two above together, writes `data/corpus.jsonl` |
| `scripts/corpus_build/demo_destinations.txt` | The decided destination list |
| `ingestion/corpus.py` | Reads/writes `data/corpus.jsonl`; writes `data/corpus_manifest.csv` |
| `ingestion/chunk.py` | Packs an article's (already-clean) sections into a token budget |
| `ingestion/embed_and_load.py` | Embeds in batches, loads Postgres, records the embedding contract |
| `ingestion/run.py` | The everyday entrypoint: `corpus.jsonl` → chunk → embed → load |
| `ingestion/sql/` | Schema, indexes, the read-only role participants connect as |

## Five decisions worth arguing about

1. **Only `star`, `guide` and `usable` articles.** Thinner statuses
   (`outline`, `stub`) match query terms fine and then answer nothing —
   worse than not being in the index at all.

2. **A fixed, named list, not a stratified sample.** Every one of the ten
   destinations is a place a guide, demo or golden-set question actually
   names — nothing left for a selector to *decide*, only to *resolve*.

3. **Section structure first, then pack to 400 tokens with 64 overlap.**
   `See`, `Eat`, `Sleep` are boundaries a traveller already thinks in, so a
   chunk rarely straddles two subjects. 400 tokens ≈ one idea plus its
   immediate context. Overlap means a sentence spanning a section boundary
   survives whole on at least one side. Changing either number invalidates
   every cached vector — the cache is keyed on the exact text embedded.

4. **The section path is embedded, but not stored.** The vector is built
   from `"Barcelona > See > Museums"` plus the text, so it encodes *where*
   and *what*, not just the sentences. The stored `content` stays clean — an
   answer should quote the prose, not a breadcrumb.

5. **Listing content is kept, and "hollow" parents are caught — but not
   auto-fixed.** A listing like `{{see|name=Sagrada Família|price=€26}}` is
   flattened to its values, not dropped — proper nouns and prices are
   exactly what keyword search is for. Separately: Wikivoyage "hollows out"
   its big cities — past a point, venue listings move into district
   sub-articles, each linked back to its parent with `{{IsPartOf|Lisbon}}`
   (the same template continent/country resolution walks), and the parent
   keeps only prose. `{{regionlist}}` is the tell that a page is hollow.
   **The actual fix is manual:** a hollow parent's districts have to be
   listed explicitly in `demo_destinations.txt`, right alongside it — that's
   why Lisbon is 7 lines, not 1. `build_corpus.py`'s check doesn't add
   anything; it only warns if a selected hollow parent's districts were
   never listed.

## The frozen-embedding contract

Every load writes `index_metadata` (embedding model, vector width, dump
date, chunk strategy, corpus size). The app reads it at startup and
**refuses to run** if its own configuration disagrees.

Why: querying an index with a different embedding model doesn't raise an
error — it returns confident, wrong results from a different vector space. A
crash at startup is far cheaper than a day spent wondering why retrieval
"feels off." Trigger it once on purpose: set `EMBEDDING_DIM` to something
else and read the error.

> `EMBEDDING_DIM` caps at **2000** — pgvector won't build an HNSW index (the
> vector index type) above that. `text-embedding-3-large` is natively 3072,
> truncated to 1536 via the API's `dimensions` parameter.

## Running it (instructors)

```bash
# 1. Get the dump (~125 MB). Pin the date — it lands in corpus.jsonl and the manifest.
mkdir -p data/dumps
curl -L -o data/dumps/enwikivoyage-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/enwikivoyage/latest/enwikivoyage-latest-pages-articles.xml.bz2

# 2. Build data/corpus.jsonl. RUN ONCE EVER — see the diagram above.
make corpus-build

# 3. The shared Azure Postgres, once. See docs/provision_azure_postgres.md.
#    Then point PGHOST/PGUSER/PGPASSWORD at it, PGSSLMODE=require.

# 4. Chunk, embed, load. Safe to re-run any time chunk size or the
#    embedding model changes — never touches the dump.
make corpus

# 5. Confirm
psql "$PG_URL" -c "SELECT count(*) FROM chunks;"
uv run travel-assist doctor
```

Then commit the regenerated `data/corpus_manifest.csv`.

**Embedding is the long pole** — it dominates the run time and scales with
corpus size, not article count. Book time for it before the teaching day.

## Licence

The corpus is a subset of English [Wikivoyage](https://en.wikivoyage.org),
**CC BY-SA** (3.0 / 4.0 dual-licensed). Every chunk carries its source URL
and every answer cites it — attribution is part of the product, not a
footnote. `data/corpus_manifest.csv` records the exact articles, revision
ids and dump date.

## Definitions:
   - *hollow: an article that points at content the corpus doesn't contain.*
   *So it's only a problem when its districts are left out*
   - *Districts are parts of the city*
   - *A listing is a single venue inside an article's wikitext (i.e. hotel, restaurants, ...)*
