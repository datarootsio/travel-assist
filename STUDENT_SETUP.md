# Getting connected

Two things to set up before you write any code: your `.env`, and (optional,
for poking around) a direct connection to the shared database.

---

## 1. Set up `.env`

```bash
cp .env.example .env
```

`.env.example` has four sections. You only touch two of them:

| Section | What you fill in | Where it comes from |
|---|---|---|
| **Model** | `AZURE_OPENAI_API_KEY` | shared key, given out in class |
| **Index** | `PGUSER`, `PGPASSWORD` | **your own row** from the credential handout |
| Observability | nothing | leave as-is |
| Context management | nothing | leave as-is |

Everything else in the file (`AZURE_OPENAI_ENDPOINT`, `PGHOST`,
`EMBEDDING_DIM`, …) is already correct — don't change it.

**Verify:**

```bash
uv run travel-assist doctor
```

All rows should read `PASS` (`tracing: WARN` is fine, not fatal). Anything
else names exactly which value is wrong.

---

## 2. Connect to the database directly (optional)

You don't need this to do the exercises — `travel-assist` handles the
database for you. It's here for when you want to poke at the corpus
yourself.

**Easiest way** — your `.env` already uses Postgres's own variable names
(`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGSSLMODE`), so
loading it into your shell is enough for `psql` to pick them up with zero
arguments:

```bash
set -a
source .env
set +a

psql
```

**Or**, without sourcing anything, build the connection string by hand from
your handout row:

```bash
psql "postgresql://<your PGUSER>:<your PGPASSWORD>@rootsacademy-travel-assist.postgres.database.azure.com:5432/travel_assist?sslmode=require"
```

(Substitute your actual username/password — don't type the `<...>`.)

**Prefer a GUI client** (TablePlus, DBeaver, pgAdmin, whatever you already
use)? Same five values from your `.env`, into a new connection — you know
that interface already, nothing special here.

---

## 3. A few queries to try

```sql
SELECT * FROM documents LIMIT 5;
SELECT * FROM chunks LIMIT 5;
SELECT count(*) FROM chunks;
```

Your role is **read-only** on `documents`, `chunks`, and `index_metadata` —
`INSERT`/`UPDATE`/`DELETE` will fail with `permission denied`. That's
expected, not a bug: this is a shared index and nobody can break it for
anyone else.

`\q` to quit `psql`. From here, explore however you're used to.

---

## 4. (extra — once your implementation works) Look at your own traces

Not required for the exercises. Worth a look once `chat` or `search` actually
runs, to see what your pipeline did under the hood.

```bash
make mlflow-ui
# → http://127.0.0.1:5000
```

macOS: use `127.0.0.1:5000`, not `localhost:5000` — AirPlay Receiver listens
on the same port and can silently hijack it.

This is your own machine's local trace store — only your runs show up.
Worth checking out once it's open:

| Where | Why worth a look |
|---|---|
| **Traces** list | your run history |
| a trace's span tree | call order |
| a span's input/output | exact prompt sent |
| token counts per span | cost sanity |

Have fun exploring! :) 
