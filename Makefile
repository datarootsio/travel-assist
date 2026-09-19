.DEFAULT_GOAL := help
.PHONY: help install lint fmt test test-db test-all eval eval-prompts doctor mlflow-ui corpus-build corpus skeleton test-skeleton smoke

# Targets that depend on code not yet written fail loudly and name the step that
# will build them, rather than succeeding vacuously.
define not_yet
	@echo "not built yet — $(1)."; exit 1
endef

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the pinned environment
	uv sync --all-groups

lint:  ## ruff + mypy --strict
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:  ## Autoformat and autofix
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run the test suite (live and db tests excluded)
	uv run pytest

test-db:  ## Run @pytest.mark.db tests once, to confirm your local Postgres connection
	uv run pytest -m db

test-all:  ## Include @pytest.mark.live and @pytest.mark.db tests — hits real services
	uv run pytest -m ""

eval:  ## Run the retrieval evaluation — ten queries, recall@5, one table
	@echo "instructor-only tooling, not shipped in the skeleton" && exit 1

eval-prompts:  ## Run the prompt comparison — one table, one number: fraction of claims with a source
	@echo "instructor-only tooling, not shipped in the skeleton" && exit 1

doctor:  ## Diagnose the local environment
	uv run travel-assist doctor

mlflow-ui:  ## Open the local trace dashboard (reads ./mlruns, no server to provision)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# There is no local database and no Docker. The index is one shared Azure
# Postgres, built once by an instructor with `make corpus`; everyone else points
# PGHOST at it and reads. The database-backed tests need *a* writable Postgres —
# any one you can create a database on, never the shared index — and skip
# cleanly when they cannot reach one. That is 38 of the 131 tests, so if you are
# working on retrieval, bring your own Postgres or you are testing nothing.

corpus-build:  ## INSTRUCTOR, RUN ONCE EVER — Wikivoyage dump -> data/corpus.jsonl
	@echo "instructor-only — not part of the exercise" && exit 1

corpus:  ## INSTRUCTOR — data/corpus.jsonl -> chunk -> embed -> load the shared Postgres
	uv run python -m ingestion.run

SKELETON_DIR ?= ../travel-assist-skeleton

skeleton:  ## Generate the student skeleton from the solution
	@echo "instructor-only — you're already looking at the skeleton" && exit 1

test-skeleton: skeleton  ## Prove the skeleton fails exactly at unimplemented milestones
	@echo "instructor-only — you're already looking at the skeleton" && exit 1

smoke:  ## Clean-clone smoke test — clone, install, doctor, search the shared index
	$(call not_yet,the smoke test)
