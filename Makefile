# Makefile for PriorAuth Agent

.PHONY: test test-all benchmark batch-run clean help

PYTHON := python
PYTEST := pytest

help:
	@echo "PriorAuth Agent Commands:"
	@echo "  make test          - Run deterministic tests (fast, CI-friendly)"
	@echo "  make test-all      - Run all tests including LLM integration (requires Ollama)"
	@echo "  make benchmark     - Run performance benchmark (Deterministic vs LLM)"
	@echo "  make batch-run     - Run full batch processing in LLM mode"
	@echo "  make batch-fast    - Run full batch processing in Deterministic mode"
	@echo "  make clean         - Remove cache and temporary files"
	@echo "  make list-files    - List repo files up to configured depth (excludes hidden)"

test:
	$(PYTEST) tests/test_guardrails.py tests/test_adversarial.py tests/test_json_extraction.py -v

test-all:
	$(PYTEST) tests/ -v

benchmark:
	$(PYTHON) benchmark.py

batch-run:
	$(PYTHON) batch_runner.py

batch-fast:
	PA_USE_DETERMINISTIC=true $(PYTHON) batch_runner.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f .coverage

list-files:
	@python -c 'import os; from config import MAX_SEARCH_DEPTH; print(f"Listing files (depth={MAX_SEARCH_DEPTH}, exclude hidden)...")'
	@find . -maxdepth 5 -not -path '*/.*'
