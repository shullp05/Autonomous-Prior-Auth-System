# Runbook

## Local setup (WSL2/conda)
```bash
# Optional: create env
conda create -n priorauth python=3.11
conda activate priorauth

python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```
Input: [requirements.lock](../requirements.lock).

## Build policy artifacts and RAG index
```bash
python -m priorauth.apps.agent.chaos_monkey     # synthetic dataset + scenario manifest
python -m priorauth.apps.agent.setup_rag         # builds chroma_db policy index
```
Modules: [src/priorauth/apps/agent/chaos_monkey.py](../src/priorauth/apps/agent/chaos_monkey.py), [src/priorauth/apps/agent/setup_rag.py](../src/priorauth/apps/agent/setup_rag.py).

## Run the agent
```bash
python -m priorauth.apps.agent.batch_runner
```
Module: [src/priorauth/apps/agent/batch_runner.py](../src/priorauth/apps/agent/batch_runner.py).

## Letter modes
```bash
# Default: deterministic letters (no LLM calls)
PA_LETTER_MODE=deterministic python -m priorauth.apps.agent.batch_runner

# Ollama letters (requires local Ollama + model)
PA_LETTER_MODE=ollama python -m priorauth.apps.agent.batch_runner

# Explicit fallback if Ollama is unavailable
PA_ALLOW_LETTER_FALLBACK=1 PA_LETTER_MODE=ollama python -m priorauth.apps.agent.batch_runner
```
Module: [src/priorauth/apps/agent/batch_runner.py](../src/priorauth/apps/agent/batch_runner.py).

## Offline mode (localhost-safe)
```bash
export PA_OFFLINE_MODE=true
python -m priorauth.apps.agent.batch_runner
```
Module: [src/priorauth/apps/agent/batch_runner.py](../src/priorauth/apps/agent/batch_runner.py).

## Offline scripts
```bash
# Deterministic (no sockets allowed)
scripts/run_offline_deterministic.sh

# Ollama (loopback allowed, external blocked)
scripts/run_offline_ollama.sh
```
Scripts: [scripts/run_offline_deterministic.sh](../scripts/run_offline_deterministic.sh), [scripts/run_offline_ollama.sh](../scripts/run_offline_ollama.sh).

## Tests
```bash
python -m compileall -q .
pytest -q
pytest -q tests/test_offline_mode.py
pytest -q tests/test_letter_mode_offline.py
```
Tests: [tests/test_offline_mode.py](../tests/test_offline_mode.py), [tests/test_letter_mode_offline.py](../tests/test_letter_mode_offline.py).

## Benchmark
```bash
python -m priorauth.apps.agent.benchmark --deterministic-only
# To include LLM runs (requires Ollama models):
# python -m priorauth.apps.agent.benchmark --flavor nemo8b
```
Module: [src/priorauth/apps/agent/benchmark.py](../src/priorauth/apps/agent/benchmark.py). Evidence: [reports/benchmark.txt](../reports/benchmark.txt).
Note: current-year filtering uses the reference date/year if set. For
year-stable runs, set `PA_REFERENCE_DATE` (ISO) or `PA_REFERENCE_YEAR`.

## Docker (optional)
Requires Docker Desktop WSL integration.
```bash
# CPU-only default
docker build -f docker/Dockerfile --build-arg REQUIREMENTS_FILE=requirements-docker-cpu.txt -t priorauth:local .
docker run --rm priorauth:local pytest -q

# CUDA build (downloads large NVIDIA CUDA wheels)
docker build -f docker/Dockerfile --build-arg REQUIREMENTS_FILE=requirements.txt -t priorauth:cuda .
docker run --rm priorauth:cuda pytest -q

# Black-box offline (no network)
docker compose -f docker/docker-compose.blackbox.yml up --build
```
Files: [docker/Dockerfile](../docker/Dockerfile), [requirements-docker-cpu.txt](../requirements-docker-cpu.txt), [requirements.txt](../requirements.txt), [docker/docker-compose.blackbox.yml](../docker/docker-compose.blackbox.yml).

## CI (GitHub Actions)
Workflows run the same local truth:
```bash
python -m pip install -r requirements.lock
python -m pip check
python -m compileall -q .
pytest -q
```
Input: [requirements.lock](../requirements.lock).

## Key environment variables
- `PA_AUDIT_MODEL_FLAVOR` (default `nemo8b`)
- `PA_USE_RAW_MODELS` (use upstream model names)
- `PA_EMBED_MODEL` (embeddings model)
- `PA_ENABLE_RERANK` / `PA_RERANK_MODEL` / `PA_RERANK_DEVICE`
- `PA_OFFLINE_MODE` (opt-in outbound block)
- `PA_LETTER_MODE` (`deterministic` or `ollama`)
- `PA_ALLOW_LETTER_FALLBACK` (explicit fallback opt-in)
- `PA_REFERENCE_DATE` (ISO date used for current-year filters)
- `PA_REFERENCE_YEAR` (explicit year override for current-year filters)
