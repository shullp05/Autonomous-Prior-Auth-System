# Reproducibility

This project favors deterministic policy logic with evidence-first artifacts. Use the steps below to regenerate core outputs.

## Dependency lock
```bash
bash scripts/freeze_dependencies.sh
```
Generates `requirements.lock` from the active environment.

## Policy snapshot (source of truth)
```bash
python -m priorauth.policy_snapshot
```
Writes the policy snapshot (default `policies/RX-WEG-2025.json`) from `UpdatedPAGuidelines.txt`.

## RAG index
```bash
python -m priorauth.apps.agent.setup_rag
```
Builds `chroma_db/` and validates embedding dimensions.

## Synthetic dataset + scenario manifest
```bash
python -m priorauth.apps.agent.chaos_monkey
```
Generates `output/` CSVs and `output/scenario_manifest.json`.

## Batch run outputs
```bash
python -m priorauth.apps.agent.batch_runner
```
Writes dashboard outputs and audit artifacts to `output/`.

## Governance report
```bash
python -m priorauth.governance_audit
```
Writes `output/governance_report.json`.

## Benchmark
```bash
python -m priorauth.apps.agent.benchmark --deterministic-only
# To include LLM runs (requires Ollama models):
# python -m priorauth.apps.agent.benchmark --flavor nemo8b
```

## Tests
```bash
python -m compileall -q .
pytest -q
```

## Key environment variables
- `PA_AUDIT_MODEL_FLAVOR`, `PA_USE_RAW_MODELS`
- `PA_EMBED_MODEL`, `PA_RAG_K_VECTOR`, `PA_RAG_TOP_K_DOCS`, `PA_RAG_SCORE_FLOOR`, `PA_RAG_MIN_DOCS`
- `PA_ENABLE_RERANK`, `PA_RERANK_MODEL`, `PA_RERANK_DEVICE`
- `PA_OFFLINE_MODE`
