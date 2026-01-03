# Reproducibility

This project favors deterministic policy logic with evidence-first artifacts. Use the steps below to regenerate core outputs.

## Dependency lock
```bash
bash scripts/freeze_dependencies.sh
```
Generates `requirements.lock` from the active environment.

## Policy snapshot (source of truth)
```bash
python policy_snapshot.py
```
Writes the policy snapshot (default `policies/RX-WEG-2025.json`) from `UpdatedPAGuidelines.txt`.

## RAG index
```bash
python setup_rag.py
```
Builds `chroma_db/` and validates embedding dimensions.

## Synthetic dataset + scenario manifest
```bash
python chaos_monkey.py
```
Generates `output/` CSVs and `output/scenario_manifest.json`.

## Batch run outputs
```bash
python batch_runner.py
```
Writes dashboard outputs and audit artifacts to `output/`.

## Governance report
```bash
python governance_audit.py
```
Writes `output/governance_report.json`.

## Benchmark
```bash
python benchmark.py --deterministic-only
# To include LLM runs (requires Ollama models):
# python benchmark.py --flavor nemo8b
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
