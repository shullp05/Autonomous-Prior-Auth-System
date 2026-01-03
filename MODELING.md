# Modeling

## Scope
The deterministic policy engine is the system of record for eligibility decisions. LLMs are used for narrative/audit output only, using retrieved policy evidence from the local RAG store when available.

## Embeddings + RAG configuration
- Embedding model default: `kronos483/MedEmbed-large-v0.1:latest` (override with `PA_EMBED_MODEL`).
- Embedding dimension guard: `setup_rag.py` enforces 1024-dim embeddings and records the dimension in `chroma_db/embedding_dim.txt`.
- Chunking: not used. RAG documents are policy “atoms” constructed from structured policy sections in `setup_rag.py`.
- Vector search: `Chroma.similarity_search` with `k=PA_RAG_K_VECTOR` (default 25) and `policy_id` filter.
- LLM context cap: `PA_RAG_TOP_K_DOCS` (default 8) after optional rerank/score filtering.
- Score filtering: `PA_RAG_SCORE_FLOOR` (default 0.35) and `PA_RAG_MIN_DOCS` (default 3).
- Reranker: BCE reranker (`PA_ENABLE_RERANK=true` by default) with model `PA_RERANK_MODEL` (default `maidalun1020/bce-reranker-base_v1`) and device `PA_RERANK_DEVICE` (default `cuda`).

## LLM model inventory (custom Modelfiles)
Defaults come from `config.py` and the Modelfiles in `models/`.

- `mistral` → `pa-audit-mistral`
  - Base: `mistral-nemo:latest`
  - Params: `num_ctx=4096`, `num_predict=512`, `temperature=0.2`, `top_p=0.9`, `repeat_penalty=1.1`
- `qwen25` → `pa-audit-qwen25`
  - Base: `qwen2.5:14b-instruct-q4_K_M`
  - Params: `num_ctx=4096`, `num_predict=768`, `temperature=0.3`, `top_p=0.9`, `repeat_penalty=1.1`
- `qwen3` → `pa-audit-qwen3`
  - Base: `qwen3:14b`
  - Params: `num_ctx=4096`, `num_predict=512`, `temperature=0.25`, `top_p=0.95`, `repeat_penalty=1.1`
- `nemo8b` → `pa-audit-nemotron-cascade8b:latest`
  - Base: `hf.co/bartowski/nvidia_Nemotron-Cascade-8B-GGUF:Q6_K`
  - Params: `num_ctx=4096`, `num_predict=768`, `temperature=0.2`, `top_p=0.9`, `top_k=20`, `repeat_penalty=1.1`, `repeat_last_n=256`, `seed=42`
- `nemo4b` → `pa-audit-nemo-cascade4b`
  - Base: `hf.co/bartowski/nvidia_Nemotron-Cascade-8B-GGUF:Q4_K_M`
  - Params: `num_ctx=4096`, `num_predict=768`, `temperature=0.2`, `top_p=0.9`, `top_k=20`, `repeat_penalty=1.1`, `repeat_last_n=256`, `seed=42`

If `PA_USE_RAW_MODELS=true`, the raw upstream models in `config.py` are used instead of the custom Modelfiles.

## Scenario manifest fields
`output/scenario_manifest.json` (also `tests/fixtures/baselines/v0_8/scenario_manifest.json`) includes:

- Metadata: `claim_count`, `claim_rate`, `notes`, `scenario_mix`, `seed`, `timestamp_utc`
- Claim fields: `patient_id`, `scenario`, `scenario_description`, `expected_verdict`, `gender`, `race`,
  `obs_date`, `med_date`, `onset_date`, `injected_bmi`, `injected_height_cm`, `injected_weight_kg`,
  `injected_condition`, `injected_med`, `wegovy_injected`, `bmi_deleted`, `height_deleted`, `weight_deleted`, `seed`

## Benchmark summary (local run)
The benchmark harness is implemented in `benchmark.py`. The most recent runs used local Ollama with reference-year filtering enabled in the benchmark data loader; sample sizes reflect the requested coverage (nemo8b 200, qwen25 100, qwen3/mistral 50).

RAG/rerank sanity checks ran against the Wegovy policy index and passed 6/6 scenarios with coverage_ok and priority_ok (see `reports/rag_rerank_sanity.txt`).

| Run | Sample size | Avg latency | JSON validity | Citation correctness | Notes |
| --- | --- | --- | --- | --- | --- |
| Deterministic (nemo8b run) | 200 | 11.46 ms | n/a | n/a | Synthetic Wegovy cohort; bucket agreement 100% |
| LLM (nemo8b) | 200 | 2335 ms | n/a | n/a | Agreement 100% |
| LLM (qwen25) | 100 | 37240 ms | n/a | n/a | Agreement 100% |
| LLM (mistral) | 50 | 2917 ms | n/a | n/a | Agreement 100% |
| LLM (qwen3) | 50 | 66904 ms | n/a | n/a | Agreement 100%; audit JSON parse warnings observed (see reports/llm_audit_json_debug.md) |
