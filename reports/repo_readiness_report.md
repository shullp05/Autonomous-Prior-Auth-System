# Repo Readiness Report

## Summary
Completed repo hardening, offline-mode fixes, CI alignment, documentation updates, and container validation. Evidence is captured under `reports/`.

## Evidence (commands + outputs)
- Baseline toolchain: `reports/baseline.md`
- Repo map: `reports/repo_files.txt`
- Repo identification: `reports/repo_identification.md`
- Secret scan: `reports/secrets_scan.md`
- Dependency install log: `reports/pip_install.txt`
- Dependency health: `reports/pip_check.txt`
- CPU image freeze: `reports/pip_freeze_cpu.txt`
- Byte-compile: `reports/compileall.txt`
- Tests: `reports/pytest.txt`
- Offline-mode tests: `reports/offline_mode_tests.txt`
- Debug check: `reports/debug_check.txt`
- Verify code enforcement: `reports/verify_code_enforcement.txt`
- Bash syntax checks: `reports/bash_n.txt`
- Docker CPU build (default): `reports/docker_build_cpu.txt`
- Docker CPU image size: `reports/docker_image_cpu.txt`
- Docker CPU test run: `reports/docker_test_cpu.txt`
- Docker CUDA build (optional): `reports/docker_build_cuda.txt`
- Docker CUDA image size: `reports/docker_image_cuda.txt`
- Docker CUDA test run: `reports/docker_test_cuda.txt`
- Synthetic cohort generation: `reports/chaos_monkey.txt`
- Batch run + governance audit: `reports/batch_runner.txt`
- Benchmark run: `reports/benchmark.txt`
- LLM benchmark (qwen25): `reports/benchmark_qwen25.txt`
- LLM benchmark (mistral): `reports/benchmark_mistral.txt`
- LLM benchmark (qwen3): `reports/benchmark_qwen3.txt`
- LLM benchmark (nemo8b): `reports/benchmark_nemo8b.txt`
- CI review: `reports/ci_review.md`
- Repo audit: `reports/repo_audit.txt`

## Changes applied
- Fixed offline mode patch state so re-enabling works after global restores; UDP/DNS patching now consistently applied (`offline_mode.py`).
- Added deterministic dependency lock (`requirements.lock`) from the CPU Docker install and updated CI workflows to install it, run `pip check`, `compileall`, and `pytest -q`.
- Updated `scripts/freeze_dependencies.sh` to prepend the PyTorch CPU extra index for lockfile generation.
- Hardened shell scripts (`scripts/freeze_dependencies.sh`, `models/build_models.sh`) with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Added a CPU-only Docker requirements file and build arg to avoid CUDA wheels (`requirements-docker-cpu.txt`, `Dockerfile`).
- Added repo-grade docs: `SECURITY.md`, `GOVERNANCE.md`, `MODELING.md`, `RUNBOOK.md`, `REPRODUCIBILITY.md`.
- Updated README/ARCHITECTURE for accuracy (defaults, RAG settings, offline mode behavior).

## Test results
- `pytest -q` passed: 265 passed, 1 skipped (see `reports/pytest.txt`).
- Offline-mode specific tests passed (see `reports/offline_mode_tests.txt`).
- `python -m compileall -q .` produced no errors.
- `tests/verify_code_enforcement.py` ran successfully (see `reports/verify_code_enforcement.txt`).
- Repo audit completed (see `reports/repo_audit.txt`).
- Dockerized pytest (CPU default) passed: 265 passed, 1 skipped (see `reports/docker_test_cpu.txt`).
- Dockerized pytest (CUDA build) passed: 265 passed, 1 skipped, 2 warnings (see `reports/docker_test_cuda.txt`).
- Batch runner (deterministic) processed 200 Wegovy claims, generated governance output, and produced deterministic approval letters without LLM errors (see `reports/batch_runner.txt`).
- Benchmark evidence captured in `reports/benchmark.txt` (latest run: nemo8b sample=200; deterministic avg 11.46 ms).
- LLM benchmarks ran with Ollama reachable: nemo8b sample 200 avg 2335 ms; qwen25 sample 100 avg 37240 ms; mistral sample 50 avg 2917 ms; qwen3 sample 50 avg 66904 ms; bucket agreement 100% per run (see `reports/benchmark.txt`).
- RAG/rerank sanity run passed 6/6 scenarios (coverage_ok and priority_ok) (see `reports/rag_rerank_sanity.txt`).
- RAG/rerank sanity run short-circuited cleanly when Ollama was unavailable (status `OLLAMA_UNAVAILABLE`) (see `reports/rag_rerank_sanity.txt`).

## Limitations / notes
- `conda info` failed in this environment due to a permission error (see `reports/baseline.md`).
- Default Docker image is CPU-only (3.64GB disk usage); CUDA build is larger (13.8GB) (`reports/docker_image_cpu.txt`, `reports/docker_image_cuda.txt`).
- CUDA image was validated via CPU-only tests; GPU runtime validation requires NVIDIA Container Toolkit and a `--gpus all` run.
- Benchmark + governance evidence uses synthetic Wegovy claims generated locally (see `reports/chaos_monkey.txt`).
- LLM advisory audit JSON parse warnings were observed during qwen3 runs (see `reports/llm_audit_json_debug.md`).
- qwen3 benchmark completed with high per-patient latency (see `reports/benchmark_qwen3.txt`).
- nemo8b, mistral, and qwen25 benchmarks logged Chroma telemetry enabled and Hugging Face download warnings at startup; rag/rerank sanity run logged the same (see `reports/benchmark_nemo8b.txt`, `reports/benchmark_mistral.txt`, `reports/benchmark_qwen25.txt`, `reports/rag_rerank_sanity.txt`).
- Secret scan matches are false positives due to `sk-` substring in `flask-cors` (see `reports/secrets_scan.md`).
- `audit_log.jsonl` is treated as a runtime artifact (gitignored) and regenerated by batch runs.

## Next steps
- If GPU validation is needed, run CUDA container with NVIDIA runtime and verify `torch.cuda.is_available()`.
- If a larger qwen3 sample is required, expect a long runtime and plan accordingly.
- Investigate advisory audit JSON parse warnings for qwen3/mistral if those outputs are required to be machine-parseable.
- Decide whether CI should keep coverage upload or run a single coverage-enabled pytest pass.
