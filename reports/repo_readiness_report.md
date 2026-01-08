# Repo Readiness Report

## Summary
Phase 3 checks, offline-mode tests, Docker CPU/CUDA builds/tests, and Docker offline socket tests were re-run in this update. Benchmark evidence remains from prior runs. Based on the evidence captured below (including prior runs for benchmarks), the repo is ready for GitHub deployment.

## Commands run (latest update)
- `python -m compileall -q .` (see `reports/compileall.txt`)
- `pytest -q` (see `reports/pytest.txt`)
- `python tests/verify_code_enforcement.py` (see `reports/verify_code_enforcement.txt`)
- `python debug_check.py` (see `reports/debug_check.txt`)
- `python repo_audit.py` (see `reports/repo_audit.txt`)
- `pytest -q tests/test_offline_mode.py` (see `reports/offline_mode_tests.txt`)
- `pytest -q tests/test_letter_mode_offline.py` (see `reports/letter_mode_offline_tests.txt`)
- `docker build -f docker/Dockerfile --build-arg REQUIREMENTS_FILE=requirements-docker-cpu.txt -t priorauth:local .` (see `reports/docker_build_cpu.txt`)
- `docker run --rm priorauth:local pytest -q` (see `reports/docker_test_cpu.txt`)
- `docker image ls priorauth:local --format '{{.Repository}}:{{.Tag}} {{.Size}}'` (see `reports/docker_image_cpu.txt`)
- `docker build -f docker/Dockerfile --build-arg REQUIREMENTS_FILE=requirements.txt -t priorauth:cuda .` (see `reports/docker_build_cuda.txt`)
- `docker run --rm priorauth:cuda pytest -q` (see `reports/docker_test_cuda.txt`)
- `docker image ls priorauth:cuda --format '{{.Repository}}:{{.Tag}} {{.Size}}'` (see `reports/docker_image_cuda.txt`)
- `docker run --rm --network none priorauth:local pytest -q tests/test_offline_mode.py tests/test_letter_mode_offline.py` (see `reports/offline_socket_tests_docker.txt`)

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
- Letter-mode offline tests: `reports/letter_mode_offline_tests.txt`
- Offline socket tests in Docker: `reports/offline_socket_tests_docker.txt`
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
- Fixed offline mode patch state so re-enabling works after global restores; UDP/DNS patching now consistently applied (`src/priorauth/offline_mode.py`).
- Added deterministic dependency lock (`requirements.lock`) from the CPU Docker install and updated CI workflows to install it, run `pip check`, `compileall`, and `pytest -q`.
- Updated `scripts/freeze_dependencies.sh` to prepend the PyTorch CPU extra index for lockfile generation.
- Hardened shell scripts (`scripts/freeze_dependencies.sh`, `models/build_models.sh`) with `#!/usr/bin/env bash` and `set -euo pipefail`.
- Added a CPU-only Docker requirements file and build arg to avoid CUDA wheels (`requirements-docker-cpu.txt`, `docker/Dockerfile`).
- Added repo-grade docs: `docs/SECURITY.md`, `docs/GOVERNANCE.md`, `docs/MODELING.md`, `docs/RUNBOOK.md`, `docs/REPRODUCIBILITY.md`.
- Updated README/ARCHITECTURE for accuracy (defaults, RAG settings, offline mode behavior).
- Refreshed benchmark evidence logs and modeling table (see `reports/benchmark*.txt`, `docs/MODELING.md`).
- Refreshed pytest evidence (see `reports/pytest.txt`).
- Moved benchmark result JSON outputs to `reports/benchmark_results/` and updated defaults in `src/priorauth/apps/agent/benchmark.py`.
- Moved audit log and model trace outputs under `output/` (see `src/priorauth/audit_logger.py`, `src/priorauth/agent_logic.py`).
- Moved source guidelines into `policies/source/`, input data into `data/`, bug artifacts into `docs/bug_data/`, and static analysis outputs into `reports/static_analysis/`.

## Test results (latest rerun)
- `pytest -q`: 265 passed, 1 skipped, 2 warnings (see `reports/pytest.txt`).
- `python -m compileall -q .` produced no errors (see `reports/compileall.txt`).
- `python tests/verify_code_enforcement.py` ran successfully (see `reports/verify_code_enforcement.txt`).
- `python debug_check.py` ran (see `reports/debug_check.txt`).
- `python repo_audit.py` ran (see `reports/repo_audit.txt`).
- `pytest -q tests/test_offline_mode.py`: 6 passed (see `reports/offline_mode_tests.txt`).
- `pytest -q tests/test_letter_mode_offline.py`: 2 passed, 1 warning (see `reports/letter_mode_offline_tests.txt`).
- `docker run --rm priorauth:local pytest -q`: 265 passed, 1 skipped (see `reports/docker_test_cpu.txt`).
- `docker run --rm priorauth:cuda pytest -q`: 265 passed, 1 skipped, 2 warnings (see `reports/docker_test_cuda.txt`).
- `docker run --rm --network none priorauth:local pytest -q tests/test_offline_mode.py tests/test_letter_mode_offline.py`: 8 passed (see `reports/offline_socket_tests_docker.txt`).

## Prior evidence (not re-run in this update)
- Batch runner + governance audit (see `reports/batch_runner.txt`).
- RAG/rerank sanity runs (see `reports/rag_rerank_sanity.txt`).
- Benchmark runs (see `reports/benchmark.txt`, `reports/benchmark_nemo8b.txt`, `reports/benchmark_qwen25.txt`, `reports/benchmark_qwen3.txt`, `reports/benchmark_mistral.txt`).

## Limitations / notes
- `conda info` failed in this environment due to a permission error (see `reports/baseline.md`).
- Default Docker image is CPU-only; CUDA build is larger (see `reports/docker_image_cpu.txt`, `reports/docker_image_cuda.txt`).
- CUDA image was validated via CPU-only tests; GPU runtime validation requires NVIDIA Container Toolkit and a `--gpus all` run.
- Benchmark + governance evidence uses synthetic Wegovy claims generated locally (see `reports/chaos_monkey.txt`).
- LLM advisory audit JSON parse warnings were observed in prior qwen3 runs (see `reports/llm_audit_json_debug.md`).
- Benchmark logs available: deterministic (`reports/benchmark.txt`), nemo8b (`reports/benchmark_nemo8b.txt`), qwen25 (`reports/benchmark_qwen25.txt`), qwen3 (`reports/benchmark_qwen3.txt`), and mistral (`reports/benchmark_mistral.txt`).
- Secret scan matches are false positives due to `sk-` substring in `flask-cors` (see `reports/secrets_scan.md`).
- `output/audit_log.jsonl` is treated as a runtime artifact (gitignored) and regenerated by batch runs.

## Risk / rollback
- Evidence-only updates; rollback by reverting `reports/repo_readiness_report.md`, `reports/offline_mode_tests.txt`, `reports/letter_mode_offline_tests.txt`, `reports/docker_test_cpu.txt`, `reports/docker_image_cpu.txt`, and `reports/docker_test_cuda.txt`.
- Local Docker images can be removed with `docker image rm priorauth:local priorauth:cuda` if disk space is needed.

## Next steps
- If GPU validation is needed, run CUDA container with NVIDIA runtime and verify `torch.cuda.is_available()`.
- If a larger qwen3 sample is required, expect a long runtime and plan accordingly.
- Investigate advisory audit JSON parse warnings for qwen3/mistral if those outputs are required to be machine-parseable.
- Decide whether CI should keep coverage upload or run a single coverage-enabled pytest pass.
