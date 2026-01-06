# CI review

## Findings
- `.github/workflows/ci.yml` and `.github/workflows/tests.yml` were using `requirements-ci.txt` and filtered pytest arguments that did not match local `pytest -q` runs.
- Neither workflow ran `pip check` or `compileall`, and CI commands diverged from local evidence runs.

## Changes applied
- Switched dependency install to `requirements.lock` for deterministic CI installs.
- Updated the lock to CPU-only PyTorch and added the CPU extra index for CI compatibility.
- Added `python -m pip check` and `python -m compileall -q .` steps.
- Updated test execution to `pytest -q` to match local truth.
- Enabled pip caching via `actions/setup-python@v5` `cache: pip`.
- Retained a separate coverage run in `ci.yml` to support the Codecov upload.
- Documented local RAG/rerank sanity evidence in `reports/rag_rerank_sanity.txt` (see file for recorded run).

## Files
- `.github/workflows/ci.yml`
- `.github/workflows/tests.yml`
