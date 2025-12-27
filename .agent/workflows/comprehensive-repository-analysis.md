---
description: Conduct exhaustive analysis of PriorAuth codebase to identify, prioritize, fix, and document ALL verifiable bugs, security vulnerabilities, and critical issues.
---

# Comprehensive Repository Analysis & Bug Fix Workflow

This workflow guides the agent through a Deep Clean of the `PriorAuth` repository. It covers discovery, systematic bug hunting (static analysis + logic review), fixing, and validation.

**Prerequisites**:
- Active Environment: `revenue_agent`
- Working Directory: `/root/projects/PriorAuth`

---

## Phase 1: Repository Discovery & Mapping

1. **Architecture Analysis**
   - Map the directory structure to understand the separation between `src` (Python agent), `dashboard` (React), and `tests`.
   - Identify key configuration files: `config.py`, `policy_constants.py`, `vite.config.js`.

2. **Environment Inventory**
   - Confirm active linters: `ruff` (Python), `eslint` (React).
   - Check testing frameworks: `pytest` (Python), `vitest` (React).
   - **Action**: Run a quick inventory command.
   ```bash
   ls -la pyproject.toml dashboard/package.json
   ```

---

## Phase 2: Systematic Bug Discovery

1. **Static Analysis (Python)**
   - Run `bandit` for security scanning.
   - Run `ruff` for code quality and potential bugs.
   // turbo
   ```bash
   bandit -r . -f json -o bandit_report.json 2>/dev/null
   ruff check . --output-format json > ruff_report.json 2>/dev/null
   ```

2. **Static Analysis (React)**
   - Run `eslint` on the dashboard.
   // turbo
   ```bash
   cd dashboard && npm run lint > ../eslint_report.txt 2>&1 || true && cd ..
   ```

3. **Logic & Security Review**
   - Scan for logic gaps in critical paths: `policy_engine.py`, `agent_logic.py`.
   - **Critical Check**: Verify Audit Logging coverage. Ensure `_audit_logger.log_event` is called for *every* decision path (Approved, Denied, Safety, Ambiguity), not just happy paths.
   - Check for React anti-patterns (e.g., synchronous state updates in `useEffect`).

4. **Keyword Scan**
   - Check for leftover TODOs or FIXMEs that indicate technical debt.
   ```bash
   grep -rE "TODO|FIXME|XXX|BUG" . --exclude-dir=node_modules --exclude-dir=.git
   ```

---

## Phase 3: Bug Documentation & Prioritization

1. **Consolidate Findings**
   - Parse `bandit_report.json`, `ruff_report.json`, and `eslint_report.txt`.
   - Filter for overlapping or critical issues.

2. **Create Bug Data Artifact**
   - Output a JSON file `bug_data_{date}.json` containing the schema:
     - `bug_id`: (BUG-001, ...)
     - `severity`: (CRITICAL/HIGH/MEDIUM/LOW)
     - `location`: (File/Line)
     - `description`: (Behavior/Expected/RootCause)
     - `status`: (OPEN)

---

## Phase 4: Fix Implementation

**Constraint**: Fixes must be minimal, focused, and verifiable.

1. **Reproduce via Test** (TDD)
   - For logic bugs (e.g., Audit Logging), create a specific reproduction test in `tests/test_audit_bug.py`.
   - **Goal**: Test must FAIL before fix and PASS after fix.

2. **Apply Fixes**
   - **Python**: Refactor functions to ensure safety/logging guarantees (e.g. wrapper functions).
   - **React**: Fix `useEffect` dependencies or anti-patterns.
   - **General**: Remove unused imports or dead code found by linters.

3. **Verify Fixes**
   - Run the specific reproduction test.
   - Run the full relevant suite (`pytest` or `npm test`).

---

## Phase 5: Testing & Validation

1. **Regression Testing**
   - Ensure no side effects.
   // turbo
   ```bash
   pytest tests/
   ```

2. **Cleanup**
   - Remove temporary report files (`bandit_report.json`, `ruff_report.json`, etc.).

---

## Phase 6: Documentation & Reporting

1. **Generate Final Report**
   - Create `bug_report_{date}.md` summarizing:
     - Total bugs found/fixed.
     - Critical findings (Security/Compliance).
     - Code quality improvements.
     - Remaining risks or technical debt.

2. **Notify User**
   - Present the `bug_report` and `bug_data` artifacts.

---
