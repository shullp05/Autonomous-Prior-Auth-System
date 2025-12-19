---
description: Refactor dashboard metrics to ensure consistency, auditability, and enterprise-grade reliability
---

# Dashboard Metrics Refactoring Workflow

This workflow fixes metric inconsistencies, establishes a single source of truth for status taxonomy and calculations, and ensures the dashboard is auditable and credible.

---

## Phase 0 — Freeze the Ground Truth

### Step 0.1 — Identify the Single "Case List" Source

**Goal:** Locate the one canonical array used for table rows, Sankey, and KPIs.

1. Open `dashboard/src/App.jsx`
2. Find where the table gets its rows (likely `data` state from `useState`)
3. Verify if Sankey (`SankeyChart.jsx`) and KPIs read from the same array
4. Document the variable name that represents "current population in view"

**Done when:** You can point to ONE variable that all components use.

---

## Phase 1 — Lock the Status Taxonomy

### Step 1.1 — Define Status Enum + Mapping

1. Create file: `dashboard/src/statusConfig.js`
2. Define the canonical status enum:
```javascript
export const STATUS = {
  MEETS_CRITERIA: 'MEETS_CRITERIA',
  NOT_ELIGIBLE: 'NOT_ELIGIBLE', 
  SAFETY_CONTRAINDICATION: 'SAFETY_CONTRAINDICATION',
  NEEDS_CLARIFICATION: 'NEEDS_CLARIFICATION',
  MISSING_REQUIRED_DATA: 'MISSING_REQUIRED_DATA',
};

export const NEEDS_REVIEW_STATUSES = [
  STATUS.NEEDS_CLARIFICATION,
  STATUS.MISSING_REQUIRED_DATA,
];
```

3. Create mapping function: `getDisplayStatus(rawStatus, reason)`
4. Create grouping function: `getReviewBucket(status)` → returns `'AUTO_RESOLVED'` or `'NEEDS_REVIEW'`

**Done when:** Table pills only use values from this enum, and all grouping logic calls the same mapping function.

### Step 1.2 — Update Components to Use Status Config

1. Update `App.jsx` to import and use `getDisplayStatus()` and `getReviewBucket()`
2. Update `SankeyChart.jsx` to use the same status mapping
3. Remove all inline `if status === ...` logic scattered through components

**Done when:** All status determination logic is centralized.

---

## Phase 2 — Create Metrics Contract Module

### Step 2.1 — Build `computeMetrics()` Function

1. Create file: `dashboard/src/metricsEngine.js`
2. Implement the function:

```javascript
export function computeMetrics(population, config = {}, scope = 'run') {
  const {
    minutesPerCasePoint = 25,
    minutesPerCaseMin = 15,
    minutesPerCaseMax = 45,
    hoursDecimals = 1,
  } = config;

  // Compute all metrics here
  // Return structured object with:
  // - total_screened
  // - needs_clarification_count
  // - missing_required_data_count  
  // - needs_review_total
  // - auto_resolved_count
  // - hours_saved_point_raw / hours_saved_point_display
  // - hours_saved_range_display
  // - labels (for tooltips)
  // - explain (preformatted breakdown)
}
```

**Done when:** KPI row and tooltip don't do any math—they only render this object.

### Step 2.2 — Enforce Rounding in One Place

1. Add formatting helpers to `metricsEngine.js`:
```javascript
export function formatHours(x, decimals = 1) {
  return Number(x).toFixed(decimals);
}

export function formatCurrency(x) {
  // consistent formatting
}
```

2. All display values must go through these formatters

**Done when:** Impossible to get `52.9` in KPI and `52.87` in tooltip.

---

## Phase 3 — Align UI Language

### Step 3.1 — Rename KPI Labels

1. In `App.jsx`, change "Flagged for Review" → "Needs Review"
2. Add tooltip breakdown showing:
   - Needs Clarification: X
   - Missing Required Data: Y

**Done when:** KPI label matches table taxonomy exactly.

### Step 3.2 — Make Tooltip Numerators Explicit

Replace formula display in tooltip:
```
OLD: (Total - Flagged) × 0.41
NEW: Auto-resolved cases = Total screened − Needs review
     Hours saved = Auto-resolved × minutes_per_case / 60
```

**Done when:** Formula is self-explanatory to unfamiliar users.

---

## Phase 4 — Fix Industry Standard Range Contradiction

### Step 4.1 — Show Point Assumption + Sensitivity Range

Update tooltip to display:
- **Assumption used:** 25 min/case (configurable)
- **Sensitivity range:** 15–45 min/case → X–Y hrs saved

**Done when:** The range is visible, not hidden.

### Step 4.2 — Add Config for Minutes Per Case

1. Add `minutesPerCase` to config (default: 25)
2. Consider adding UI control in "Advanced / Governance" section
3. Persist setting (localStorage or config file)

**Done when:** Changing the config updates both KPI and tooltip together.

---

## Phase 5 — Scope Correctness

### Step 5.1 — Define Scope Rules

Choose ONE and commit:

**Option A (recommended):** KPIs follow current view (filters/search affect numbers)
- Label: "Hours saved (current view)"

**Option B:** KPIs follow entire run, regardless of filters
- Label: "Hours saved (entire run)"

Update labels to be explicit about scope.

### Step 5.2 — Wire Metrics to Correct Population

1. If table uses `filteredCases`, metrics must use `filteredCases`
2. Add `scope` parameter to `computeMetrics()` call
3. Ensure KPI re-computes when filters change

**Done when:** Filtering to "Needs Review" changes KPI numbers consistently (or explicitly doesn't, based on scope choice).

---

## Phase 6 — Popover UX Improvements

### Step 6.1 — Make Tooltip Click-to-Pin

1. Update the metrics tooltip component:
   - Click info icon → opens and stays open
   - Click outside / ESC → closes
   - Tab-focus + Enter/Space toggles (keyboard accessible)

2. Consider replacing hover-only with click-toggle for touchscreen support

**Done when:** Works on touchscreens and with keyboard-only navigation.

---

## Phase 7 — Add Tests

### Step 7.1 — Unit Tests for `computeMetrics`

Create `dashboard/src/__tests__/metricsEngine.test.js`:

```javascript
describe('computeMetrics', () => {
  test('needs_review_total equals sum of subtypes', () => {});
  test('auto_resolved equals total minus needs_review', () => {});
  test('hours_saved calculated correctly', () => {});
  test('display rounding is always 1 decimal', () => {});
  test('sensitivity range computed correctly', () => {});
});
```

// turbo
Run tests: `cd dashboard && npm test`

**Done when:** Cannot ship a mismatch without test failure.

### Step 7.2 — Integration Test (Optional but High ROI)

If Playwright/Cypress available:
1. Load dashboard with known fixture
2. Assert KPI "Needs review" equals visible row counts
3. Assert tooltip value equals KPI displayed value

---

## Phase 8 — Documentation

### Step 8.1 — Add Metric Definitions to README

Add section to `dashboard/README.md` or project README:

```markdown
## Metric Definitions

### Status Categories
- **Meets Criteria**: [definition]
- **Needs Review**: Sum of Needs Clarification + Missing Required Data
- etc.

### Hours Saved Calculation
- Default: 25 min/case (MGMA/AMA complex PA benchmark)
- Formula: (Total screened − Needs review) × minutes_per_case / 60
- Sensitivity range: 15–45 min/case

### Scope
- KPIs reflect [current view / entire run]
```

**Done when:** Dashboard math is explainable without you present.

---

## Execution Order (Minimal Ripple)

Execute phases in this exact order to minimize breaking changes:

1. ✅ Status enum + mapping (Phase 1)
2. ✅ `computeMetrics` contract module + formatting (Phase 2)
3. ✅ Wire KPIs and tooltip to contract (Phase 3)
4. ✅ Fix tooltip wording + sensitivity range (Phase 4)
5. ✅ Scope labeling + population alignment (Phase 5)
6. ✅ Popover click-to-pin + keyboard/ESC (Phase 6)
7. ✅ Unit tests + integration test (Phase 7)
8. ✅ README metric definitions (Phase 8)

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `dashboard/src/statusConfig.js` | NEW: Status enum, mapping functions |
| `dashboard/src/metricsEngine.js` | NEW: computeMetrics(), formatters |
| `dashboard/src/App.jsx` | Wire to new modules, update KPI labels |
| `dashboard/src/SankeyChart.jsx` | Use status config for grouping |
| `dashboard/src/__tests__/metricsEngine.test.js` | NEW: Unit tests |
| `README.md` | Add metric definitions section |

---

## Verification Commands

// turbo
```bash
cd /root/projects/PriorAuth/dashboard && npm run build
```

// turbo
```bash
cd /root/projects/PriorAuth/dashboard && npm test
```
