# PriorAuth Dashboard (React + Vite)

This React/Vite app renders the outputs produced by `batch_runner.py`. After each batch run, the backend automatically mirrors `dashboard_data.json`, `governance_report.json`, and `.last_model_trace.json` into this folder’s `public/` directory so the UI always reflects the latest snapshot—no manual `cp` required.

## Metric Definitions

### Status Categories
- **Meets Criteria**: Case met all clinical requirements and was APPROVED.
- **Auto-Resolved**: Sum of all finalized decisions (APPROVED + DENIED + CDI_REQUIRED). These cases did not require human manual review.
- **Needs Review**: Sum of (Needs Clarification + Missing Required Data). These cases require human intervention.
- **Needs Clarification**: LLM found ambiguous medical terms that require human judgment (e.g., "Elevated BP" without context).
- **Missing Required Data**: Case lacked critical fields (BMI, Height/Weight) needed for decision.

### Hours Saved Calculation
- **Default Assumption:** 25 minutes per complex PA (Based on MGMA/AMA benchmarks).
- **Formula:** `(Total Screened − Needs Review) × 25 min / 60`
- **Sensitivity Range:** Displays estimates for 15 min (optimistic) to 45 min (conservative) per case.
- **Note on CDI:** Claims requiring Clinical Documentation Improvement (CDI) are excluded from "Hours Saved" calculations as they effectively pause the workflow for physician query.

### Revenue Metrics
- **Revenue Secured:** Total value of APPROVED claims.
- **Cost Avoidance:** Total value of DENIED claims (clinically inappropriate or unsafe).
- **Revenue at Risk:** Total value of CDI_REQUIRED claims (clinically eligible but missing anchor codes).

### Scope
- **KPI Cards:** Reflect totals for the **entire run/batch**, regardless of table filters.
- **Table/Sankey:** Reflect the **entire run** by default, but table rows react to search.

## Development Workflow

1. Generate data from the root project:
   ```bash
   cd ..  # repo root
   make batch-run      # or python batch_runner.py
   ```
   This refreshes:
   - `dashboard/public/dashboard_data.json` (decision log + metadata)
   - `dashboard/public/governance_report.json` (FNR parity report)
   - `dashboard/public/.last_model_trace.json` (TraceBanner telemetry)

2. Start the dashboard:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```
   `TraceBanner` will show the last model invocation, KPI cards will use the new totals, and the governance card/Sankey will update automatically.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Moving Toward Live APIs

The dashboard currently reads static JSON. When you switch to live APIs:

1. **Centralize fetching** – create hooks like `useDashboardData()` or use React Query to encapsulate `/api/dashboard` and `/api/governance` calls, including auth headers and retries.
2. **Validate payloads** – use client-side schemas (e.g., Zod) matching the JSON Schemas already used on the backend. Reject malformed responses before they reach `App.jsx` or `SankeyChart.jsx`.
3. **Handle loading/errors** – show skeletons/spinners while data arrives, and guard the chart/table against `null` data to avoid runtime crashes.
4. **Add streaming support** – if you plan to push live deltas, batch updates and memoize heavy layouts (Sankey already uses `useMemo`) to keep renders smooth.

Example (React Query + Zod):
```ts
const DashboardSchema = z.object({
  metadata: z.object({ timestamp: z.string(), total_claims: z.number() }),
  results: z.array(z.object({ patient_id: z.string(), status: z.string() }))
});

export function useDashboardData() {
  return useQuery({
    queryKey: ['dashboard'],
    queryFn: async () => DashboardSchema.parse(await (await fetch('/api/dashboard')).json())
  });
}
```
Replace the current `fetch("/dashboard_data.json")` call with this hook and wire in `isLoading`/`error` states. Add auth (bearer tokens or session cookies) and CSRF protection consistent with your backend.

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
