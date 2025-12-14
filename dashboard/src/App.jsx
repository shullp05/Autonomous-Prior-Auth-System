import React, { useEffect, useMemo, useState } from "react";
import TraceBanner from "./TraceBanner";
import RuntimeStats from "./RuntimeStats";
import SankeyChart from "./SankeyChart";
import { FNR_THRESHOLD } from "./constants";
import "./App.css";

// --- Helper Functions ---

function safeUpper(s) {
  return String(s ?? "").toUpperCase();
}

function formatPct(x) {
  if (x === null || x === undefined) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  return `${(n * 100).toFixed(1)}%`;
}

function formatMoney(n) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  // Adaptive formatting: $M for millions, $K for thousands, full for smaller
  if (Math.abs(x) >= 1_000_000) return `$${(x / 1_000_000).toFixed(1)}M`;
  if (Math.abs(x) >= 1_000) return `$${(x / 1_000).toFixed(1)}K`;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(x);
}

// Normalizes the governance data structure (supports both new and legacy formats)
function getAudit(governance, attribute) {
  if (!governance) return null;

  // New format support
  if (Array.isArray(governance.attribute_audits)) {
    const found = governance.attribute_audits.find((a) => a?.attribute === attribute);
    if (found) return found;
  }

  // Fallback for old structure (race only)
  if (attribute === "race" && governance.race_metrics) {
    return {
      attribute: "race",
      group_metrics: Object.fromEntries(
        Object.entries(governance.race_metrics || {}).map(([group, v]) => {
          if (typeof v === "number") {
            return [group, { fnr_access: v, eligible_n: null, insufficient_data: false }];
          }
          return [group, { fnr_access: null, eligible_n: null, insufficient_data: true, note: String(v) }];
        })
      ),
      bias_detected: Boolean(governance.bias_detected),
      bias_warning: governance.bias_warning ?? "",
    };
  }
  return null;
}

// --- Sub-Components ---

function MetricBar({ label, value, eligibleN }) {
  const n = typeof value === "number" ? value : null;
  const pct = n !== null ? n : null;
  const isBad = pct !== null && pct >= FNR_THRESHOLD;
  // Visual scale caps at 30% error to keep bars readable
  const widthPct = pct === null ? 0 : Math.min(100, (pct / 0.3) * 100);

  return (
    <div style={{ marginBottom: "12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px", fontSize: "13px" }}>
        <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
          {label}
          {eligibleN !== null && <span style={{ color: "var(--text-tertiary)", fontWeight: 500 }}> (N={eligibleN})</span>}
        </span>
        <span style={{ fontWeight: 600, fontFamily: "JetBrains Mono", color: isBad ? "var(--status-red)" : "var(--text-primary)" }}>
          {pct === null ? "Insufficient Data" : formatPct(pct)}
        </span>
      </div>
      <div style={{ height: "6px", background: "var(--bg-element)", borderRadius: "99px", overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${widthPct}%`,
            background: isBad ? "var(--status-red)" : "var(--status-green)",
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

// --- Main Application ---

export default function App() {
  const [data, setData] = useState([]);
  const [metadata, setMetadata] = useState(null);  // Run metadata from batch_runner
  const [governance, setGovernance] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedAppeal, setSelectedAppeal] = useState(null);
  const [activeAuditAttr, setActiveAuditAttr] = useState("race");

  // 1. Load Data
  useEffect(() => {
    async function loadAll() {
      try {
        const base = import.meta.env.BASE_URL || '/';
        const [dataRes, govRes] = await Promise.all([
          fetch(`${base}dashboard_data.json`),
          fetch(`${base}governance_report.json`),
        ]);

        const d = await dataRes.json();
        const g = await govRes.json();

        // Support both new format (with metadata) and legacy format (array only)
        if (d && typeof d === 'object' && 'results' in d && Array.isArray(d.results)) {
          setData(d.results);
          setMetadata(d.metadata ?? null);
        } else if (Array.isArray(d)) {
          setData(d);
          setMetadata(null);
        } else {
          console.warn('Unexpected dashboard data shape:', d);
          setData([]);
          setMetadata(null);
        }
        setGovernance(g);

        // Auto-select first available audit attribute
        if (g?.attribute_audits?.length > 0) {
          const firstAttr = g.attribute_audits[0]?.attribute;
          if (firstAttr) setActiveAuditAttr(firstAttr);
        }
      } catch (e) {
        console.error("Load failed", e);
      } finally {
        setLoading(false);
      }
    }
    loadAll();
  }, []);

  // 2. KPI Logic
  const kpis = useMemo(() => {
    const total = data.length;
    const approved = data.filter((d) => safeUpper(d.status) === "APPROVED").length;
    const denied = data.filter((d) => safeUpper(d.status) === "DENIED").length;
    // Items that are Flagged or require Action are "Manual"
    const manuallyHandled = total - approved - denied;

    const totalValue = data.reduce((sum, r) => sum + (r.value || 0), 0);
    const protectedRevenue = data
      .filter((d) => safeUpper(d.status) === "APPROVED")
      .reduce((sum, r) => sum + (r.value || 0), 0);
    // Cost avoidance usually calculated on Denied claims
    const avoidedCosts = data
      .filter((d) => safeUpper(d.status) === "DENIED")
      .reduce((sum, r) => sum + (r.value || 0), 0);

    const autoResolutionRate = total > 0 ? ((approved + denied) / total) : 0;

    return { total, approved, denied, manuallyHandled, totalValue, protectedRevenue, avoidedCosts, autoResolutionRate };
  }, [data]);

  // 3. Governance Logic
  const activeAudit = useMemo(() => getAudit(governance, activeAuditAttr), [governance, activeAuditAttr]);

  const fnrRows = useMemo(() => {
    const gm = activeAudit?.group_metrics ?? {};
    return Object.entries(gm)
      .map(([group, m]) => ({
        group,
        fnr: typeof m?.fnr_access === "number" ? m.fnr_access : null,
        eligibleN: m?.eligible_n ?? null,
        insufficient: Boolean(m?.insufficient_data),
      }))
      .sort((a, b) => (b.fnr || 0) - (a.fnr || 0));
  }, [activeAudit]);

  const overallFNR = governance?.overall?.fnr_access;

  if (loading) return <div className="dashboard-shell"><p>Loading Dashboard...</p></div>;

  return (
    <div className="dashboard-shell">
      <div className="dashboard-content">

        {/* --- Header --- */}
        <div className="header">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
            <div>
              <h1>Agentic Utilization Review</h1>
              <p className="subtitle">
                {(metadata?.model_name === "DETERMINISTIC_ENGINE" && "Deterministic Engine") ||
                  metadata?.model_name ||
                  "Mistral Nemo"}{" "}
                • {metadata?.policy_version ?? "v2.4.1"}
                {metadata?.timestamp && ` • ${new Date(metadata.timestamp).toLocaleDateString()}`}
              </p>
            </div>

            {governance && (
              <div style={{ display: "flex", gap: 10 }}>
                <span className={`badge ${activeAudit?.bias_detected ? "badge-red" : "badge-green"}`}>
                  {activeAudit?.bias_detected ? "Bias Threshold Exceeded" : "Governance Active"}
                </span>
              </div>
            )}
          </div>
        </div>
        <TraceBanner metadata={metadata} />
        {metadata && (
          <RuntimeStats metadata={metadata} />
        )}

        {/* --- KPI Grid --- */}
        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">Total Pipeline</div>
            <div className="kpi-value">{formatMoney(kpis.totalValue)}</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Auto-Resolution</div>
            <div className="kpi-value">{formatPct(kpis.autoResolutionRate)}</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Revenue Recovered</div>
            <div className="kpi-value" style={{ color: 'var(--status-green)' }}>{formatMoney(kpis.protectedRevenue)}</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Claims Flagged</div>
            <div className="kpi-value" style={{ color: 'var(--status-amber)' }}>{kpis.manuallyHandled}</div>
          </div>
          <div className="kpi-card">
            <div className="kpi-label">Avg Duration (ms)</div>
            <div className="kpi-value">
              {metadata?.avg_duration_ms ? metadata.avg_duration_ms.toFixed(1) : "—"}
            </div>
          </div>
        </div>

        {/* --- Visual Analysis Section --- */}
        <div className="split-layout">
          {/* Left: Sankey Chart */}
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h2>Revenue Flow Analysis</h2>
            <div style={{ flex: 1 }}>
              <SankeyChart data={data} />
            </div>
          </div>

          {/* Right: Governance Panel */}
          <div className="card">
            <h2>Fairness Audit (FNR)</h2>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 20 }}>
              Monitoring <strong>False Negative Rate</strong> (Wrongful Denials) across protected groups.
            </p>

            {typeof overallFNR === "number" && (
              <MetricBar label="Overall FNR" value={overallFNR} />
            )}

            <div style={{ height: 1, background: 'var(--border-subtle)', margin: '16px 0' }}></div>

            {/* Audit Attribute Selectors */}
            {governance?.attribute_audits?.length > 1 && (
              <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
                {governance.attribute_audits.map(a => (
                  <button
                    key={a.attribute}
                    className={`chip ${activeAuditAttr === a.attribute ? 'chip-active' : ''}`}
                    onClick={() => setActiveAuditAttr(a.attribute)}>
                    {a.attribute}
                  </button>
                ))}
              </div>
            )}

            {/* Metric Bars */}
            {fnrRows.map(r => (
              <MetricBar key={r.group} label={r.group} value={r.fnr} eligibleN={r.eligibleN} />
            ))}
          </div>
        </div>

        {/* --- Data Table --- */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '24px 24px 0' }}>
            <h2>Clinical Decision Log</h2>
          </div>
          <div className="table-wrap" style={{ border: 'none', borderRadius: 0 }}>
            <table className="claims-table">
              <thead>
                <tr>
                  <th>Patient ID</th>
                  <th>Status</th>
                  <th>Value</th>
                  <th>Reasoning</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.patient_id}>
                    <td style={{ fontFamily: "JetBrains Mono", color: 'var(--text-secondary)' }}>{r.patient_id}</td>
                    <td>
                      <span className={`status-pill status-${safeUpper(r.status).toLowerCase()}`}>
                        {r.status === 'PROVIDER_ACTION_REQUIRED' ? 'Action Req' : r.status}
                      </span>
                    </td>
                    <td style={{ fontFamily: 'JetBrains Mono' }}>{formatMoney(r.value)}</td>
                    <td style={{ maxWidth: 450, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {r.reason || "—"}
                    </td>
                    <td>
                      {r.appeal_letter ? (
                        <button
                          className="link-button"
                          onClick={() => setSelectedAppeal(r)}
                        >
                          Review Draft
                        </button>
                      ) : (
                        <span style={{ color: 'var(--border-subtle)' }}>—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {/* --- Document Preview Modal (Split View) --- */}
      {selectedAppeal && (
        <div className="modal-overlay" onClick={() => setSelectedAppeal(null)}>
          <div className="modal-workspace" onClick={(e) => e.stopPropagation()}>

            {/* Sidebar: Context & Actions */}
            <div className="modal-sidebar">
              <div className="sidebar-header">
                <h3 style={{ margin: 0, fontSize: 18 }}>
                  {safeUpper(selectedAppeal.status) === 'FLAGGED' ? 'Prior Authorization Draft' : 'Appeal Review'}
                </h3>
                <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                  <span className={`status-pill status-${safeUpper(selectedAppeal.status).toLowerCase()}`}>
                    {selectedAppeal.status}
                  </span>
                  <span className="badge" style={{ background: '#F1F5F9', color: '#64748B' }}>
                    {safeUpper(selectedAppeal.status) === 'FLAGGED' ? 'PA Template' : 'Auto-Drafted'}
                  </span>
                </div>
              </div>

              <div className="sidebar-content">
                <div className="detail-label">Patient Reference</div>
                <div className="detail-value" style={{ fontFamily: 'JetBrains Mono' }}>{selectedAppeal.patient_id}</div>

                <div className="detail-label">
                  {safeUpper(selectedAppeal.status) === 'FLAGGED' ? 'Flag Reason' : 'Denial Reason (Payer)'}
                </div>
                <div className="detail-value">{selectedAppeal.reason}</div>

                {selectedAppeal.appeal_note && (
                  <>
                    <div className="detail-label">Provider Guidance</div>
                    <div className="ai-reasoning-box">
                      {selectedAppeal.appeal_note}
                    </div>
                  </>
                )}

                {!selectedAppeal.appeal_note && (
                  <>
                    <div className="detail-label">AI Strategy</div>
                    <div className="ai-reasoning-box">
                      Draft PA generated to assist provider review. Please verify clinical criteria and update as needed.
                    </div>
                  </>
                )}
              </div>

              <div className="sidebar-footer">
                <button className="link-button" style={{
                  background: 'var(--text-primary)', color: 'white',
                  padding: '12px', textAlign: 'center', fontSize: 14
                }}>
                  Approve & Fax (eRx)
                </button>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <button className="link-button" style={{ padding: '10px', textAlign: 'center' }}>Edit Text</button>
                  <button className="link-button" style={{ padding: '10px', textAlign: 'center', color: 'var(--status-red)', borderColor: 'var(--status-red)' }}>Reject</button>
                </div>
              </div>
            </div>

            {/* Document Area: The "Paper" */}
            <div className="modal-document-area">
              <div className="document-sheet">
                {/* Simulated Letterhead */}
                <div className="letter-header">
                  <div className="letter-logo">Mercy General Health</div>
                  <div className="letter-meta">
                    Department of Internal Medicine<br />
                    Auth Appeal Unit<br />
                    {new Date().toLocaleDateString()}
                  </div>
                </div>

                {/* Content */}
                <div style={{ marginBottom: 30 }}>
                  <strong>
                    RE: {safeUpper(selectedAppeal.status) === 'FLAGGED'
                      ? 'Prior Authorization Request for Wegovy (Semaglutide)'
                      : 'Appeal for Coverage of Wegovy (Semaglutide)'}
                  </strong><br />
                  <strong>Patient ID:</strong> {selectedAppeal.patient_id}
                </div>

                <div style={{ whiteSpace: 'pre-wrap' }}>
                  {selectedAppeal.appeal_letter}
                </div>

                <div style={{ marginTop: 60 }}>
                  Sincerely,<br /><br />
                  <em style={{ fontFamily: 'Cursive', fontSize: 18, color: '#444' }}>[AI Generated Signature]</em><br />
                  <strong>Dr. AI Assistant, MD</strong><br />
                  Chief Medical Information Officer
                </div>
              </div>
            </div>

            {/* Close Button */}
            <button
              onClick={() => setSelectedAppeal(null)}
              style={{
                position: 'absolute', top: 20, right: 20,
                background: 'rgba(0,0,0,0.5)', border: 'none',
                color: 'white', width: 32, height: 32, borderRadius: '50%',
                cursor: 'pointer', zIndex: 10, display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}
            >
              ✕
            </button>

          </div>
        </div>
      )}
    </div>
  );
}
