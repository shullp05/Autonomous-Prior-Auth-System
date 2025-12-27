import React, { useEffect, useMemo, useState } from "react";
import TraceBanner from "./TraceBanner";
import MetricTooltip from "./MetricTooltip";
import RuntimeStats from "./RuntimeStats";
import SankeyChart from "./SankeyChart";
import { FNR_THRESHOLD } from "./constants";
import { FileText, Ban, AlertTriangle, Info, MessageSquare, HelpCircle, Upload } from "lucide-react";
import "./App.css";

// Centralized modules
import { safeUpper, getStatusDisplayLabel, getStatusPillClass } from "./statusConfig";
import { computeMetrics } from "./metricsEngine";



function cleanAppealLetter(letter) {
  if (!letter) return "";
  let text = String(letter).trim();

  // Handle JSON wrapper like {"letter": "..."}
  if (text.startsWith("{") && text.includes('"letter"')) {
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.letter === "string") {
        text = parsed.letter;
      }
    } catch {
      // Not valid JSON, continue with original
    }
  }

  // Handle escaped newlines
  text = text.replace(/\\n/g, "\n");

  return text;
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

// Smart formatting for Sankey chart values
function formatSankeyValue(value) {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  return `$${(value / 1_000).toFixed(1)}K`;
}



function formatVelocity(ms) {
  if (ms === null || ms === undefined) return "—";
  const n = Number(ms);
  if (!Number.isFinite(n)) return "—";
  // Always display in seconds (User Rule: Internally ms, Externally seconds)
  return `${(n / 1000).toFixed(2)}s/case`;

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

  // Filter & Sort State
  const [searchTerm, setSearchTerm] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: 'date', direction: 'desc' }); // sorting by date desc by default if available, else patient_id

  // Letter editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editedLetter, setEditedLetter] = useState("");

  // Save edited letter with audit trail
  const handleSaveEdit = async () => {
    if (!selectedAppeal) return;

    const auditRecord = {
      patient_id: selectedAppeal.patient_id,
      timestamp: new Date().toISOString(),
      original_letter: selectedAppeal.appeal_letter,
      modified_letter: editedLetter,
      status: selectedAppeal.status,
      editor: "Manual Edit"
    };

    // Save to localStorage for persistence
    const storageKey = `letter_edit_${selectedAppeal.patient_id}`;
    localStorage.setItem(storageKey, JSON.stringify(auditRecord));

    // Update the data array with modified letter
    setData(prev => prev.map(item =>
      item.patient_id === selectedAppeal.patient_id
        ? { ...item, appeal_letter: editedLetter, manually_edited: true }
        : item
    ));

    // Update selectedAppeal to show new content
    setSelectedAppeal(prev => ({ ...prev, appeal_letter: editedLetter, manually_edited: true }));

    // Save to server (output/manual_changes/ folder)
    try {
      const response = await fetch('http://localhost:5052/api/save-edit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(auditRecord)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('[Audit] Letter edit saved to server:', result.path);
      } else {
        console.warn('[Audit] Server save failed, using localStorage only');
      }
    } catch (err) {
      console.warn('[Audit] Could not reach edit server, using localStorage only:', err.message);
    }

    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditedLetter(selectedAppeal?.appeal_letter || "");
  };

  const handleStartEdit = () => {
    setEditedLetter(cleanAppealLetter(selectedAppeal?.appeal_letter || ""));
    setIsEditing(true);
  };

  // Tooltip State (Ref for performance)
  const tooltipRef = React.useRef(null);
  const [tooltipContent, setTooltipContent] = useState(null);

  const handleReasonEnter = (e, content) => {
    if (!content) return;
    setTooltipContent(content);
    if (tooltipRef.current) {
      tooltipRef.current.style.display = "block";
      tooltipRef.current.style.left = `${e.clientX + 5}px`;
      tooltipRef.current.style.top = `${e.clientY + 5}px`;
    }
  };

  const handleReasonMove = (e) => {
    if (tooltipRef.current) {
      tooltipRef.current.style.left = `${e.clientX + 5}px`;
      tooltipRef.current.style.top = `${e.clientY + 5}px`;
    }
  };

  const handleReasonLeave = () => {
    if (tooltipRef.current) {
      tooltipRef.current.style.display = "none";
    }
    setTooltipContent(null);
  };

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

  // 2. KPI Logic (using centralized metrics engine)
  const metrics = useMemo(() => computeMetrics(data, {}, 'run'), [data]);
  const kpis = metrics; // Use the structured object directly

  // 3. Governance Logic
  const activeAudit = useMemo(() => getAudit(governance, activeAuditAttr), [governance, activeAuditAttr]);

  const fnrRows = useMemo(() => {
    if (!governance) return [];
    const audit = getAudit(governance, activeAuditAttr);
    if (!audit || !audit.group_metrics) return [];

    return Object.entries(audit.group_metrics)
      .map(([group, m]) => ({
        group,
        fnr: m.fnr_access,
        eligibleN: m.eligible_n,
        insufficient: m.insufficient_data
      }))
      .sort((a, b) => (b.fnr || 0) - (a.fnr || 0));
  }, [governance, activeAuditAttr]);
  // 3. Filter & Sort Logic
  const filteredData = useMemo(() => {
    let res = [...data];

    // Filter
    if (searchTerm) {
      const lower = searchTerm.toLowerCase();
      res = res.filter(r =>
        (r.patient_id && r.patient_id.toLowerCase().includes(lower)) ||
        (r.status && r.status.toLowerCase().includes(lower)) ||
        (r.reason && r.reason.toLowerCase().includes(lower))
      );
    }

    // Sort
    res.sort((a, b) => {
      let valA = a[sortConfig.key];
      let valB = b[sortConfig.key];

      // Handle numeric
      if (sortConfig.key === 'value') {
        valA = Number(valA) || 0;
        valB = Number(valB) || 0;
      } else {
        // String comparison
        valA = String(valA || "").toLowerCase();
        valB = String(valB || "").toLowerCase();
      }

      if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
      if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });

    return res;
  }, [data, searchTerm, sortConfig]);

  const overallFNR = governance?.overall?.fnr_access;

  if (loading) return <div className="dashboard-shell"><p>Loading Dashboard...</p></div>;

  return (
    <div className="dashboard-shell">
      <div className="dashboard-content">

        {/* --- Header --- */}
        <div className="header">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <h1>Provider-side “Pre-Submission PA Copilot”</h1>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 8 }}>
                <p className="subtitle" style={{ margin: 0 }}>
                  {(metadata?.model_name === "DETERMINISTIC_ENGINE" && "Deterministic Engine") ||
                    metadata?.model_name ||
                    "Qwen 2.5"}{" "}
                  • {metadata?.policy_version ?? "v2.4.1"}
                  {metadata?.timestamp && ` • ${new Date(metadata.timestamp).toLocaleDateString()}`}
                </p>
                <span className="protocol-badge">
                  Protocol: Wegovy (Semaglutide) - Weight Loss
                </span>
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <div className="system-status">
                <span className="pulse-dot" />
                <span>Local Engine: Online</span>
              </div>
              {governance && (
                <span className={`badge ${activeAudit?.bias_detected ? "badge-red" : "badge-green"}`}>
                  {activeAudit?.bias_detected ? "Bias Threshold Exceeded" : "Governance Active"}
                </span>
              )}
            </div>
          </div>
        </div>
        <TraceBanner metadata={metadata} />
        {metadata && (
          <RuntimeStats metadata={metadata} />
        )}

        {/* --- KPI Grid --- */}
        <div className="kpi-grid">
          <div className="kpi-card">
            <div className="kpi-label">{kpis.labels.hoursKpi}</div>
            <div className="kpi-label-row">
              {/* Tooltip Wrapper */}
              <MetricTooltip>
                <h4 className="tooltip-header">Hours Saved Calculation</h4>

                {/* Section 1: Formula */}
                <div className="tooltip-section">
                  <p className="tooltip-label">Auto-Resolved Cases</p>
                  <p className="tooltip-value">
                    {metrics.explain.autoResolvedBreakdown} = <strong>{metrics.autoResolvedCount}</strong>
                  </p>
                </div>

                {/* Section 2: Assumption */}
                <div className="tooltip-section">
                  <p className="tooltip-label">Assumption</p>
                  <p className="tooltip-value">
                    {metrics.explain.assumptionNote}
                  </p>
                </div>

                {/* Section 3: Math */}
                <div className="tooltip-math-box">
                  <div className="tooltip-math-header">
                    <span>Auto-resolved</span>
                    <span>× {metrics.config.minutesPerCasePoint} min / 60</span>
                  </div>
                  <div className="tooltip-math-values">
                    <span>{metrics.autoResolvedCount}</span>
                    <span>× {metrics.config.minutesPerCasePoint} / 60</span>
                  </div>
                  <div className="tooltip-math-result">
                    <span className="tooltip-result-label">Point Estimate</span>
                    <span className="tooltip-result-value">{metrics.display.hoursSavedPoint} Hrs</span>
                  </div>
                </div>

                {/* Section 4: Sensitivity Range */}
                <div className="tooltip-section" style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid var(--border-subtle)' }}>
                  <p className="tooltip-label">Sensitivity Range</p>
                  <p className="tooltip-value">
                    {metrics.explain.sensitivityNote}
                  </p>
                </div>
              </MetricTooltip>
            </div>
            <div className="kpi-value">{kpis.display.hoursSavedPoint}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">{kpis.labels.autoResolutionKpi}</div>
            <div className="kpi-value">{kpis.display.autoResolutionRate}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">{kpis.labels.revenueKpi}</div>
            <div className="kpi-value">{formatSankeyValue(kpis.approvedValue)}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">{kpis.labels.riskKpi}</div>
            <div className="kpi-label-row">
              <MetricTooltip>
                <h4 className="tooltip-header">Potential Revenue Impact</h4>
                <div className="tooltip-section">
                  <p className="tooltip-label">CDI Required</p>
                  <p className="tooltip-value">
                    <strong>{kpis.cdiRequiredCount}</strong> claims clinically eligible but missing anchor codes
                  </p>
                </div>
                <div className="tooltip-section">
                  <p className="tooltip-value">
                    {kpis.explain.cdiBreakdown}
                  </p>
                </div>
              </MetricTooltip>
            </div>
            <div className="kpi-value" style={{ color: '#7C3AED' }}>{kpis.display.revenueAtRisk}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">{kpis.labels.needsReviewKpi}</div>
            <div className="kpi-label-row">
              {/* Breakdown Tooltip */}
              <MetricTooltip>
                <h4 className="tooltip-header">Breakdown</h4>
                <div className="tooltip-section">
                  <p className="tooltip-label">Needs Clarification</p>
                  <p className="tooltip-value"><strong>{kpis.needsClarificationCount}</strong> cases with ambiguous terms</p>
                </div>
                <div className="tooltip-section">
                  <p className="tooltip-label">Missing Required Data</p>
                  <p className="tooltip-value"><strong>{kpis.missingRequiredDataCount}</strong> cases missing BMI/documentation</p>
                </div>
              </MetricTooltip>
            </div>
            <div className="kpi-value" style={{ color: 'var(--status-amber)' }}>{kpis.display.needsReviewTotal}</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Processing Velocity</div>
            <div className="kpi-value">{formatVelocity(metadata?.avg_duration_ms)}</div>
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
          <div style={{ padding: '24px 24px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>Clinical Decision Log</h2>

            <div style={{ display: 'flex', gap: '12px' }}>
              {/* Search Input */}
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  placeholder="Search ID, Status, Reason..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    paddingLeft: '32px',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: '6px',
                    fontSize: '13px',
                    width: '240px',
                    outline: 'none'
                  }}
                />
                <FileText size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-tertiary)' }} />
              </div>

              {/* Sort Select */}
              <select
                value={sortConfig.key}
                onChange={(e) => setSortConfig({ ...sortConfig, key: e.target.value })}
                style={{
                  padding: '8px 12px',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '6px',
                  fontSize: '13px',
                  background: 'white',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="patient_id">Sort by ID</option>
                {/* Assuming date exists, if not fall back to ID order */}
                <option value="status">Sort by Status</option>
                <option value="value">Sort by Value</option>
              </select>

              <button
                onClick={() => setSortConfig(p => ({ ...p, direction: p.direction === 'asc' ? 'desc' : 'asc' }))}
                style={{
                  padding: '8px',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '6px',
                  background: 'white',
                  cursor: 'pointer'
                }}
                title="Toggle Sort Direction"
              >
                {sortConfig.direction === 'asc' ? 'Asc' : 'Desc'}
              </button>
            </div>
          </div>
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
              {filteredData.map((r) => (
                <tr key={r.patient_id}>
                  <td style={{ fontFamily: "JetBrains Mono", color: 'var(--text-secondary)' }}>{r.patient_id}</td>
                  <td>
                    <span className={`status-pill ${getStatusPillClass(r.status, r.reason)}`}>
                      {getStatusDisplayLabel(r.status, r.reason)}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'JetBrains Mono' }}>{formatMoney(r.value)}</td>
                  <td
                    style={{ maxWidth: 450 }}
                    onMouseEnter={(e) => handleReasonEnter(e, r.reason)}
                    onMouseMove={handleReasonMove}
                    onMouseLeave={handleReasonLeave}
                  >
                    <div className="truncated-text">{r.reason || "—"}</div>
                  </td>
                  <td>
                    {(() => {
                      const status = safeUpper(r.status);
                      const isHardStop = getStatusPillClass(r.status, r.reason) === 'status-safety_contraindication';
                      const isSignal = getStatusPillClass(r.status, r.reason) === 'status-safety_signal';

                      // 1. APPROVED
                      if (status === 'APPROVED') {
                        return (
                          <button
                            className="link-button"
                            style={{ background: 'var(--status-green-bg)', color: 'var(--status-green)', borderColor: 'var(--status-green)' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'READ_ONLY' })}
                          >
                            <FileText size={14} style={{ marginRight: 6 }} />
                            Open Packet
                          </button>
                        );
                      }

                      // 2. NEEDS CLARIFICATION (Ambiguity)
                      if (status === 'FLAGGED' || status === 'MANUAL_REVIEW') {
                        return (
                          <button
                            className="link-button"
                            style={{ background: '#FFFBEB', color: '#D97706', borderColor: '#FDE68A' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'AMBIGUITY_RESOLUTION' })}
                          >
                            <HelpCircle size={14} style={{ marginRight: 6 }} />
                            Resolve Clarification
                          </button>
                        );
                      }

                      // 3. MISSING REQUIRED DATA
                      if (status === 'PROVIDER_ACTION_REQUIRED') {
                        return (
                          <button
                            className="link-button"
                            style={{ background: '#EFF6FF', color: '#3B82F6', borderColor: '#BFDBFE' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'MISSING_DATA_RESOLUTION' })}
                          >
                            <Upload size={14} style={{ marginRight: 6 }} />
                            Add Required Data
                          </button>
                        );
                      }

                      // 4. CDI REQUIRED
                      if (status === 'CDI_REQUIRED') {
                        return (
                          <button
                            className="link-button"
                            style={{ background: '#F5F3FF', color: '#7C3AED', borderColor: '#DDD6FE' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'CDI_QUERY' })}
                          >
                            <MessageSquare size={14} style={{ marginRight: 6 }} />
                            Open Query
                          </button>
                        );
                      }

                      // 5. SAFETY CONTRAINDICATION (Hard Stop)
                      if (isHardStop) {
                        return (
                          <button
                            className="link-button"
                            style={{ background: '#FEF2F2', color: '#DC2626', borderColor: '#FECACA' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'READ_ONLY_DENIAL' })}
                          >
                            <Ban size={14} style={{ marginRight: 6 }} />
                            View Hard Stop
                          </button>
                        );
                      }

                      // 6. SAFETY SIGNAL
                      if (isSignal) {
                        return (
                          <button
                            className="link-button"
                            style={{ background: '#FFF7ED', color: '#EA580C', borderColor: '#FED7AA' }}
                            onClick={() => setSelectedAppeal({ ...r, mode: 'EDIT', reason: r.reason + " (Safety Signal Verified)" })}
                          >
                            <AlertTriangle size={14} style={{ marginRight: 6 }} />
                            Verify Safety Signal
                          </button>
                        );
                      }

                      // 7. NOT ELIGIBLE (Clinical Denial)
                      return (
                        <button
                          className="link-button"
                          style={{ background: '#F1F5F9', color: '#64748B', borderColor: '#E2E8F0' }}
                          onClick={() => setSelectedAppeal({ ...r, mode: 'READ_ONLY_DENIAL' })}
                        >
                          <Info size={14} style={{ marginRight: 6 }} />
                          View Rationale
                        </button>
                      );
                    })()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div >



      {/* --- Document Preview Modal (Split View) --- */}
      {
        selectedAppeal && (
          <div className="modal-overlay" onClick={() => setSelectedAppeal(null)}>
            <div className="modal-workspace" onClick={(e) => e.stopPropagation()}>

              {/* Sidebar: Context & Actions */}
              <div className="modal-sidebar">
                {/* Mode-based Header */}
                <div className="sidebar-header" style={{
                  background: selectedAppeal.mode === 'READ_ONLY'
                    ? 'var(--status-green-bg)'
                    : selectedAppeal.mode === 'READ_ONLY_DENIAL'
                      ? 'var(--status-red-bg)'
                      : selectedAppeal.mode === 'CDI_QUERY'
                        ? '#F5F3FF'
                        : selectedAppeal.mode === 'AMBIGUITY_RESOLUTION'
                          ? '#FFFBEB'
                          : selectedAppeal.mode === 'MISSING_DATA_RESOLUTION'
                            ? '#EFF6FF'
                            : 'var(--status-amber-bg)',
                  borderBottom: `2px solid ${selectedAppeal.mode === 'READ_ONLY'
                    ? 'var(--status-green)'
                    : selectedAppeal.mode === 'READ_ONLY_DENIAL'
                      ? 'var(--status-red)'
                      : selectedAppeal.mode === 'CDI_QUERY'
                        ? '#7C3AED'
                        : selectedAppeal.mode === 'AMBIGUITY_RESOLUTION'
                          ? '#D97706'
                          : selectedAppeal.mode === 'MISSING_DATA_RESOLUTION'
                            ? '#3B82F6'
                            : 'var(--status-amber)'
                    }`
                }}>
                  <h3 style={{
                    margin: 0,
                    fontSize: 18,
                    color: selectedAppeal.mode === 'READ_ONLY'
                      ? 'var(--status-green)'
                      : selectedAppeal.mode === 'READ_ONLY_DENIAL'
                        ? 'var(--status-red)'
                        : selectedAppeal.mode === 'CDI_QUERY'
                          ? '#7C3AED'
                          : selectedAppeal.mode === 'AMBIGUITY_RESOLUTION'
                            ? '#D97706'
                            : selectedAppeal.mode === 'MISSING_DATA_RESOLUTION'
                              ? '#3B82F6'
                              : 'var(--status-amber)'
                  }}>
                    {selectedAppeal.mode === 'READ_ONLY' && 'Ready for Signature'}
                    {selectedAppeal.mode === 'READ_ONLY_DENIAL' && 'Hard Stop Notification'}
                    {selectedAppeal.mode === 'EDIT' && 'Clinical Clarification Required'}
                    {selectedAppeal.mode === 'CDI_QUERY' && 'Physician Query Required'}
                    {selectedAppeal.mode === 'AMBIGUITY_RESOLUTION' && 'Ambiguity Resolution'}
                    {selectedAppeal.mode === 'MISSING_DATA_RESOLUTION' && 'Missing Required Data'}
                    {!selectedAppeal.mode && 'Prior Authorization Draft'}
                  </h3>
                  <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
                    <span className={`status-pill ${getStatusPillClass(selectedAppeal.status, selectedAppeal.reason)}`}>
                      {getStatusDisplayLabel(selectedAppeal.status, selectedAppeal.reason)}
                    </span>
                    <span className="badge" style={{ background: '#F1F5F9', color: '#64748B' }}>
                      {selectedAppeal.mode === 'EDIT' ? 'Needs Review' :
                        selectedAppeal.mode === 'READ_ONLY' ? 'Auto-Approved' :
                          selectedAppeal.mode === 'READ_ONLY_DENIAL' ? 'Auto-Denied' :
                            selectedAppeal.mode === 'CDI_QUERY' ? 'Admin Loop' :
                              selectedAppeal.mode === 'AMBIGUITY_RESOLUTION' ? 'Clarification' :
                                selectedAppeal.mode === 'MISSING_DATA_RESOLUTION' ? 'Input Needed' : 'PA Template'}
                    </span>
                  </div>
                </div>

                <div className="sidebar-content">
                  <div className="detail-label">Patient Reference</div>
                  <div className="detail-value" style={{ fontFamily: 'JetBrains Mono' }}>{selectedAppeal.patient_id}</div>

                  <div className="detail-label">
                    {selectedAppeal.mode === 'READ_ONLY_DENIAL' ? 'Denial Reason' :
                      selectedAppeal.mode === 'READ_ONLY' ? 'Approval Criteria' :
                        selectedAppeal.mode === 'CDI_QUERY' ? 'Missing Documentation' :
                          selectedAppeal.mode === 'AMBIGUITY_RESOLUTION' ? 'Ambiguous Term' :
                            selectedAppeal.mode === 'MISSING_DATA_RESOLUTION' ? 'Missing Field' : 'Flag Reason'}
                  </div>
                  <div className="detail-value">{selectedAppeal.reason}</div>

                  {selectedAppeal.appeal_note && (
                    <>
                      <div className="detail-label">
                        {selectedAppeal.mode === 'CDI_QUERY' ? 'Query Context' : 'Provider Guidance'}
                      </div>
                      <div className="ai-reasoning-box">
                        {selectedAppeal.appeal_note}
                      </div>
                    </>
                  )}

                  {!selectedAppeal.appeal_note && selectedAppeal.mode !== 'READ_ONLY_DENIAL' && (
                    <>
                      <div className="detail-label">AI Strategy</div>
                      <div className="ai-reasoning-box">
                        {selectedAppeal.mode === 'READ_ONLY'
                          ? 'Patient meets all eligibility criteria. Ready for provider signature.'
                          : selectedAppeal.mode === 'CDI_QUERY'
                            ? 'Patient is clinically eligible but lacks required administrative anchor codes (E66.x). Send query to provider.'
                            : selectedAppeal.mode === 'AMBIGUITY_RESOLUTION'
                              ? 'Ambiguous clinical terms detected. Human review required to determine intent.'
                              : selectedAppeal.mode === 'MISSING_DATA_RESOLUTION'
                                ? 'Key clinical data points (e.g. BMI, Labs) are missing or incomplete.'
                                : 'Draft PA generated to assist provider review. Please verify clinical criteria and update as needed.'}
                      </div>
                    </>
                  )}
                </div>

                {/* Mode-based Footer */}
                <div className="sidebar-footer">
                  {selectedAppeal.mode === 'AMBIGUITY_RESOLUTION' && (
                    <>
                      <button className="link-button" style={{
                        background: '#D97706', color: 'white',
                        padding: '12px', textAlign: 'center', fontSize: 14, borderColor: '#D97706'
                      }}>
                        Confirm Clinical Context
                      </button>
                      <button className="link-button" style={{
                        padding: '10px', textAlign: 'center'
                      }}>
                        Route for Peer Review
                      </button>
                    </>
                  )}
                  {selectedAppeal.mode === 'MISSING_DATA_RESOLUTION' && (
                    <>
                      <button className="link-button" style={{
                        background: '#3B82F6', color: 'white',
                        padding: '12px', textAlign: 'center', fontSize: 14, borderColor: '#3B82F6'
                      }}>
                        Request Missing Info
                      </button>
                      <button className="link-button" style={{
                        padding: '10px', textAlign: 'center'
                      }}>
                        Enter Data Manually
                      </button>
                    </>
                  )}
                  {selectedAppeal.mode === 'CDI_QUERY' && (
                    <>
                      <button className="link-button" style={{
                        background: '#7C3AED', color: 'white',
                        padding: '12px', textAlign: 'center', fontSize: 14, borderColor: '#7C3AED'
                      }}>
                        Approve & Send Query
                      </button>
                      <button className="link-button" style={{
                        padding: '10px', textAlign: 'center'
                      }}>
                        Edit Query Text
                      </button>
                    </>
                  )}
                  {selectedAppeal.mode === 'READ_ONLY' && (
                    <>
                      <button className="link-button" style={{
                        background: 'var(--status-green)', color: 'white',
                        padding: '12px', textAlign: 'center', fontSize: 14, borderColor: 'var(--status-green)'
                      }}>
                        Confirm & Sign
                      </button>
                      {!isEditing ? (
                        <button
                          className="link-button"
                          style={{ padding: '10px', textAlign: 'center' }}
                          onClick={handleStartEdit}
                        >
                          Edit Manually
                        </button>
                      ) : (
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                          <button
                            className="link-button"
                            style={{ padding: '10px', textAlign: 'center', background: 'var(--status-green)', color: 'white', borderColor: 'var(--status-green)' }}
                            onClick={handleSaveEdit}
                          >
                            Save Changes
                          </button>
                          <button
                            className="link-button"
                            style={{ padding: '10px', textAlign: 'center', color: 'var(--status-red)', borderColor: 'var(--status-red)' }}
                            onClick={handleCancelEdit}
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </>
                  )}
                  {selectedAppeal.mode === 'READ_ONLY_DENIAL' && (
                    <button className="link-button" style={{
                      background: 'var(--status-red)', color: 'white',
                      padding: '12px', textAlign: 'center', fontSize: 14, borderColor: 'var(--status-red)'
                    }}>
                      Acknowledge
                    </button>
                  )}
                  {(selectedAppeal.mode === 'EDIT' || !selectedAppeal.mode) && (
                    <>
                      <button className="link-button" style={{
                        background: 'var(--text-primary)', color: 'white',
                        padding: '12px', textAlign: 'center', fontSize: 14
                      }}>
                        Approve & Sign
                      </button>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                        <button className="link-button" style={{ padding: '10px', textAlign: 'center' }}>Edit Text</button>
                        <button className="link-button" style={{ padding: '10px', textAlign: 'center', color: 'var(--status-red)', borderColor: 'var(--status-red)' }}>Reject</button>
                      </div>
                    </>
                  )}
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
                        : safeUpper(selectedAppeal.status) === 'CDI_REQUIRED'
                          ? 'Physician Query: Clinical Documentation Improvement'
                          : 'Appeal for Coverage of Wegovy (Semaglutide)'}
                    </strong><br />
                    <strong>Patient ID:</strong> {selectedAppeal.patient_id}
                  </div>

                  {isEditing ? (
                    <textarea
                      value={editedLetter}
                      onChange={(e) => setEditedLetter(e.target.value)}
                      style={{
                        width: '100%',
                        minHeight: 400,
                        fontFamily: 'inherit',
                        fontSize: 14,
                        lineHeight: 1.6,
                        padding: 16,
                        border: '2px solid var(--status-amber)',
                        borderRadius: 8,
                        resize: 'vertical',
                        background: '#FFFBEB'
                      }}
                    />
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>
                      {cleanAppealLetter(selectedAppeal.appeal_letter)}
                      {selectedAppeal.manually_edited && (
                        <div style={{ marginTop: 16, padding: 8, background: '#FEF3C7', borderRadius: 4, fontSize: 12, color: '#92400E' }}>
                          ✎ This letter has been manually edited
                        </div>
                      )}
                    </div>
                  )}


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
        )
      }
      {/* Global Cursor Tooltip */}
      <div
        ref={tooltipRef}
        style={{
          position: "fixed",
          display: "none",
          backgroundColor: "#0F172A", /* Slate-900 */
          color: "white",
          padding: "8px 12px",
          borderRadius: "6px",
          fontSize: "12px",
          lineHeight: "1.4",
          maxWidth: "300px",
          zIndex: 9999,
          pointerEvents: "none",
          boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
          whiteSpace: "normal"
        }}
      >
        {tooltipContent}
      </div>
    </div >
  );
}
