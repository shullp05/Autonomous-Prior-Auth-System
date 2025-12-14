import React from "react";

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

export default function RuntimeStats({ metadata }) {
  if (!metadata) return null;

  const rows = [
    { label: "Run Timestamp", value: metadata.timestamp ? new Date(metadata.timestamp).toLocaleString() : "—" },
    { label: "Mode", value: metadata.mode ?? "—" },
    { label: "Policy Version", value: metadata.policy_version ?? "—" },
    { label: "Model", value: metadata.model_name ?? "—" },
    { label: "Model Flavor", value: metadata.model_flavor ?? "—" },
    { label: "Claims Processed", value: formatNumber(metadata.total_claims) },
    { label: "Drug", value: metadata.drug_queried ?? "—" },
    { label: "p50 / p95 (ms)", value: metadata.p50_duration_ms && metadata.p95_duration_ms ? `${metadata.p50_duration_ms.toFixed(2)} / ${metadata.p95_duration_ms.toFixed(2)}` : "—" },
  ];

  return (
    <div className="runtime-stats">
      <h3>Run Telemetry</h3>
      <div className="runtime-grid">
        {rows.map((row) => (
          <div key={row.label} className="runtime-row">
            <span className="runtime-label">{row.label}</span>
            <span className="runtime-value">{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

import PropTypes from 'prop-types';

RuntimeStats.propTypes = {
  metadata: PropTypes.shape({
    timestamp: PropTypes.string,
    mode: PropTypes.string,
    policy_version: PropTypes.string,
    model_name: PropTypes.string,
    model_flavor: PropTypes.string,
    total_claims: PropTypes.number,
    drug_queried: PropTypes.string,
    p50_duration_ms: PropTypes.number,
    p95_duration_ms: PropTypes.number,
  }),
};

RuntimeStats.defaultProps = {
  metadata: null,
};
