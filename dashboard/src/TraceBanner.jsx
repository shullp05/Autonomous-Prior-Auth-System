// TraceBanner.jsx - Displays LLM model trace information
import React, { useEffect, useState } from "react";

const RUN_PARAMS = {
  "pa-audit-qwen25": { temperature: 0.2, top_p: 0.9, num_predict: 768 },
  "pa-audit-mistral": { temperature: 0.2, top_p: 0.9, num_predict: 512 },
  "pa-audit-qwen3": { temperature: 0.25, top_p: 0.95, num_predict: 512 },
};

function TraceBanner({ metadata }) {
  const [trace, setTrace] = useState(null);
  const isDeterministic = metadata?.mode === "deterministic";
  const traceFallback = metadata?.model_params || (metadata?.model_name ? RUN_PARAMS[metadata.model_name] : null);

  useEffect(() => {
    if (isDeterministic) {
      setTrace(null);
      return;
    }

    let isMounted = true;
    const base = import.meta.env.BASE_URL || '/';
    fetch(`${base}.last_model_trace.json`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => {
        if (isMounted) setTrace(data);
      })
      .catch(() => {
        if (isMounted) setTrace(null);
      });

    return () => {
      isMounted = false;
    };
  }, [isDeterministic]);

  if (isDeterministic || (!trace && !traceFallback)) return null;

  const params = (trace && trace.params && Object.keys(trace.params).length > 0 ? trace.params : traceFallback) || {};
  const modelName = trace?.model_name ?? metadata?.model_name ?? "—";
  const role = trace?.role ?? metadata?.audit_role ?? "clinical_audit";
  const ramRequired = metadata?.ram_required_gb ?? trace?.ram_required_gb ?? 10;
  const ramAvailable = trace?.ram_available_gb;

  return (
    <div className="trace-banner">
      <div className="trace-meta">
        <div>
          <span className="trace-label">Model</span>
          <span className="trace-value">{modelName}</span>
        </div>
        <div>
          <span className="trace-label">Role</span>
          <span className="trace-value">{role}</span>
        </div>
        <div>
          <span className="trace-label">Params</span>
          <span className="trace-value">
            <span className="trace-value">temp={params.temperature ?? "—"}</span>,
            <span className="trace-value">top_p={params.top_p ?? "—"}</span>,
            <span className="trace-value">max={params.num_predict ?? "—"}</span>
          </span>
        </div>
        <div>
          <span className="trace-label">RAM</span>
          <span className="trace-value">
            {ramRequired ?? "—"} GB{" "}
            {ramAvailable !== undefined && (
              <span className="trace-ram-required">(available: {ramAvailable} GB)</span>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

import PropTypes from 'prop-types';

TraceBanner.propTypes = {
  metadata: PropTypes.shape({
    mode: PropTypes.string,
    model_params: PropTypes.object,
    model_name: PropTypes.string,
    ram_required_gb: PropTypes.number,
    audit_role: PropTypes.string,
  }),
};

TraceBanner.defaultProps = {
  metadata: null,
};

export default TraceBanner;
