#!/usr/bin/env python3
"""
Benchmark: Deterministic vs LLM Engine Comparison

This tool runs both the deterministic policy engine and the LLM-augmented agent
side-by-side on the same patient cohort, measuring:

1. **Agreement Rate**: How often do both engines reach the same verdict?
2. **Performance**: Timing comparison (deterministic is typically 100-1000x faster)
3. **Discrepancy Analysis**: Where do they disagree and why?

Usage:
    python benchmark.py [--sample N] [--output benchmark_results.json]

This is useful for:
- Validating that the deterministic engine matches LLM behavior
- Identifying edge cases where LLM reasoning differs
- Performance benchmarking for production deployment decisions
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Suppress LangChain warnings during benchmark
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from priorauth import paths
from priorauth.policy_engine import evaluate_eligibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_REFERENCE_YEAR: int | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _env_reference_year() -> int | None:
    ref_date = os.getenv("PA_REFERENCE_DATE", "").strip()
    if ref_date:
        try:
            return int(pd.to_datetime(ref_date, errors="raise").year)
        except Exception:
            logger.warning("Invalid PA_REFERENCE_DATE=%s; ignoring.", ref_date)
    ref_year = os.getenv("PA_REFERENCE_YEAR", "").strip()
    if ref_year:
        try:
            return int(ref_year)
        except Exception:
            logger.warning("Invalid PA_REFERENCE_YEAR=%s; ignoring.", ref_year)
    return None


def _resolve_reference_year(candidates: list[tuple[pd.DataFrame, str]] | None = None) -> int:
    env_year = _env_reference_year()
    if env_year is not None:
        return env_year
    years: list[int] = []
    if candidates:
        for df, col in candidates:
            if df is None or df.empty or col not in df.columns:
                continue
            dates = pd.to_datetime(df[col], errors="coerce")
            if dates.empty:
                continue
            max_year = dates.dt.year.max()
            if pd.notna(max_year):
                years.append(int(max_year))
    if years:
        return max(years)
    return datetime.now(UTC).year


def _current_year() -> int:
    return _REFERENCE_YEAR or _resolve_reference_year()


def _filter_current_year_rows(
    df: pd.DataFrame,
    date_col: str,
    reference_year: int | None = None,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    if reference_year is None:
        reference_year = _REFERENCE_YEAR or _resolve_reference_year([(df, date_col)])
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return df.loc[dates.dt.year == reference_year].copy()


def _latest_bmi_observation(p_obs: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
    if p_obs.empty or "date" not in p_obs.columns:
        return None, None
    bmi_rows = p_obs[p_obs["type"] == "BMI"].copy()
    if bmi_rows.empty:
        return None, None
    bmi_rows["date_parsed"] = pd.to_datetime(bmi_rows["date"], errors="coerce")
    bmi_rows = bmi_rows.dropna(subset=["date_parsed"]).sort_values("date_parsed", ascending=False)
    if bmi_rows.empty:
        return None, None
    try:
        return bmi_rows.iloc[0]["date_parsed"], float(bmi_rows.iloc[0]["value"])
    except Exception:
        return None, None


def _latest_height_weight_pair(p_obs: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None, float | None]:
    if p_obs.empty or "date" not in p_obs.columns:
        return None, None, None
    obs = p_obs[p_obs["type"].isin(["Height", "Weight"])].copy()
    if obs.empty:
        return None, None, None
    obs["date_parsed"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date_parsed"])
    if obs.empty:
        return None, None, None
    height_rows = obs[obs["type"] == "Height"]
    weight_rows = obs[obs["type"] == "Weight"]
    if height_rows.empty or weight_rows.empty:
        return None, None, None
    height_by_date = height_rows.sort_values("date_parsed").groupby("date_parsed").tail(1)
    weight_by_date = weight_rows.sort_values("date_parsed").groupby("date_parsed").tail(1)
    common_dates = set(height_by_date["date_parsed"]) & set(weight_by_date["date_parsed"])
    if not common_dates:
        return None, None, None
    latest_date = max(common_dates)
    try:
        height_cm = float(height_by_date[height_by_date["date_parsed"] == latest_date].iloc[0]["value"])
        weight_kg = float(weight_by_date[weight_by_date["date_parsed"] == latest_date].iloc[0]["value"])
    except Exception:
        return None, None, None
    return latest_date, height_cm, weight_kg


def _load_patient_data(
    pid: str,
    df_obs: pd.DataFrame,
    df_conds: pd.DataFrame,
    df_meds: pd.DataFrame
) -> dict:
    """Load patient data for deterministic engine."""
    # Get latest BMI
    p_obs = df_obs[df_obs["patient_id"] == pid].copy()
    p_obs = _filter_current_year_rows(p_obs, "date")
    bmi_date, bmi_val = _latest_bmi_observation(p_obs)
    hw_date, height_cm, weight_kg = _latest_height_weight_pair(p_obs)

    if bmi_val is None and height_cm is None:
        latest_bmi = "MISSING_DATA"
    elif bmi_val is not None and (hw_date is None or (bmi_date is not None and bmi_date >= hw_date)):
        latest_bmi = str(round(bmi_val, 1))
    else:
        # Try to calculate from height/weight (most recent visit with both values)
        if height_cm is not None and weight_kg is not None:
            try:
                ht = float(height_cm) / 100.0
                wt = float(weight_kg)
                if ht > 0:
                    calculated_bmi = wt / (ht ** 2)
                    latest_bmi = f"{calculated_bmi:.1f} (Calculated)"
                else:
                    latest_bmi = "MISSING_DATA"
            except (ValueError, TypeError):
                latest_bmi = "MISSING_DATA"
        else:
            latest_bmi = "MISSING_DATA"

    cond_rows = df_conds[df_conds["patient_id"] == pid]
    conditions = cond_rows.to_dict(orient="records")

    meds = (
        df_meds[df_meds["patient_id"] == pid]["medication_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

    return {
        "latest_bmi": latest_bmi,
        "conditions": conditions,
        "meds": meds,
    }


def run_deterministic(pid: str, df_obs, df_conds, df_meds) -> tuple[str, str, float]:
    """Run deterministic engine, return (verdict, reasoning, duration_ms)."""
    start = time.perf_counter()
    patient_data = _load_patient_data(pid, df_obs, df_conds, df_meds)
    result = evaluate_eligibility(patient_data)
    duration_ms = (time.perf_counter() - start) * 1000
    return result.verdict, result.reasoning, duration_ms


def run_llm(agent, pid: str, drug: str = "Wegovy") -> tuple[str, str, float, dict]:
    """Run LLM agent, return (verdict, reasoning, duration_ms, audit_findings)."""
    start = time.perf_counter()
    try:
        response = agent.invoke({"patient_id": pid, "drug_requested": drug})
        verdict = response.get("final_decision", "ERROR")
        reasoning = response.get("reasoning", "") or ""
        audit_findings = response.get("audit_findings", {}) or {}
    except Exception as e:
        verdict = "ERROR"
        reasoning = str(e)
        audit_findings = {}
    duration_ms = (time.perf_counter() - start) * 1000
    return verdict, reasoning, duration_ms, audit_findings


def normalize_verdict(verdict: str) -> str:
    """
    Normalize verdict into review buckets to align with dashboard taxonomy:
    APPROVED, DENIED, PENDING_CDI, NEEDS_REVIEW, MISSING_INFO, OTHER.
    """
    v = verdict.upper()
    if v == "APPROVED":
        return "APPROVED"
    if v == "CDI_REQUIRED":
        return "PENDING_CDI"
    if v in {"DENIED_MISSING_INFO", "PROVIDER_ACTION_REQUIRED"}:
        return "MISSING_INFO"
    if v in {"FLAGGED", "MANUAL_REVIEW", "SAFETY_SIGNAL_NEEDS_REVIEW"}:
        return "NEEDS_REVIEW"
    if v.startswith("DENIED"):
        return "DENIED"
    return "OTHER"


def run_benchmark(
    sample_size: int | None = None,
    output_path: str = "benchmark_results.json",
    skip_llm: bool = False,
    model_flavor: str = "nemo8b"
):
    """
    Run benchmark comparing deterministic vs LLM engines.
    """
    print("\n" + "=" * 70)
    print("  PRIORAUTH BENCHMARK: Deterministic vs LLM Engine Comparison")
    print("=" * 70 + "\n")

    # Load data
    print("Loading data...")
    df_meds = pd.read_csv(paths.DATA_DIR / "data_medications.csv")
    df_obs = pd.read_csv(paths.DATA_DIR / "data_observations.csv")
    df_conds = pd.read_csv(paths.DATA_DIR / "data_conditions.csv")

    global _REFERENCE_YEAR
    _REFERENCE_YEAR = _resolve_reference_year(
        [
            (df_meds, "date"),
            (df_obs, "date"),
            (df_conds, "onset_date"),
        ]
    )

    df_meds = _filter_current_year_rows(df_meds, "date", _REFERENCE_YEAR)
    df_obs = _filter_current_year_rows(df_obs, "date", _REFERENCE_YEAR)
    df_conds = _filter_current_year_rows(df_conds, "onset_date", _REFERENCE_YEAR)

    # Normalize IDs
    for df in [df_meds, df_obs, df_conds]:
        df["patient_id"] = df["patient_id"].astype(str)

    # Get target patients (Wegovy claims)
    target_meds = df_meds[df_meds["medication_name"].str.contains("Wegovy", case=False, na=False)]
    all_pids = target_meds["patient_id"].dropna().unique().tolist()

    if sample_size and sample_size < len(all_pids):
        import random
        random.seed(42)  # Reproducible sampling
        pids = random.sample(all_pids, sample_size)
    else:
        pids = all_pids

    print(f"Benchmarking {len(pids)} patients (total pool: {len(all_pids)})")

    # Initialize LLM agent if needed
    agent = None
    if not skip_llm and pids:
        print(f"Initializing LLM agent (Flavor: {model_flavor})...")
        os.environ["PA_AUDIT_MODEL_FLAVOR"] = model_flavor
        # Force reload agent_logic to pick up new env var
        import importlib
        if "priorauth.config" in sys.modules:
            importlib.reload(sys.modules["priorauth.config"])
        if "priorauth.agent_logic" in sys.modules:
            importlib.reload(sys.modules["priorauth.agent_logic"])
        from priorauth.agent_logic import build_agent
        agent = build_agent()

    # Run benchmark
    results = []
    det_times = []
    llm_times = []
    agreements = 0
    disagreements = []
    agreement_rate = 0.0
    llm_avg = None
    llm_total = None
    speedup = None

    print("\nRunning benchmark...")
    print("-" * 70)

    if not pids:
        print("No patients available after filtering; check PA_REFERENCE_DATE/PA_REFERENCE_YEAR.")

    for i, pid in enumerate(pids):
        # Deterministic
        det_verdict, det_reason, det_ms = run_deterministic(pid, df_obs, df_conds, df_meds)
        det_times.append(det_ms)

        # LLM
        llm_verdict = "SKIPPED"
        llm_reason = ""
        llm_ms = 0.0

        if not skip_llm:
            llm_verdict, llm_reason, llm_ms, _ = run_llm(agent, pid)
            llm_times.append(llm_ms)

        # Compare (bucket-level agreement)
        det_norm = normalize_verdict(det_verdict)
        llm_norm = normalize_verdict(llm_verdict)

        match = (det_norm == llm_norm)
        if match:
            agreements += 1
        elif not skip_llm:
            disagreements.append({
                "patient_id": pid,
                "deterministic": det_verdict,
                "llm": llm_verdict,
                "det_bucket": det_norm,
                "llm_bucket": llm_norm,
                "det_reason": det_reason,
                "llm_reason": llm_reason
            })

        status_icon = "✓" if match else "✗"
        if skip_llm: status_icon = "-"

        print(f"[{i+1:3d}/{len(pids)}] {pid[:12]}... DET: {det_verdict:<20} LLM: {llm_verdict:<20} {status_icon} ({det_ms:.1f}ms vs {llm_ms:.0f}ms)")

        results.append({
            "patient_id": pid,
            "decision": llm_verdict if not skip_llm else det_verdict,
            "det_verdict": det_verdict,
            "llm_verdict": llm_verdict,
            "det_bucket": det_norm,
            "llm_bucket": llm_norm,
            "match": match,
            "duration_sec": (llm_ms if not skip_llm else det_ms) / 1000.0,
            "det_duration_ms": det_ms,
            "llm_duration_ms": llm_ms,
            "reason_preview": llm_reason if not skip_llm else det_reason
        })

    # Summary statistics
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    det_avg = sum(det_times) / len(det_times) if det_times else 0
    det_total = sum(det_times)

    print("\n📊 Deterministic Engine Performance:")
    print(f"   • Average: {det_avg:.2f} ms/patient")
    print(f"   • Total:   {det_total:.0f} ms ({det_total/1000:.2f} sec)")
    if det_times:
        print(f"   • Min:     {min(det_times):.2f} ms")
        print(f"   • Max:     {max(det_times):.2f} ms")
    else:
        print("   • Min:     n/a")
        print("   • Max:     n/a")

    if not skip_llm and llm_times:
        llm_avg = sum(llm_times) / len(llm_times)
        llm_total = sum(llm_times)
        speedup = llm_avg / det_avg if det_avg > 0 else 0

        print(f"\n🤖 LLM Engine Performance (Flavor: {model_flavor}):")
        print(f"   • Average: {llm_avg:.0f} ms/patient")
        print(f"   • Total:   {llm_total:.0f} ms ({llm_total/1000:.1f} sec)")
        print(f"   • Min:     {min(llm_times):.0f} ms")
        print(f"   • Max:     {max(llm_times):.0f} ms")

        print(f"\n⚡ Speedup: Deterministic is {speedup:.0f}x faster than LLM")

        agreement_rate = (agreements / len(pids)) * 100 if pids else 0
        print(f"\n🎯 Agreement Rate (bucket-level): {agreements}/{len(pids)} ({agreement_rate:.1f}%)")

        if disagreements:
            print(f"\n⚠️  Disagreements ({len(disagreements)} cases):")
            for d in disagreements[:5]:  # Show first 5
                print(f"   • {d['patient_id'][:12]}... DET={d['deterministic']} vs LLM={d['llm']}")
            if len(disagreements) > 5:
                print(f"   ... and {len(disagreements) - 5} more")

    # Save results
    output = {
        "metadata": {
            "timestamp": _now_iso(),
            "total_patients": len(pids),
            "sample_size": sample_size,
            "skip_llm": skip_llm,
            "model_flavor": model_flavor if not skip_llm else None,
            "agreement_basis": "review_bucket",
            "reference_year": _REFERENCE_YEAR,
        },
        "performance": {
            "deterministic": {
                "avg_ms": round(det_avg, 2),
                "total_ms": round(det_total, 2),
                "min_ms": round(min(det_times), 2) if det_times else None,
                "max_ms": round(max(det_times), 2) if det_times else None,
            },
            "llm": {
                "avg_ms": round(llm_avg, 2) if not skip_llm and llm_times else None,
                "total_ms": round(llm_total, 2) if not skip_llm and llm_times else None,
            } if not skip_llm else None,
            "speedup_factor": round(speedup, 1) if not skip_llm and llm_times else None,
        },
        "accuracy": {
            "agreement_rate": round(agreement_rate, 2) if not skip_llm else None,
            "agreements": agreements if not skip_llm else None,
            "disagreements_count": len(disagreements) if not skip_llm else None,
        } if not skip_llm else None,
        "disagreements": disagreements if not skip_llm else None,
        "detailed_results": results,
    }

    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = paths.REPO_ROOT / out_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n📁 Results saved to: {out_path}")
    print("=" * 70 + "\n")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic vs LLM policy engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                     # Full benchmark, all patients (default nemo8b)
  python benchmark.py --sample 50         # Benchmark 50 random patients
  python benchmark.py --flavor mistral     # Use mistral model
  python benchmark.py --flavor nemo8b      # Use nemotron-cascade8b model
  python benchmark.py --deterministic-only # Only time deterministic engine
        """
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=None,
        help="Number of patients to sample (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.json",
        help="Output file path (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--deterministic-only", "-d",
        action="store_true",
        help="Skip LLM evaluation (only benchmark deterministic engine)"
    )
    parser.add_argument(
        "--flavor", "-f",
        type=str,
        default="nemo8b",
        choices=["mistral", "qwen25", "qwen3", "nemo8b"],
        help="LLM model flavor to use (default: nemo8b)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_benchmark(
        sample_size=args.sample,
        output_path=args.output,
        skip_llm=args.deterministic_only,
        model_flavor=args.flavor
    )


if __name__ == "__main__":
    main()
