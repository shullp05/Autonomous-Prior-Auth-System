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

import pandas as pd

# Suppress LangChain warnings during benchmark
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

from policy_engine import evaluate_eligibility

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_patient_data(
    pid: str,
    df_obs: pd.DataFrame,
    df_conds: pd.DataFrame,
    df_meds: pd.DataFrame
) -> dict:
    """Load patient data for deterministic engine."""
    # Get latest BMI
    p_obs = df_obs[df_obs["patient_id"] == pid].copy()
    bmi_obs = p_obs[p_obs["type"] == "BMI"].copy()

    if not bmi_obs.empty:
        if "date" in bmi_obs.columns:
            bmi_obs["date_parsed"] = pd.to_datetime(bmi_obs["date"], errors="coerce")
            bmi_obs = bmi_obs.sort_values("date_parsed", ascending=False)
        latest_bmi = str(bmi_obs.iloc[0]["value"])
    else:
        # Try to calculate from height/weight
        ht_obs = p_obs[p_obs["type"] == "Height"]
        wt_obs = p_obs[p_obs["type"] == "Weight"]
        if not ht_obs.empty and not wt_obs.empty:
            try:
                ht = float(ht_obs.iloc[0]["value"]) / 100.0
                wt = float(wt_obs.iloc[0]["value"])
                if ht > 0:
                    calculated_bmi = wt / (ht ** 2)
                    latest_bmi = f"{calculated_bmi:.1f} (Calculated)"
                else:
                    latest_bmi = "MISSING_DATA"
            except (ValueError, TypeError):
                latest_bmi = "MISSING_DATA"
        else:
            latest_bmi = "MISSING_DATA"

    conditions = (
        df_conds[df_conds["patient_id"] == pid]["condition_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

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
    """Normalize verdict for comparison (APPROVED vs non-APPROVED)."""
    v = verdict.upper()
    if v == "APPROVED":
        return "APPROVED"
    elif v.startswith("DENIED"):
        return "DENIED"
    elif v == "MANUAL_REVIEW":
        return "MANUAL_REVIEW"
    else:
        return "OTHER"


def run_benchmark(
    sample_size: int | None = None,
    output_path: str = "benchmark_results.json",
    skip_llm: bool = False,
    model_flavor: str = "mistral"
):
    """
    Run benchmark comparing deterministic vs LLM engines.
    """
    print("\n" + "=" * 70)
    print("  PRIORAUTH BENCHMARK: Deterministic vs LLM Engine Comparison")
    print("=" * 70 + "\n")

    # Load data
    print("Loading data...")
    df_meds = pd.read_csv("data_medications.csv")
    df_obs = pd.read_csv("data_observations.csv")
    df_conds = pd.read_csv("data_conditions.csv")

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
    if not skip_llm:
        print(f"Initializing LLM agent (Flavor: {model_flavor})...")
        os.environ["PA_AUDIT_MODEL_FLAVOR"] = model_flavor
        # Force reload agent_logic to pick up new env var
        import importlib
        if "config" in sys.modules:
            importlib.reload(sys.modules["config"])
        if "agent_logic" in sys.modules:
            importlib.reload(sys.modules["agent_logic"])
        from agent_logic import build_agent
        agent = build_agent()

    # Run benchmark
    results = []
    det_times = []
    llm_times = []
    agreements = 0
    disagreements = []

    print("\nRunning benchmark...")
    print("-" * 70)

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

        # Compare
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
    print(f"   • Min:     {min(det_times):.2f} ms")
    print(f"   • Max:     {max(det_times):.2f} ms")

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
        print(f"\n🎯 Agreement Rate: {agreements}/{len(pids)} ({agreement_rate:.1f}%)")

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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n📁 Results saved to: {output_path}")
    print("=" * 70 + "\n")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic vs LLM policy engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                     # Full benchmark, all patients (default qwen25)
  python benchmark.py --sample 50         # Benchmark 50 random patients
  python benchmark.py --flavor mistral     # Use mistral:nemo model
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
        default="qwen25",
        choices=["mistral", "qwen25", "qwen3"],
        help="LLM model flavor to use (default: qwen25)"
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
