import json
from dataclasses import asdict

from rag_rerank_sanity import (
    _ensure_vectorstore,
    build_scenarios,
    run_scenario,
)


def test_rag_rerank_scenarios_all_pass():
    """
    Smoke test over all synthetic Wegovy scenarios.

    Fails if:
      - Expected policy atoms are missing after filtering, or
      - They don't appear in the LLM doc pack, or
      - Overall pass_flag is false.
    """
    vectordb = _ensure_vectorstore()
    scenarios = build_scenarios()
    results = []

    for cfg in scenarios:
        res = run_scenario(cfg, vectordb)
        results.append(res)

        assert res.coverage_ok, (
            f"{cfg.name}: coverage_ok=False; coverage_hits={res.coverage_hits}; "
            f"filtered_sections_all={res.filtered_sections_all}"
        )
        assert res.priority_ok, (
            f"{cfg.name}: priority_ok=False; hits_llm={res.hits_llm}; "
            f"llm_sections={res.llm_sections}"
        )
        assert res.pass_flag, f"{cfg.name}: pass_flag=False"

    # Ensure the summary is JSON serializable (for dashboards / logs)
    json.dumps({"scenarios": [asdict(r) for r in results]})
