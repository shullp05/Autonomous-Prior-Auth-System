import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup_rag import _section_documents
from policy_snapshot import parse_guidelines


def test_rag_sections_minimum_atoms():
    snapshot = parse_guidelines()
    docs = _section_documents(snapshot)
    sections = {doc.metadata["section"] for doc in docs}

    # Eligibility and diagnosis atoms
    assert "eligibility:pathway1" in sections
    assert "eligibility:pathway2" in sections
    assert "diagnosis:obesity_strings" in sections
    assert "diagnosis:overweight_strings" in sections

    # Comorbidities
    for key in [
        "comorbidity:hypertension",
        "comorbidity:type2_diabetes",
        "comorbidity:dyslipidemia",
        "comorbidity:obstructive_sleep_apnea",
        "comorbidity:cardiovascular_disease",
    ]:
        assert key in sections

    # Safety exclusions (pregnancy/nursing must be present; SNF phrases are not a safety section)
    safety_sections = {s for s in sections if s.startswith("safety_exclusions:")}
    assert any("pregnancy_nursing" in s for s in safety_sections)
    assert len(safety_sections) >= 6

    # Documentation, drug conflicts, ambiguities
    assert "documentation:requirements" in sections
    assert "drug_conflicts:glp1_glp1_gip" in sections
    assert any(s.startswith("ambiguity:") for s in sections)

