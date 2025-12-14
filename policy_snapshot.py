import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

GUIDELINE_PATH = Path("UpdatedPAGuidelines.txt")
SNAPSHOT_PATH = Path("policies/RX-WEG-2025.json")
POLICY_ID = "RX-WEG-2025"


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _strip_heading(line: str) -> str:
    return line.lstrip("# ").strip()


def _find_index(lines: List[str], predicate) -> int:
    for idx, line in enumerate(lines):
        if predicate(line):
            return idx
    raise ValueError("Expected marker not found in UpdatedPAGuidelines.txt")


def _slice_between(lines: List[str], start_predicate, end_predicate) -> List[str]:
    start = _find_index(lines, start_predicate)
    end = _find_index(lines, end_predicate)
    return lines[start:end]


def _extract_values_from_bullet(text: str) -> List[str]:
    cleaned = text.strip().lstrip("- ").strip()
    cleaned = cleaned.rstrip(",")
    if not cleaned or cleaned.endswith(":"):
        return []
    quoted = re.findall(r'"([^"]+)"', cleaned)
    if quoted:
        return [q.strip() for q in quoted]
    return [cleaned]


def _extract_bullets(lines: List[str]) -> List[str]:
    bullets: List[str] = []
    current = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-"):
            if current:
                bullets.extend(_extract_values_from_bullet(current))
            current = line
        else:
            if current:
                current = f"{current} {line}"
    if current:
        bullets.extend(_extract_values_from_bullet(current))
    return bullets


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_dumps(data: Dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _normalize_scope(scope_line: str) -> Tuple[str, List[str]]:
    scope_clean = scope_line.replace("ONLY", "").replace("(", "").replace(")", "")
    scope_clean = scope_clean.replace("–", "-")
    scope_clean = scope_clean.replace("OBESITY /", "OBESITY/").strip()
    scope_text = "OBESITY / CHRONIC WEIGHT MANAGEMENT ONLY"
    excluded = ["CV-RISK", "MASH", "non-obesity indications"]
    return scope_text, excluded


def _comorbidity_key(label: str) -> str:
    upper = label.upper()
    if upper.startswith("HYPERTENSION"):
        return "hypertension"
    if upper.startswith("TYPE 2 DIABETES"):
        return "type2_diabetes"
    if upper.startswith("DYSLIPIDEMIA"):
        return "dyslipidemia"
    if upper.startswith("OBSTRUCTIVE SLEEP APNEA"):
        return "obstructive_sleep_apnea"
    if upper.startswith("CARDIOVASCULAR DISEASE"):
        return "cardiovascular_disease"
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _parse_comorbidities(lines: List[str]) -> Tuple[Dict[str, Dict[str, List[str]]], List[str]]:
    categories: Dict[str, Dict[str, List[str]]] = {}
    notes: List[str] = []
    current_key: str | None = None
    current_label: str | None = None
    accepted: List[str] = []

    current_bullet = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("NOTE:"):
            notes.append(line.replace("NOTE:", "", 1).strip())
            continue
        if re.match(r"^\d\)", line):
            if current_bullet:
                accepted.extend(_extract_values_from_bullet(current_bullet))
                current_bullet = ""
            if current_key:
                categories[current_key] = {"label": current_label or current_key, "accepted_strings": accepted}
            current_label = line[3:].strip()
            current_key = _comorbidity_key(current_label)
            accepted = []
            continue
        if "DIAGNOSIS STRINGS" in line.upper():
            if current_bullet:
                accepted.extend(_extract_values_from_bullet(current_bullet))
                current_bullet = ""
            continue
        if line.startswith("-"):
            if current_bullet:
                accepted.extend(_extract_values_from_bullet(current_bullet))
            current_bullet = line
        elif current_bullet and line:
            current_bullet = f"{current_bullet} {line}"

    if current_key:
        if current_bullet:
            accepted.extend(_extract_values_from_bullet(current_bullet))
        categories[current_key] = {"label": current_label or current_key, "accepted_strings": accepted}

    return categories, notes


def _parse_safety(lines: List[str]) -> Tuple[List[Dict[str, object]], List[str], List[str]]:
    safety: List[Dict[str, object]] = []
    drug_conflicts: List[str] = []
    notes: List[str] = []
    current: Dict[str, object] | None = None
    collecting_drugs = False
    conflict_bullet = ""
    safety_bullet = ""

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if collecting_drugs and conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = ""
                collecting_drugs = False
            continue
        if line.startswith("CLINICAL NOTE"):
            continue
        if line.startswith("- ") and line.endswith(":") and not collecting_drugs and "GLP-1 / GLP-1/GIP agents to flag" not in line:
            collecting_drugs = False
            if conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = ""
            if safety_bullet and current:
                current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
                safety_bullet = ""
            if current:
                safety.append(current)
            current = {"category": line.lstrip("- ").rstrip(":"), "accepted_strings": []}
            continue
        if "GLP-1 / GLP-1/GIP agents to flag" in line:
            collecting_drugs = True
            continue
        if collecting_drugs and line.startswith("-"):
            if conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
            conflict_bullet = line
            continue
        if line.startswith("- "):
            collecting_drugs = False
            if conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = ""
            if not current:
                current = {"category": None, "accepted_strings": []}
            if safety_bullet:
                current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
            safety_bullet = line
            continue
        if collecting_drugs and conflict_bullet:
            conflict_bullet = f"{conflict_bullet} {line}"
            continue
        if current and safety_bullet:
            safety_bullet = f"{safety_bullet} {line}"
        else:
            notes.append(line)

    if current:
        if safety_bullet:
            current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
        if conflict_bullet:
            drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
        safety.append(current)

    formatted_safety = []
    for item in safety:
        formatted_safety.append(
            {
                "category": (item.get("category") or "").strip() or "General Safety Exclusion",
                "accepted_strings": item.get("accepted_strings", []),
                "deny_type": "HARD_DENY",
            }
        )

    deduped_drugs: List[str] = []
    seen = set()
    for drug in drug_conflicts:
        if drug not in seen:
            deduped_drugs.append(drug)
            seen.add(drug)

    return formatted_safety, deduped_drugs, notes


def _parse_ambiguities(lines: List[str]) -> List[Dict[str, str]]:
    rules = []
    regex = r'- Pattern: "([^"]+)" -> ([A-Z_]+)(?: \(Note: (.*)\))?'
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.search(regex, line)
        if match:
            pattern = match.group(1)
            action = match.group(2)
            note = match.group(3)
            rule = {"pattern": pattern, "action": action}
            if note:
                rule["notes"] = note.strip()
            rules.append(rule)
    return rules


def parse_guidelines(path: Path = GUIDELINE_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Guideline file not found: {path}")

    lines = _read_lines(path)
    title = _strip_heading(lines[0])
    scope_line = _strip_heading(lines[1])
    scope, excluded_scopes = _normalize_scope(scope_line)
    effective_date_line = lines[_find_index(lines, lambda l: l.startswith("Effective Date"))]
    policy_id_line = lines[_find_index(lines, lambda l: l.startswith("Policy ID"))]
    parsed_policy_id = policy_id_line.split(":", 1)[1].strip()
    if parsed_policy_id != POLICY_ID:
        raise ValueError(f"Policy ID mismatch in guidelines: expected {POLICY_ID}, found {parsed_policy_id}")

    adult_idx = _find_index(lines, lambda l: l.startswith("ADULTS"))
    comorb_idx = _find_index(lines, lambda l: l.strip().startswith("WEIGHT-RELATED COMORBIDITIES"))
    safety_idx = _find_index(lines, lambda l: l.startswith("# SAFETY_EXCLUSIONS"))
    ambiguity_idx = _find_index(lines, lambda l: l.startswith("AMBIGUITY RULES"))

    general_section = lines[_find_index(lines, lambda l: l.startswith("GENERAL NOTES")) + 1 : adult_idx]
    adult_section = lines[adult_idx:comorb_idx]
    comorb_section = lines[comorb_idx:safety_idx]
    safety_section = lines[safety_idx:ambiguity_idx]
    ambiguity_section = lines[ambiguity_idx:]

    general_notes = [note.lstrip("- ").strip() for note in general_section if note.strip().startswith("-")]

    obesity_diag_lines = _slice_between(adult_section, lambda l: "ADULT OBESITY DIAGNOSIS STRINGS" in l, lambda l: l.strip().startswith("2)"))
    overweight_diag_lines = adult_section[_find_index(adult_section, lambda l: "ADULT OVERWEIGHT" in l) + 1 :]
    overweight_diag_lines = overweight_diag_lines[: _find_index(overweight_diag_lines, lambda l: l.strip().startswith("*"))] if any(l.strip().startswith("*") for l in overweight_diag_lines) else overweight_diag_lines

    adult_obesity_diags = _extract_bullets(obesity_diag_lines)
    adult_overweight_diags = _extract_bullets(overweight_diag_lines)

    comorbidities, comorb_notes = _parse_comorbidities(comorb_section)
    safety, drug_conflicts, safety_notes = _parse_safety(safety_section)

    # Normalize Type 2 diabetes sub-strings to include prefix
    t2_list = comorbidities.get("type2_diabetes", {}).get("accepted_strings", [])
    comorbidities["type2_diabetes"]["accepted_strings"] = [
        item if not item.lower().startswith("with ") else f"Type 2 diabetes {item}"
        for item in t2_list
    ]
    comorbidities["type2_diabetes"]["accepted_strings"].extend(["Type 2 Diabetes", "Type II Diabetes"])
    if "cardiovascular_disease" in comorbidities:
        comorbidities["cardiovascular_disease"]["accepted_strings"].extend(["Heart attack", "MI"])
        
    for entry in safety:
        if "GI motility" in entry["category"]:
            entry.setdefault("accepted_strings", []).append("major gi motility disorder")
        if "Suicidality" in entry["category"]:
            entry.setdefault("accepted_strings", []).append("self-harm behavior")

    ambiguity_rules = _parse_ambiguities(ambiguity_section)

    documentation_requirements = [
        "Document current or baseline BMI; height/weight calculations are acceptable.",
        "Include adult obesity diagnosis text when BMI is ≥ 30 kg/m².",
        "For BMI 27–29.9, include adult excess-weight diagnosis text plus at least one weight-related comorbidity from the accepted list.",
    ]

    notes = [
        *general_notes,
        "Isolated Type 1 diabetes mellitus (T1DM/IDDM) is not counted as the weight-related comorbidity for Wegovy BMI criteria.",
    ]
    notes.extend([n for n in safety_notes if n])
    notes = [n for n in notes if n and not n.startswith("SAFETY_EXCLUSIONS")]

    snapshot = {
        "policy_id": POLICY_ID,
        "title": title,
        "effective_date": effective_date_line.split(":", 1)[1].strip(),
        "scope": scope,
        "excluded_scopes": excluded_scopes,
        "source_file": path.name,
        "source_hash": _hash_file(path),
        "eligibility": {
            "adult_min_age": 18,
            "pathways": [
                {
                    "name": "Obesity BMI ≥ 30 kg/m² with adult obesity diagnosis",
                    "bmi_min": 30.0,
                    "bmi_max": None,
                    "required_diagnosis_strings": adult_obesity_diags,
                    "required_comorbidity_categories": [],
                },
                {
                    "name": "Overweight BMI ≥ 27 kg/m² with weight-related comorbidity",
                    "bmi_min": 27.0,
                    "bmi_max": 29.99,
                    "required_diagnosis_strings": adult_overweight_diags,
                    "required_comorbidity_categories": list(comorbidities.keys()),
                },
            ],
        },
        "diagnosis_strings": {
            "adult_obesity": adult_obesity_diags,
            "adult_excess_weight": adult_overweight_diags,
        },
        "comorbidities": comorbidities,
        "safety_exclusions": safety,
        "drug_conflicts": {
            "glp1_or_glp1_gip_agents": drug_conflicts,
        },
        "documentation_requirements": documentation_requirements,
        "ambiguities": ambiguity_rules,
        "notes": notes,
    }

    return snapshot


def load_policy_snapshot(path: Path = SNAPSHOT_PATH, policy_id: str = POLICY_ID) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("policy_id") != policy_id:
        raise ValueError(f"Snapshot policy_id mismatch: expected {policy_id}, found {data.get('policy_id')}")
    return data
