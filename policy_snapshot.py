from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# Allow env overrides so this stays aligned with config without hard-coding.
GUIDELINE_PATH = Path(os.getenv("PA_GUIDELINE_PATH", "UpdatedPAGuidelines.txt"))
SNAPSHOT_PATH = Path(os.getenv("PA_SNAPSHOT_PATH", "policies/RX-WEG-2025.json"))
POLICY_ID = os.getenv("PA_POLICY_ID", "RX-WEG-2025")


def _read_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = text.lstrip("\ufeff")  # strip BOM if present
    lines = text.splitlines()
    # drop leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def _strip_heading(line: str) -> str:
    # Robust against "#", "##", etc.
    return line.strip().lstrip("#").strip()


def _norm(line: str) -> str:
    return _strip_heading(line).strip()


def _norm_upper(line: str) -> str:
    return _norm(line).upper()


def _find_index(lines: list[str], predicate, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if predicate(lines[idx]):
            return idx
    raise ValueError("Expected marker not found in guidelines file")


def _slice_between(lines: list[str], start_predicate, end_predicate) -> list[str]:
    """
    Safer slicing:
      - Finds start
      - Finds end AFTER start
      - Returns content BETWEEN markers (excluding marker lines)
    """
    try:
        start = _find_index(lines, start_predicate, start=0)
        end = _find_index(lines, end_predicate, start=start + 1)
        return lines[start + 1 : end]
    except ValueError:
        return []


def _extract_values_from_bullet(text: str) -> list[str]:
    cleaned = text.strip().lstrip("- ").strip()
    cleaned = cleaned.rstrip(",")
    if not cleaned or cleaned.endswith(":"):
        return []
    quoted = re.findall(r'"([^"]+)"', cleaned)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]
    return [cleaned] if cleaned else []


def _extract_bullets(lines: list[str]) -> list[str]:
    bullets: list[str] = []
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


def canonical_dumps(data: dict[str, object]) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _dedupe_preserve(seq: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for s in seq:
        if s is None:
            continue
        v = str(s).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _normalize_scope(scope_line: str) -> tuple[str, list[str]]:
    scope_clean = _norm(scope_line).replace("ONLY", "").replace("(", "").replace(")", "")
    scope_clean = scope_clean.replace("–", "-").replace("OBESITY /", "OBESITY/").strip()
    scope_text = scope_clean or "OBESITY / CHRONIC WEIGHT MANAGEMENT ONLY"
    excluded = ["CV-RISK", "MASH", "non-obesity indications"]
    return scope_text, excluded


def _comorbidity_key(label: str) -> str:
    upper = (label or "").upper()
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
    return re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")


def _parse_comorbidities(lines: list[str]) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    categories: dict[str, dict[str, list[str]]] = {}
    notes: list[str] = []

    current_key: str | None = None
    current_label: str | None = None
    accepted: list[str] = []
    current_bullet = ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.upper().startswith("NOTE:"):
            notes.append(line.split(":", 1)[1].strip() if ":" in line else line[5:].strip())
            continue

        if re.match(r"^[a-zA-Z0-9]+\)", line):
            if current_bullet:
                accepted.extend(_extract_values_from_bullet(current_bullet))
                current_bullet = ""

            if current_key:
                categories[current_key] = {
                    "label": current_label or current_key,
                    "accepted_strings": _dedupe_preserve(accepted),
                }

            current_label = line.split(")", 1)[1].strip()
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
        elif current_bullet:
            current_bullet = f"{current_bullet} {line}"

    if current_key:
        if current_bullet:
            accepted.extend(_extract_values_from_bullet(current_bullet))
        categories[current_key] = {
            "label": current_label or current_key,
            "accepted_strings": _dedupe_preserve(accepted),
        }

    return categories, notes


def _parse_safety(lines: list[str]) -> tuple[list[dict[str, object]], list[str], list[str]]:
    safety_items: list[dict[str, object]] = []
    drug_conflicts: list[str] = []
    notes: list[str] = []

    current: dict[str, object] | None = None
    collecting_drugs = False
    conflict_bullet = ""
    safety_bullet = ""

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            if collecting_drugs and conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = ""
            continue

        if _norm_upper(line).startswith("CLINICAL NOTE"):
            continue

        if line.startswith("- ") and line.endswith(":") and "GLP-1 / GLP-1/GIP agents to flag" not in line:
            if collecting_drugs and conflict_bullet:
                drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = ""
                collecting_drugs = False

            if safety_bullet and current:
                current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
                safety_bullet = ""

            if current:
                safety_items.append(current)

            current = {"category": line.lstrip("- ").rstrip(":"), "accepted_strings": []}
            continue

        if "GLP-1 / GLP-1/GIP agents to flag" in line:
            collecting_drugs = True
            continue

        if collecting_drugs:
            if line.startswith("-"):
                if conflict_bullet:
                    drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))
                conflict_bullet = line
            elif conflict_bullet:
                conflict_bullet = f"{conflict_bullet} {line}"
            continue

        if line.startswith("-"):
            if not current:
                current = {"category": "General Safety Exclusion", "accepted_strings": []}
            if safety_bullet:
                current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
            safety_bullet = line
        else:
            if current and safety_bullet:
                safety_bullet = f"{safety_bullet} {line}"
            else:
                notes.append(line)

    if current:
        if safety_bullet:
            current.setdefault("accepted_strings", []).extend(_extract_values_from_bullet(safety_bullet))
        safety_items.append(current)

    if conflict_bullet:
        drug_conflicts.extend(_extract_values_from_bullet(conflict_bullet))

    formatted_safety: list[dict[str, object]] = []
    for item in safety_items:
        formatted_safety.append(
            {
                "category": (str(item.get("category") or "")).strip() or "General Safety Exclusion",
                "accepted_strings": _dedupe_preserve([str(x) for x in (item.get("accepted_strings") or [])]),
                "deny_type": "HARD_DENY",
            }
        )

    return formatted_safety, _dedupe_preserve(drug_conflicts), _dedupe_preserve(notes)


def _parse_ambiguities(lines: list[str]) -> list[dict[str, str]]:
    rules: list[dict[str, str]] = []
    regex = r'-\s*Pattern:\s*"([^"]+)"\s*->\s*([A-Z_]+)(?:\s*\(Note:\s*(.*)\))?'
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        match = re.search(regex, line)
        if match:
            pattern = match.group(1).strip()
            action = match.group(2).strip()
            note = (match.group(3) or "").strip()
            rule = {"pattern": pattern, "action": action}
            if note:
                rule["notes"] = note
            rules.append(rule)
    return rules


def parse_guidelines(path: Path = GUIDELINE_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Guideline file not found: {path}")

    lines = _read_lines(path)
    if len(lines) < 2:
        raise ValueError("Guideline file is too short to parse (missing title/scope lines).")

    title = _strip_heading(lines[0])
    scope_line = lines[1]
    scope, excluded_scopes = _normalize_scope(scope_line)

    eff_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("EFFECTIVE DATE"))
    pid_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("POLICY ID"))

    effective_date_line = _norm(lines[eff_idx])
    policy_id_line = _norm(lines[pid_idx])

    parsed_policy_id = policy_id_line.split(":", 1)[1].strip() if ":" in policy_id_line else policy_id_line.strip()
    if parsed_policy_id != POLICY_ID:
        raise ValueError(f"Policy ID mismatch in guidelines: expected {POLICY_ID}, found {parsed_policy_id}")

    # Major Sections
    adult_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("ADULTS"))
    comorb_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("WEIGHT-RELATED COMORBIDITIES"))
    safety_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("SAFETY_EXCLUSIONS"))
    ambiguity_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("AMBIGUITY RULES"))
    general_notes_idx = _find_index(lines, lambda l: _norm_upper(l).startswith("GENERAL NOTES"))

    general_section = lines[general_notes_idx + 1 : adult_idx]
    adult_section = lines[adult_idx:comorb_idx]
    comorb_section = lines[comorb_idx:safety_idx]
    safety_section = lines[safety_idx:ambiguity_idx]
    ambiguity_section = lines[ambiguity_idx:]

    general_notes = [raw.strip().lstrip("- ").strip() for raw in general_section if raw.strip().startswith("-")]

    # --- OBESITY PATHWAY PARSING ---
    # Strings
    obesity_diag_lines = _slice_between(
        adult_section,
        lambda l: "ADULT OBESITY DIAGNOSIS STRINGS" in _norm_upper(l),
        lambda l: "ADULT OBESITY E66" in _norm_upper(l),
    )
    # E66
    obesity_e66_lines = _slice_between(
        adult_section,
        lambda l: "ADULT OBESITY E66" in _norm_upper(l),
        lambda l: "ADULT OBESITY Z68" in _norm_upper(l),
    )
    # Z68
    obesity_z68_lines = _slice_between(
        adult_section,
        lambda l: "ADULT OBESITY Z68" in _norm_upper(l),
        lambda l: re.match(r"^\s*2\)", l.strip()) is not None,
    )

    # --- OVERWEIGHT PATHWAY PARSING ---
    # Strings
    overweight_diag_lines = _slice_between(
        adult_section,
        lambda l: "ADULT OVERWEIGHT DIAGNOSIS STRINGS" in _norm_upper(l),
        lambda l: "ADULT OVERWEIGHT E66" in _norm_upper(l),
    )
    # E66
    overweight_e66_lines = _slice_between(
        adult_section,
        lambda l: "ADULT OVERWEIGHT E66" in _norm_upper(l),
        lambda l: "ADULT OVERWEIGHT Z68" in _norm_upper(l),
    )
    # Z68 (goes until end of adult section, which is implicitly the start of comorb section)
    overweight_z68_lines = adult_section[_find_index(adult_section, lambda l: "ADULT OVERWEIGHT Z68" in _norm_upper(l)) + 1 :]

    # Extract bullets
    adult_obesity_diags = _dedupe_preserve(_extract_bullets(obesity_diag_lines))
    adult_obesity_e66 = _dedupe_preserve(_extract_bullets(obesity_e66_lines))
    adult_obesity_z68 = _dedupe_preserve(_extract_bullets(obesity_z68_lines))

    adult_overweight_diags = _dedupe_preserve(_extract_bullets(overweight_diag_lines))
    adult_overweight_e66 = _dedupe_preserve(_extract_bullets(overweight_e66_lines))
    adult_overweight_z68 = _dedupe_preserve(_extract_bullets(overweight_z68_lines))

    comorbidities, comorb_notes = _parse_comorbidities(comorb_section)
    safety, drug_conflicts, safety_notes = _parse_safety(safety_section)
    ambiguity_rules = _parse_ambiguities(ambiguity_section)

    if "type2_diabetes" in comorbidities:
        t2_list = comorbidities.get("type2_diabetes", {}).get("accepted_strings", [])
        normalized = [
            (item if not item.lower().startswith("with ") else f"Type 2 diabetes {item}")
            for item in t2_list
        ]
        normalized.extend(["Type 2 Diabetes", "Type II Diabetes"])
        comorbidities["type2_diabetes"]["accepted_strings"] = _dedupe_preserve(normalized)

    if "cardiovascular_disease" in comorbidities:
        cv = comorbidities["cardiovascular_disease"].get("accepted_strings", [])
        cv.extend(["Heart attack", "MI"])
        comorbidities["cardiovascular_disease"]["accepted_strings"] = _dedupe_preserve(cv)

    for entry in safety:
        cat = str(entry.get("category", ""))
        acc = list(entry.get("accepted_strings", []) or [])
        if "GI motility" in cat:
            acc.append("major gi motility disorder")
        if "Suicidality" in cat:
            acc.append("self-harm behavior")
        entry["accepted_strings"] = _dedupe_preserve(acc)

    documentation_requirements = [
        "Document current or baseline BMI; height/weight calculations are acceptable.",
        "Include adult obesity diagnosis text when BMI is ≥ 30 kg/m².",
        "For BMI 27–29.9, include adult excess-weight diagnosis text plus at least one weight-related comorbidity from the accepted list.",
    ]

    notes = [
        *_dedupe_preserve(general_notes),
        "Isolated Type 1 diabetes mellitus (T1DM/IDDM) is not counted as the weight-related comorbidity for Wegovy BMI criteria.",
        *_dedupe_preserve(comorb_notes),
        *_dedupe_preserve([n for n in safety_notes if n and not str(n).startswith("SAFETY_EXCLUSIONS")]),
    ]

    snapshot = {
        "policy_id": POLICY_ID,
        "title": title,
        "effective_date": (effective_date_line.split(":", 1)[1].strip() if ":" in effective_date_line else effective_date_line),
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
                    "required_diagnosis_codes_e66": adult_obesity_e66,
                    "required_diagnosis_codes_z68": adult_obesity_z68,
                    "required_comorbidity_categories": [],
                },
                {
                    "name": "Overweight BMI ≥ 27 kg/m² with weight-related comorbidity",
                    "bmi_min": 27.0,
                    "bmi_max": 29.99,
                    "required_diagnosis_strings": adult_overweight_diags,
                    "required_diagnosis_codes_e66": adult_overweight_e66,
                    "required_diagnosis_codes_z68": adult_overweight_z68,
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


def write_policy_snapshot(snapshot: dict[str, object], path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(snapshot), encoding="utf-8")


def load_policy_snapshot(path: Path = SNAPSHOT_PATH, policy_id: str = POLICY_ID) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("policy_id") != policy_id:
        raise ValueError(f"Snapshot policy_id mismatch: expected {policy_id}, found {data.get('policy_id')}")
    return data


if __name__ == "__main__":
    snap = parse_guidelines(GUIDELINE_PATH)
    write_policy_snapshot(snap, SNAPSHOT_PATH)
    print(f"Wrote snapshot: {SNAPSHOT_PATH} (policy_id={snap.get('policy_id')})")
