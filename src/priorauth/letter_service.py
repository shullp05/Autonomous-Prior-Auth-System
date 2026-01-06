"""
letter_service.py - Explicit letter generation modes (deterministic vs ollama)

PA_LETTER_MODE:
  - deterministic (default): no LLM calls
  - ollama: use ChatOllama; fail-fast or explicit LLM_UNAVAILABLE
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from priorauth.config import (
    APPEAL_MODEL_NAME,
    AUDIT_MODEL_RAM_GB,
    require_provider_context,
)

logger = logging.getLogger(__name__)

LETTER_MODE_ENV = "PA_LETTER_MODE"
LETTER_MODE_DETERMINISTIC = "deterministic"
LETTER_MODE_OLLAMA = "ollama"
LETTER_MODES = {LETTER_MODE_DETERMINISTIC, LETTER_MODE_OLLAMA}


def _env_truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_letter_mode() -> str:
    if _env_truthy("PA_USE_DETERMINISTIC", "false"):
        return LETTER_MODE_DETERMINISTIC
    mode = os.getenv(LETTER_MODE_ENV, LETTER_MODE_DETERMINISTIC).strip().lower()
    if mode not in LETTER_MODES:
        raise RuntimeError(
            f"{LETTER_MODE_ENV} must be one of {sorted(LETTER_MODES)} (got: {mode!r})"
        )
    return mode


def allow_letter_fallback() -> bool:
    return _env_truthy("PA_ALLOW_LETTER_FALLBACK", "false")


@dataclass(frozen=True)
class LetterResult:
    status: str
    letter: str | None
    note: str | None = None


class ProviderContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    provider_credentials: str = ""
    practice_name: str
    npi: str
    phone: str
    fax: str
    address: str = ""


class PARequestLetterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_name: str
    patient_dob: str
    patient_id: str
    drug_name: str = "Wegovy (semaglutide)"
    strength: str = "2.4 mg"
    route: str = "subcutaneous"
    frequency: str = "weekly"
    indication: str = "Chronic weight management"

    bmi_value: float
    bmi_date: str = ""
    qualifying_pathway: str
    qualifying_comorbidity: str = ""

    adult_obesity_diags: list[str] = Field(default_factory=list)
    adult_overweight_diags: list[str] = Field(default_factory=list)

    qualifying_obesity_icd10_codes: list[str] = Field(default_factory=list)
    qualifying_obesity_icd_z_codes: list[str] = Field(default_factory=list)
    qualifying_overweight_icd10_codes: list[str] = Field(default_factory=list)
    qualifying_overweight_icd_z_codes: list[str] = Field(default_factory=list)

    contraindications_checked: list[str] = Field(default_factory=list)
    contraindications_found: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)

    found_diagnosis_string: str | None = None
    found_e66_code: str | None = None
    found_z68_code: str | None = None
    found_comorbidity_evidence: str | None = None


class PARequestLetterDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient_org: str
    recipient_department: str
    attention_line: str

    subject_line: str

    opening_paragraph: str
    clinical_summary_bullets: list[str]
    criteria_bullets: list[str]
    safety_paragraph: str
    requested_action_paragraph: str
    attachments_bullets: list[str]


def _extract_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 1)[1]
        if "```" in raw:
            raw = raw.split("```", 1)[0]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    candidate = (
        raw[first_brace : last_brace + 1].strip()
        if (first_brace != -1 and last_brace != -1 and last_brace > first_brace)
        else raw
    )

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Parsed JSON is not an object.", candidate, 0)
    return obj


def _guard_letter_text(text: str) -> None:
    bad_phrases = ["Dr. AI"]
    lowered = (text or "").lower()
    if "needappeal" in lowered:
        raise ValueError("Letter contains 'needappeal' language; incorrect for PA request.")
    for p in bad_phrases:
        if p.lower() in lowered:
            raise ValueError(f"Letter contains prohibited phrase: {p!r}")


def _get_provider_context() -> ProviderContext:
    ctx = require_provider_context()
    return ProviderContext(**ctx)


def _deterministic_letter_template(patient_data: dict, approval_reasoning: str, findings: dict) -> str:
    name = str((patient_data or {}).get("name", "")).strip() or "Patient"
    dob = str((patient_data or {}).get("dob", "")).strip() or ""
    patient_id = str((patient_data or {}).get("patient_id", "")).strip()
    bmi = (findings or {}).get("bmi_numeric")
    comorbidity = str((findings or {}).get("comorbidity_category", "NONE") or "NONE")

    bmi_text = "[See chart]"
    try:
        if bmi is not None:
            bmi_text = f"{float(bmi):.1f} kg/m²"
    except Exception as e:
        logger.warning("Failed to format BMI in deterministic letter: %s", e)

    comorbidity_line = f"Qualifying Comorbidity: {comorbidity}" if comorbidity and comorbidity != "NONE" else ""

    template = f"""LETTER OF MEDICAL NECESSITY
Prior Authorization Request — Wegovy (Semaglutide)

To: Medical Director, Utilization Management

RE: {name}
DOB: {dob}
Patient ID: {patient_id}

Dear Medical Director,

I am writing to request prior authorization for Wegovy (semaglutide) for chronic weight management.

CLINICAL JUSTIFICATION:
{approval_reasoning}

Patient's current BMI: {bmi_text}
{comorbidity_line}

Based on the clinical criteria outlined above, this patient meets coverage requirements for Wegovy.

Sincerely,
_______________________________
Prescriber Signature / Date
"""
    return template


def _get_ambiguity_clarification(evidence: str) -> str:
    ev_lower = (evidence or "").lower()

    if "prediabetes" in ev_lower or "pre-diabetes" in ev_lower or "borderline diabetes" in ev_lower or "impaired fasting" in ev_lower:
        return """CLARIFICATION NEEDED:
The term "prediabetes" (or similar) does NOT qualify as a weight-related comorbidity for Wegovy coverage.

To support approval, document ONE of the following (if present):
- Type 2 Diabetes Mellitus
- Hypertension / High Blood Pressure
- Dyslipidemia / Hyperlipidemia
- Obstructive Sleep Apnea (OSA)
- Cardiovascular Disease (ASCVD)

If the patient has any of these, update the chart to clearly document it."""

    if "sleep apnea" in ev_lower and "obstructive" not in ev_lower:
        return """CLARIFICATION NEEDED:
Generic "sleep apnea" does not qualify—documentation must specify "Obstructive Sleep Apnea (OSA)".

To support approval:
- Confirm diagnosis is obstructive (not central/mixed)
- Update chart to clearly state "Obstructive Sleep Apnea" or "OSA"

Alternatively, document another qualifying comorbidity (HTN, T2DM, dyslipidemia, CVD)."""

    if "thyroid" in ev_lower:
        return """CLARIFICATION NEEDED:
Thyroid terminology requires clarification for safety determination.

- Medullary Thyroid Carcinoma (MTC) is a contraindication for Wegovy
- Other thyroid cancers (papillary/follicular) are not contraindications per policy

Please clarify the specific thyroid diagnosis/history to determine Wegovy safety."""

    if "blood pressure" in ev_lower or "borderline hypertension" in ev_lower:
        return """CLARIFICATION NEEDED:
"Elevated blood pressure" or "borderline hypertension" may not meet criteria for a qualifying comorbidity.

To support approval:
- Confirm documented Hypertension / HTN requiring treatment
- Update the problem list to clearly state "Hypertension" or "HTN"

Alternatively, document another qualifying comorbidity (T2DM, dyslipidemia, OSA, CVD)."""

    return f"""CLARIFICATION NEEDED:
The term "{evidence}" requires clarification before this PA can be processed.

Please provide documentation of a qualifying weight-related comorbidity:
- Hypertension
- Type 2 Diabetes Mellitus
- Dyslipidemia
- Obstructive Sleep Apnea (OSA)
- Cardiovascular Disease"""


def _fallback_pa_template(patient_data: dict, reason: str, findings: dict) -> str:
    name = str((patient_data or {}).get("name", "")).strip() or "Patient"
    dob = str((patient_data or {}).get("dob", "")).strip() or ""
    bmi = (findings or {}).get("bmi_numeric")
    evidence = str((findings or {}).get("evidence_quoted", "")).strip()
    verdict = str((findings or {}).get("verdict", "")).upper()

    is_ambiguity_case = verdict == "MANUAL_REVIEW" or "ambiguous" in (reason or "").lower() or "flagged" in (reason or "").lower()

    bmi_text = "Current BMI: REQUIRES VERIFICATION"
    bmi_analysis = "BMI could not be reliably extracted. Please verify from chart."
    if bmi is not None:
        try:
            bmi_val = float(bmi)
            bmi_text = f"Current BMI: {bmi_val:.1f} kg/m²"
            if bmi_val >= 30:
                bmi_analysis = "Patient meets BMI threshold (≥30) for obesity pathway."
            elif bmi_val >= 27:
                bmi_analysis = "Patient meets BMI threshold (≥27) but requires documented qualifying comorbidity."
            else:
                bmi_analysis = "Patient BMI is below coverage threshold (<27). Coverage unlikely without exceptional circumstances."
        except Exception as e:
            logger.warning("Failed to parse BMI in fallback template: %s", e)

    if is_ambiguity_case and evidence:
        cond_text = f'FLAGGED TERM REQUIRING CLARIFICATION: "{evidence}"'
    else:
        cond_text = "Review chart for qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD) if BMI is 27–29.9."

    template = f"""PRIOR AUTHORIZATION DOCUMENTATION TEMPLATE — WEGOVY (SEMAGLUTIDE)
Indication: Chronic weight management

Patient: {name}
Date of Birth: {dob}

CLINICAL PROFILE:
{bmi_text}
Assessment: {bmi_analysis}

{cond_text}

CASE STATUS:
{reason}

NEXT STEPS:
1) Ensure BMI is current and documented
2) If BMI 27–29.9, document a clearly qualifying comorbidity (HTN, T2DM, dyslipidemia, OSA, ASCVD)
3) Review for safety exclusions (MTC/MEN2 history, pregnancy/lactation, concurrent GLP-1/GLP-1-GIP)

_____________________________________________
Provider Signature / Date
"""
    return template


def generate_approved_letter(patient_data: dict, approval_reasoning: str, findings: dict) -> LetterResult:
    mode = get_letter_mode()
    if mode == LETTER_MODE_DETERMINISTIC:
        letter = _deterministic_letter_template(patient_data, approval_reasoning, findings)
        return LetterResult(status="DETERMINISTIC", letter=letter, note="PA_LETTER_MODE=deterministic")

    try:
        provider_ctx = _get_provider_context()
    except Exception as e:
        msg = f"LLM_UNAVAILABLE: provider context missing ({e})"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    patient_name = str(patient_data.get("name", "")).strip()
    patient_dob = str(patient_data.get("dob", "")).strip()
    patient_id = str(patient_data.get("patient_id", "")).strip()

    if not patient_name or not patient_dob or not patient_id:
        msg = "LLM_UNAVAILABLE: missing patient identifiers"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    bmi_value = (findings or {}).get("bmi_numeric")
    if bmi_value is None:
        msg = "LLM_UNAVAILABLE: BMI missing"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    try:
        bmi_value_f = float(bmi_value)
    except Exception:
        msg = "LLM_UNAVAILABLE: BMI invalid"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    comorbidity = str((findings or {}).get("comorbidity_category", "NONE") or "NONE").upper()

    adult_obesity_diags = []
    adult_overweight_diags = []
    qualifying_obesity_icd10_codes = []
    qualifying_obesity_icd_z_codes = []
    qualifying_overweight_icd10_codes = []
    qualifying_overweight_icd_z_codes = []

    if bmi_value_f >= 30.0:
        pathway = "BMI_30_PLUS"
        adult_obesity_diags = ["obesity", "obes"]
        qualifying_obesity_icd10_codes = ["E66.01", "E66.9", "E66.09", "E66.0", "E66.x"]
        qualifying_obesity_icd_z_codes = ["Z68.27", "Z68.28", "Z68.29", "Z68.41", "Z68.42", "Z68.43", "Z68.44", "Z68.45", "Z68.x"]
        qualifying_comorbidity = ""
    else:
        pathway = "BMI_27_29_WITH_COMORBIDITY"
        adult_overweight_diags = ["overweight", "overwt"]
        qualifying_overweight_icd10_codes = ["E66.3", "E66.x"]
        qualifying_overweight_icd_z_codes = ["Z68.27", "Z68.28", "Z68.29", "Z68.x"]
        qualifying_comorbidity = (
            "Hypertension"
            if comorbidity == "HYPERTENSION"
            else ("Type 2 Diabetes Mellitus" if comorbidity == "DIABETES" else comorbidity.title())
        )

    letter_input = PARequestLetterInput(
        patient_name=patient_name,
        patient_dob=patient_dob,
        patient_id=patient_id,
        bmi_value=bmi_value_f,
        qualifying_pathway=pathway,
        adult_obesity_diags=adult_obesity_diags,
        adult_overweight_diags=adult_overweight_diags,
        qualifying_obesity_icd10_codes=qualifying_obesity_icd10_codes,
        qualifying_obesity_icd_z_codes=qualifying_obesity_icd_z_codes,
        qualifying_overweight_icd_z_codes=qualifying_overweight_icd_z_codes,
        qualifying_overweight_icd10_codes=qualifying_overweight_icd10_codes,
        qualifying_comorbidity=qualifying_comorbidity,
        contraindications_checked=[
            "pregnancy/lactation",
            "MTC/MEN2 (personal/family history)",
            "concurrent GLP-1/GLP-1-GIP therapy",
            "pancreatitis history (if documented)",
        ],
        contraindications_found=[],
        attachments=[
            "Most recent vitals or BMI documentation",
            "Problem list reflecting qualifying comorbidity (if applicable)",
            "Medication list",
        ],
        found_diagnosis_string=(findings or {}).get("found_diagnosis_string"),
        found_e66_code=(findings or {}).get("found_e66_code"),
        found_z68_code=(findings or {}).get("found_z68_code"),
        found_comorbidity_evidence=(findings or {}).get("found_comorbidity_evidence") or (findings or {}).get("evidence_quoted"),
    )

    system = """You draft a payer-ready Prior Authorization Request / Letter of Medical Necessity for a PCP office.
HARD RULES:
- Draft for provider review and signature. Do not mention AI, automation, models, or internal systems.
- Do NOT use the word "appeal". This is an initial PA request.
- Do NOT say "approved" or "we approved". The provider is requesting authorization.
- Do NOT invent facts, labs, diagnoses, prior therapies, dates, or contraindications. Use only the provided JSON.
- Safety language must be non-absolute. Use: "No contraindications were identified in the reviewed record" if none found.
- Output MUST be a single valid JSON object matching the schema. No markdown. No extra text.
- IMPORTANT: You MUST explicitly list the following specific criteria values found in the patient record:
  1. Documented BMI Value
  2. Diagnosis String used (found_diagnosis_string)
  3. ICD-10 E66 Code found (found_e66_code)
  4. ICD-10 Z68 Code found (found_z68_code)
  5. Qualifying Documented Comorbidity (found_comorbidity_evidence) - NOTE: If BMI >= 30, write "Not Applicable (BMI >= 30)". If found_comorbidity_evidence is missing/empty but BMI < 30, state "None Documented".
  Include these in the 'criteria_bullets' section.

STYLE:
- Professional, concise, clinical-administrative tone.
- One-page structure. Bullets where appropriate.

OUTPUT JSON SCHEMA (exact keys):
{
  "recipient_org": "...",
  "recipient_department": "...",
  "attention_line": "...",
  "subject_line": "...",
  "opening_paragraph": "...",
  "clinical_summary_bullets": ["..."],
  "criteria_bullets": ["..."],
  "safety_paragraph": "...",
  "requested_action_paragraph": "...",
  "attachments_bullets": ["..."]
}
"""

    user = {
        "provider_context": provider_ctx.model_dump(),
        "letter_input": letter_input.model_dump(),
        "approval_reasoning": approval_reasoning,
    }

    try:
        from langchain_ollama import ChatOllama
    except Exception as e:
        msg = f"LLM_UNAVAILABLE: ChatOllama import failed ({e})"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    try:
        llm = ChatOllama(model=APPEAL_MODEL_NAME, temperature=0.0, format="json")
        resp = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=True)},
            ]
        )

        try:
            from priorauth.agent_logic import write_model_trace  # lazy import

            write_model_trace(
                model_name=APPEAL_MODEL_NAME,
                role="approved_letter_generator",
                params={"temperature": 0.0, "format": "json"},
                required_ram_gb=AUDIT_MODEL_RAM_GB,
            )
        except Exception:
            pass

        raw = str(resp.content or "").strip()
        obj = _extract_json_object(raw)
        draft = PARequestLetterDraft(**obj)

        date_line = time.strftime("%Y-%m-%d")
        provider_line = provider_ctx.provider_name + (
            f", {provider_ctx.provider_credentials}" if provider_ctx.provider_credentials else ""
        )
        practice_block = "\n".join(
            [provider_line, provider_ctx.practice_name, f"NPI: {provider_ctx.npi}", f"Phone: {provider_ctx.phone}  Fax: {provider_ctx.fax}"]
            + ([provider_ctx.address] if provider_ctx.address else [])
        )

        recipient_block = "\n".join([draft.recipient_department, f"Attn: {draft.attention_line}", draft.recipient_org])

        def bullets(items: list[str]) -> str:
            return "\n".join([f"- {str(i).strip()}" for i in items if str(i).strip()])

        letter_text = f"""{practice_block}

{date_line}

{recipient_block}

Subject: {draft.subject_line}

Re: {letter_input.patient_name} (DOB: {letter_input.patient_dob}) | Patient ID: {letter_input.patient_id}

Dear Medical Director,

{draft.opening_paragraph}

Clinical Summary:
{bullets(draft.clinical_summary_bullets)}

Medical Necessity & Coverage Criteria:
{bullets(draft.criteria_bullets)}

Safety Review:
{draft.safety_paragraph}

Requested Action:
{draft.requested_action_paragraph}

Attachments:
{bullets(draft.attachments_bullets)}

Sincerely,

______________________________
{provider_line}
{provider_ctx.practice_name}
"""

        _guard_letter_text(letter_text)
        return LetterResult(status="OLLAMA", letter=letter_text, note="PA_LETTER_MODE=ollama")

    except ValidationError as ve:
        msg = f"LLM_UNAVAILABLE: letter schema validation failed ({str(ve)[:300]})"
        logger.error(msg)
        if allow_letter_fallback():
            letter = _deterministic_letter_template(patient_data, approval_reasoning, findings)
            return LetterResult(status="FALLBACK_USED", letter=letter, note=msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)
    except Exception as e:
        msg = f"LLM_UNAVAILABLE: {e}"
        logger.error(msg)
        if allow_letter_fallback():
            letter = _deterministic_letter_template(patient_data, approval_reasoning, findings)
            return LetterResult(status="FALLBACK_USED", letter=letter, note=msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)


def generate_appeal_letter(patient_data: dict, denial_reason: str, findings: dict) -> LetterResult:
    mode = get_letter_mode()
    if mode == LETTER_MODE_DETERMINISTIC:
        return LetterResult(
            status="DETERMINISTIC",
            letter=_fallback_pa_template(patient_data, denial_reason, findings),
            note="PA_LETTER_MODE=deterministic",
        )

    if not isinstance(patient_data, dict):
        msg = "LLM_UNAVAILABLE: patient_data invalid"
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    verdict = str((findings or {}).get("verdict", "")).upper()
    if verdict == "DENIED_SAFETY":
        return LetterResult(status="SKIPPED", letter=None, note="Safety denial: no appeal letter")

    patient_name = str(patient_data.get("name", "")).strip()
    patient_dob = str(patient_data.get("dob", "")).strip()
    if not patient_name or not patient_dob:
        msg = "LLM_UNAVAILABLE: missing patient name/dob"
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    bmi_value = (findings or {}).get("bmi_numeric")
    evidence = str((findings or {}).get("evidence_quoted", "")).strip()

    is_flagged_case = verdict == "MANUAL_REVIEW" or "ambiguous" in (denial_reason or "").lower() or "flagged" in (denial_reason or "").lower()

    try:
        from langchain_ollama import ChatOllama
    except Exception as e:
        msg = f"LLM_UNAVAILABLE: ChatOllama import failed ({e})"
        logger.error(msg)
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

    llm = ChatOllama(model=APPEAL_MODEL_NAME, temperature=0.2)

    try:
        if is_flagged_case and evidence:
            clarification_guidance = _get_ambiguity_clarification(evidence)
            prompt = f"""You are a board-certified Clinical Pharmacist drafting a prior authorization clarification request.

PATIENT: {patient_name}, DOB: {patient_dob}
CURRENT BMI: {bmi_value if bmi_value is not None else "REQUIRES VERIFICATION"} kg/m²

ISSUE: This prior authorization was flagged for manual review due to ambiguous terminology.
AMBIGUOUS TERM FOUND: "{evidence}"

{clarification_guidance}

INSTRUCTIONS:
Write a brief, focused letter that:
1. States the medication (Wegovy/semaglutide) and indication (chronic weight management)
2. Notes the patient's BMI (or requests a current BMI if not documented)
3. Explains ONLY why the specific term "{evidence}" requires clarification
4. States exactly what documentation or clarification is needed
5. Does NOT list unrelated medical conditions

Do NOT include markdown. Do NOT include placeholders in brackets.
Use direct language appropriate for a clinical document."""
        else:
            prompt = f"""You are a board-certified Clinical Pharmacist drafting a prior authorization documentation guidance letter.

PATIENT: {patient_name}, DOB: {patient_dob}
BMI: {bmi_value if bmi_value is not None else "REQUIRES VERIFICATION"} kg/m²

CURRENT STATUS:
{denial_reason}

INSTRUCTIONS:
Write a professional letter that:
1. States the medication being requested (Wegovy/semaglutide) and indication
2. Summarizes only clinically relevant information
3. Identifies what specific documentation is needed to meet criteria
4. Closes professionally

Do NOT include markdown. Do NOT include placeholders in brackets."""

        resp = llm.invoke(prompt)

        try:
            from priorauth.agent_logic import write_model_trace  # lazy import

            write_model_trace(
                model_name=APPEAL_MODEL_NAME,
                role="appeal_generator",
                params={"temperature": 0.2},
                required_ram_gb=AUDIT_MODEL_RAM_GB,
            )
        except Exception:
            pass

        text = str(resp.content or "").strip()

        if text.startswith("{") and "letter" in text[:80].lower():
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("letter"), str):
                    text = parsed["letter"].strip()
            except Exception as e:
                logger.warning("Failed to parse JSON wrapper in appeal letter: %s", e)

        if "\\n" in text:
            text = text.replace("\\n", "\n")

        if not text or len(text) < 120:
            msg = "LLM_UNAVAILABLE: appeal letter too short"
            if allow_letter_fallback():
                return LetterResult(
                    status="FALLBACK_USED",
                    letter=_fallback_pa_template(patient_data, denial_reason, findings),
                    note=msg,
                )
            return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

        if "appeal" in text.lower():
            msg = "LLM_UNAVAILABLE: appeal letter contains prohibited 'appeal' language"
            if allow_letter_fallback():
                return LetterResult(
                    status="FALLBACK_USED",
                    letter=_fallback_pa_template(patient_data, denial_reason, findings),
                    note=msg,
                )
            return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)

        return LetterResult(status="OLLAMA", letter=text, note="PA_LETTER_MODE=ollama")

    except Exception as e:
        msg = f"LLM_UNAVAILABLE: {e}"
        if allow_letter_fallback():
            return LetterResult(
                status="FALLBACK_USED",
                letter=_fallback_pa_template(patient_data, denial_reason, findings),
                note=msg,
            )
        return LetterResult(status="LLM_UNAVAILABLE", letter=None, note=msg)
