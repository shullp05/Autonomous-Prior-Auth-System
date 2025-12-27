import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from policy_snapshot import (
    GUIDELINE_PATH,
    POLICY_ID,
    SNAPSHOT_PATH,
    canonical_dumps,
    load_policy_snapshot,
    parse_guidelines,
)
from schema_validation import validate_policy_snapshot

logger = logging.getLogger(__name__)

# Default to medical embeddings for better domain accuracy
# Override with PA_EMBED_MODEL environment variable
DEFAULT_EMBED_MODEL = "kronos483/MedEmbed-large-v0.1:latest"
EMBED_MODEL = os.getenv("PA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
POLICY_PDF = Path("Policy_Weight_Mgmt_2025.pdf")
EMBED_DIM_FILE = Path("chroma_db/embedding_dim.txt")


def _bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(item, style)) for item in items], bulletType="bullet", start="•")


def _write_snapshot(snapshot: dict[str, object], path: Path = SNAPSHOT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(snapshot), encoding="utf-8")
    logger.info("Wrote policy snapshot to %s", path)
    return path


def generate_snapshot() -> dict[str, object]:
    snapshot = parse_guidelines(GUIDELINE_PATH)
    validate_policy_snapshot(snapshot)
    _write_snapshot(snapshot)
    return snapshot


def _ensure_embedding_dim(embedding_fn: OllamaEmbeddings, expected: int = 1024) -> int:
    """Validate embedding dimension against persisted expectation to avoid Chroma mismatch."""
    EMBED_DIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    dim = len(embedding_fn.embed_query("dimension probe"))
    if EMBED_DIM_FILE.exists():
        recorded = int(EMBED_DIM_FILE.read_text().strip())
        if recorded != dim or dim != expected:
            raise RuntimeError(
                f"Chroma embedding dimension mismatch (expected {expected}, recorded {recorded}, got {dim}). "
                "Delete ./chroma_db and rerun setup_rag.py to rebuild with 1024-dim MedEmbed."
            )
    else:
        EMBED_DIM_FILE.write_text(str(dim), encoding="utf-8")
    if dim != expected:
        raise RuntimeError(
            f"Embedding dimension {dim} does not match expected {expected} for kronos483/MedEmbed-large-v0.1:latest. "
            "Delete ./chroma_db and rerun setup_rag.py."
        )
    return dim


def create_policy_pdf(snapshot: dict[str, object], filename: Path = POLICY_PDF) -> Path:
    doc = SimpleDocTemplate(
        str(filename),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.8 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, leading=18, spaceAfter=8, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="SubHeader", fontSize=12, leading=16, spaceAfter=4, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="BodyTextSmall", fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="PolicyBullet", fontSize=10, leading=14))

    story: list = []
    story.append(Paragraph(snapshot["title"], styles["Title"]))
    story.append(Paragraph(snapshot["scope"], styles["Italic"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(
        Paragraph(
            f"Effective Date: {snapshot['effective_date']} &nbsp;&nbsp;&nbsp; Policy ID: {snapshot['policy_id']}",
            styles["BodyTextSmall"],
        )
    )
    story.append(Paragraph(f"Source: {snapshot['source_file']} (sha256={snapshot['source_hash']})", styles["BodyTextSmall"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Eligibility", styles["SectionHeader"]))
    story.append(Paragraph(f"Adults ≥ {snapshot['eligibility']['adult_min_age']} years", styles["BodyTextSmall"]))
    for pathway in snapshot["eligibility"]["pathways"]:
        bmi_min = pathway["bmi_min"]
        bmi_max = pathway.get("bmi_max")
        bmi_text = f"BMI ≥ {bmi_min}" if bmi_max is None else f"BMI ≥ {bmi_min} and < {bmi_max}"
        story.append(Paragraph(pathway["name"], styles["SubHeader"]))
        story.append(_bullet_list([bmi_text], styles["PolicyBullet"]))
        if pathway.get("required_diagnosis_strings"):
            story.append(Paragraph("Required diagnosis strings:", styles["BodyTextSmall"]))
            story.append(_bullet_list(pathway["required_diagnosis_strings"], styles["PolicyBullet"]))
        if pathway.get("required_comorbidity_categories"):
            labels = [
                snapshot["comorbidities"][key]["label"]
                for key in pathway["required_comorbidity_categories"]
                if key in snapshot["comorbidities"]
            ]
            story.append(Paragraph("At least one weight-related comorbidity:", styles["BodyTextSmall"]))
            story.append(_bullet_list(labels, styles["PolicyBullet"]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Diagnosis Strings", styles["SectionHeader"]))
    story.append(Paragraph("Adult Obesity", styles["SubHeader"]))
    story.append(_bullet_list(snapshot["diagnosis_strings"]["adult_obesity"], styles["PolicyBullet"]))
    story.append(Paragraph("Adult Excess Weight / Overweight", styles["SubHeader"]))
    story.append(_bullet_list(snapshot["diagnosis_strings"]["adult_excess_weight"], styles["PolicyBullet"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Weight-Related Comorbidities", styles["SectionHeader"]))
    for key, comorb in snapshot["comorbidities"].items():
        story.append(Paragraph(comorb["label"], styles["SubHeader"]))
        story.append(_bullet_list(comorb["accepted_strings"], styles["PolicyBullet"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Safety Exclusions & Contraindications", styles["SectionHeader"]))
    for section in snapshot["safety_exclusions"]:
        story.append(Paragraph(section["category"], styles["SubHeader"]))
        story.append(_bullet_list(section["accepted_strings"], styles["PolicyBullet"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Drug Conflicts (No concurrent GLP-1 / GLP-1-GIP use)", styles["SectionHeader"]))
    story.append(_bullet_list(snapshot["drug_conflicts"]["glp1_or_glp1_gip_agents"], styles["PolicyBullet"]))
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Documentation Requirements", styles["SectionHeader"]))
    story.append(_bullet_list(snapshot["documentation_requirements"], styles["PolicyBullet"]))
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Ambiguities (policy-directed handling)", styles["SectionHeader"]))
    for amb in snapshot["ambiguities"]:
        line = f"{amb['pattern']} → {amb['action']}"
        if amb.get("notes"):
            line = f"{line} ({amb['notes']})"
        story.append(Paragraph(line, styles["BodyTextSmall"]))
    story.append(Spacer(1, 0.05 * inch))

    story.append(Paragraph("Notes", styles["SectionHeader"]))
    story.append(_bullet_list(snapshot["notes"], styles["Bullet"]))

    doc.build(story)
    logger.info("Generated Policy Document: %s", filename)
    return filename


def _section_documents(snapshot: dict[str, object]) -> list[Document]:
    base_meta = {
        "policy_id": snapshot["policy_id"],
        "effective_date": snapshot["effective_date"],
        "source_hash": snapshot["source_hash"],
        "source_file": snapshot["source_file"],
        "doc_type": "policy_atom",
    }

    docs: list[Document] = []
    order = 0

    def add_doc(section: str, summary: str, lines: list[str]):
        nonlocal order
        content = summary.strip()
        if lines:
            content += "\n" + "\n".join(lines)
        docs.append(
            Document(
                page_content=content,
                metadata={**base_meta, "section": section, "order": order},
            )
        )
        order += 1

    # Eligibility atoms
    for idx, pathway in enumerate(snapshot["eligibility"]["pathways"], start=1):
        bmi_min = pathway["bmi_min"]
        bmi_max = pathway.get("bmi_max")
        bmi_clause = f"BMI ≥ {bmi_min}" if bmi_max is None else f"BMI ≥ {bmi_min} and < {bmi_max}"
        summary = (
            f"Policy: Adults ≥18 years qualify under {pathway['name']} when {bmi_clause} "
            f"and required diagnosis/comorbidity criteria are documented."
        )
        lines = [f"BMI requirement: {bmi_clause}"]
        if pathway.get("required_diagnosis_strings"):
            lines.append("Required diagnosis strings:")
            lines.extend([f"- {d}" for d in pathway["required_diagnosis_strings"]])
        if pathway.get("required_diagnosis_codes_e66"):
             lines.append("Required ICD-10 E66 Codes:")
             lines.extend([f"- {c}" for c in pathway["required_diagnosis_codes_e66"]])
        if pathway.get("required_diagnosis_codes_z68"):
             lines.append("Required ICD-10 Z68 Codes:")
             lines.extend([f"- {c}" for c in pathway["required_diagnosis_codes_z68"]])
        if pathway.get("required_comorbidity_categories"):
            lines.append("Required weight-related comorbidities (any one):")
            labels = [
                snapshot["comorbidities"][key]["label"]
                for key in pathway["required_comorbidity_categories"]
                if key in snapshot["comorbidities"]
            ]
            lines.extend([f"- {lbl}" for lbl in labels])
        add_doc(f"eligibility:pathway{idx}", summary, lines)

    # Diagnosis strings atoms
    add_doc(
        "diagnosis:obesity_strings",
        "Policy: Adult obesity diagnosis strings that satisfy BMI ≥ 30 pathway.",
        [f"- {d}" for d in snapshot["diagnosis_strings"]["adult_obesity"]],
    )
    add_doc(
        "diagnosis:overweight_strings",
        "Policy: Adult overweight/excess-weight diagnosis strings required for BMI 27–29.9 pathway.",
        [f"- {d}" for d in snapshot["diagnosis_strings"]["adult_excess_weight"]],
    )

    # Comorbidity atoms (stable order)
    comorb_order = ["hypertension", "type2_diabetes", "dyslipidemia", "obstructive_sleep_apnea", "cardiovascular_disease"]
    for key in comorb_order:
        if key not in snapshot["comorbidities"]:
            continue
        comorb = snapshot["comorbidities"][key]
        summary = f"Policy: {comorb['label']} counts as a weight-related comorbidity for BMI 27–29.9 pathway."
        lines = [f"- {s}" for s in comorb["accepted_strings"]]
        add_doc(f"comorbidity:{key}", summary, lines)

    # Safety exclusions atoms
    def safety_code(category: str) -> str:
        cat = category.lower()
        if "medullary thyroid" in cat:
            return "mtc_men2"
        if "multiple endocrine neoplasia" in cat:
            return "mtc_men2"
        if "pregnant" in cat or "nursing" in cat:
            return "pregnancy_nursing"
        if "hypersensitivity" in cat:
            return "hypersensitivity"
        if "pancreatitis" in cat:
            return "pancreatitis"
        if "suicid" in cat:
            return "suicidality"
        if "motility" in cat or "gastroparesis" in cat:
            return "gi_motility"
        if "glp-1" in cat:
            return "concurrent_glp1"
        return "safety_other"

    for exclusion in snapshot["safety_exclusions"]:
        code = safety_code(exclusion["category"])
        if code == "safety_other":
            continue
        summary = f"Policy: Safety exclusion for {exclusion['category']}."
        lines = [f"- {s}" for s in exclusion.get("accepted_strings", [])]
        add_doc(f"safety_exclusions:{code}", summary, lines)

    # Drug conflicts
    add_doc(
        "drug_conflicts:glp1_glp1_gip",
        "Policy: Concurrent GLP-1 or GLP-1/GIP agonist use is not allowed with Wegovy.",
        [f"- {d}" for d in snapshot["drug_conflicts"]["glp1_or_glp1_gip_agents"]],
    )

    # Documentation requirements
    add_doc(
        "documentation:requirements",
        "Policy: Documentation requirements for Wegovy prior authorization.",
        [f"- {req}" for req in snapshot["documentation_requirements"]],
    )

    # Ambiguities grouped
    ambiguity_groups = {
        "thyroid": [],
        "diabetes_borderline": [],
        "bp_borderline": [],
        "sleep_apnea_unclear": [],
        "other": [],
    }
    for amb in snapshot["ambiguities"]:
        pattern = amb["pattern"].lower()
        if "thyroid" in pattern:
            ambiguity_groups["thyroid"].append(amb)
        elif "diabetes" in pattern or "glucose" in pattern:
            ambiguity_groups["diabetes_borderline"].append(amb)
        elif "blood pressure" in pattern or "hypertension" in pattern:
            ambiguity_groups["bp_borderline"].append(amb)
        elif "sleep apnea" in pattern:
            ambiguity_groups["sleep_apnea_unclear"].append(amb)
        else:
            ambiguity_groups["other"].append(amb)

    for key, group in ambiguity_groups.items():
        if not group:
            continue
        summary = f"Policy: Ambiguity handling for {key.replace('_', ' ')} terms."
        lines = [
            f"- {amb['pattern']} -> {amb['action']}" + (f" ({amb['notes']})" if amb.get("notes") else "")
            for amb in group
        ]
        add_doc(f"ambiguity:{key}", summary, lines)

    return docs


def setup_vector_db():
    """Set up ChromaDB vector store with snapshot-driven policy documents."""
    snapshot = generate_snapshot()
    validated_snapshot = load_policy_snapshot(SNAPSHOT_PATH, POLICY_ID)
    validate_policy_snapshot(validated_snapshot)

    pdf_path = create_policy_pdf(validated_snapshot)
    PyPDFLoader(pdf_path).load()  # ensures PDF is readable for downstream loaders

    documents = _section_documents(validated_snapshot)
    logger.info(f"Generating embeddings with {EMBED_MODEL}")

    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)
    _ensure_embedding_dim(embedding_function)

    vectorstore = Chroma(
        collection_name="priorauth_policies",
        embedding_function=embedding_function,
        persist_directory="./chroma_db",
        collection_metadata={
            "policy_id": POLICY_ID,
            "effective_date": validated_snapshot["effective_date"],
            "source_hash": validated_snapshot["source_hash"],
        },
    )
    ids = [f"{doc.metadata['section']}:{doc.metadata['order']}" for doc in documents]
    try:
        vectorstore.delete(ids=ids)
    except Exception:
        pass
    vectorstore.add_documents(documents=documents, ids=ids)
    logger.info("Vector store updated successfully with Wegovy policy guidelines")
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"Using embedding model: {EMBED_MODEL}")
    logger.info("To use a different model, set PA_EMBED_MODEL environment variable")
    setup_vector_db()
