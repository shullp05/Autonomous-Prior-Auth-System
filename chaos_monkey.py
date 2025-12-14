import logging
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CLAIM_RATE
from etl_pipeline import run_etl

logger = logging.getLogger(__name__)


def inject_complex_scenarios(
    df_p: pd.DataFrame,
    df_c: pd.DataFrame,
    df_m: pd.DataFrame,
    df_o: pd.DataFrame,
    *,
    claim_rate: float = DEFAULT_CLAIM_RATE,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inject synthetic Wegovy prior-auth scenarios aligned to agent_logic.py:

      - Approvals:
          * BMI >= 30, or
          * BMI 27–29.9 + QUALIFYING comorbidity (HTN, Dyslipidemia, T2DM, OSA, CVD)
      - Denials:
          * BMI < 27, or
          * BMI 27–29.9 with no qualifying comorbidity
      - Safety denials:
          * MTC/MEN2, pregnancy/nursing, concurrent GLP-1
      - Manual review:
          * ambiguous terms like Prediabetes, generic Sleep apnea, Elevated BP, Obesity term,
            generic thyroid malignancy terms without medullary specification.

    Key fix: Ambiguous terms are NOT treated as “ground truth eligibility positives.”
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    n_pat = len(df_p)
    logger.info(f"Injecting scenarios into cohort of {n_pat} patients")
    if n_pat == 0:
        return df_p, df_c, df_m, df_o

    sample_size = max(1, int(round(n_pat * claim_rate)))
    cohort_ids = df_p["patient_id"].dropna().astype(str).sample(sample_size, random_state=seed).values.tolist()
    logger.info(f"Generating {len(cohort_ids)} Wegovy claims")

    new_meds: List[dict] = []
    new_obs: List[dict] = []
    new_conds: List[dict] = []

    scenarios = [
        ("SAFETY_MTC_MEN2", 0.10),
        ("SAFETY_DUPLICATE_GLP1", 0.10),
        ("SAFETY_PREGNANCY_NURSING", 0.05),
        ("DATA_GAP_HEIGHT_WEIGHT_ONLY", 0.20),
        ("OVERWEIGHT_VALID_COMORBIDITY", 0.25),
        ("OVERWEIGHT_AMBIGUOUS_TERM", 0.15),
        ("OVERWEIGHT_NO_COMORBIDITY", 0.10),
        ("BMI_UNDER_27", 0.05),
    ]
    labels, probs = zip(*scenarios)

    def add_wegovy(pid: str):
        new_meds.append(
            {
                "patient_id": pid,
                "medication_name": "Wegovy 2.4 MG Injection",
                "rx_code": "",
                "date": "2024-02-01",
                "status": "active",
            }
        )

    def add_bmi(pid: str, bmi: float, date: str = "2024-01-01"):
        new_obs.append(
            {"patient_id": pid, "type": "BMI", "value": round(float(bmi), 1), "unit": "kg/m2", "date": date}
        )

    def add_height_weight(pid: str, height_cm: float, weight_kg: float, date: str = "2024-01-01"):
        new_obs.append({"patient_id": pid, "type": "Height", "value": round(float(height_cm), 1), "unit": "cm", "date": date})
        new_obs.append({"patient_id": pid, "type": "Weight", "value": round(float(weight_kg), 1), "unit": "kg", "date": date})

    def add_condition(pid: str, name: str, onset: str = "2023-01-01"):
        new_conds.append(
            {
                "patient_id": pid,
                "condition_name": name,
                "code": "",
                "onset_date": onset,
            }
        )

    def add_concurrent_glp1(pid: str):
        new_meds.append(
            {
                "patient_id": pid,
                "medication_name": "Ozempic (semaglutide) injection",
                "rx_code": "",
                "date": "2024-01-15",
                "status": "active",
            }
        )

    for pid in cohort_ids:
        add_wegovy(pid)

        scenario = rng.choices(labels, probs, k=1)[0]

        gender = ""
        try:
            gender = (
                df_p.loc[df_p["patient_id"].astype(str) == str(pid), "gender"]
                .astype(str)
                .iloc[0]
                .lower()
            )
        except Exception:
            gender = ""

        if scenario == "SAFETY_MTC_MEN2":
            add_bmi(pid, rng.uniform(32.0, 40.0))
            add_condition(
                pid,
                rng.choice(
                    ["Medullary Thyroid Carcinoma (MTC)", "MTC", "Multiple Endocrine Neoplasia type 2 (MEN2)"]
                ),
                onset="2020-01-01",
            )

        elif scenario == "SAFETY_DUPLICATE_GLP1":
            add_bmi(pid, rng.uniform(30.0, 38.0))
            add_concurrent_glp1(pid)

        elif scenario == "SAFETY_PREGNANCY_NURSING":
            add_bmi(pid, rng.uniform(27.0, 36.0))
            if "female" in gender:
                add_condition(pid, rng.choice(["Pregnant", "Currently pregnant", "Breastfeeding", "Nursing"]), onset="2024-01-10")
            else:
                add_concurrent_glp1(pid)

        elif scenario == "DATA_GAP_HEIGHT_WEIGHT_ONLY":
            add_height_weight(pid, height_cm=175.0, weight_kg=105.0)

        elif scenario == "OVERWEIGHT_VALID_COMORBIDITY":
            add_bmi(pid, rng.uniform(27.0, 29.9))
            qualifying = [
                "Essential hypertension",
                "Hypertension",
                "Hyperlipidemia",
                "Dyslipidemia",
                "Type 2 diabetes mellitus",
                "T2DM",
                "Obstructive Sleep Apnea",
                "OSA",
                "Coronary Artery Disease",
                "Myocardial Infarction",
                "Ischemic Stroke",
                "Peripheral Arterial Disease",
            ]
            add_condition(pid, rng.choice(qualifying))

        elif scenario == "OVERWEIGHT_AMBIGUOUS_TERM":
            add_bmi(pid, rng.uniform(27.0, 29.9))
            ambiguous = [
                "Prediabetes",
                "Impaired fasting glucose",
                "Elevated BP",
                "Borderline hypertension",
                "Obesity",
                "Body mass index 30+",
                "Sleep apnea",
                "Malignant tumor of thyroid",
            ]
            add_condition(pid, rng.choice(ambiguous))

        elif scenario == "OVERWEIGHT_NO_COMORBIDITY":
            add_bmi(pid, rng.uniform(27.0, 29.9))

        elif scenario == "BMI_UNDER_27":
            add_bmi(pid, rng.uniform(22.0, 26.9))

        else:
            add_bmi(pid, rng.uniform(27.0, 29.9))

    if new_meds:
        df_m = pd.concat([df_m, pd.DataFrame(new_meds)], ignore_index=True)
    if new_obs:
        df_o = pd.concat([df_o, pd.DataFrame(new_obs)], ignore_index=True)
    if new_conds:
        df_c = pd.concat([df_c, pd.DataFrame(new_conds)], ignore_index=True)

    return df_p, df_c, df_m, df_o


if __name__ == "__main__":
    # Configure logging for standalone runs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df_p, df_c, df_m, df_o = run_etl()
    if df_p.empty:
        raise SystemExit("No patients found. Check your FHIR output directory and rerun.")

    df_p, df_c, df_m, df_o = inject_complex_scenarios(df_p, df_c, df_m, df_o)

    df_p.to_csv("output/data_patients.csv", index=False)
    df_c.to_csv("output/data_conditions.csv", index=False)
    df_m.to_csv("output/data_medications.csv", index=False)
    df_o.to_csv("output/data_observations.csv", index=False)

    logger.info("Data generation complete")
