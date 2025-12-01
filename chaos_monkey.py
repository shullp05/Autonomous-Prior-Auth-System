import pandas as pd
import numpy as np
import random
from etl_pipeline import run_etl

def inject_complex_scenarios(df_p, df_c, df_m, df_o):
    print(f"--- INJECTING SCENARIOS INTO COHORT OF {len(df_p)} PATIENTS ---")
    
    # Target 15% of the total population for Wegovy requests
    sample_size = int(len(df_p) * 0.15)
    cohort_ids = df_p['patient_id'].sample(sample_size).values
    
    print(f"Generating {len(cohort_ids)} Wegovy Claims...")
    
    new_meds = []
    new_obs = []
    new_conds = []
    
    for pid in cohort_ids:
        # 1. Prescribe Wegovy
        new_meds.append({
            'patient_id': pid, 'medication_name': 'Wegovy 2.4 MG Injection', 
            'date': '2024-02-01', 'status': 'active'
        })
        
        roll = random.random()
        
        # --- SCENARIO A: THE TRAP (Safety Exclusion) --- (10%)
        # Patient is Obese (Eligible) BUT has Thyroid Cancer (Contraindicated)
        if roll < 0.10:
            bmi = random.uniform(32.0, 40.0)
            new_obs.append({'patient_id': pid, 'type': 'BMI', 'value': round(bmi, 1), 'unit': 'kg/m2', 'date': '2024-01-01'})
            # The Poison Pill
            new_conds.append({'patient_id': pid, 'condition_name': 'Malignant tumor of thyroid', 'onset_date': '2020-01-01'})

        # --- SCENARIO B: "Hidden" Obese (Data Gap) --- (20%)
        elif roll < 0.30:
            new_obs.append({'patient_id': pid, 'type': 'Height', 'value': 175.0, 'unit': 'cm', 'date': '2024-01-01'})
            new_obs.append({'patient_id': pid, 'type': 'Weight', 'value': 105.0, 'unit': 'kg', 'date': '2024-01-01'})
            
        # --- SCENARIO C: Borderline + Comorbidity (Approval) --- (30%)
        elif roll < 0.60:
            bmi = random.uniform(27.0, 29.9)
            new_obs.append({'patient_id': pid, 'type': 'BMI', 'value': round(bmi, 1), 'unit': 'kg/m2', 'date': '2024-01-01'})
            comorbs = ['Essential hypertension', 'Sleep apnea', 'Prediabetes', 'Dyslipidemia']
            new_conds.append({'patient_id': pid, 'condition_name': random.choice(comorbs), 'onset_date': '2023-01-01'})
            
        # --- SCENARIO D: Borderline Healthy (Denial) --- (20%)
        elif roll < 0.80:
             bmi = random.uniform(27.0, 29.9)
             new_obs.append({'patient_id': pid, 'type': 'BMI', 'value': round(bmi, 1), 'unit': 'kg/m2', 'date': '2024-01-01'})
             
        # --- SCENARIO E: Underweight (Denial) --- (20%)
        else:
            bmi = random.uniform(22.0, 26.9)
            new_obs.append({'patient_id': pid, 'type': 'BMI', 'value': round(bmi, 1), 'unit': 'kg/m2', 'date': '2024-01-01'})

    # Append Data
    df_m = pd.concat([df_m, pd.DataFrame(new_meds)], ignore_index=True)
    df_o = pd.concat([df_o, pd.DataFrame(new_obs)], ignore_index=True)
    if new_conds:
        df_c = pd.concat([df_c, pd.DataFrame(new_conds)], ignore_index=True)
        
    return df_p, df_c, df_m, df_o

if __name__ == "__main__":
    df_p, df_c, df_m, df_o = run_etl()
    if df_p.empty: exit()
    df_p, df_c, df_m, df_o = inject_complex_scenarios(df_p, df_c, df_m, df_o)
    
    df_p.to_csv("data_patients.csv", index=False)
    df_c.to_csv("data_conditions.csv", index=False)
    df_m.to_csv("data_medications.csv", index=False)
    df_o.to_csv("data_observations.csv", index=False)
    print("Data Generation 3.0 Complete.")