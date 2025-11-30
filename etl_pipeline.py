import json
import glob
import pandas as pd
import os

FHIR_DIR = "./output/fhir/*.json"

def parse_fhir_bundle(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    patient_id = None
    demographics = {}
    conditions = []
    medications = []
    observations = [] 

    for entry in data.get('entry', []):
        resource = entry['resource']
        r_type = resource['resourceType']

        if r_type == 'Patient':
            patient_id = resource['id']
            demographics = {
                'patient_id': patient_id,
                'name': f"{resource['name'][0]['given'][0]} {resource['name'][0]['family']}",
                'gender': resource['gender'],
                'dob': resource['birthDate']
            }
        
        elif r_type == 'Condition':
            code_data = resource.get('code', {}).get('coding', [{}])[0]
            conditions.append({
                'patient_id': patient_id,
                'condition_name': code_data.get('display'),
                'code': code_data.get('code'),
                'onset_date': resource.get('onsetDateTime', '')[:10]
            })
            
        elif r_type == 'MedicationRequest':
            med_data = resource.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
            medications.append({
                'patient_id': patient_id,
                'medication_name': med_data.get('display'),
                'rx_code': med_data.get('code'),
                'date': resource.get('authoredOn', '')[:10],
                'status': resource.get('status')
            })
            
        # NEW: Extract BMI, Height, and Weight
        elif r_type == 'Observation':
            code_data = resource.get('code', {}).get('coding', [{}])[0]
            code = code_data.get('code')
            
            # 39156-5 = BMI
            # 29463-7 = Body Weight
            # 8302-2  = Body Height
            target_codes = ['39156-5', '29463-7', '8302-2']
            
            if code in target_codes and 'valueQuantity' in resource:
                # Map codes to readable types
                obs_type = "Unknown"
                if code == '39156-5': obs_type = 'BMI'
                elif code == '29463-7': obs_type = 'Weight'
                elif code == '8302-2': obs_type = 'Height'

                observations.append({
                    'patient_id': patient_id,
                    'type': obs_type,
                    'value': round(resource['valueQuantity']['value'], 2),
                    'unit': resource['valueQuantity']['unit'],
                    'date': resource.get('effectiveDateTime', '')[:10]
                })

    return demographics, conditions, medications, observations

def run_etl():
    print(f"Scanning for FHIR files in {FHIR_DIR}...")
    files = glob.glob(FHIR_DIR)
    
    if not files:
        print("ERROR: No FHIR files found. Please run Synthea generation first.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_demographics, all_conditions, all_medications, all_observations = [], [], [], []
    
    for f in files:
        demo, cond, meds, obs = parse_fhir_bundle(f)
        if demo:
            # Backfill Patient ID
            for c in cond: c['patient_id'] = demo['patient_id']
            for m in meds: m['patient_id'] = demo['patient_id']
            for o in obs: o['patient_id'] = demo['patient_id']
            
            all_demographics.append(demo)
            all_conditions.extend(cond)
            all_medications.extend(meds)
            all_observations.extend(obs)

    return (pd.DataFrame(all_demographics), 
            pd.DataFrame(all_conditions), 
            pd.DataFrame(all_medications), 
            pd.DataFrame(all_observations))

if __name__ == "__main__":
    df_pat, df_con, df_med, df_obs = run_etl()
    print(f"ETL Complete. Processed {len(df_pat)} patients.")