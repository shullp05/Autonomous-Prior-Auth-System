import pandas as pd
import json
import time
from agent_logic import build_agent

# Load Data
df_meds = pd.read_csv("data_medications.csv")

# Filter for Wegovy
target_meds = df_meds[df_meds['medication_name'].str.contains('Wegovy', case=False, na=False)]
target_ids = target_meds['patient_id'].unique()

print(f"--- BATCH STARTING: Found {len(target_ids)} Wegovy claims to audit ---")

results = []
agent = build_agent()

def process_patient(pid):
    start_time = time.time()
    
    try:
        response = agent.invoke({"patient_id": pid, "drug_requested": "Wegovy"})
        decision = response['final_decision']
        reason = response['reasoning']
    except Exception as e:
        decision = "ERROR"
        reason = str(e)
        
    duration = time.time() - start_time
    
    # Wegovy cost ~ $1,350
    claim_value = 1350.00 
    
    return {
        "patient_id": pid,
        "status": decision,
        "reason": reason,
        "value": claim_value,
        "duration_sec": round(duration, 2)
    }

for i, pid in enumerate(target_ids):
    res = process_patient(pid)
    results.append(res)
    print(f"[{i+1}/{len(target_ids)}] Patient {pid} -> {res['status']}")

# Save
with open('dashboard_data.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nBatch Complete. Results saved to 'dashboard_data.json'.")