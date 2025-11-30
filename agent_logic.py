import pandas as pd
import json
import os
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_chroma import Chroma

# --- 1. DATA LOADING ---
try:
    if not os.path.exists("data_patients.csv"):
        raise FileNotFoundError("Data missing. Run chaos_monkey.py")
    df_patients = pd.read_csv("data_patients.csv")
    df_meds = pd.read_csv("data_medications.csv")
    df_conditions = pd.read_csv("data_conditions.csv")
    df_obs = pd.read_csv("data_observations.csv") 
except Exception as e:
    sys.exit(f"Data Error: {e}")

def calculate_bmi_if_missing(patient_id, df_obs):
    """
    Python Logic Layer:
    If BMI is missing, calculate it from Height/Weight history.
    """
    p_obs = df_obs[df_obs['patient_id'] == patient_id]
    
    # 1. Try explicit BMI first
    bmi_rows = p_obs[p_obs['type'] == 'BMI'].sort_values('date', ascending=False)
    if not bmi_rows.empty:
        return f"{bmi_rows.iloc[0]['value']} (Source: EMR Record)"
    
    # 2. Fallback: Calculation
    try:
        wt_rows = p_obs[p_obs['type'] == 'Weight'].sort_values('date', ascending=False)
        ht_rows = p_obs[p_obs['type'] == 'Height'].sort_values('date', ascending=False)
        
        if wt_rows.empty or ht_rows.empty:
            return "Not Found (Data Missing)"
            
        weight_kg = wt_rows.iloc[0]['value']
        height_cm = ht_rows.iloc[0]['value']
        
        # Formula: kg / m^2
        # Convert cm to m
        height_m = height_cm / 100.0
        if height_m == 0: return "Error (Height is 0)"
        
        calculated_bmi = round(weight_kg / (height_m ** 2), 1)
        
        return f"{calculated_bmi} (Source: AI Calculated from Ht/Wt)"
        
    except Exception as e:
        return f"Calculation Error"

def look_up_patient_data(patient_id):
    pat_rows = df_patients[df_patients['patient_id'] == patient_id].to_dict('records')
    if not pat_rows: return None
    pat = pat_rows[0]
    meds = df_meds[df_meds['patient_id'] == patient_id]['medication_name'].tolist()
    conds = df_conditions[df_conditions['patient_id'] == patient_id]['condition_name'].tolist()
    
    # SMART LOOKUP
    latest_bmi = calculate_bmi_if_missing(patient_id, df_obs)
    
    return {
        "name": pat['name'], 
        "meds": meds, 
        "conditions": conds,
        "latest_bmi": latest_bmi
    }

class AgentState(TypedDict):
    patient_id: str
    drug_requested: str
    patient_data: dict
    policy_text: str
    audit_findings: dict
    final_decision: str
    reasoning: str

# --- NODE: RETRIEVAL ---
def retrieve_policy(state: AgentState):
    print(f"--- [Node: RAG] Retrieving Policy for {state['drug_requested']} ---")
    embedding_function = OllamaEmbeddings(model="all-minilm:latest")
    
    if not os.path.exists("./chroma_db"):
        return {"policy_text": "ERROR: DB Missing"}

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.invoke(f"criteria for {state['drug_requested']}")
    return {"policy_text": docs[0].page_content if docs else "No policy found."}

# --- NODE: CLINICAL AUDIT (Mistral Nemo + Safety Guardrails) ---
def clinical_audit(state: AgentState):
    print(f"--- [Node: Audit] Checking Patient {state['patient_id']} ---")
    p_data = look_up_patient_data(state['patient_id'])
    
    llm = ChatOllama(model="mistral-nemo", temperature=0)
    
    # ADVANCED PROMPT ENGINEERING:
    # 1. Negative Constraints (STOP commands)
    # 2. Hierarchical Logic (Safety > Eligibility)
    
    system_prompt = f"""
    You are an expert Medical Director auditing a Prior Authorization.
    
    ### PROTOCOL: SAFETY FIRST
    Before evaluating BMI, you MUST check for CONTRAINDICATIONS.
    
    ### CONTRAINDICATIONS (Absolute Denials)
    - Medullary Thyroid Carcinoma (MTC)
    - Thyroid Cancer / Malignant Tumor of Thyroid
    - Multiple Endocrine Neoplasia (MEN 2)
    
    IF any contraindication is found:
       - STOP immediately.
       - Verdict: DENIED.
       - Reasoning: "Safety Exclusion: [Condition Name] found."

    ### ELIGIBILITY RULES (Only if Safety Check Passes)
    1. BMI >= 30: APPROVE.
    2. BMI 27-29.9 + Comorbidity (Hypertension, Diabetes, Apnea, Lipids): APPROVE.
    3. BMI < 27: DENY.
    4. BMI 27-29.9 without Comorbidity: DENY.

    ### PATIENT DATA
    - BMI: {p_data['latest_bmi']}
    - Conditions: {p_data['conditions']}
    
    ### OUTPUT FORMAT (JSON)
    {{
        "safety_check": "PASS" or "FAIL",
        "contraindication_found": "Name of condition" or "None",
        "bmi_value": number or string,
        "category": "Obese" or "Overweight" or "Normal",
        "approval_status": "APPROVED" or "DENIED",
        "reasoning": "Step-by-step logic."
    }}
    """
    
    try:
        response = llm.invoke(system_prompt)
        content = response.content.strip()
        print(f"\n[MISTRAL REASONING]:\n{content}\n") 

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        # Logic to handle Safety Fails explicitly
        status = data.get("approval_status", "DENIED")
        if data.get("safety_check") == "FAIL":
            status = "DENIED" # Force denial just in case
            
        audit_result = {
            "status": status,
            "notes": data.get("reasoning", "No reasoning provided.")
        }
        
    except Exception as e:
        print(f"Error: {e}")
        audit_result = {"status": "DENIED", "notes": "System Error"}
        
    return {"patient_data": p_data, "audit_findings": audit_result}
# --- NODE: DECISION ---
def make_decision(state: AgentState):
    f = state['audit_findings']
    return {"final_decision": f['status'], "reasoning": f['notes']}

# --- GRAPH ---
def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve_policy", retrieve_policy)
    workflow.add_node("clinical_audit", clinical_audit)
    workflow.add_node("make_decision", make_decision)
    workflow.set_entry_point("retrieve_policy")
    workflow.add_edge("retrieve_policy", "clinical_audit")
    workflow.add_edge("clinical_audit", "make_decision")
    workflow.add_edge("make_decision", END)
    return workflow.compile()

if __name__ == "__main__":
    app = build_agent()
    # Find a Wegovy patient
    try:
        target = df_meds[df_meds['medication_name'].str.contains("Wegovy", na=False)].iloc[0]['patient_id']
        print(f"--- TESTING WEGOVY LOGIC FOR: {target} ---")
        res = app.invoke({"patient_id": target, "drug_requested": "Wegovy"})
        print(f"RESULT: {res['final_decision']} | {res['reasoning']}")
    except IndexError:
        print("No Wegovy patients found. Run chaos_monkey.py.")