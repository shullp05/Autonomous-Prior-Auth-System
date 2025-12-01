import pandas as pd
import json
import os
import sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings 
from langchain_chroma import Chroma

# --- 1. ROBUST DATA LOADING ---
try:
    if not os.path.exists("data_patients.csv"):
        raise FileNotFoundError("Data CSVs not found. Please run 'chaos_monkey.py' first.")
        
    df_patients = pd.read_csv("data_patients.csv")
    df_meds = pd.read_csv("data_medications.csv")
    df_conditions = pd.read_csv("data_conditions.csv")
except Exception as e:
    print(f"CRITICAL ERROR: Data loading failed. {e}")
    sys.exit(1)

def look_up_patient_data(patient_id):
    """Retrieves patient history from the 'EMR' (CSVs)."""
    pat_rows = df_patients[df_patients['patient_id'] == patient_id].to_dict('records')
    if not pat_rows: 
        return None
    
    pat = pat_rows[0]
    meds = df_meds[df_meds['patient_id'] == patient_id]['medication_name'].tolist()
    conds = df_conditions[df_conditions['patient_id'] == patient_id]['condition_name'].tolist()
    
    return {
        "name": pat['name'],
        "dob": pat['dob'],
        "medication_history": meds,
        "conditions": conds
    }

# --- 2. STATE DEFINITION ---
class AgentState(TypedDict):
    patient_id: str
    drug_requested: str
    patient_data: dict
    policy_text: str
    audit_findings: dict
    final_decision: str
    reasoning: str

# --- 3. NODE: RETRIEVAL (RAG) ---
def retrieve_policy(state: AgentState):
    print(f"--- [Node: RAG] Retrieving Policy for {state['drug_requested']} ---")
    
    # UPDATED MODEL TAG: Must match setup_rag.py exactly
    embedding_function = OllamaEmbeddings(model="all-minilm:latest")
    
    if not os.path.exists("./chroma_db"):
        return {"policy_text": "ERROR: Policy Database not found. Run setup_rag.py."}

    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    try:
        docs = retriever.invoke(f"policy criteria for {state['drug_requested']}")
        policy_text = docs[0].page_content if docs else "No policy found."
    except Exception as e:
        print(f"RAG Error: {e}")
        policy_text = "Policy Retrieval Failed."
        
    return {"policy_text": policy_text}

# --- 4. NODE: CLINICAL AUDIT (The "Brain") ---
def clinical_audit(state: AgentState):
    print(f"--- [Node: Audit] Checking Patient {state['patient_id']} ---")
    
    p_data = look_up_patient_data(state['patient_id'])
    if not p_data:
        return {"audit_findings": {"has_diagnosis": False, "has_step_therapy": False, "notes": "Patient ID not found in EMR."}}

    policy = state['policy_text']
    
    # UPDATED MODEL TAG: Explicitly using 3b
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    
    system_prompt = f"""
    You are an expert Clinical Pharmacist Auditor.
    
    PAYER POLICY RULES:
    1. Diagnosis Check: Patient MUST have Type 2 Diabetes (or Metabolic Syndrome).
    2. Step Therapy Check: Patient MUST have a history of Metformin (or Glucophage).

    PATIENT RECORD:
    - Condition List: {p_data['conditions']}
    - Medication History: {p_data['medication_history']}
    
    INSTRUCTIONS:
    Review the patient record against the rules.
    Return a SINGLE JSON OBJECT. Do not use Markdown.
    
    JSON FORMAT:
    {{
        "diagnosis_met": "YES" or "NO",
        "step_therapy_met": "YES" or "NO",
        "reasoning": "Brief 1 sentence explanation citing specific dates or drugs found."
    }}
    """
    
    try:
        response = llm.invoke(system_prompt)
        content = response.content.strip()

        # Debug Output
        print(f"\n[LLM RAW OUTPUT]: {content}\n")

        # Parsing Logic
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        # Normalize Keys
        diag_str = str(data.get("diagnosis_met", "NO")).upper()
        step_str = str(data.get("step_therapy_met", "NO")).upper()
        
        # Convert to Python Booleans
        has_diag = (diag_str == "YES") or (diag_str == "TRUE")
        has_step = (step_str == "YES") or (step_str == "TRUE")
        
        audit_result = {
            "has_diagnosis": has_diag,
            "has_step_therapy": has_step,
            "notes": data.get("reasoning", "No reasoning provided.")
        }
        
    except json.JSONDecodeError:
        print("[ERROR] Failed to parse JSON from LLM.")
        audit_result = {"has_diagnosis": False, "has_step_therapy": False, "notes": "AI Output Error (Format)"}
    except Exception as e:
        print(f"[ERROR] Unexpected error in audit: {e}")
        audit_result = {"has_diagnosis": False, "has_step_therapy": False, "notes": f"System Error: {str(e)}"}
        
    return {"patient_data": p_data, "audit_findings": audit_result}

# --- 5. NODE: DECISION ENGINE ---
def make_decision(state: AgentState):
    f = state['audit_findings']
    
    # Deterministic Business Logic
    if f['has_diagnosis'] and f['has_step_therapy']:
        decision = "APPROVED"
        reason = f"Approved. {f['notes']}"
    else:
        decision = "DENIED"
        missing = []
        if not f['has_diagnosis']: missing.append("Missing Diagnosis")
        if not f['has_step_therapy']: missing.append("Missing Step Therapy")
        
        reason = f"Denied due to: {', '.join(missing)}. AI Findings: {f['notes']}"
        
    return {"final_decision": decision, "reasoning": reason}

# --- 6. GRAPH BUILDER ---
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

# --- 7. MAIN TESTER ---
if __name__ == "__main__":
    app = build_agent()
    test_id = df_patients.iloc[0]['patient_id']
    
    print(f"--- STARTING SINGLE PATIENT TEST: {test_id} ---")
    res = app.invoke({"patient_id": test_id, "drug_requested": "Ozempic"})
    
    print(f"\n=== FINAL VERDICT ===")
    print(f"DECISION: {res['final_decision']}")
    print(f"REASON:   {res['reasoning']}")
    print("=====================")
