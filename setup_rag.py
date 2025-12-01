import os
from reportlab.pdfgen import canvas
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_complex_policy():
    filename = "Policy_Weight_Mgmt_2025.pdf"
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "PAYER POLICY: Chronic Weight Management (Wegovy/Saxenda)")
    c.drawString(100, 780, "Effective Date: 2025-01-01")
    
    text = [
        "CRITERIA FOR APPROVAL:",
        "1. Patient must have a calculated Body Mass Index (BMI) meeting one of the following:",
        "   a. BMI >= 30 kg/m2 (Obesity)",
        "   b. BMI >= 27 kg/m2 (Overweight) AND presence of at least one weight-related comorbidity:",
        "      - Hypertension",
        "      - Dyslipidemia (High Cholesterol)",
        "      - Obstructive Sleep Apnea",
        "      - Type 2 Diabetes",
        " ",
        "EXCLUSIONS / CONTRAINDICATIONS:",
        "1. Personal or family history of Medullary Thyroid Carcinoma (MTC).",
        "2. Multiple Endocrine Neoplasia syndrome type 2 (MEN 2).",
        "3. Pregnancy.",
    ]
    
    y = 750
    for line in text:
        c.drawString(100, y, line)
        y -= 20
        
    c.save()
    print(f"Generated Policy Document: {filename}")
    return filename

def setup_vector_db():
    pdf_path = create_complex_policy()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    print("--- Generating Embeddings (all-minilm:latest) ---")
    embedding_function = OllamaEmbeddings(model="all-minilm:latest")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    print("Vector Store Updated Successfully.")
    return vectorstore

if __name__ == "__main__":
    setup_vector_db()