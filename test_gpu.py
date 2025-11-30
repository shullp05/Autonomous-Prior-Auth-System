

from langchain_ollama import ChatOllama



print("--- Testing GPU Acceleration ---")

llm = ChatOllama(model="llama3.2:3b", temperature=0)



# A complex query to force the GPU to work

response = llm.invoke("Explain the pathophysiology of Type 2 Diabetes in 2 sentences.")



print("\nResponse:")

print(response.content)

print("\n--- SUCCESS: Setup Complete ---")


