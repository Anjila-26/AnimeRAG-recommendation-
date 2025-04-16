#Import the Langchain libraries
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

#import the Model
model = OllamaLLM(model="llama3.2:1b")

#Create a prompt template
template = """
You are an Otaku assistant specialized in anime and manga recommendations. 
Your task is to recommend similar anime based on the user's preferences.

Given these anime entries:
{context}

User query: {query}

Provide 3-5 recommendations in this format:
- **Title**: [name]
  - **Type**: [TV/Movie/OVA]
  - **Genre**: [genre]
  - **Why Recommended**: [brief explanation]

Focus only on anime recommendations and avoid any off-topic responses.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


#Result
while True:
    print("\n\n------------------------------------")
    question = input("Tell me your loved anime and I will recomment something similar (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break

    retrieved_docs = retriever.invoke(question)
    print("Retrieved docs:", retrieved_docs)  # Check if documents are being retrieved

    result = chain.invoke({"context" : retrieved_docs, "query": question})
    print(result)