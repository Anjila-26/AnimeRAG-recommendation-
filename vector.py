from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import hashlib
import os
import pandas as pd
import time
from tqdm import tqdm  # For progress bar

# Start overall timer
start_time = time.time()

df = pd.read_csv("animedata.csv")
df = df.dropna(subset=["name","Type","Plot Summary","Genre"])
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Initialize Chroma
vector_store = Chroma(
    collection_name="anime",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    print("Creating new vector database...")
    documents = []
    total_rows = len(df)
    
    # Document creation with progress bar
    doc_start_time = time.time()
    for i, row in tqdm(df.iterrows(), total=total_rows, desc="Processing documents"):
        unique_id = hashlib.md5(f"{row['name']}{row['Type']}".encode()).hexdigest()

        document = Document(
            page_content=(
                f"Title: {row['name']}\n"
                f"Type: {row['Type']}\n"
                f"Genres: {row['Genre']}\n"
                f"Plot: {row['Plot Summary']}"
            ),
            metadata={
                "title": row["name"],
                "type": row["Type"],
                "genres": row["Genre"],
                "status": row["Status"],
                "aliases": row["Other name"],
                "id": unique_id,
                "original_index": i
            }
        )
        documents.append(document)
    
    print(f"Document creation completed in {time.time() - doc_start_time:.2f} seconds")
    
    # Add documents in batches with progress tracking
    batch_size = 100
    add_start_time = time.time()
    for i in tqdm(range(0, len(documents), batch_size), 
                 desc="Adding documents to vector store",
                 total=len(documents)//batch_size + 1):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"Added batch {i//batch_size + 1}: {len(batch)} documents ({i + len(batch)}/{len(documents)})")
    
    print(f"Document addition completed in {time.time() - add_start_time:.2f} seconds")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
    return_metadata=True,
    search_type="similarity"
)

# Final stats
total_time = time.time() - start_time
doc_count = len(documents) if add_documents else "existing"
print(f"\nDatabase initialization complete!")
print(f"Total documents: {doc_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Database location: {os.path.abspath(db_location)}")