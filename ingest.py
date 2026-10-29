"""
Ingestion script: Creates the Pinecone index and seeds sample institute documents.

Usage:
    python ingest.py
"""

import os
import time

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "institute-knowledge-base"
DIMENSION = 1024
METRIC = "cosine"

# --- Step 1: Create the index if it doesn't exist ---
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}' (dims={DIMENSION}, metric={METRIC})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait for the index to be ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("  Waiting for index to be ready...")
        time.sleep(2)
    print(f"Index '{INDEX_NAME}' created and ready!")
else:
    print(f"Index '{INDEX_NAME}' already exists, skipping creation.")

index = pc.Index(INDEX_NAME)

# --- Step 2: Sample institute document chunks ---
docs = [
    {
        "id": "doc_1",
        "text": "The ECE department labs are located on the 2nd floor of the Academic Block. Timings are 9 AM to 5 PM.",
        "dept": "ECE",
    },
    {
        "id": "doc_2",
        "text": "Hostel mess fee for the semester is 22,000 INR and must be submitted to the accounts section before term starts.",
        "dept": "Administration",
    },
    {
        "id": "doc_3",
        "text": "To apply for provisional transcripts or grade sheets, submit an application to the Academic Section signed by the Faculty-in-Charge.",
        "dept": "Academics",
    },
]

# --- Step 3: Embed and upsert ---
print(f"Embedding {len(docs)} documents with Cohere embed-english-v3.0...")
vectors = []
for d in docs:
    emb = co.embed(
        texts=[d["text"]],
        model="embed-english-v3.0",
        input_type="search_document",
    ).embeddings[0]
    vectors.append(
        {
            "id": d["id"],
            "values": emb,
            "metadata": {"text": d["text"], "department": d["dept"]},
        }
    )

index.upsert(vectors=vectors)
print("Data seeded successfully!")

# --- Step 4: Quick verification ---
time.sleep(2)  # brief pause for indexing
stats = index.describe_index_stats()
print(f"Index stats: {stats}")
