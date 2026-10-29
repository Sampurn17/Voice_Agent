"""
Real File Ingestion Script.

Reads PDFs, text, and markdown files from the data/ directory,
chunks them, embeds via Cohere, and upserts to Pinecone.

Usage:
    python -m src.scripts.ingest_real_files
"""

import os
import uuid

import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("institute-knowledge-base")

DATA_DIR = "data"


def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        dept_label = os.path.basename(root)
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                loaded = loader.load()
            elif file.endswith(".txt") or file.endswith(".md"):
                loader = TextLoader(file_path, encoding="utf-8")
                loaded = loader.load()
            else:
                continue

            for doc in loaded:
                doc.metadata["department"] = (
                    dept_label.capitalize() if dept_label != "data" else "General"
                )
                docs.append(doc)

            print(f"  Loaded: {file_path} ({len(loaded)} page(s)) -> dept={dept_label.capitalize()}")
    return docs


def run_pipeline():
    raw_docs = load_documents()
    if not raw_docs:
        print("No documents found in data/ folder!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=60, separators=["\n\n", "\n", ". ", " "]
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"\nPrepared {len(split_docs)} document chunks.")

    batch_size = 64
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        texts = [d.page_content for d in batch]

        res = co.embed(
            texts=texts, model="embed-english-v3.0", input_type="search_document"
        )

        vectors = []
        for j, d in enumerate(batch):
            vectors.append({
                "id": f"doc_{uuid.uuid4().hex[:10]}",
                "values": res.embeddings[j],
                "metadata": {
                    "text": d.page_content,
                    "department": d.metadata.get("department", "General"),
                },
            })

        index.upsert(vectors=vectors)
        print(f"Indexed batch {i // batch_size + 1} ({len(batch)} vectors)")

    print("\nAll real institute documents successfully synced to Pinecone.")


if __name__ == "__main__":
    run_pipeline()
