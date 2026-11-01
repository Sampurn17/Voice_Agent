"""
Real File Ingestion Script.

Reads PDFs, text, and markdown files from the data/ directory,
chunks them, embeds via Cohere, and upserts to Pinecone.

Usage:
    python -m src.scripts.ingest_real_files
"""

import os
import sys
import uuid
import time

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "institute-knowledge-base"
DIMENSION = 1024
METRIC = "cosine"
DATA_DIR = "data"

CATEGORY_MAP = {
    "academics": "Academics",
    "administration": "Administration",
    "hostels": "Hostels",
    "student_life": "Student Life",
    "placement": "Placement",
    "data": "General",
}


def ensure_index():
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' (dims={DIMENSION}) ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while True:
            desc = pc.describe_index(INDEX_NAME)
            status = getattr(desc, "status", {})
            ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
            if ready:
                break
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print(f"Index '{INDEX_NAME}' created and ready!")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)


def resolve_category(folder_name: str) -> str:
    """Map a folder name to a human-readable category label."""
    return CATEGORY_MAP.get(folder_name.lower(), folder_name.replace("_", " ").title())


def detect_branch_and_title(file_name: str) -> str:
    """Detect branch and document subject to inject into chunk context."""
    fname = file_name.upper()
    if "CSECY" in fname:
        return "B.Tech Computer Science & Engineering (Cyber Security - CY)"
    elif "CSEDS" in fname:
        return "B.Tech Computer Science & Engineering (Data Science - DS)"
    elif "GCSE" in fname or "CSE" in fname:
        return "B.Tech Computer Science & Engineering (CSE)"
    elif "GECE" in fname or "ECE" in fname:
        return "B.Tech Electronics & Communication Engineering (ECE)"
    elif "ORDINANCE" in fname:
        return "Academic Ordinances & Rules"
    elif "PLACEMENT" in fname:
        return "Training & Placement (T&P) Cell & Statistics"
    elif "PG" in fname:
        return "Postgraduate (M.Tech) Curriculum"
    return "IIIT Una Institute Information"


def load_documents():
    """Walk the data/ tree and load every supported file."""
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        category = resolve_category(folder_name)

        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_lower = file.lower()
            tag = detect_branch_and_title(file)

            try:
                if file_lower.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    loaded = loader.load()
                elif file_lower.endswith((".txt", ".md")):
                    loader = TextLoader(file_path, encoding="utf-8")
                    loaded = loader.load()
                else:
                    continue

                for doc in loaded:
                    doc.metadata["department"] = category
                    doc.metadata["source_file"] = file
                    doc.metadata["branch_title"] = tag
                    doc.page_content = f"[{tag} | {file}]\n{doc.page_content}"

                docs.extend(loaded)
                print(f"  [OK] Loaded: {file} ({len(loaded)} page(s)) -> {tag}")

            except Exception as e:
                print(f"  [ERROR] Failed to load {file_path}: {e}")

    return docs


def run_pipeline():
    """Main ingestion pipeline: load -> chunk -> embed -> upsert."""
    index = ensure_index()

    print("\n1. Loading all PDF and text documents from data/ ...")
    raw_docs = load_documents()
    if not raw_docs:
        print("No documents found in data/ folder!")
        return

    print("\n2. Chunking documents with RecursiveCharacterTextSplitter...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=150,  # Increased overlap to preserve continuity
        separators=["\n\n", "\n", ". ", " "],
    )
    split_docs = splitter.split_documents(raw_docs)
    split_docs = [d for d in split_docs if d.page_content and d.page_content.strip()]
    print(f"Prepared {len(split_docs)} document chunks from {len(raw_docs)} source pages.\n")

    batch_size = 64
    total_batches = (len(split_docs) + batch_size - 1) // batch_size

    print(f"3. Embedding and upserting into Pinecone in {total_batches} batches...")
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        texts = [d.page_content for d in batch]

        # Retry logic for Cohere embedding with backoff
        for attempt in range(5):
            try:
                res = co.embed(
                    texts=texts,
                    model="embed-english-v3.0",
                    input_type="search_document",
                )
                break
            except Exception as e:
                print(f"  Warning: Cohere embed retry {attempt + 1}/5 ({e}). Waiting 3s...")
                time.sleep(3)
        else:
            print("  Skipping batch due to persistent Cohere embed error.")
            continue

        raw_embs = res.embeddings
        embs_list = raw_embs if isinstance(raw_embs, list) else list(raw_embs)

        vectors = []
        for j, d in enumerate(batch):
            vectors.append(
                {
                    "id": f"doc_{uuid.uuid4().hex[:10]}",
                    "values": embs_list[j],
                    "metadata": {
                        "text": d.page_content[:2500],  # Store more text per chunk
                        "department": d.metadata.get("department", "General"),
                        "source_file": d.metadata.get("source_file", "unknown"),
                        "branch_title": d.metadata.get("branch_title", "General"),
                    },
                }
            )

        index.upsert(vectors=vectors)
        batch_num = i // batch_size + 1
        print(f"  Indexed Batch {batch_num}/{total_batches} ({len(batch)} vectors)")
        time.sleep(0.3)

    # Final verification
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"\n[DONE] Ingestion complete! Index stats: {stats}")


if __name__ == "__main__":
    run_pipeline()
