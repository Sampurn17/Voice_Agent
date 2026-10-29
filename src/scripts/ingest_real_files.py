"""
Real File Ingestion Script.

Reads PDFs, text, and markdown files from the data/ directory,
chunks them, embeds via Cohere, and upserts to Pinecone.

Supports nested directory structures — the immediate parent folder
name is used as the department/category label, and the filename is
preserved as a source reference.

Usage:
    python -m src.scripts.ingest_real_files
"""

import os
import uuid
import time

import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "institute-knowledge-base"
DIMENSION = 1024
METRIC = "cosine"
DATA_DIR = "data"

# --- Category label mapping (folder name → human-readable label) ---
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
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print(f"Index '{INDEX_NAME}' created and ready!")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    return pc.Index(INDEX_NAME)


def resolve_category(folder_name: str) -> str:
    """Map a folder name to a human-readable category label."""
    return CATEGORY_MAP.get(folder_name.lower(), folder_name.replace("_", " ").title())


def load_documents():
    """Walk the data/ tree and load every supported file."""
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        category = resolve_category(folder_name)

        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_lower = file.lower()

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
                docs.extend(loaded)
                print(f"  ✓ Loaded: {file_path} ({len(loaded)} page(s)) → category={category}")

            except Exception as e:
                print(f"  ✗ Failed to load {file_path}: {e}")

    return docs


def run_pipeline():
    """Main ingestion pipeline: load → chunk → embed → upsert."""
    index = ensure_index()

    raw_docs = load_documents()
    if not raw_docs:
        print("No documents found in data/ folder!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    split_docs = splitter.split_documents(raw_docs)
    split_docs = [d for d in split_docs if d.page_content and d.page_content.strip()]
    print(f"\nPrepared {len(split_docs)} document chunks from {len(raw_docs)} source pages.\n")

    batch_size = 64
    total_batches = (len(split_docs) + batch_size - 1) // batch_size

    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i : i + batch_size]
        texts = [d.page_content for d in batch]

        res = co.embed(
            texts=texts, model="embed-english-v3.0", input_type="search_document"
        )

        vectors = []
        for j, d in enumerate(batch):
            vectors.append(
                {
                    "id": f"doc_{uuid.uuid4().hex[:10]}",
                    "values": res.embeddings[j],
                    "metadata": {
                        "text": d.page_content,
                        "department": d.metadata.get("department", "General"),
                        "source_file": d.metadata.get("source_file", "unknown"),
                    },
                }
            )

        index.upsert(vectors=vectors)
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} — indexed {len(batch)} vectors")

    # Final verification
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"\n✓ Ingestion complete! Index stats: {stats}")


if __name__ == "__main__":
    run_pipeline()
