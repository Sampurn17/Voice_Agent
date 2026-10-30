"""
Ingest Administration & Specialization PDFs into Pinecone:
- UG Ordinance.pdf (Rules, grading, examinations)
- IIITU_Placement_Brochure_2026-27.pdf (Placement stats, packages, recruiters)
- IIITUGCSEDS251.pdf (CSE Data Science full semester structure)
"""

import os
import sys
import uuid
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cohere
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

load_dotenv()

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("institute-knowledge-base")

TARGET_FILES = [
    {
        "path": "data/administration/UG Ordinance.pdf",
        "tag": "IIIT Una Academic UG Ordinance, Grading, Exam & Attendance Rules",
        "dept": "Administration",
        "pages": 47,
    },
    {
        "path": "data/administration/IIITU_Placement_Brochure_2026-27.pdf",
        "tag": "IIIT Una Placement Brochure, Companies, Average Packages & Statistics",
        "dept": "Placement",
        "pages": 40,
    },
    {
        "path": "data/academics/IIITUGCSEDS251.pdf",
        "tag": "B.Tech Computer Science & Engineering (Data Science - DS) Curriculum",
        "dept": "Academics",
        "pages": 50,  # First 50 pages contains the entire semester-wise curriculum and core courses
    },
]

def run():
    splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    all_chunks = []

    for item in TARGET_FILES:
        path = item["path"]
        tag = item["tag"]
        dept = item["dept"]
        max_pages = item["pages"]

        print(f"Loading {path} (up to {max_pages} pages)...")
        loader = PyPDFLoader(path)
        pages = loader.load()
        selected_pages = pages[:max_pages]

        for p in selected_pages:
            p.metadata["department"] = dept
            p.metadata["source_file"] = os.path.basename(path)
            p.metadata["branch_title"] = tag
            p.page_content = f"[{tag} | {os.path.basename(path)}]\n{p.page_content}"

        split_pages = splitter.split_documents(selected_pages)
        split_pages = [s for s in split_pages if s.page_content.strip()]
        all_chunks.extend(split_pages)
        print(f"  -> Generated {len(split_pages)} chunks for {os.path.basename(path)}")

    print(f"\nTotal chunks to index: {len(all_chunks)}")
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [b.page_content for b in batch]

        for attempt in range(5):
            try:
                res = co.embed(
                    texts=texts,
                    model="embed-english-v3.0",
                    input_type="search_document"
                )
                break
            except Exception as e:
                print(f"  Rate limit hit ({e}). Sleeping 15s...")
                time.sleep(15)

        raw_embs = res.embeddings
        embs_list = raw_embs if isinstance(raw_embs, list) else list(raw_embs)

        vectors = []
        for j, doc in enumerate(batch):
            vectors.append({
                "id": f"admin_doc_{uuid.uuid4().hex[:10]}",
                "values": embs_list[j],
                "metadata": {
                    "text": doc.page_content[:1500],
                    "department": doc.metadata.get("department", "Administration"),
                    "source_file": doc.metadata.get("source_file", "unknown"),
                    "branch_title": doc.metadata.get("branch_title", "General"),
                }
            })

        index.upsert(vectors=vectors)
        print(f"  Indexed batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size} ({len(batch)} vectors)")
        time.sleep(1.0)

    print("\nTargeted Ingestion finished successfully!")
    print("New Index Stats:", index.describe_index_stats())

if __name__ == "__main__":
    run()
