"""
IIIT Una Knowledge Base Ingestion Script.

Chunks institute-specific content, embeds via Cohere, and upserts to Pinecone.

Usage:
    python -m src.scripts.ingest_iiit_una
"""

import os
import uuid

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("institute-knowledge-base")

# 1. Structure the raw data or load text extracted from circulars/PDFs
DATA_SOURCES = [
    {
        "department": "Academics",
        "content": """
        Indian Institute of Information Technology Una (IIIT Una) is an Institute of National Importance established by the Ministry of Education, Government of India.
        IIIT Una offers four-year Bachelor of Technology (B.Tech) programs in Computer Science and Engineering (CSE), Electronics and Communication Engineering (ECE), and Information Technology (IT).
        Academic grading operates on a 10-point scale (CGPA and SGPA). Minimum passing criteria and continuation in the degree requires satisfying credit quotas each semester.
        Provisional transcripts and grade records can be issued by the Academic Section upon submission of a formal request letter forwarded by the Faculty-in-Charge (Academics).
        Official working hours for the academic section are 9:00 AM to 5:30 PM on Monday through Friday.
        """,
    },
    {
        "department": "ECE",
        "content": """
        The Department of Electronics and Communication Engineering (ECE) at IIIT Una prepares students in core electronics, signal processing, and communication networks.
        Specialized hardware and software laboratories for ECE include the VLSI Design Lab, DSP (Digital Signal Processing) Lab, Embedded Systems and IoT Lab, and the Analog & Digital Communication Lab.
        Labs are situated in the main Academic Block. Laboratory hours run from 9:00 AM to 5:00 PM on working days.
        Coursework covers Microprocessors, Semiconductor Physics, Antennas and Wave Propagation, Microwave Engineering, and VLSI Architectures.
        """,
    },
    {
        "department": "Administration",
        "content": """
        The permanent campus of IIIT Una is located in Saloh, Tehsil Haroli, District Una, Himachal Pradesh, PIN code 177209.
        Administrative inquiries regarding bonafide certificates, fee refund policies, institute identity cards, and transit passes are handled by the main Administrative Block.
        Semester tuition fees and mess caution deposits must be deposited into the institute's designated bank account via SBI Collect before the start of each semester.
        """,
    },
    {
        "department": "Campus Facilities",
        "content": """
        IIIT Una maintains on-campus student hostels equipped with high-speed internet connectivity, reading spaces, and recreation facilities.
        The mess committee comprises student representatives and faculty wardens who decide the mess menu and enforce hygiene standards.
        Basic health services and emergency first-aid are provided at the campus dispensary by an on-duty medical assistant. Severe health cases are directed to the Civil Hospital in Una.
        Sports amenities include courts and equipment for badminton, table tennis, volleyball, and outdoor cricket/football grounds.
        """,
    },
    {
        "department": "Placement",
        "content": """
        The Training and Placement (T&P) Cell handles internship allocations and campus recruitments for 3rd and 4th-year students across CSE, IT, and ECE.
        The T&P Cell organizes mock technical interviews, DSA coding rounds, and aptitude evaluations to prepare candidates for corporate recruitment drives.
        """,
    },
]


def run_ingestion():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )

    documents_to_upsert = []

    for source in DATA_SOURCES:
        dept = source["department"]
        chunks = text_splitter.split_text(source["content"].strip())

        for chunk in chunks:
            documents_to_upsert.append({"department": dept, "text": chunk.strip()})

    print(f"Total chunks created: {len(documents_to_upsert)}")

    # 2. Convert texts to Cohere embeddings using input_type='search_document'
    texts_only = [item["text"] for item in documents_to_upsert]
    embedding_response = co.embed(
        texts=texts_only,
        model="embed-english-v3.0",
        input_type="search_document",
    )

    # 3. Assemble vectors with metadata for Pinecone
    upsert_payload = []
    for idx, item in enumerate(documents_to_upsert):
        upsert_payload.append({
            "id": f"iiit_una_{uuid.uuid4().hex[:8]}",
            "values": embedding_response.embeddings[idx],
            "metadata": {"text": item["text"], "department": item["department"]},
        })

    # 4. Upsert into Pinecone
    print("Upserting vectors into Pinecone...")
    index.upsert(vectors=upsert_payload)
    print("IIIT Una knowledge base ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
