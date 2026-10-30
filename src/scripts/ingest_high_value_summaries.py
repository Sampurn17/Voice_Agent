"""
Targeted Ingestion for High-Value Placement and Curriculum Summary
"""

import os
import uuid
from dotenv import load_dotenv
import cohere
from pinecone import Pinecone

load_dotenv()

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("institute-knowledge-base")

HIGH_VALUE_SUMMARIES = [
    {
        "id": "placement_summary_stats_2025_2026",
        "dept": "Placement",
        "tag": "IIIT Una Placement Statistics and Packages (AY 2025-26)",
        "text": (
            "[IIIT Una Placement Statistics & Packages | Highest and Average CTC]\n"
            "Official Placement Statistics for IIIT Una (Academic Year 2025-26):\n"
            "- Overall Maximum CTC (Highest Package): 53.60 LPA (Lakhs Per Annum)\n"
            "- Overall Average CTC: 13.46 LPA\n"
            "- Overall Minimum CTC: 5.00 LPA\n"
            "- CSE (Computer Science and Engineering) Maximum CTC: 53.60 LPA\n"
            "- CSE Average CTC: 15.64 LPA\n"
            "- CSE Minimum CTC: 5.00 LPA\n"
            "- ECE (Electronics & Communication Engineering) Maximum CTC: 53.60 LPA\n"
            "- ECE Average CTC: 12.25 LPA\n"
            "- ECE Minimum CTC: 5.00 LPA\n"
            "- IT (Information Technology) Maximum CTC: 53.60 LPA\n"
            "- IT Average CTC: 12.45 LPA\n"
            "- IT Minimum CTC: 5.00 LPA\n"
            "Placement recruiters and companies visiting campus include top tech firms, product companies, and MNCs.\n"
            "Training & Placement Cell (CDC) Contact: Dr. Naman Garg (Faculty-In-Charge, fic.cdc@iiitu.ac.in)."
        ),
    },
    {
        "id": "cybersecurity_sem3_curriculum_summary",
        "dept": "Academics",
        "tag": "B.Tech Computer Science & Engineering (Cyber Security - CY) Semester 3 Curriculum",
        "text": (
            "[B.Tech CSE (Cyber Security) | Semester 3 (Third Semester) Subject Breakdown]\n"
            "In the 3rd Semester (Semester III) of the B.Tech CSE (Cyber Security) program at IIIT Una, students study 6 courses totaling 24 credits:\n"
            "1. CYMC201: Introduction to Cryptography (3 Lecture, 0 Tutorial, 2 Practical - 4 Credits) [Core Cyber Security Course]\n"
            "2. CSMC201: Discrete Structures (3-1-0 - 4 Credits)\n"
            "3. CSMC202: Computer Organization (3-1-0 - 4 Credits)\n"
            "4. CSMC203: Design and Analysis of Algorithms (3-1-0 - 4 Credits)\n"
            "5. CSMC204: Object Oriented Programming (3-0-2 - 4 Credits)\n"
            "6. ECMC202: Digital Circuits and Systems (3-0-2 - 4 Credits)\n"
            "Total Credits for Semester 3: 24 credits."
        ),
    },
    {
        "id": "cybersecurity_sem4_curriculum_summary",
        "dept": "Academics",
        "tag": "B.Tech Computer Science & Engineering (Cyber Security - CY) Semester 4 Curriculum",
        "text": (
            "[B.Tech CSE (Cyber Security) | Semester 4 (Fourth Semester) Subject Breakdown]\n"
            "In the 4th Semester (Semester IV) of the B.Tech CSE (Cyber Security) program at IIIT Una, students study:\n"
            "1. CYMC202: Fundamentals of Data Science (3-0-2 - 4 Credits)\n"
            "2. CYMC203: Mobile Forensics and Security (3-0-2 - 4 Credits)\n"
            "3. CSMC205: Operating Systems (3-0-2 - 4 Credits)\n"
            "4. CSMC206: Database Management Systems (3-0-2 - 4 Credits)\n"
            "5. CSMC207: Theory of Computation (3-1-0 - 4 Credits)\n"
            "6. CSMC208: Computer Networks (3-0-2 - 4 Credits)\n"
            "7. ECMC213: Product Development Lab (0-0-2 - 1 Credit)\n"
            "Total Credits for Semester 4: 25 credits."
        ),
    },
    {
        "id": "cybersecurity_sem5_curriculum_summary",
        "dept": "Academics",
        "tag": "B.Tech Computer Science & Engineering (Cyber Security - CY) Semester 5 Curriculum",
        "text": (
            "[B.Tech CSE (Cyber Security) | Semester 5 (Fifth Semester) Subject Breakdown]\n"
            "In the 5th Semester (Semester V) of the B.Tech CSE (Cyber Security) program at IIIT Una, students study:\n"
            "1. CYMC301: Blockchain and Cryptocurrencies (3-0-2 - 4 Credits)\n"
            "2. CYMC302: Multimedia Security and Forensics (3-0-0 - 3 Credits)\n"
            "3. Program Elective I & II (CSSE3XX)\n"
            "4. Open Elective I (XXMS3XX)\n"
            "5. Professional Communication and Soft Skills (ICAE301)\n"
            "6. Professional Ethics (ICVA301)\n"
            "Total Credits: 23 credits."
        ),
    },
]

def run():
    texts = [item["text"] for item in HIGH_VALUE_SUMMARIES]
    res = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
    raw_embs = res.embeddings
    embs_list = raw_embs if isinstance(raw_embs, list) else list(raw_embs)

    vectors = []
    for i, item in enumerate(HIGH_VALUE_SUMMARIES):
        vectors.append({
            "id": item["id"],
            "values": embs_list[i],
            "metadata": {
                "text": item["text"],
                "department": item["dept"],
                "source_file": "curriculum_and_placement_official_records",
                "branch_title": item["tag"],
            }
        })

    index.upsert(vectors=vectors)
    print(f"Successfully upserted {len(vectors)} high-value summary records!")

if __name__ == "__main__":
    run()
