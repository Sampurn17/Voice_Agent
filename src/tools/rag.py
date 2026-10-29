import os

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("institute-knowledge-base")


def retrieve_institute_context(query: str, department: str = None) -> str:
    try:
        print(f"\n--- INCOMING QUERY: {query} (dept: {department}) ---")

        # 1. Embed query
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        query_vector = embed_response.embeddings[0]

        # 2. Query WITHOUT restrictive filters first
        query_results = index.query(
            vector=query_vector, top_k=3, include_metadata=True
        )

        matches = query_results.get("matches", [])
        print(f"Pinecone matches found: {len(matches)}")

        if not matches:
            return (
                "No specific institute records found for this query in the database."
            )

        for m in matches:
            print(f"Match score: {m.get('score')} | Text: {m['metadata'].get('text')}")

        # 3. Concatenate and return text
        retrieved_chunks = [
            match["metadata"]["text"]
            for match in matches
            if "text" in match.get("metadata", {})
        ]

        return "\n---\n".join(retrieved_chunks)

    except Exception as e:
        print(f"Error in retriever: {e}")
        return f"Error querying knowledge base: {str(e)}"
