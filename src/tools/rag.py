import os
import logging

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

logger = logging.getLogger(__name__)

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("institute-knowledge-base")

_SYNTHESIS_PROMPT = """You are a helpful voice assistant for IIIT Una. \
A student asked: "{query}"

Here is the relevant information retrieved from the institute knowledge base:
---
{context}
---

Your task: Write a SHORT, CLEAR, SPOKEN answer (2-4 sentences max) that directly answers the student's question.
Rules:
- Use plain spoken English. NO bullet points, NO asterisks, NO markdown.
- Do NOT read out course codes like CYMC201 — say the course NAME instead.
- Do NOT list every single subject. Summarise key highlights.
- If numbers like LPA or credits are relevant, state them naturally (e.g. "fifteen point six LPA average").
- End with one helpful follow-up offer if appropriate.
Answer:"""


def retrieve_institute_context(query: str, department: str = None) -> str:
    try:
        logger.info(f"RAG query: {query!r} | dept: {department}")

        # 1. Embed query
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        query_vector = embed_response.embeddings[0]

        # 2. Retrieve top-k chunks from Pinecone
        query_results = index.query(
            vector=query_vector, top_k=5, include_metadata=True
        )
        matches = query_results.get("matches", [])
        logger.info(f"Pinecone matches: {len(matches)}")

        if not matches:
            return "I don't have specific records for that in the institute knowledge base. Please check with the relevant office at IIIT Una."

        retrieved_chunks = [
            match["metadata"]["text"]
            for match in matches
            if "text" in match.get("metadata", {})
        ]
        raw_context = "\n---\n".join(retrieved_chunks)

        # 3. Synthesise a short, spoken-word-friendly answer via Cohere
        try:
            synthesis = co.chat(
                model="command-r-08-2024",
                message=_SYNTHESIS_PROMPT.format(query=query, context=raw_context[:3000]),
                temperature=0.2,
            )
            answer = synthesis.text.strip()
            if answer:
                logger.info(f"Synthesised answer ({len(answer)} chars)")
                return answer
        except Exception as synth_err:
            logger.warning(f"Synthesis failed ({synth_err}), falling back to raw context")

        # 4. Fallback: return first chunk trimmed to 400 chars
        return retrieved_chunks[0][:400] if retrieved_chunks else "No information found."

    except Exception as e:
        logger.error(f"Error in RAG retriever: {e}", exc_info=True)
        return f"I'm having trouble accessing the knowledge base right now. Please try again in a moment."

