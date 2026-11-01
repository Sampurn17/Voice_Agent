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

Your task: Write a CLEAR, COMPREHENSIVE, SPOKEN answer that thoroughly addresses the student's question.
Rules:
- Use plain spoken English. NO bullet points, NO asterisks, NO markdown.
- Do NOT read out course codes like CYMC201 — say the course NAME instead.
- Provide detailed information but organize it naturally in spoken format.
- When discussing curriculum, mention key subjects, credit structure, and important requirements.
- If numbers like LPA or credits are relevant, state them naturally (e.g. "fifteen point six LPA average").
- For complex topics like curriculum, provide a complete overview covering all important aspects.
- Speak naturally as if explaining to a student in person - be thorough but conversational.
- End with a helpful follow-up offer if appropriate.
Answer:"""


def retrieve_institute_context(query: str, department: str = None) -> str:
    try:
        logger.info(f"RAG query: {query!r} | dept: {department}")

        # FAST MODE: Skip slow synthesis, return formatted chunks directly
        FAST_MODE = True  # Set to False to re-enable synthesis
        
        # 1. Embed query
        embed_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
        )
        query_vector = embed_response.embeddings[0]

        # 2. Retrieve top-k chunks from Pinecone (increased for better coverage)
        query_results = index.query(
            vector=query_vector, top_k=8, include_metadata=True
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

        # FAST MODE: Return formatted chunks immediately (5-8 second response)
        if FAST_MODE:
            if not retrieved_chunks:
                return "I don't have specific information on that in my knowledge base right now. For accurate details, please check with the administrative office at Triple I T Oona campus in Saloh."
            
            # Check if this is a curriculum/subject list query
            query_lower = query.lower()
            is_curriculum_query = any(word in query_lower for word in ['semester', 'curriculum', 'subjects', 'courses', 'syllabus'])
            
            if is_curriculum_query:
                # For curriculum queries, combine more chunks and clean for speech
                combined_text = " ".join(retrieved_chunks[:5])  # Use top 5 for better coverage
                
                # Clean up table formatting for natural speech
                combined_text = combined_text.replace("\n", " ")
                combined_text = combined_text.replace("  ", " ")
                combined_text = combined_text.replace(" S.No. ", " Number ")
                combined_text = combined_text.replace("L T P C", "")
                combined_text = combined_text.replace("Total", "The total is")
                
                # Limit to prevent timeout but allow more for curriculum
                final_response = combined_text[:1600].strip()
                logger.info(f"Fast mode curriculum response ({len(final_response)} chars, {len(retrieved_chunks)} chunks used)")
                return final_response
            
            # For non-curriculum queries, use existing format
            response_parts = []
            for i, chunk in enumerate(retrieved_chunks[:3]):  # Use top 3 chunks
                # Clean up chunk text for speech
                clean_chunk = chunk.replace("\n", " ").replace("  ", " ").strip()
                response_parts.append(clean_chunk)
            
            # Join with natural transitions
            natural_response = " Additionally, ".join(response_parts[:2])
            if len(response_parts) > 2:
                natural_response += " Also, " + response_parts[2]
            
            # Limit total length for voice
            final_response = natural_response[:1800]
            logger.info(f"Fast mode response ({len(final_response)} chars, skipped synthesis)")
            return final_response

        # 3. Synthesise a comprehensive, spoken-word-friendly answer via Cohere (SLOW)
        try:
            # Optimized for speed while maintaining quality
            synthesis = co.chat(
                model="command-r-08-2024",
                message=_SYNTHESIS_PROMPT.format(query=query, context=raw_context[:6000]),
                temperature=0.3,
                max_tokens=500,  # Optimized for speed (15-18 second response)
            )
            answer = synthesis.text.strip()
            if answer:
                logger.info(f"Synthesised answer ({len(answer)} chars)")
                return answer
        except Exception as synth_err:
            logger.warning(f"Synthesis failed ({synth_err}), falling back to raw context")

        # 4. Fallback: return comprehensive context (not truncated too early)
        fallback_text = "\n".join(retrieved_chunks[:3])
        return fallback_text[:2000] if fallback_text else "No information found."

    except Exception as e:
        logger.error(f"Error in RAG retriever: {e}", exc_info=True)
        return f"I'm having trouble accessing the knowledge base right now. Please try again in a moment."

