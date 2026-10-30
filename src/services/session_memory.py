import logging
import os
import time
from typing import Any, Dict, List, Optional

import cohere
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


class ConversationSessionManager:
    """Maintains session-level conversation history and resolves follow-up questions
    into standalone search queries for the RAG pipeline.
    """

    def __init__(self, max_history_turns: int = 6, session_ttl_seconds: int = 1800):
        self.max_history_turns = max_history_turns
        self.session_ttl_seconds = session_ttl_seconds
        # sessions: {session_id: {"last_active": float, "history": [{"query": str, "response": str, "timestamp": float}]}}
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self.cohere_client: Optional[cohere.Client] = None
        if COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(COHERE_API_KEY)
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client for session memory: {e}")

    def _cleanup_old_sessions(self):
        """Prune sessions that have been idle longer than session_ttl_seconds."""
        now = time.time()
        expired = [
            sid
            for sid, data in self.sessions.items()
            if now - data.get("last_active", 0) > self.session_ttl_seconds
        ]
        for sid in expired:
            del self.sessions[sid]

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns for a session."""
        self._cleanup_old_sessions()
        session_data = self.sessions.get(session_id)
        if not session_data:
            return []
        return session_data.get("history", [])

    def add_turn(self, session_id: str, query: str, response: str):
        """Record a completed question-and-answer turn in the session."""
        if not session_id or not query:
            return

        self._cleanup_old_sessions()
        now = time.time()

        if session_id not in self.sessions:
            self.sessions[session_id] = {"last_active": now, "history": []}

        session_entry = self.sessions[session_id]
        session_entry["last_active"] = now

        # Add new turn and keep only recent turns
        session_entry["history"].append(
            {
                "query": query.strip(),
                "response": (response or "").strip()[:500],  # keep concise snippet
                "timestamp": now,
            }
        )

        if len(session_entry["history"]) > self.max_history_turns:
            session_entry["history"] = session_entry["history"][-self.max_history_turns :]

    def clear_session(self, session_id: str):
        """Remove a session from memory (e.g., when a call ends)."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session history for call {session_id}")

    def resolve_standalone_query(self, session_id: str, current_query: str) -> str:
        """Resolve a student's follow-up question into a standalone, self-contained query.

        Example:
          Turn 1: "What are the hostel rules?"
          Turn 2: "What about weekends?" -> "What are the hostel rules on weekends?"
          Turn 3: "And what about night-outs?" -> "What are the hostel rules for night-outs?"
        """
        clean_query = (current_query or "").strip()
        if not clean_query:
            return clean_query

        history = self.get_history(session_id)
        # If no prior turns exist or no Cohere client available, return query as-is
        if not history or not self.cohere_client:
            return clean_query

        # Build formatted history string
        history_lines = []
        for turn in history[-4:]:  # last 4 turns for context
            q = turn.get("query", "")
            r = turn.get("response", "")
            if q:
                history_lines.append(f"Student: {q}")
            if r:
                # Include short snippet of previous response
                snippet = r.replace("\n", " ")[:200]
                history_lines.append(f"Assistant: {snippet}")

        history_context = "\n".join(history_lines)

        prompt = f"""You are a query rewriting assistant for a university voice assistant (IIIT Una).
Given the previous conversation turns and a student's follow-up question, rewrite the follow-up question into a concise, self-contained standalone search query that preserves all necessary subject context.

Guidelines:
- Maintain the student's exact intent and incorporate missing subjects/entities from previous turns.
- If the question is already fully complete and self-contained, or introduces a completely new topic, return it unchanged.
- Output ONLY the rewritten standalone query. Do NOT answer the question. Do NOT include quotes, explanations, or labels.

Conversation History:
{history_context}

Follow-up Question: {clean_query}
Standalone Question:"""

        try:
            response = self.cohere_client.chat(
                model="command-r-08-2024",
                message=prompt,
                temperature=0.0,
            )
            resolved = response.text.strip().strip('"\'')
            if resolved:
                logger.info(
                    f"Session [{session_id}] Resolved follow-up: '{clean_query}' -> '{resolved}'"
                )
                return resolved
        except Exception as e:
            logger.warning(
                f"Failed to resolve follow-up query via Cohere ({e}). Falling back to original query."
            )

        return clean_query


# Global singleton instance
session_manager = ConversationSessionManager()
