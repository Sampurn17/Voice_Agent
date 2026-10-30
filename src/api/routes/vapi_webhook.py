import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

from src.models.base import SessionLocal
from src.services.session_memory import session_manager
from src.services.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

router = APIRouter()


def _extract_session_id(payload: Dict[str, Any], request: Request) -> str:
    """Extract a consistent session/call identifier from the incoming Vapi payload."""
    message = payload.get("message", {})
    call_obj = message.get("call") or payload.get("call") or {}

    # Check common Vapi identifiers
    if isinstance(call_obj, dict) and call_obj.get("id"):
        return str(call_obj.get("id"))
    if message.get("callId"):
        return str(message.get("callId"))
    if payload.get("callId"):
        return str(payload.get("callId"))
    if payload.get("sessionId"):
        return str(payload.get("sessionId"))
    if payload.get("conversationId"):
        return str(payload.get("conversationId"))

    header_id = request.headers.get("x-vapi-call-id")
    if header_id:
        return header_id

    return "default_session"


def _parse_arguments(raw_args: Any) -> Dict[str, Any]:
    """Ensure tool arguments are parsed into a dictionary."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"query": raw_args}
    return {}


@router.post("/vapi/webhook")
async def handle_vapi_webhook(request: Request):
    payload = await request.json()
    logger.info("--- INCOMING VAPI WEBHOOK PAYLOAD ---")
    logger.info(json.dumps(payload, indent=2))

    session_id = _extract_session_id(payload, request)
    message = payload.get("message", {})
    msg_type = message.get("type") or payload.get("type")

    # 1. Handle end-of-call or status-update to clean up session memory
    if msg_type in ("end-of-call-report", "status-update"):
        status = message.get("status") or payload.get("status")
        if msg_type == "end-of-call-report" or status in ("ended", "forwarded"):
            session_manager.clear_session(session_id)
            return {"status": "session cleared"}

    db = SessionLocal()
    try:
        # 2. Format used by Vapi's Server URL "tool-calls" event
        if msg_type == "tool-calls":
            tool_calls = message.get("toolCallList") or message.get("toolCalls") or []
            results = []

            for call in tool_calls:
                call_id = call.get("id")
                func = call.get("function", {})
                func_name = func.get("name", "")
                raw_args = func.get("arguments", {})
                args = _parse_arguments(raw_args)

                result_text = await ToolDispatcher.dispatch(
                    function_name=func_name,
                    arguments=args,
                    session_id=session_id,
                    db=db,
                )
                results.append({"toolCallId": call_id, "result": result_text})

            return {"results": results}

        # 3. Format used by Vapi's "apiRequest" Tool
        # In an apiRequest tool, Vapi sends the arguments directly at the root
        func_name = (
            payload.get("function")
            or payload.get("name")
            or payload.get("action")
            or ("get_institute_info" if "query" in payload else "")
        )

        result_text = await ToolDispatcher.dispatch(
            function_name=func_name,
            arguments=payload,
            session_id=session_id,
            db=db,
        )
        logger.info(f"Returning direct context to Vapi: {result_text[:100]}...")
        return {"result": result_text}

    finally:
        db.close()
