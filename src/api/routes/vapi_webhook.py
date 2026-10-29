import json
from fastapi import APIRouter, Request
from src.tools.rag import retrieve_institute_context

router = APIRouter()


@router.post("/vapi/webhook")
async def handle_vapi_webhook(request: Request):
    payload = await request.json()
    print("\n--- INCOMING VAPI WEBHOOK PAYLOAD ---")
    print(json.dumps(payload, indent=2))

    # 1. Format used by Vapi's "apiRequest" Tool
    # In an apiRequest tool, Vapi posts the body properties directly at the root
    if "query" in payload:
        user_query = payload.get("query", "")
        department = payload.get("department", None)
        context = retrieve_institute_context(
            query=user_query, department=department
        )
        print(f"Returning direct context to Vapi: {context[:100]}...")
        return {"result": context}

    # 2. Format used by Vapi's Server URL "tool-calls" event
    message = payload.get("message", {})
    if message.get("type") == "tool-calls":
        tool_calls = message.get("toolCallList", [])
        results = []

        for call in tool_calls:
            call_id = call.get("id")
            func = call.get("function", {})
            raw_args = func.get("arguments", {})
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            user_query = args.get("query", "")
            department = args.get("department", None)

            context = retrieve_institute_context(
                query=user_query, department=department
            )
            results.append({"toolCallId": call_id, "result": context})

        return {"results": results}

    return {"result": "No matching handler found."}
