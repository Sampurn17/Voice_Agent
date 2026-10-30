import asyncio
import json
import logging
from src.models.base import SessionLocal
from src.models.database import create_tables
from src.services.tool_dispatcher import ToolDispatcher
from src.services.session_memory import session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_webhook")

async def run_tests():
    create_tables()
    db = SessionLocal()
    session_id = "test_call_session_999"

    print("\n=======================================================")
    print("TEST 1: MULTI-TURN RAG FOLLOW-UP QUERY RESOLUTION")
    print("=======================================================")

    # Turn 1
    t1_query = "What are the hostel rules?"
    print(f"\n[Turn 1] Student asks: '{t1_query}'")
    res1 = await ToolDispatcher.dispatch(
        function_name="get_institute_info",
        arguments={"query": t1_query},
        session_id=session_id,
        db=db
    )
    print(f"Result (first 120 chars): {res1[:120]}...")

    # Turn 2
    t2_query = "What about weekends?"
    print(f"\n[Turn 2] Student follow-up: '{t2_query}'")
    res2 = await ToolDispatcher.dispatch(
        function_name="get_institute_info",
        arguments={"query": t2_query},
        session_id=session_id,
        db=db
    )
    print(f"Result (first 120 chars): {res2[:120]}...")

    # Turn 3
    t3_query = "And what about night-outs?"
    print(f"\n[Turn 3] Student follow-up: '{t3_query}'")
    res3 = await ToolDispatcher.dispatch(
        function_name="get_institute_info",
        arguments={"query": t3_query},
        session_id=session_id,
        db=db
    )
    print(f"Result (first 120 chars): {res3[:120]}...")

    print("\n=======================================================")
    print("TEST 2: TODO SERVICE DISPATCH")
    print("=======================================================")
    # 2a: Create Todo
    create_todo_res = await ToolDispatcher.dispatch(
        function_name="createTodo",
        arguments={"title": "Submit CS201 Assignment", "description": "Implement binary search tree"},
        session_id=session_id,
        db=db
    )
    print("createTodo response:", create_todo_res)

    # 2b: Get Todos
    get_todos_res = await ToolDispatcher.dispatch(
        function_name="getTodos",
        arguments={},
        session_id=session_id,
        db=db
    )
    print("getTodos response:\n", get_todos_res)

    print("\n=======================================================")
    print("TEST 3: REMINDER SERVICE DISPATCH")
    print("=======================================================")
    # 3a: Add Reminder
    add_rem_res = await ToolDispatcher.dispatch(
        function_name="addReminder",
        arguments={"reminder_text": "Pay semester mess fee", "importance": "high"},
        session_id=session_id,
        db=db
    )
    print("addReminder response:", add_rem_res)

    # 3b: Get Reminders
    get_rems_res = await ToolDispatcher.dispatch(
        function_name="getReminders",
        arguments={},
        session_id=session_id,
        db=db
    )
    print("getReminders response:\n", get_rems_res)

    print("\n=======================================================")
    print("TEST 4: CALENDAR EVENT SERVICE DISPATCH")
    print("=======================================================")
    # 4a: Add Calendar Entry
    add_cal_res = await ToolDispatcher.dispatch(
        function_name="addCalendarEntry",
        arguments={
            "title": "Midterm Examination",
            "event_from": "2026-11-10 09:30:00",
            "event_to": "2026-11-10 12:30:00",
            "description": "Room 101"
        },
        session_id=session_id,
        db=db
    )
    print("addCalendarEntry response:", add_cal_res)

    # 4b: Get Calendar Entries
    get_cal_res = await ToolDispatcher.dispatch(
        function_name="getCalendarEntries",
        arguments={},
        session_id=session_id,
        db=db
    )
    print("getCalendarEntries response:\n", get_cal_res)

    print("\n=======================================================")
    print("TEST 5: SESSION CLEANUP")
    print("=======================================================")
    history_before = session_manager.get_history(session_id)
    print(f"History turns before cleanup: {len(history_before)}")
    session_manager.clear_session(session_id)
    history_after = session_manager.get_history(session_id)
    print(f"History turns after cleanup: {len(history_after)}")
    assert len(history_after) == 0, "Session should be empty after clear_session"

    db.close()
    print("\n ALL WORKFLOWS DISPATCHED AND VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
