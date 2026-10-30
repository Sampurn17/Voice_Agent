import datetime as dt
import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from src.models.domain.calendar_event import (
    CalendarEventCreate,
    CalendarEventId,
)
from src.models.domain.call import CallRequest, CustomerInfo
from src.models.domain.reminder import ReminderCreate, ReminderId
from src.models.domain.todo import TodoCreate, TodoId
from src.repositories.calendar_event_repository import CalendarEventRepository
from src.repositories.reminder_repository import ReminderRepository
from src.repositories.todo_repository import TodoRepository
from src.services.calendar_event_service import CalendarEventService
from src.services.call_service import CallService
from src.services.reminder_service import ReminderService
from src.services.session_memory import session_manager
from src.services.todo_service import TodoService
from src.tools.rag import retrieve_institute_context

logger = logging.getLogger(__name__)


def _parse_datetime(value: Any) -> dt.datetime:
    """Safely parse various datetime representations."""
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, str):
        # Clean common string formats (e.g. trailing 'Z')
        clean_val = value.replace("Z", "+00:00")
        try:
            return dt.datetime.fromisoformat(clean_val)
        except Exception:
            pass
        # Common alternative formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(clean_val, fmt)
            except Exception:
                continue

    # Fallback to current time if parsing fails
    logger.warning(f"Unable to parse datetime from '{value}', defaulting to now.")
    return dt.datetime.now()


class ToolDispatcher:
    """Dispatches incoming Vapi tool calls to the appropriate application services

    (Todo, Reminder, Calendar, Outbound Call, or Institute RAG with session memory).
    """

    @staticmethod
    async def dispatch(
        function_name: str,
        arguments: Dict[str, Any],
        session_id: str,
        db: Session,
    ) -> str:
        """Route the tool call based on function_name and arguments."""
        fn_raw = (function_name or "").strip()
        fn = fn_raw.lower().replace("_", "")

        logger.info(
            f"Dispatching tool '{fn_raw}' (normalized: '{fn}') for session '{session_id}'"
        )

        try:
            # -------------------------------------------------------------
            # 1. Todo Service Dispatches
            # -------------------------------------------------------------
            if fn in ("createtodo", "addtodo", "newtodo"):
                title = arguments.get("title") or arguments.get("task") or "Untitled Todo"
                description = arguments.get("description")
                todo_service = TodoService(TodoRepository(db))
                todo = await todo_service.create_todo(
                    TodoCreate(title=str(title), description=description)
                )
                return f"To-do '{todo.title}' created successfully with ID {todo.id}."

            elif fn in ("gettodos", "listtodos", "fetchtodos", "showtodos"):
                todo_service = TodoService(TodoRepository(db))
                todos = await todo_service.get_todos()
                if not todos:
                    return "You have no to-dos on your list."
                lines = [
                    f"{t.id}. {t.title} ({'completed' if t.completed else 'pending'})"
                    for t in todos
                ]
                return f"You have {len(todos)} to-do(s):\n" + "\n".join(lines)

            elif fn in ("completetodo", "marktododone", "finishtodo"):
                raw_id = arguments.get("id") or arguments.get("todo_id") or arguments.get("todoId")
                if not raw_id:
                    return "Error: Todo ID is required to mark a to-do as completed."
                todo_service = TodoService(TodoRepository(db))
                todo = await todo_service.complete_todo(TodoId(id=int(raw_id)))
                return f"To-do '{todo.title}' (ID {todo.id}) has been marked as completed."

            elif fn in ("deletetodo", "removetodo"):
                raw_id = arguments.get("id") or arguments.get("todo_id") or arguments.get("todoId")
                if not raw_id:
                    return "Error: Todo ID is required to delete a to-do."
                todo_service = TodoService(TodoRepository(db))
                res = await todo_service.delete_todo(TodoId(id=int(raw_id)))
                return res.get("message", f"Todo {raw_id} deleted successfully.")

            # -------------------------------------------------------------
            # 2. Reminder Service Dispatches
            # -------------------------------------------------------------
            elif fn in ("addreminder", "createreminder", "setreminder", "newreminder"):
                text = (
                    arguments.get("reminder_text")
                    or arguments.get("reminderText")
                    or arguments.get("text")
                    or arguments.get("reminder")
                    or "Untitled Reminder"
                )
                importance = arguments.get("importance") or "medium"
                reminder_service = ReminderService(ReminderRepository(db))
                reminder = await reminder_service.create_reminder(
                    ReminderCreate(reminder_text=str(text), importance=str(importance))
                )
                return f"Reminder set: '{reminder.reminder_text}' (Priority: {reminder.importance}, ID: {reminder.id})."

            elif fn in ("getreminders", "listreminders", "fetchreminders", "showreminders"):
                reminder_service = ReminderService(ReminderRepository(db))
                reminders = await reminder_service.get_reminders()
                if not reminders:
                    return "You have no active reminders."
                lines = [
                    f"{r.id}. {r.reminder_text} [Priority: {r.importance}]"
                    for r in reminders
                ]
                return f"You have {len(reminders)} reminder(s):\n" + "\n".join(lines)

            elif fn in ("deletereminder", "removereminder"):
                raw_id = (
                    arguments.get("id")
                    or arguments.get("reminder_id")
                    or arguments.get("reminderId")
                )
                if not raw_id:
                    return "Error: Reminder ID is required to delete a reminder."
                reminder_service = ReminderService(ReminderRepository(db))
                res = await reminder_service.delete_reminder(ReminderId(id=int(raw_id)))
                return res.get("message", f"Reminder {raw_id} deleted successfully.")

            # -------------------------------------------------------------
            # 3. Calendar Event Service Dispatches
            # -------------------------------------------------------------
            elif fn in (
                "addcalendarentry",
                "createcalendarentry",
                "addevent",
                "createevent",
                "scheduleevent",
            ):
                title = arguments.get("title") or arguments.get("event") or "Untitled Event"
                desc = arguments.get("description")
                start_raw = (
                    arguments.get("event_from")
                    or arguments.get("eventFrom")
                    or arguments.get("start")
                    or arguments.get("start_time")
                )
                end_raw = (
                    arguments.get("event_to")
                    or arguments.get("eventTo")
                    or arguments.get("end")
                    or arguments.get("end_time")
                )

                start_dt = _parse_datetime(start_raw)
                end_dt = _parse_datetime(end_raw)
                if end_dt <= start_dt:
                    end_dt = start_dt + dt.timedelta(hours=1)

                calendar_service = CalendarEventService(CalendarEventRepository(db))
                event = await calendar_service.create_event(
                    CalendarEventCreate(
                        title=str(title),
                        description=desc,
                        event_from=start_dt,
                        event_to=end_dt,
                    )
                )
                return (
                    f"Calendar event '{event.title}' scheduled from "
                    f"{event.event_from.strftime('%Y-%m-%d %H:%M')} to "
                    f"{event.event_to.strftime('%Y-%m-%d %H:%M')} (ID {event.id})."
                )

            elif fn in (
                "getcalendarentries",
                "listcalendarentries",
                "getevents",
                "listevents",
                "showevents",
            ):
                calendar_service = CalendarEventService(CalendarEventRepository(db))
                events = await calendar_service.get_events()
                if not events:
                    return "You have no scheduled calendar entries."
                lines = [
                    f"{e.id}. {e.title} ({e.event_from.strftime('%Y-%m-%d %H:%M')} - {e.event_to.strftime('%H:%M')})"
                    for e in events
                ]
                return f"You have {len(events)} calendar event(s):\n" + "\n".join(lines)

            elif fn in ("deletecalendarentry", "deleteevent", "removeevent"):
                raw_id = arguments.get("id") or arguments.get("event_id") or arguments.get("eventId")
                if not raw_id:
                    return "Error: Calendar Event ID is required to delete an event."
                calendar_service = CalendarEventService(CalendarEventRepository(db))
                res = await calendar_service.delete_event(CalendarEventId(id=int(raw_id)))
                return res.get("message", f"Calendar event {raw_id} deleted successfully.")

            # -------------------------------------------------------------
            # 4. Outbound Call Service Dispatch
            # -------------------------------------------------------------
            elif fn in ("makecall", "placecall", "outboundcall"):
                assistant_id = arguments.get("assistant_id") or arguments.get("assistantId")
                phone_number_id = arguments.get("phone_number_id") or arguments.get("phoneNumberId")
                cust_arg = arguments.get("customer") or {}
                number = (
                    cust_arg.get("number")
                    if isinstance(cust_arg, dict)
                    else arguments.get("phone_number") or arguments.get("number")
                )

                if not assistant_id or not phone_number_id or not number:
                    return (
                        "Error: assistant_id, phone_number_id, and customer number "
                        "are required to make an outbound call."
                    )

                call_service = CallService()
                call_res = await call_service.make_call(
                    CallRequest(
                        assistant_id=str(assistant_id),
                        phone_number_id=str(phone_number_id),
                        customer=CustomerInfo(number=str(number)),
                    )
                )
                return f"Call initiated to {number} with ID {call_res.call_id} (Status: {call_res.status})."

            # -------------------------------------------------------------
            # 5. Institute Knowledge Base / RAG (Default Fallback)
            # -------------------------------------------------------------
            else:
                # If function is get_institute_info, retrieve_institute_context, or generic query
                raw_query = (
                    arguments.get("query")
                    or arguments.get("question")
                    or arguments.get("input")
                    or ""
                )

                # Fallback argument checks if function name was ambiguous
                if not raw_query:
                    if "reminder_text" in arguments:
                        return await ToolDispatcher.dispatch("addReminder", arguments, session_id, db)
                    if "event_from" in arguments or "eventFrom" in arguments:
                        return await ToolDispatcher.dispatch("addCalendarEntry", arguments, session_id, db)
                    if "title" in arguments:
                        return await ToolDispatcher.dispatch("createTodo", arguments, session_id, db)

                if not raw_query:
                    return f"No query provided for institute knowledge search (tool '{fn_raw}')."

                dept = arguments.get("department")

                # Multi-turn conversation resolution: rewrite follow-up questions
                # into self-contained standalone queries
                resolved_query = session_manager.resolve_standalone_query(
                    session_id=session_id, current_query=raw_query
                )

                logger.info(
                    f"RAG query | Raw: '{raw_query}' | Resolved: '{resolved_query}' | Dept: {dept}"
                )

                context = retrieve_institute_context(
                    query=resolved_query, department=dept
                )

                # Save turn into session memory
                session_manager.add_turn(
                    session_id=session_id,
                    query=raw_query,
                    response=context,
                )

                return context

        except Exception as e:
            logger.error(f"Error handling tool '{fn_raw}': {e}", exc_info=True)
            return f"An error occurred while executing {fn_raw}: {str(e)}"
