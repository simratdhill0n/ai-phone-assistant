"""Caller memory: turns a caller's history into a greeting and LLM context.

Security rule (requirement 5): caller ID can be spoofed. So the phone number
alone only lets us *ask* "am I speaking with Ahmed?". Details of previous
calls are only shared after the caller confirms who they are.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from config import settings
from db import Call, get_contact, get_recent_calls

LOCAL_TZ = ZoneInfo(settings.owner_timezone)


def _when(dt: datetime) -> str:
    """Friendly relative time: 'earlier today', 'yesterday', 'on Monday', 'on March 3'."""
    if dt.tzinfo is None:            # SQLite returns naive datetimes; we stored UTC
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local = dt.astimezone(LOCAL_TZ)
    days_ago = (datetime.now(LOCAL_TZ).date() - local.date()).days

    if days_ago == 0:
        return "earlier today"
    if days_ago == 1:
        return "yesterday"
    if days_ago < 7:
        return f"on {local.strftime('%A')}"          # weekday name
    return f"on {local.strftime('%B')} {local.day}"   # month and day


def _describe_call(call: Call) -> str:
    when = _when(call.started_at)
    if call.status == "missed":
        return f"called {when} but hung up before saying why"
    if call.reason:
        return f"called {when} about: {call.reason}"
    return f"called {when}"


def build_caller_context(phone: str, default_greeting: str) -> tuple[str, str, str | None]:
    """Look up the caller. Returns (greeting, extra_system_prompt, known_name).

    Plain (not async) function: it queries the database, so main.py runs it
    with asyncio.to_thread.
    """
    if not phone.startswith("+"):
        return default_greeting, "", None   # hidden caller ID: nothing to look up

    contact = get_contact(phone)
    calls = get_recent_calls(phone, limit=3)
    if contact is None and not calls:
        return default_greeting, "", None   # first-time caller

    greeting = default_greeting
    lines = ["\nCALLER HISTORY (from our records, linked to this phone number):"]

    if contact and contact.name:
        # Ask, don't assume: the number could be spoofed or shared.
        greeting = (
            f"Hi, you've reached the office of {settings.owner_name}. "
            f"I'm {settings.assistant_name}, his AI assistant, and this call may be recorded. "
            f"Am I speaking with {contact.name}?"
        )
        lines.append(f"- This number belongs to {contact.name}. You have asked if you are speaking with them.")

    for call in calls:
        lines.append(f"- They {_describe_call(call)}")

    lines.append(
        "Rules for using this history:\n"
        "- Only mention details of previous calls AFTER the caller confirms who they are.\n"
        "- If they confirm, you may briefly mention their last call and ask if this is about the same thing.\n"
        "- If their last call was missed, you may say you saw they called earlier but didn't catch why.\n"
        "- If they say they are someone else, ignore this history completely and treat them as a new caller.\n"
        "- Still collect all details for THIS call, but don't ask for things they have just confirmed."
    )

    known_name = contact.name if contact else None
    return greeting, "\n".join(lines) + "\n", known_name