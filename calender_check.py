"""Google Calendar, read only, through a service account.

Two calendars, two levels of access (set in Google Calendar's sharing settings):
  - Your MAIN calendar: "See only free/busy". We only ever learn WHEN you're
    busy, never what you're doing. Google enforces that, not our code.
  - The NOVA calendar: "See event details". Appointments you put here are
    ones Nova may tell the matching caller about. Write the caller's phone
    number in the event description, e.g.  nova: +15195550142

Nothing here talks to the LLM. These functions return plain facts and
finished sentences; the conversation code decides when they may be said.

Test on its own:  python calendar_check.py +15195550142
"""

import re
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import google.auth.transport.requests
import httpx
from google.oauth2 import service_account

from config import settings

SCOPES = [
    "https://www.googleapis.com/auth/calendar.freebusy",         # main calendar: times only
    "https://www.googleapis.com/auth/calendar.events.readonly",  # Nova calendar: read events
]
API = "https://www.googleapis.com/calendar/v3"
LOOK_AHEAD_HOURS = 12        # how far ahead "busy until" looks
APPOINTMENT_DAYS = 30        # how far ahead shared appointments are searched

_creds = None   # cached: one token lasts about an hour, no need to fetch it every call


def _token() -> str:
    """A short-lived access token for the service account, refreshed when expired."""
    global _creds
    if _creds is None:
        _creds = service_account.Credentials.from_service_account_file(
            settings.google_credentials_path, scopes=SCOPES
        )
    if not _creds.valid:
        _creds.refresh(google.auth.transport.requests.Request())
    return _creds.token


def _headers() -> dict:
    return {"Authorization": f"Bearer {_token()}"}


def _parse(value: str) -> datetime:
    """Google's '2026-10-05T14:00:00Z' -> an aware datetime."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------- Free / busy ----------

def busy_blocks(now: datetime) -> list[tuple[datetime, datetime]]:
    """All busy periods in the next few hours, from BOTH calendars, sorted."""
    calendar_ids = [c for c in (settings.google_calendar_id, settings.google_nova_calendar_id) if c]
    response = httpx.post(
        f"{API}/freeBusy",
        headers=_headers(),
        json={
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(hours=LOOK_AHEAD_HOURS)).isoformat(),
            "items": [{"id": c} for c in calendar_ids],
        },
        timeout=10,
    )
    response.raise_for_status()
    blocks = []
    for calendar_id, result in response.json()["calendars"].items():
        if result.get("errors"):
            # Usually: the calendar isn't shared with the service account
            print(f"Calendar {calendar_id}: {result['errors']}")
        for block in result.get("busy", []):
            blocks.append((_parse(block["start"]), _parse(block["end"])))
    return sorted(blocks)


def free_at(blocks: list[tuple[datetime, datetime]], now: datetime) -> datetime | None:
    """Pure logic, no API: None if free now, else the moment you're free again.
    Back-to-back or overlapping blocks count as one long busy stretch."""
    busy_end = None
    for start, end in blocks:          # sorted by start time
        if busy_end is None:
            if start <= now < end:     # this block covers right now
                busy_end = end
        elif start <= busy_end:        # touches or overlaps the stretch: extend it
            busy_end = max(busy_end, end)
        else:
            break                      # a gap: free from busy_end
    return busy_end


def busy_until(now: datetime) -> datetime | None:
    """If the owner is busy right now, when they're free again. None if free."""
    return free_at(busy_blocks(now), now)


# ---------- Shared appointments (Nova calendar only) ----------

PHONE_IN_TEXT = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def _last_10_digits(text: str) -> str:
    return re.sub(r"\D", "", text)[-10:]


def appointments_for(phone: str, now: datetime) -> list[datetime]:
    """Start times of upcoming Nova-calendar events whose description holds
    this caller's number. Only start times leave this function: titles and
    descriptions are never handed to the rest of the app."""
    if not settings.google_nova_calendar_id or not phone.startswith("+"):
        return []
    response = httpx.get(
        f"{API}/calendars/{settings.google_nova_calendar_id}/events",
        headers=_headers(),
        params={
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=APPOINTMENT_DAYS)).isoformat(),
            "singleEvents": "true",   # expand repeating events into real dates
            "orderBy": "startTime",
        },
        timeout=10,
    )
    response.raise_for_status()
    wanted = _last_10_digits(phone)
    starts = []
    for event in response.json().get("items", []):
        numbers = {_last_10_digits(m) for m in PHONE_IN_TEXT.findall(event.get("description", ""))}
        start = event.get("start", {}).get("dateTime")   # all-day events have only "date": skipped
        if wanted in numbers and start:
            starts.append(_parse(start))
    return starts


# ---------- Turning times into speech ----------

def spoken_time(moment: datetime, now: datetime, say_today: bool = False) -> str:
    """In the owner's timezone: '3 PM' (or 'today at 3 PM'), 'tomorrow at 9:30 AM',
    'on Monday, October 12 at 11 AM'."""
    tz = ZoneInfo(settings.owner_timezone)
    local, today = moment.astimezone(tz), now.astimezone(tz).date()
    # %I is "03", so strip the zero; drop ":00" so it's read "3 PM", not "3 o-o PM"
    clock = local.strftime("%I:%M %p").lstrip("0").replace(":00", "")
    if local.date() == today:
        return f"today at {clock}" if say_today else clock
    if local.date() == today + timedelta(days=1):
        return f"tomorrow at {clock}"
    return f"on {local.strftime('%A, %B')} {local.day} at {clock}"


def availability_sentence(free_again: datetime | None, now: datetime) -> str | None:
    """The one calendar fact Nova may tell any caller. None when free."""
    if free_again is None:
        return None
    return f"{settings.owner_name} is busy until {spoken_time(free_again, now)}."


def appointment_sentence(starts: list[datetime], now: datetime) -> str | None:
    if not starts:
        return None
    times = [spoken_time(s, now, say_today=True) for s in starts[:2]]   # at most two: it's read aloud
    if len(times) == 1:
        return f"Your appointment with {settings.owner_name} is {times[0]}."
    return f"You have appointments with {settings.owner_name} {times[0]}, and {times[1]}."


def call_facts(phone: str, known_contact: bool) -> tuple[str | None, str | None]:
    """Everything a call needs from the calendar, fetched once at call start:
    (availability sentence, appointment sentence). A calendar problem must
    never break a call, so any error just means "no calendar facts"."""
    if not settings.google_calendar_id:
        return None, None   # calendar not set up: feature off
    now = datetime.now(timezone.utc)
    try:
        availability = availability_sentence(busy_until(now), now)
        # Appointments only for numbers we know: they're only ever read out
        # after the caller confirms they're that contact.
        appointment = appointment_sentence(appointments_for(phone, now), now) if known_contact else None
        return availability, appointment
    except Exception as e:
        print(f"Calendar unavailable: {e!r}")
        return None, None


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    free_again = busy_until(now)
    print("busy until:", free_again, "|", availability_sentence(free_again, now) or "free now")
    if len(sys.argv) > 1:
        starts = appointments_for(sys.argv[1], now)
        print("appointments:", starts, "|", appointment_sentence(starts, now))