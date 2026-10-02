"""Database: contacts, calls and transcripts, stored in SQLite via SQLModel.

SQLModel classes are Pydantic models and database tables at the same time.
Switching to PostgreSQL later only means changing DATABASE_URL.
"""

from datetime import datetime, timezone

from sqlalchemy import inspect, text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from config import settings
from llm import CallDetails


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- Tables ----------

class Contact(SQLModel, table=True):
    """One row per person (phone number) who has called."""
    phone: str = Field(primary_key=True)            # E.164, e.g. +15485771772
    name: str | None = None
    company: str | None = None
    relationship: str | None = None                 # e.g. "recruiter", "friend"
    status_summary: str | None = None               # running summary, used in Stage 4
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Call(SQLModel, table=True):
    """One row per phone call."""
    call_sid: str = Field(primary_key=True)          # Twilio's ID for the call
    phone: str = Field(index=True)                   # index = fast "all calls from this number"
    started_at: datetime
    ended_at: datetime
    status: str                                      # "completed", "incomplete" or "missed"
    caller_name: str | None = None
    reason: str | None = None
    urgency: str | None = None
    callback: str | None = None
    recording_path: str | None = None


class TranscriptTurn(SQLModel, table=True):
    """One line of a call's conversation."""
    id: int | None = Field(default=None, primary_key=True)   # auto-numbered
    call_sid: str = Field(foreign_key="call.call_sid", index=True)
    position: int                                             # order within the call
    speaker: str                                              # "caller" or "assistant"
    text: str


class Note(SQLModel, table=True):
    """A note the owner wrote about a contact, sent by SMS."""
    id: int | None = Field(default=None, primary_key=True)
    phone: str = Field(foreign_key="contact.phone", index=True)
    text: str
    visibility: str            # "private" (never said to caller) or "shareable"
    created_at: datetime = Field(default_factory=utcnow)
    delivered_at: datetime | None = None   # when a shareable note was passed on


# ---------- Setup ----------

# check_same_thread=False: SQLite normally refuses to be used from more than
# one thread. We call it through asyncio.to_thread, so we must allow that.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


def init_db() -> None:
    """Create any tables that don't exist yet. Safe to run on every startup."""
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """A tiny hand-made migration.

    create_all() only creates MISSING TABLES. It never changes a table that
    already exists, so a column added to a model later (like Note.delivered_at)
    doesn't appear in an existing database. Real projects use a migration tool
    such as Alembic for this. For now, add the column if it's missing.
    """
    existing = {column["name"] for column in inspect(engine).get_columns("note")}
    if "delivered_at" not in existing:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE note ADD COLUMN delivered_at DATETIME"))
        print("Database migrated: added note.delivered_at")


# ---------- Writing ----------

def save_call(
    call_sid: str,
    phone: str,
    started_at: datetime,
    status: str,
    details: CallDetails,
    transcript: list[tuple[str, str]],
    recording_path: str | None,
) -> None:
    """Save a finished call, its transcript, and create/update the contact.

    Plain (not async) function: database calls block, so main.py runs it
    with asyncio.to_thread.
    """
    real_number = phone.startswith("+")   # hidden caller ID can't become a contact

    # A session is one "unit of work". Everything added inside it is saved
    # together by commit(), or not at all if something fails.
    with Session(engine) as session:
        if real_number:
            contact = session.get(Contact, phone)   # look up by primary key
            if contact is None:
                contact = Contact(phone=phone)
            # Only trust a name from a call where the caller confirmed details
            if status == "completed" and details.name:
                contact.name = details.name
            contact.updated_at = utcnow()
            session.add(contact)

        session.add(Call(
            call_sid=call_sid,
            phone=phone or "unknown",
            started_at=started_at,
            ended_at=utcnow(),
            status=status,
            caller_name=details.name,
            reason=details.reason,
            urgency=details.urgency,
            callback=details.callback,
            recording_path=recording_path,
        ))

        for position, (speaker, text) in enumerate(transcript):
            session.add(TranscriptTurn(
                call_sid=call_sid, position=position, speaker=speaker, text=text,
            ))

        session.commit()


# ---------- Reading ----------

def get_contact(phone: str) -> Contact | None:
    with Session(engine) as session:
        return session.get(Contact, phone)


def get_recent_calls(phone: str, limit: int = 3) -> list[Call]:
    """Most recent calls from this number, newest first."""
    with Session(engine) as session:
        statement = (
            select(Call)
            .where(Call.phone == phone)
            .order_by(Call.started_at.desc())
            .limit(limit)
        )
        return list(session.exec(statement))


def find_contacts_by_name(name: str) -> list[Contact]:
    """Contacts whose name contains this text, ignoring case.
    'ahmed' matches 'Ahmed Khan'."""
    with Session(engine) as session:
        statement = select(Contact).where(Contact.name.ilike(f"%{name}%"))
        return list(session.exec(statement))


def add_note(phone: str, text: str, visibility: str, name: str | None = None) -> Contact:
    """Save a note for this number. Creates the contact if it's new."""
    with Session(engine) as session:
        contact = session.get(Contact, phone)
        if contact is None:
            contact = Contact(phone=phone)
        if name and not contact.name:
            contact.name = name
        contact.updated_at = utcnow()
        session.add(contact)
        session.add(Note(phone=phone, text=text, visibility=visibility))
        session.commit()
        session.refresh(contact)   # reload so it's usable after the session closes
        return contact


def get_notes(phone: str) -> list[Note]:
    """All notes about this number, oldest first."""
    with Session(engine) as session:
        statement = select(Note).where(Note.phone == phone).order_by(Note.created_at)
        return list(session.exec(statement))


def mark_notes_delivered(note_ids: list[int]) -> None:
    """Record that these shareable notes were passed on, so they aren't repeated."""
    if not note_ids:
        return
    with Session(engine) as session:
        for note in session.exec(select(Note).where(Note.id.in_(note_ids))):
            note.delivered_at = utcnow()
            session.add(note)
        session.commit()