"""Notes by SMS: the owner texts the assistant's number to leave notes about people.

Examples:
    Note for Ahmed: interview moved to Monday. Share
    Ahmed 519-555-0123: he's a recruiter at Shopify, private
    note for sara - send her the invoice when she calls, shareable

The LLM turns free-form text into structured fields. Our code then finds the
contact and saves the note.
"""

import asyncio
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from config import settings
from db import add_note, find_contacts_by_name
from llm import client


class ParsedNote(BaseModel):
    person_name: str | None = None     # who the note is about
    phone: str | None = None           # if the owner included a number
    note: str                          # the note itself, without name/tag
    visibility: Literal["private", "shareable"] = "private"


PARSE_PROMPT = """You extract notes from text messages. The owner writes notes about people who may call him.

Return JSON with:
- person_name: the name of the person the note is about, or null
- phone: their phone number if one is written in the message, or null
- note: the note itself, rewritten as a short clear sentence, without the name prefix or the visibility word
- visibility: "shareable" ONLY if the owner clearly says it can be shared with the person
  (e.g. "share", "shareable", "tell him", "you can tell her"). Otherwise "private".
"""


def normalize_phone(raw: str) -> str | None:
    """Turn '519-555-0123' or '+1 (519) 555 0123' into '+15195550123'."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return None


async def parse_note(message: str) -> ParsedNote | None:
    response = await client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user", "content": message},
        ],
        format=ParsedNote.model_json_schema(),
        options={"temperature": 0},   # extraction: we want the same answer every time
    )
    try:
        return ParsedNote.model_validate_json(response["message"]["content"])
    except ValidationError:
        return None


async def handle_owner_sms(message: str) -> str:
    """Process a note from the owner. Returns the text to reply with."""
    parsed = await parse_note(message)
    if parsed is None or not parsed.note:
        return "Sorry, I couldn't understand that note. Try: Note for Ahmed: interview moved to Monday. Share"

    # 1. Work out which contact the note is about
    phone = normalize_phone(parsed.phone) if parsed.phone else None

    if phone is None:
        if not parsed.person_name:
            return "Who is this note about? Try: Note for Ahmed: ..."

        matches = await asyncio.to_thread(find_contacts_by_name, parsed.person_name)
        if len(matches) == 0:
            return (
                f"I don't have a contact named {parsed.person_name} yet. "
                f"Resend with their number, e.g. Note for {parsed.person_name} 519-555-0123: ..."
            )
        if len(matches) > 1:
            options = ", ".join(f"{c.name} ({c.phone})" for c in matches)
            return f"I know more than one {parsed.person_name}: {options}. Resend with their number."
        phone = matches[0].phone

    # 2. Save it (database call = blocking, so run it in a thread)
    contact = await asyncio.to_thread(
        add_note, phone, parsed.note, parsed.visibility, parsed.person_name
    )

    who = contact.name or contact.phone
    return f"Saved {parsed.visibility} note for {who}: {parsed.note}"