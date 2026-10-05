"""Warm transfer: put an urgent caller through to the owner's real phone.

How it works:
1. The conversation decides a transfer is worth it (urgent, details confirmed).
2. We tell Twilio, through its REST API, to stop running our TwiML for this
   live call and run new TwiML instead: <Dial> the owner. Our media stream
   ends, and the caller hears ringing.
3. When the owner picks up, Twilio fetches /whisper and plays it to the
   OWNER only: "Urgent call from Dave about a burst pipe. Press 1 to take it."
4. Owner presses 1: the two calls are joined. No key (or their voicemail
   answered): the owner's leg hangs up, and the caller hears a short
   "couldn't reach him" message instead.

Why "press 1" and not just connect? If the owner doesn't answer, their
voicemail does, and Twilio can't tell a voicemail from a person. The caller
would end up leaving a message on the owner's voicemail. A keypress proves
a human picked up.
"""

import httpx

from config import settings

TWILIO_API = "https://api.twilio.com/2010-04-01"

# Live transfers, keyed by the CALLER's call SID (the "parent" call).
# Plain dicts are fine: one server process, and entries live for seconds.
_whispers: dict[str, str] = {}   # what the owner hears before accepting
_accepted: set[str] = set()      # transfers the owner said yes to


def whisper_text(details, caller_number: str) -> str:
    """What the owner hears when they pick up. Short: they're busy."""
    text = f"Urgent call from {details.name or 'someone'}"
    if caller_number.startswith("+"):
        # Last 4 digits, spaced so they're read one by one: "0 1 4 2"
        text += f", number ending in {' '.join(caller_number[-4:])}"
    return text + f", about {details.reason}."


async def transfer_call(call_sid: str, whisper: str) -> bool:
    """Redirect a live call to the owner. Returns False if Twilio refused."""
    _whispers[call_sid] = whisper
    host = settings.public_host
    # callerId is OUR Twilio number, so the owner knows it's a transfer from
    # the assistant (and Twilio only allows numbers we own or have verified).
    twiml = (
        "<Response>"
        f'<Dial timeout="{settings.transfer_ring_seconds}" '
        f'action="https://{host}/transfer-result" callerId="{settings.twilio_phone_number}">'
        f'<Number url="https://{host}/whisper">{settings.owner_phone}</Number>'
        "</Dial>"
        "</Response>"
    )
    sid = settings.twilio_account_sid
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{TWILIO_API}/Accounts/{sid}/Calls/{call_sid}.json",
                data={"Twiml": twiml},
                auth=(sid, settings.twilio_auth_token.get_secret_value()),
            )
        response.raise_for_status()
        print(f"Transferring {call_sid} to {settings.owner_name}...")
        return True
    except httpx.HTTPError as e:
        print(f"Transfer failed: {e}")
        _whispers.pop(call_sid, None)
        return False


def get_whisper(parent_call_sid: str) -> str:
    # Fallback, in case Twilio didn't send ParentCallSid for some reason
    return _whispers.get(parent_call_sid, "You have an urgent call from your assistant.")


def mark_accepted(parent_call_sid: str) -> None:
    _accepted.add(parent_call_sid)


def finish(parent_call_sid: str) -> bool:
    """Clean up a transfer. Returns True if the owner took the call."""
    _whispers.pop(parent_call_sid, None)
    if parent_call_sid in _accepted:
        _accepted.discard(parent_call_sid)
        return True
    return False