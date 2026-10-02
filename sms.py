"""Text the owner a summary after each call, using Twilio's SMS API.

We call Twilio's REST API directly with httpx instead of installing the
Twilio SDK, so you can see exactly what the request looks like.
"""

import httpx

from config import settings
from llm import CallDetails

TWILIO_MESSAGES_URL = (
    f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
)


def format_summary(
    details: CallDetails,
    caller_number: str,
    completed: bool,
    private_notes: list[str] | None = None,
    delivered_notes: list[str] | None = None,
) -> str:
    """Build a short SMS. Short matters: every 160 characters is billed as
    a separate message segment."""
    number = caller_number or "unknown number"

    if not completed and not details.name and not details.reason:
        # Caller hung up before saying anything useful
        return f"Missed call from {number}. They hung up without leaving a message."

    urgent = details.urgency == "urgent"
    lines = [
        f"{'URGENT: ' if urgent else ''}Call from {details.name or 'unknown caller'} ({number})",
        f"Reason: {details.reason or 'not given'}",
    ]
    if details.urgency and not urgent:
        lines.append(f"Urgency: {details.urgency}")
    lines.append(f"Callback: {details.callback or number}")
    if not completed:
        lines.append("(Caller hung up before confirming)")

    # Context for YOU, never said to the caller. Latest two keep the SMS short.
    if delivered_notes:
        lines.append("Passed on: " + " / ".join(delivered_notes))
    if private_notes:
        lines.append("Your notes: " + " / ".join(private_notes[-2:]))

    return "\n".join(lines)


async def send_sms(to: str, body: str) -> None:
    """Send one SMS through Twilio's REST API.

    Sending is network I/O, so we use httpx's async client and await it,
    like the Ollama call. No thread needed.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            TWILIO_MESSAGES_URL,
            # HTTP Basic auth: Account SID as username, Auth Token as password
            auth=(
                settings.twilio_account_sid,
                settings.twilio_auth_token.get_secret_value(),
            ),
            # Twilio's API takes form data, same format its webhooks send us
            data={
                "To": to,
                "From": settings.twilio_phone_number,
                "Body": body,
            },
        )
        response.raise_for_status()  # turn 4xx/5xx replies into exceptions


async def notify_owner(
    details: CallDetails,
    caller_number: str,
    completed: bool,
    private_notes: list[str] | None = None,
    delivered_notes: list[str] | None = None,
) -> None:
    """Text the owner a call summary. Never raises: a failed SMS must not
    crash the server."""
    body = format_summary(details, caller_number, completed, private_notes, delivered_notes)
    try:
        await send_sms(settings.owner_phone, body)
        print(f"Summary texted to owner:\n{body}")
    except httpx.HTTPError as e:
        # Twilio's error details are in the response body when there is one
        detail = getattr(getattr(e, "response", None), "text", "")
        print(f"Failed to send summary SMS: {e} {detail}")