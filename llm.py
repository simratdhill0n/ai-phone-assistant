"""The conversation brain: talks to a local LLM through Ollama.

The LLM returns structured JSON every turn: the details it has collected so
far, plus what to say next. Our code, not the model, decides when the call
is complete and ends it.
"""

from typing import Literal

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from config import settings

REQUIRED_FIELDS = ("name", "reason", "urgency", "callback")


class TurnOutput(BaseModel):
    """The exact shape the LLM must reply with on every turn."""
    name: str | None = None
    reason: str | None = None
    urgency: Literal["low", "normal", "urgent"] | None = None
    callback: str | None = None
    caller_confirmed: bool = False   # caller said the read-back details are correct
    reply: str                       # what the assistant says out loud next


class CallDetails(BaseModel):
    """What we've collected so far. This becomes the SMS summary later."""
    name: str | None = None
    reason: str | None = None
    urgency: str | None = None
    callback: str | None = None

    def is_complete(self) -> bool:
        return all(getattr(self, field) for field in REQUIRED_FIELDS)


SYSTEM_PROMPT = f"""You are {settings.assistant_name}, the AI phone assistant for {settings.owner_name}.
{settings.owner_name} is unavailable, and you are taking a message on a live phone call.

Collect these details, one question at a time:
- name: the caller's name
- reason: why they are calling
- urgency: "low", "normal" or "urgent"
- callback: the best phone number or time to call them back

Every reply must be JSON with these keys: name, reason, urgency, callback, caller_confirmed, reply.
- Fill in every detail you know so far, from the whole conversation. Use null for unknown ones.
- "reply" is what you say out loud next.
- Once all four details are known, use "reply" to briefly read them back and ask if they are correct.
- Set caller_confirmed to true only when the caller has just said the read-back details are correct.
- If the caller corrects something, update the detail and read it back again.

Rules for "reply":
- You are speaking on the phone. One or two short sentences, plain spoken language.
- No lists, markdown, emojis or special characters. It is read aloud.
- Ask only one question at a time. Never ask again for something you already know.
- Never share personal information about {settings.owner_name}, and never promise what he will do.
- If asked, say honestly that you are an AI assistant.
"""

client = AsyncClient()  # connects to the Ollama server at http://localhost:11434


class Conversation:
    """One phone call's conversation: message history plus collected details."""

    def __init__(self, greeting: str, caller_number: str = "", caller_context: str = ""):
        system_prompt = SYSTEM_PROMPT + caller_context
        if caller_number.startswith("+"):
            # A real number: let the assistant offer it instead of asking cold.
            system_prompt += (
                f"\nThe caller is calling from {caller_number}. For callback, ask if "
                "this number is the best one to reach them, instead of asking for a number. "
                "If they say yes, set callback to this number.\n"
            )
        else:
            # Blocked or unknown caller ID: the assistant has to ask.
            system_prompt += "\nThe caller's number is hidden, so ask for a callback number.\n"

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": greeting},
        ]
        # Human-readable record of the call: (speaker, text) pairs.
        # self.messages holds raw JSON for the model, this holds what was said.
        self.transcript: list[tuple[str, str]] = [("assistant", greeting)]
        self.details = CallDetails()
        self._read_back_done = False  # have we read the details back to the caller?

    async def reply(self, caller_text: str) -> tuple[str, bool]:
        """Add what the caller said, get the next reply.

        Returns (reply_text, end_call).
        """
        self.messages.append({"role": "user", "content": caller_text})
        self.transcript.append(("caller", caller_text))

        response = await client.chat(
            model=settings.ollama_model,
            messages=self.messages,
            # Structured output: Ollama forces the reply to match this JSON schema
            format=TurnOutput.model_json_schema(),
            options={"temperature": 0.2},
            keep_alive="30m",
        )
        raw = response["message"]["content"]

        try:
            turn = TurnOutput.model_validate_json(raw)
        except ValidationError:
            # Rare with a schema, but never crash a live call over bad output
            print(f"LLM returned invalid output: {raw}")
            return self._said("Sorry, could you say that again?"), False

        # Keep the model's JSON in the history, so on the next turn it sees
        # exactly what it already collected.
        self.messages.append({"role": "assistant", "content": raw})
        self._merge(turn)
        print(f"  details so far: {self.details.model_dump()}")

        # Our code decides when the call is done: every detail collected,
        # read back to the caller, and confirmed by them.
        if self.details.is_complete() and self._read_back_done and turn.caller_confirmed:
            first_name = self.details.name.split()[0]
            goodbye = (
                f"Thanks {first_name}, I'll pass your message to {settings.owner_name}. "
                "Have a great day, goodbye."
            )
            return self._said(goodbye), True

        # If all details are now known, this reply is the read-back question.
        if self.details.is_complete():
            self._read_back_done = True

        return self._said(turn.reply), False

    def _said(self, text: str) -> str:
        """Record what the assistant is about to say, and pass it through."""
        self.transcript.append(("assistant", text))
        return text

    def _merge(self, turn: TurnOutput) -> None:
        """Update details with anything new. A null never erases a known value."""
        for field in REQUIRED_FIELDS:
            value = getattr(turn, field)
            if value:
                setattr(self.details, field, value)