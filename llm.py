"""The conversation brain: talks to a local LLM through Ollama.

The LLM returns structured JSON every turn: the details it has collected so
far, plus what to say next. Our code tracks progress, tells the model what
to ask next, and decides when the call ends.
"""

from typing import Literal

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from config import settings

REQUIRED_FIELDS = ("name", "reason", "urgency", "callback")

# How to describe each missing detail when steering the model
FIELD_QUESTIONS = {
    "name": "the caller's name",
    "reason": "the reason for the call",
    "urgency": "how urgent it is (low, normal or urgent)",
    "callback": "the best number to call them back on",
}

# Safety net: no call goes on forever, whatever the model does
MAX_CALLER_TURNS = 15


class TurnOutput(BaseModel):
    """The exact shape the LLM must reply with on every turn."""
    name: str | None = None
    reason: str | None = None
    urgency: Literal["low", "normal", "urgent"] | None = None
    callback: str | None = None
    caller_confirmed: bool = False      # caller said the read-back details are correct
    caller_wants_to_end: bool = False   # caller is saying goodbye or wants to hang up
    reply: str                          # what the assistant says out loud next


class CallDetails(BaseModel):
    """What we've collected so far. This becomes the SMS summary later."""
    name: str | None = None
    reason: str | None = None
    urgency: str | None = None
    callback: str | None = None

    def is_complete(self) -> bool:
        return all(getattr(self, field) for field in REQUIRED_FIELDS)

    def missing(self) -> list[str]:
        return [field for field in REQUIRED_FIELDS if not getattr(self, field)]


SYSTEM_PROMPT = f"""You are {settings.assistant_name}, the AI phone assistant for {settings.owner_name}.
{settings.owner_name} is unavailable, and you are taking a message on a live phone call.
You speak ONLY as the assistant. Never speak as the caller.

Collect these details, one question at a time:
- name: the caller's name
- reason: why they are calling
- urgency: "low", "normal" or "urgent"
- callback: the best phone number or time to call them back

Every reply must be JSON with these keys: name, reason, urgency, callback, caller_confirmed, caller_wants_to_end, reply.
- Fill in every detail you know so far, from the whole conversation. Use null for unknown ones.
- Only fill a detail when the caller actually said it. Never guess.
- Callers rarely say "low", "normal" or "urgent". Map what they mean: "no rush", "whenever", "it's okay" = low. "Soon", "today" = normal. "ASAP", "emergency", "right away" = urgent.
- Sounds like "mhm", "yeah" or "uh huh" on their own are not answers or confirmations.
- "reply" is what you say out loud next.
- Set caller_confirmed to true only when the caller has just said the read-back details are correct.
- Set caller_wants_to_end to true when the caller says goodbye or clearly wants to end the call.
- If the caller corrects something, update the detail.
- Each turn you will get a STATUS note saying what is still needed. Follow it.

Rules for "reply":
- You are speaking on the phone. One or two short sentences, plain spoken language.
- No lists, markdown, emojis or special characters. It is read aloud.
- Ask only one question at a time. Never ask again for something you already know.
- You cannot call anyone back or help with their request yourself. Never say you will.
  You only take a message and pass it to {settings.owner_name}.
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
        self.caller_number = caller_number
        self.completed = False         # True once the caller confirmed all details
        self._read_back_done = False   # have we read the details back to the caller?
        self._read_back_snapshot: CallDetails | None = None   # details as last read back
        self._caller_turns = 0

    def _status_note(self) -> str:
        """Tell the model exactly where we are. Our code tracks progress,
        so the model doesn't have to work it out from the history."""
        missing = self.details.missing()
        if missing:
            needed = ", ".join(FIELD_QUESTIONS[f] for f in missing)
            return (
                f"STATUS: still needed: {needed}. "
                f"Unless the caller just gave it, ask for {FIELD_QUESTIONS[missing[0]]} next."
            )
        if not self._read_back_done:
            return "STATUS: all details collected."
        return "STATUS: details were read back. Set caller_confirmed only if the caller clearly agreed, or update what they corrected."

    async def reply(self, caller_text: str) -> tuple[str, bool]:
        """Add what the caller said, get the next reply.

        Returns (reply_text, end_call).
        """
        self.messages.append({"role": "user", "content": caller_text})
        self.transcript.append(("caller", caller_text))
        self._caller_turns += 1

        # The status note is added for this request only, never stored in
        # history, so the model always sees the CURRENT status, not old ones.
        response = await client.chat(
            model=settings.ollama_model,
            messages=self.messages + [{"role": "system", "content": self._status_note()}],
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

        # 1. Read-back confirmed: done.
        if self._read_back_done and turn.caller_confirmed and self.details == self._read_back_snapshot:
            self.completed = True
            first_name = self.details.name.split()[0]
            return self._said(
                f"Thanks {first_name}, I'll pass your message to {settings.owner_name}. "
                "Have a great day, goodbye."
            ), True

        # 2. Caller wants to go, or the call has gone on too long: wrap up
        #    politely with whatever we have.
        if turn.caller_wants_to_end or self._caller_turns >= MAX_CALLER_TURNS:
            return self._said(
                f"No problem, I'll pass on your message to {settings.owner_name}. Goodbye."
            ), True

        # 3. All details known, and not yet read back in this exact form
        #    (first time, or the caller just corrected something): our code
        #    writes the read-back itself, so it always contains every detail.
        if self.details.is_complete() and self.details != self._read_back_snapshot:
            self._read_back_done = True
            self._read_back_snapshot = self.details.model_copy()
            return self._said(self._read_back_text()), False

        return self._said(turn.reply), False

    def _read_back_text(self) -> str:
        d = self.details
        if d.callback == self.caller_number:
            callback = "and the best number to reach you is the one you're calling from"
        else:
            callback = f"and the best way to reach you is {d.callback}"
        return (
            f"Let me make sure I have this right. Your name is {d.name}, "
            f"you're calling about {d.reason}, it's {d.urgency} urgency, "
            f"{callback}. Is that correct?"
        )

    def _said(self, text: str) -> str:
        """Record what the assistant is about to say, and pass it through."""
        self.transcript.append(("assistant", text))
        return text

    def _merge(self, turn: TurnOutput) -> None:
        """Update details with anything new.

        - A null never erases a known value.
        - Before the read-back, known values are locked. A misheard word
          (e.g. "soon" heard as "Owen") must not overwrite a good name.
        - After the read-back, corrections are allowed. Any change triggers
          a fresh read-back, so the caller always confirms the final version.
        """
        for field in REQUIRED_FIELDS:
            value = getattr(turn, field)
            if not is_real_value(value):
                continue
            current = getattr(self.details, field)
            if current and not self._read_back_done:
                if value.strip() != current:
                    print(f"  ignored change to {field}: {current!r} -> {value!r} (locked until read-back)")
                continue
            setattr(self.details, field, value.strip())


# Small models sometimes write the WORD "null" instead of a JSON null.
# "null" is a non-empty string, so Python treats it as a real value.
PLACEHOLDERS = {"null", "none", "unknown", "n/a", "na", "not given", "not provided", ""}


def is_real_value(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in PLACEHOLDERS


async def warm_up() -> None:
    """Load the model onto the GPU at startup, so the first caller doesn't wait."""
    await client.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
        keep_alive="30m",
    )