"""The conversation brain: talks to a local LLM through Ollama.

The LLM returns structured JSON every turn: any details the caller just gave,
plus what to say next. Division of labour:
  - our code decides WHAT must happen: what's collected, what's safe to say,
    when the details are confirmed, when the call ends.
  - the model decides HOW to say it: wording, order, tone.
"""

import random
import re
from typing import Literal

from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError

from config import settings

# Every detail we store. Only some must be ASKED (see Conversation._required):
# urgency is inferred, and the callback defaults to the caller ID.
ALL_FIELDS = ("name", "reason", "urgency", "callback")

# How to describe a missing detail in the status note
FIELD_DESCRIPTIONS = {
    "name": "the caller's name",
    "reason": "the reason for the call",
    "callback": "a number to call them back on (their caller ID is hidden)",
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
    identity_confirmed: bool = False    # caller confirmed they are the known contact
    reply: str                          # what the assistant says out loud next


class CallDetails(BaseModel):
    """What we've collected so far. This becomes the SMS summary later."""
    name: str | None = None
    reason: str | None = None
    urgency: str | None = None
    callback: str | None = None

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [field for field in required if not getattr(self, field)]


SYSTEM_PROMPT = f"""You are {settings.assistant_name}, the AI phone assistant for {settings.owner_name}.
{settings.owner_name} is unavailable, and you are taking a message on a live phone call.
You speak ONLY as the assistant. Never speak as the caller.

How you sound: like a friendly, relaxed human receptionist. Warm and brief.
- React to what the caller actually said, then move on. ("Ah, the interview, got it.")
- Let the caller talk in any order. If they give several details at once, take them all.
- Vary your wording. Don't start every reply with "Thank you" or the caller's name.
  Use their name now and then, not every time.
- One or two short sentences. Plain spoken language, no lists, markdown or emojis. It is read aloud.

What you need for the message: the caller's NAME and the REASON for the call.
- Don't ask how urgent it is. Only set urgency if the caller signals it:
  "no rush", "whenever" = low. "Today", "soon" = normal. "ASAP", "emergency", "right away" = urgent.
- Don't ask for a callback number unless the STATUS note says it's needed.
  Only set callback if the caller offers a different number or a time to call.

Each turn you get a STATUS note with what's known and what's missing. Use it, but in your own words.

Every reply is JSON with these keys: name, reason, urgency, callback, caller_confirmed, caller_wants_to_end, identity_confirmed, reply.
- Only fill a detail the caller gave in their LATEST message. Use null for everything else,
  including details you already know (our system remembers them).
- Never guess. "Mhm", "yeah" or "uh huh" on their own are not answers.
- "reason": a short phrase like "the interview on Monday", not "calling about the interview".
- caller_confirmed: true only when the caller has just agreed that the details read back to them are right.
- caller_wants_to_end: true when the caller says goodbye or clearly wants to hang up.
- If the caller corrects something, fill in the corrected value.
- "reply" is what you say out loud next.

Limits:
- You cannot call anyone back or help with their request yourself. Never say you will.
  You only take a message and pass it to {settings.owner_name}.
- Never share personal information about {settings.owner_name}, and never promise what he will do.
- If asked, say honestly that you are an AI assistant.

Two examples of the tone (the names here are made up, never reuse them):

Caller: Hey, it's Priya, I'm calling about the invoice I sent last week.
You: Hi Priya, sure, the invoice from last week. Is there anything you'd like me to pass on about it?
Caller: Just that it's due Friday.
You: Got it, due Friday.

Caller: Hi, is Simrat there?
You: He's not available right now, but I can take a message. Who's calling?
Caller: Marcus.
You: Thanks Marcus, and what's it about?
"""

client = AsyncClient()  # connects to the Ollama server at http://localhost:11434

# Thinking models (Qwen 3.x and others) write out reasoning before answering.
# Great for puzzles, far too slow on a phone call, so we switch it off.
# Older models don't support the setting at all, and Ollama rejects it,
# so if that happens once, we stop sending it.
_send_think = True


async def chat(**kwargs):
    """Every Ollama request goes through here: same model, same keep_alive,
    thinking off where supported."""
    global _send_think
    kwargs.setdefault("model", settings.ollama_model)
    kwargs.setdefault("keep_alive", "30m")
    if _send_think:
        try:
            return await client.chat(think=False, **kwargs)
        except (ResponseError, TypeError):
            # ResponseError: this model doesn't support thinking settings.
            # TypeError: the ollama library is too old to know "think".
            _send_think = False
    return await client.chat(**kwargs)


def _print_llm_stats(response) -> None:
    """Where the LLM time goes. Ollama reports durations in nanoseconds.

    prompt: reading the input (system prompt + history). Grows every turn.
    output: generating the reply, one token at a time.
    """
    try:
        p_tok, p_ns = response["prompt_eval_count"], response["prompt_eval_duration"]
        o_tok, o_ns = response["eval_count"], response["eval_duration"]
        print(f"  llm detail: prompt {p_tok} tok in {p_ns / 1e9:.2f}s | "
              f"output {o_tok} tok in {o_ns / 1e9:.2f}s ({o_tok / (o_ns / 1e9):.0f} tok/s)")
    except (KeyError, TypeError, ZeroDivisionError):
        pass  # some responses (e.g. cached prompts) leave fields out


class Conversation:
    """One phone call's conversation: message history plus collected details."""

    def __init__(
        self,
        greeting: str,
        caller_number: str = "",
        caller_context: str = "",
        known_name: str | None = None,
        shareable_notes: list[tuple[int, str]] | None = None,
    ):
        system_prompt = SYSTEM_PROMPT + caller_context
        if known_name:
            system_prompt += (
                f"\nSet identity_confirmed to true only when the caller clearly confirms "
                f"they are {known_name}. Otherwise keep it false.\n"
            )
        # If the caller turns out to be someone else, the history is swapped
        # out for this, so the model can't leak what it no longer sees.
        self._prompt_for_stranger = SYSTEM_PROMPT + (
            "\nThis caller is a new caller. You have no information about any other "
            "person, so never discuss anyone else's calls, plans or appointments.\n"
        )
        # What MUST be asked. With caller ID, the callback defaults to that
        # number (they can still offer another). Hidden number: we must ask.
        self._has_caller_id = caller_number.startswith("+")
        self._required = ("name", "reason") if self._has_caller_id else ("name", "reason", "callback")

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
        self._last_read_back: str | None = None
        # Once a read-back has STARTED, the caller may correct details,
        # even if they interrupted it before the end.
        self._corrections_allowed = False

        # Shareable notes: (note_id, text). The LLM never sees these. Our code
        # says them word for word, once the caller confirms who they are.
        self.known_name = known_name
        self._shareable_notes = shareable_notes or []
        self.identity_confirmed = False
        self.identity_denied = False   # caller said they're someone else: permanent for this call
        self._honesty_prefix = ""      # set when the caller asks if they're talking to a person
        self.delivered_note_ids: list[int] = []

    def _status_note(self) -> str:
        """Tell the model exactly where we are. Our code tracks progress,
        so the model doesn't have to work it out from the history."""
        known = {f: v for f, v in self.details.model_dump().items() if v}
        known_text = f"Known so far: {known}. " if known else ""
        missing = self.details.missing(self._required)
        if missing:
            needed = ", ".join(FIELD_DESCRIPTIONS[f] for f in missing)
            return f"STATUS: {known_text}Still missing: {needed}. Work it into the conversation naturally."
        if not self._read_back_done:
            return f"STATUS: {known_text}Everything needed is collected."
        return f"STATUS: {known_text}Details were read back. Set caller_confirmed only if the caller clearly agreed, or update what they corrected."

    async def reply(self, caller_text: str) -> tuple[str, bool]:
        """Add what the caller said, get the next reply.

        Returns (reply_text, end_call).
        """
        self.messages.append({"role": "user", "content": caller_text})
        self.transcript.append(("caller", caller_text))
        self._caller_turns += 1
        asked_if_ai = "?" in caller_text and AI_QUESTION.search(caller_text)
        self._honesty_prefix = HONESTY_LINE if asked_if_ai else ""

        turn = await self._ask_model()
        if turn is None:
            return self._said("Sorry, could you say that again?"), False

        # The history gets what the assistant actually SAID (see _said), as
        # plain text, not the model's raw JSON. Shorter prompts = faster replies,
        # and when our code overrides the model (read-back, notes, goodbye),
        # the history still matches what the caller really heard.
        self._merge(turn)
        print(f"  details so far: {self.details.model_dump()}")

        # A reply that opens with "no", "wrong", "actually"... is never a
        # confirmation, whatever the model says. Checked in code, because a
        # message delivered with confident wrong details is worse than none.
        disagreed = bool(DISAGREEMENT.search(caller_text))
        matches_read_back = self._read_back_done and self.details == self._read_back_snapshot
        # "Yes" to the read-back = the details are confirmed (completed), even
        # if they tack on a question. "Yeah, but..." is not a yes.
        if (matches_read_back and turn.caller_confirmed and not disagreed
                and not YES_BUT.search(caller_text)):
            self.completed = True

        # Identity is a safety decision, so code makes it, not the model alone.
        if self.known_name and not self.identity_confirmed and not self.identity_denied:
            self._check_identity(turn, disagreed)
            if self.identity_denied:
                # The reply we just got was written while the model could still
                # see the history (that's how "Sara called about the interview"
                # leaked). Throw it away and ask again without the history.
                retry = await self._ask_model()
                if retry is None:   # never fall back to the leaky reply
                    return self._said("Sorry about that. What can I help you with today?"), False
                turn = retry
                self._merge(turn)

        # 1. Details confirmed and nothing left to answer: say goodbye.
        #    If they asked something ("is there a confirmation number?"),
        #    answer it first (step 5); the goodbye comes on a later turn.
        if self.completed and matches_read_back and not disagreed and "?" not in caller_text:
            first_name = self.details.name.split()[0]
            return self._said(random.choice([
                f"Perfect, I'll make sure {settings.owner_name} gets this. Thanks {first_name}, bye now.",
                f"Great, I'll pass that on to {settings.owner_name}. Have a good one, {first_name}.",
                f"All set, {first_name}. I'll let {settings.owner_name} know. Take care.",
            ])), True

        # 1b. They said the read-back was wrong, but nothing changed: the model
        #     missed the correction. Ask plainly instead of carrying on.
        if self._read_back_done and disagreed and self.details == self._read_back_snapshot:
            self.completed = False   # an earlier "yes" no longer stands
            return self._said(random.choice([
                "Sorry about that. What should I change?",
                "Oh, sorry. What did I get wrong?",
            ])), False

        # 2. Caller wants to go, or the call has gone on too long: wrap up
        #    politely with whatever we have.
        if turn.caller_wants_to_end or self._caller_turns >= MAX_CALLER_TURNS:
            return self._said(
                f"No problem, I'll pass on your message to {settings.owner_name}. Goodbye."
            ), True

        # 3. Identity confirmed by code: pass on shareable notes, word for word,
        #    once each, then carry on with what the model wanted to say.
        if self.identity_confirmed:
            pending = [(i, t) for i, t in self._shareable_notes if i not in self.delivered_note_ids]
            if pending:
                self.delivered_note_ids.extend(i for i, _ in pending)
                notes = " ".join(_as_sentence(t) for _, t in pending)
                return self._said(
                    f"{settings.owner_name} asked me to pass on a message: {notes} {turn.reply}"
                ), False

        # 4. All details known, and not yet read back in this exact form
        #    (first time, or the caller just corrected something): our code
        #    writes the read-back itself, so it always contains every detail.
        if not self.details.missing(self._required):
            self._fill_defaults()
        if not self.details.missing(self._required) and self.details != self._read_back_snapshot:
            self._read_back_done = True
            self.completed = False   # new details: they need confirming again
            self._corrections_allowed = True
            self._read_back_snapshot = self.details.model_copy()
            self._last_read_back = self._read_back_text()
            return self._said(self._last_read_back), False

        return self._said(turn.reply), False

    async def _ask_model(self) -> TurnOutput | None:
        """One model call with the current history. None if the output was unusable."""
        # The status note is added for this request only, never stored in
        # history, so the model always sees the CURRENT status, not old ones.
        response = await chat(
            messages=self.messages + [{"role": "system", "content": self._status_note()}],
            # Structured output: Ollama forces the reply to match this JSON schema
            format=TurnOutput.model_json_schema(),
            options={
                "temperature": 0.3,   # some variety, without getting sloppy at extracting details
                "num_predict": 200,   # hard cap on output length (tokens)
            },
        )
        raw = response["message"]["content"]
        _print_llm_stats(response)
        try:
            return TurnOutput.model_validate_json(raw)
        except ValidationError:
            # Rare with a schema, but never crash a live call over bad output
            print(f"LLM returned invalid output: {raw}")
            return None

    def _check_identity(self, turn: TurnOutput, disagreed: bool) -> None:
        """Decide whether the caller is the known contact.

        Denied (for the rest of the call) if they disagree when first asked,
        or give a different name. Confirmed only if the model says so AND
        the caller didn't disagree in that same message.
        """
        known_first = self.known_name.split()[0].lower()
        gave_other_name = bool(self.details.name) and known_first not in self.details.name.lower()

        if (self._caller_turns == 1 and disagreed) or gave_other_name:
            self.identity_denied = True
            # Take the caller history out of the model's view entirely
            self.messages[0]["content"] = self._prompt_for_stranger
            print(f"  identity DENIED: caller is not {self.known_name}. History removed from prompt.")
            return

        if turn.identity_confirmed and not disagreed:
            self.identity_confirmed = True
            if not self.details.name:
                self.details.name = self.known_name   # they just confirmed it
            print(f"  identity confirmed: {self.known_name}")

    def _fill_defaults(self) -> None:
        """Details we don't ask for get sensible defaults. The read-back
        still mentions them, so the caller can correct either one."""
        if not self.details.urgency:
            self.details.urgency = "normal"
        if not self.details.callback and self._has_caller_id:
            self.details.callback = self.caller_number

    def _read_back_text(self) -> str:
        """Our code writes the read-back, so it always contains every detail
        correctly. Varied wording keeps it from sounding like a form."""
        d = self.details
        reason = _clean_reason(d.reason)
        urgency = {"urgent": ", and it's urgent", "low": ", no rush"}.get(d.urgency, "")
        if d.callback == self.caller_number:
            callback = f"{settings.owner_name} can call you back on this number"
        else:
            callback = f"{settings.owner_name} can reach you at {d.callback}"
        return random.choice([
            f"Okay, just to confirm: {d.name}, about {reason}{urgency}, and {callback}. Did I get that right?",
            f"Got it. So that's {d.name}, calling about {reason}{urgency}, and {callback}. Is that right?",
            f"Alright, let me read that back. {d.name}, {reason}{urgency}, and {callback}. Sound good?",
        ])

    def _said(self, text: str) -> str:
        """Record what the assistant is about to say, and pass it through."""
        if self._honesty_prefix and not re.search(r"\bAI\b", text):
            text = self._honesty_prefix + text
        self._honesty_prefix = ""
        self.transcript.append(("assistant", text))
        self.messages.append({"role": "assistant", "content": text})
        return text

    def mark_interrupted(self) -> None:
        """The caller talked over the last reply, so they may not have heard
        all of it. Note that in the history, so the model doesn't assume
        its whole question landed."""
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                # If the cut-off reply was the read-back, the caller never
                # heard it all, so it doesn't count. Read back again next turn.
                if message["content"] == self._last_read_back:
                    self._read_back_done = False
                    self._read_back_snapshot = None
                message["content"] += " (the caller cut in here)"
                break
        if self.transcript and self.transcript[-1][0] == "assistant":
            speaker, text = self.transcript[-1]
            self.transcript[-1] = (speaker, text + " [interrupted]")

    def _merge(self, turn: TurnOutput) -> None:
        """Update details with anything new.

        - A null never erases a known value.
        - Before the read-back, known values are locked. A misheard word
          (e.g. "soon" heard as "Owen") must not overwrite a good name.
        - After the read-back, corrections are allowed. Any change triggers
          a fresh read-back, so the caller always confirms the final version.
        """
        for field in ALL_FIELDS:
            value = getattr(turn, field)
            if not is_real_value(value):
                continue
            value = value.strip()
            if field == "callback":
                value = _normalize_phone(value)
            current = getattr(self.details, field)
            if current and not self._corrections_allowed:
                if value != current:
                    print(f"  ignored change to {field}: {current!r} -> {value!r} (locked until read-back)")
                continue
            setattr(self.details, field, value)


# Signs the caller is disagreeing or correcting, not confirming
# "Am I talking to a real person?" must always get a straight answer, even
# when our code replaces the model's reply (read-back, notes, goodbye).
AI_QUESTION = re.compile(
    r"\b(real person|(a|an) (real )?(human|person|robot|bot|machine|ai)|"
    r"are you (human|real|a robot|a bot|ai|an ai)|automated)\b",
    re.IGNORECASE,
)
HONESTY_LINE = "No, I'm an AI assistant, not a person. "

# "Yeah, but can you...?" is not a clean yes. Neither is a yes with a question.
YES_BUT = re.compile(r"^\W*(yes|yeah|yep|sure|okay|ok|right)\b\W*but\b", re.IGNORECASE)

DISAGREEMENT = re.compile(
    r"^\W*(no|nope|nah|not|wrong|incorrect|actually|wait|hold on)\b"
    r"|\b(not right|not correct|that's wrong|that is wrong|isn't right|is not right)\b",
    re.IGNORECASE,
)


# Small models sometimes write the WORD "null" instead of a JSON null.
# "null" is a non-empty string, so Python treats it as a real value.
PLACEHOLDERS = {"null", "none", "unknown", "n/a", "na", "not given", "not provided", ""}


def is_real_value(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in PLACEHOLDERS


def _normalize_phone(value: str) -> str:
    """'548-577-1772' and '+15485771772' are the same number: store both as
    +15485771772. Anything that isn't a North American number (a time, an
    email, "evenings") is kept as the caller said it."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return value


def _clean_reason(reason: str) -> str:
    """'calling regarding the interview' -> 'the interview', so the read-back
    doesn't say 'calling about calling regarding ...'."""
    cleaned = re.sub(
        r"^(i'?m\s+)?(calling\s+)?(about|regarding|re|for|because of)\s+",
        "", reason.strip(), flags=re.IGNORECASE,
    ) or reason
    # "make a dinner reservation" -> "making a dinner reservation", so
    # "calling about ..." stays grammatical
    cleaned = re.sub(r"^(wants? to|wanting to|to)\s+", "", cleaned, flags=re.IGNORECASE)
    first, _, rest = cleaned.partition(" ")
    gerund = GERUNDS.get(first.lower())
    return f"{gerund} {rest}".strip() if gerund else cleaned


GERUNDS = {
    "make": "making", "book": "booking", "check": "checking", "ask": "asking",
    "talk": "talking", "discuss": "discussing", "schedule": "scheduling",
    "reschedule": "rescheduling", "confirm": "confirming", "cancel": "cancelling",
    "change": "changing", "get": "getting", "set": "setting", "follow": "following",
    "see": "seeing", "find": "finding", "sell": "selling", "buy": "buying",
    "pick": "picking", "drop": "dropping", "return": "returning", "pay": "paying",
}


def _as_sentence(text: str) -> str:
    """Make sure a note ends with punctuation, so the voice pauses after it."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


async def warm_up() -> None:
    """Load the model onto the GPU at startup, so the first caller doesn't wait."""
    await chat(messages=[{"role": "user", "content": "hi"}], options={"num_predict": 1})