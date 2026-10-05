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
    asks_about_appointment: bool = False   # caller asks when THEIR appointment with the owner is
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
- React to what the caller actually said, then move on. ("Ah, the roof repair, got it.")
- Let the caller talk in any order. If they give several details at once, take them all.
- Vary your wording. Don't start every reply with "Thank you" or the caller's name.
  Use their name now and then, not every time.
- One or two short sentences. Plain spoken language, no lists, markdown or emojis. It is read aloud.

You only take messages for {settings.owner_name}. Never offer to pass messages to, or find
things out from, anyone else.

What you need for the message: the caller's NAME and the REASON for the call.
- Don't ask how urgent it is. Only set urgency if the caller signals it:
  "no rush", "whenever" = low. "Today", "soon" = normal. "ASAP", "emergency", "right away" = urgent.
- Don't ask for a callback number unless the STATUS note says it's needed.
  Only set callback if the caller offers a different number or a time to call.

Each turn you get a STATUS note with what's known and what's missing. Use it, but in your own words.

Every reply is JSON with these keys: name, reason, urgency, callback, caller_confirmed, caller_wants_to_end, identity_confirmed, asks_about_appointment, reply.
- Only fill a detail the caller gave in their LATEST message. Use null for everything else,
  including details you already know (our system remembers them).
- Never guess. "Mhm", "yeah" or "uh huh" on their own are not answers.
- "reason": a short phrase like "the roof repair quote", not "calling about the roof repair".
- caller_confirmed: true only when the caller has just agreed that the details read back to them are right.
- caller_wants_to_end: true when the caller says goodbye or clearly wants to hang up.
- asks_about_appointment: true when the caller asks when their own appointment or meeting with
  {settings.owner_name} is. You never know appointment times yourself, so never guess one.
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
        availability: str | None = None,
        appointment_text: str | None = None,
    ):
        system_prompt = SYSTEM_PROMPT + caller_context
        # Calendar: the model only ever gets WHEN the owner is busy, as a
        # finished sentence. It never sees what the busy time is for.
        if availability:
            system_prompt += (
                f"\nAVAILABILITY (from the calendar, fine to tell any caller): {availability}\n"
                f"Never say or guess what {settings.owner_name} is doing, where he is, or why. "
                "Just say he's busy.\n"
            )
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
        self._prefixes: list[tuple[str, str]] = []   # facts code says first (see reply)
        self._reconfirm_next = False   # read the details back again next turn (see reply)
        self.wants_transfer = False    # urgent + confirmed: main.py dials the owner
        self.availability = availability          # e.g. "Simrat is busy until 3 PM." or None (free)
        self._appointment_text = appointment_text  # spoken by code, never shown to the model
        self._appointment_answered = False
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
        # Facts code adds in front of whatever is said next, so they're said
        # even when code replaces the model's reply (read-back, notes, goodbye).
        # Each: (text to add, skip it if the reply already contains this).
        self._prefixes = []
        if "?" in caller_text and AI_QUESTION.search(caller_text):
            self._prefixes.append((HONESTY_LINE, "AI"))
        if self.availability and WHEN_FREE.search(caller_text):
            busy_time = self.availability.split(" until ")[-1].rstrip(".")   # "3 PM"
            self._prefixes.append((self.availability + " ", busy_time))

        turn = await self._ask_model()
        if turn is None:
            return self._said("Sorry, could you say that again?"), False

        # The history gets what the assistant actually SAID (see _said), as
        # plain text, not the model's raw JSON. Shorter prompts = faster replies,
        # and when our code overrides the model (read-back, notes, goodbye),
        # the history still matches what the caller really heard.
        details_before = self.details.model_copy()
        self._merge(turn)
        print(f"  details so far: {self.details.model_dump()}")

        # A reply that opens with "no", "wrong", "actually"... is never a
        # confirmation, whatever the model says. Checked in code, because a
        # message delivered with confident wrong details is worse than none.
        # ...but only when it answers the read-back. "No, that covers it" in
        # reply to "anything else?" is not a correction.
        last_nova_line = self.transcript[-2][1] if len(self.transcript) >= 2 else ""
        answering_read_back = bool(self._last_read_back) and last_nova_line.endswith(self._last_read_back)
        disagreed = bool(DISAGREEMENT.search(caller_text)) or (
            bool(LEADING_NO.search(caller_text)) and (answering_read_back or not self._read_back_done))
        # A correction after the read-back that the model missed: try code first.
        if (self._read_back_done and self.details == self._read_back_snapshot
                and CORRECTION_CUE.search(caller_text)):
            fixed = code_correction(caller_text, self.details)
            if fixed:
                field, value = fixed
                print(f"  code caught a correction the model missed: {field} -> {value!r}")
                setattr(self.details, field, value)

        matches_read_back = self._read_back_done and self.details == self._read_back_snapshot
        # Set last turn when the caller answered the read-back with something
        # other than a clean yes ("Yeah, but can you book it?"): we answered
        # them, and now the read-back must be heard again before it counts.
        reconfirm_now, self._reconfirm_next = self._reconfirm_next, False
        # "Yes" to the read-back = the details are confirmed (completed), even
        # if they tack on a question. "Yeah, but..." is not a yes.
        # It must answer the read-back itself: a "yes" to something the model
        # said in between ("so it's the quote then?") confirms nothing we stored.
        if (matches_read_back and answering_read_back and turn.caller_confirmed
                and not disagreed and not YES_BUT.search(caller_text)):
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
                # Details taken from the leaky turn may be leaky too (a reason
                # like "the interview" came from Sara's history, not Kevin).
                # Undo them, keep only the name they gave, take the retry's.
                other_name = self.details.name
                self.details = details_before
                self.details.name = other_name
                turn = retry
                self._merge(turn)

        # 1. Details confirmed and nothing left to answer: say goodbye.
        #    If they asked something ("is there a confirmation number?"),
        #    answer it first (step 5); the goodbye comes on a later turn.
        if self.completed and matches_read_back and not disagreed and "?" not in caller_text:
            first_name = self.details.name.split()[0]
            # Urgent: instead of goodbye, try to put them through (main.py
            # does the transfer once this line has been heard).
            if settings.transfer_enabled and self.details.urgency == "urgent":
                if self.availability:
                    # Calendar says busy: don't ring him, but make sure he sees it first
                    return self._said(
                        f"{self.availability} I've marked your message as urgent, "
                        f"so it's the first thing he'll see. Take care, {first_name}."
                    ), True
                self.wants_transfer = True
                return self._said(
                    f"Since it's urgent, let me try to put you through to {settings.owner_name}. "
                    "One moment."
                ), True
            return self._said(random.choice([
                f"Perfect, I'll make sure {settings.owner_name} gets this. Thanks {first_name}, bye now.",
                f"Great, I'll pass that on to {settings.owner_name}. Have a good one, {first_name}.",
                f"All set, {first_name}. I'll let {settings.owner_name} know. Take care.",
            ])), True

        # 1b. They said the read-back was wrong, but nothing changed: the model
        #     missed the correction. Ask plainly instead of carrying on.
        if self._read_back_done and disagreed and self.details == self._read_back_snapshot:
            self.completed = False   # an earlier "yes" no longer stands
            # Second chance: ask the model again, telling it plainly that this
            # is a correction. ("Actually, it's the quote, not the invoice"
            # was once answered with "so it's the quote" while we still had
            # "invoice" stored, and the caller then confirmed the wrong thing.)
            retry = await self._ask_model(
                "The caller is correcting the details that were read back. "
                "Fill in the corrected value(s) from their latest message.")
            if retry:
                self._merge(retry)
            if self.details != self._read_back_snapshot:
                turn = retry   # fixed: step 4 reads the corrected details back
            elif "?" in caller_text and not CORRECTION_CUE.search(caller_text):
                # "No, I need his cell number. Can you help?" pushes back on
                # something else, not the details: let the model answer that.
                # The read-back must be heard again before anything counts.
                self._read_back_snapshot = None
                return self._said((retry or turn).reply), False
            else:
                return self._said(random.choice([
                    "Sorry about that. What should I change?",
                    "Oh, sorry. What did I get wrong?",
                ])), False

        # 1c. "When is my appointment?" Answered by code, never by the model.
        #     Only a caller whose identity code confirmed hears a time.
        # The model sometimes flags "That's all, thanks" as asking again;
        # after one answer, only a real question gets it repeated.
        asked_again_for_real = "?" in caller_text or not self._appointment_answered
        if turn.asks_about_appointment and asked_again_for_real:
            if not self.details.reason:
                self.details.reason = "asked about their appointment"   # for your SMS
            if self.identity_confirmed and self._appointment_text:
                answer = self._appointment_text
                self._appointment_answered = True
            elif self.identity_confirmed and (self.delivered_note_ids or self._has_pending_notes()):
                answer = ""   # the note they're about to hear is the answer
            elif self.identity_confirmed:
                answer = f"I don't see anything booked for you right now. I'll let {settings.owner_name} know you asked."
            else:
                answer = (f"I'm not able to share appointment details over the phone, "
                          f"but I'll let {settings.owner_name} know you asked.")
            return self._said(
                (self._pending_notes_text() + answer).strip() + " Is there anything else you'd like me to pass on?"
            ), False

        # 2. Caller wants to go, or the call has gone on too long: wrap up
        #    politely with whatever we have. Exception: everything is known
        #    but was never read back ("...no rush. Thanks!"). One quick
        #    read-back is worth it, so fall through to step 4.
        wants_out = turn.caller_wants_to_end or self._caller_turns >= MAX_CALLER_TURNS
        # ...unless we just gave them what they called for (a note, their
        # appointment time): then "that's all, thanks" really means goodbye.
        got_their_answer = bool(self.delivered_note_ids) or self._appointment_answered
        quick_read_back = (not self.details.missing(self._required) and not self._read_back_done
                           and self._caller_turns < MAX_CALLER_TURNS and not got_their_answer)
        if wants_out and not quick_read_back:
            if self.details.reason:
                self._fill_defaults()   # so the SMS still has a callback number
                goodbye = f"No problem, I'll pass on your message to {settings.owner_name}. Goodbye."
            else:
                goodbye = "No problem, thanks for calling. Goodbye."   # nothing to pass on
            return self._said(self._pending_notes_text() + goodbye), True

        # 3. Identity confirmed by code: pass on shareable notes, word for word,
        #    once each. The model never saw the notes, so its reply may not fit
        #    after them ("what time are you hoping to confirm?"): use our own.
        notes_text = self._pending_notes_text()
        if notes_text:
            return self._said(
                notes_text + f"Was there anything else you wanted me to pass on to {settings.owner_name}?"
            ), False

        # 4. All details known, and not yet read back in this exact form
        #    (first time, or the caller just corrected something): our code
        #    writes the read-back itself, so it always contains every detail.
        if not self.details.missing(self._required):
            self._fill_defaults()
        if not self.details.missing(self._required) and (
                self.details != self._read_back_snapshot or (reconfirm_now and not self.completed)):
            self._read_back_done = True
            self.completed = False   # new details: they need confirming again
            self._corrections_allowed = True
            self._read_back_snapshot = self.details.model_copy()
            self._last_read_back = self._read_back_text()
            return self._said(self._last_read_back), False

        if answering_read_back and not self.completed:
            self._reconfirm_next = True
        return self._said(turn.reply), False

    def _has_pending_notes(self) -> bool:
        return self.identity_confirmed and any(
            i not in self.delivered_note_ids for i, _ in self._shareable_notes)

    def _pending_notes_text(self) -> str:
        """Shareable notes not yet delivered, as one spoken sentence (or "").
        Only ever for a caller whose identity our code confirmed."""
        if not self.identity_confirmed:
            return ""
        pending = [(i, t) for i, t in self._shareable_notes if i not in self.delivered_note_ids]
        if not pending:
            return ""
        self.delivered_note_ids.extend(i for i, _ in pending)
        notes = " ".join(_as_sentence(t) for _, t in pending)
        return f"{settings.owner_name} asked me to pass on a message: {notes} "

    async def _ask_model(self, extra_note: str = "") -> TurnOutput | None:
        """One model call with the current history. None if the output was unusable."""
        # The status note is added for this request only, never stored in
        # history, so the model always sees the CURRENT status, not old ones.
        status = self._status_note() + (" " + extra_note if extra_note else "")
        response = await chat(
            messages=self.messages + [{"role": "system", "content": status}],
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

        # The model sometimes misses a confirmation buried in a longer
        # sentence ("Yes, this is Sara. What time is my appointment?").
        # So code also accepts a plain yes to the greeting's question, or the
        # caller naming themselves as the contact.
        said_yes = self._caller_turns == 1 and bool(YES_START.search(self.transcript[-1][1]))
        named_self = bool(re.search(
            rf"\b(this is|it's|it is|i'm|i am)\s+{re.escape(known_first)}\b",
            self.transcript[-1][1], re.IGNORECASE))
        if (turn.identity_confirmed or said_yes or named_self) and not disagreed:
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
        for prefix, skip_if in reversed(self._prefixes):
            if skip_if not in text:
                text = prefix + text
        self._prefixes = []
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
            if field == "reason" and is_vague_reason(value):
                continue   # "something urgent" isn't a reason: keep asking
            current = getattr(self.details, field)
            if current and not self._corrections_allowed:
                if value != current:
                    print(f"  ignored change to {field}: {current!r} -> {value!r} (locked until read-back)")
                continue
            setattr(self.details, field, value)


# "Am I talking to a real person?" must always get a straight answer, even
# when our code replaces the model's reply (read-back, notes, goodbye).
AI_QUESTION = re.compile(
    r"\b(real person|(a|an) (real )?(human|person|robot|bot|machine|ai)|"
    r"are you (human|real|a robot|a bot|ai|an ai)|automated)\b",
    re.IGNORECASE,
)
HONESTY_LINE = "No, I'm an AI assistant, not a person. "

# A plain yes at the start of a sentence: "Yes", "Yeah it's me", "Speaking"
YES_START = re.compile(r"^\W*(yes|yeah|yep|yup|speaking|that's me|it's me|correct)\b", re.IGNORECASE)

# "It's the quote, not the invoice": a correction, even with a "?" on the end.
# If we failed to pick up the new value, never let the model's reply pretend
# we did ("Got it, the quote" while "invoice" is still stored).
CORRECTION_CUE = re.compile(r"\bnot (the|a|an|my|that|about)\b|\binstead\b|\bit's (actually|about)\b|\bchange it\b", re.IGNORECASE)

# Code reads the most common correction phrasings itself, because the model
# often misses them (eval: "it's the quote, not the invoice" left "invoice"
# stored, three runs out of five).
_NEW = r"(?P<new>[\w' -]{2,40}?)"
_OLD = r"(?P<old>[\w' -]{2,40})"
CORRECTION_PATTERNS = [
    re.compile(r"\b(?:it's|it is|its)\s+(?:actually\s+)?(?:about\s+|regarding\s+)?" + _NEW + r",?\s+not\s+(?:about\s+)?" + _OLD, re.IGNORECASE),
    re.compile(_NEW + r"\s+instead of\s+" + _OLD, re.IGNORECASE),
]
_LEAD_IN = re.compile(r"^.*\b(?:to|is|about|actually|it's|its)\s+", re.IGNORECASE)
_ARTICLE = re.compile(r"^(?:the|a|an|my)\s+", re.IGNORECASE)


def code_correction(text: str, details: "CallDetails") -> tuple[str, str] | None:
    """'it's the quote, not the invoice' with reason 'the invoice'
    -> ('reason', 'the quote'). None if nothing matches a stored detail."""
    for pattern in CORRECTION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        new = _ARTICLE.sub("", _LEAD_IN.sub("", match["new"].strip())).strip()
        old = _ARTICLE.sub("", match["old"].strip()).split("  ")[0].strip()
        if not new or not old:
            continue
        for field in ("reason", "name"):
            current = getattr(details, field) or ""
            if re.search(rf"\b{re.escape(old)}\b", current, re.IGNORECASE):
                return field, re.sub(rf"\b{re.escape(old)}\b", new, current, flags=re.IGNORECASE)
    return None


# "When will he be free?", "Is he around?" -> the calendar answers that
WHEN_FREE = re.compile(
    r"\b(when|what time)\b.{0,40}\b(free|available|back|around|done|finish|reach|call)"
    r"|\b(is he|is " + re.escape(settings.owner_name) + r")\s+(free|available|around|busy|in)\b",
    re.IGNORECASE,
)

# "Yeah, but can you...?" is not a clean yes. Neither is a yes with a question.
YES_BUT = re.compile(r"^\W*(yes|yeah|yep|sure|okay|ok|right)\b\W*but\b", re.IGNORECASE)

# Signs the caller is disagreeing or correcting, not confirming
# A leading "no" / "actually" / "wait" only means "you got it wrong" when it
# answers the read-back. ("No, that covers it" answers "anything else?")
LEADING_NO = re.compile(r"^\W*(no|nope|nah|not|actually|wait|hold on)\b", re.IGNORECASE)
# These mean "wrong" wherever they appear
DISAGREEMENT = re.compile(
    r"^\W*(wrong|incorrect)\b"
    r"|\b(not right|not correct|that's wrong|that is wrong|isn't right|is not right)\b",
    re.IGNORECASE,
)


# Small models sometimes write the WORD "null" instead of a JSON null.
# "null" is a non-empty string, so Python treats it as a real value.
# A "reason" made only of these words says nothing about what the call is about
VAGUE_WORDS = {
    "a", "an", "the", "some", "something", "stuff", "matter", "thing", "issue",
    "question", "personal", "private", "important", "urgent", "very", "really",
    "business", "quick", "small", "just",
}


def is_vague_reason(reason: str) -> bool:
    words = re.findall(r"[a-z']+", reason.lower())
    return all(w in VAGUE_WORDS for w in words)


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
    if len(digits) == 8 and digits.startswith("1"):
        digits = digits[1:]                   # the model turned "555-1234" into "+15551234"
    if len(digits) == 7:
        return f"{digits[:3]}-{digits[3:]}"   # no area code: keep it as said, never "+1555..."
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
    # "urgent pipe burst" -> "pipe burst": the read-back already says "it's urgent"
    cleaned = re.sub(r"^(an? )?(urgent|important)\s+", lambda m: "a " if m.group(1) else "",
                     cleaned, flags=re.IGNORECASE) or cleaned
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