"""A fake LLM for tests: no Ollama, no GPU, same answer every run.

The real model is replaced by a script of outputs, one per model call. That
lets a test say "suppose the model MISSES this correction" and check that
our code still does the safe thing. Those are exactly the cases the eval
suite found, and here they're checked in milliseconds on every push.
"""

import asyncio
import json

import llm

# Every field the model returns, empty (the reply is filled per test)
BLANK = {name: field.default for name, field in llm.TurnOutput.model_fields.items() if name != "reply"}


class FakeModel:
    def __init__(self, *outputs: dict):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []   # the messages sent on each call

    async def __call__(self, **kwargs):
        self.calls.append(kwargs["messages"])
        out = {**BLANK, "reply": "Okay."}
        if self.outputs:
            out.update(self.outputs.pop(0))
        return {"message": {"content": json.dumps(out)}}

    def everything_sent(self) -> str:
        """All text the model was ever shown, to prove secrets never reach it."""
        return " ".join(m["content"] for call in self.calls for m in call)


def call(model: FakeModel, lines: list[str], **conversation_args) -> tuple[llm.Conversation, list[tuple[str, bool]]]:
    """Play a call: each caller line in turn. Returns the conversation and
    Nova's replies as (text, end_call). Stops when Nova ends the call."""
    llm.chat = model   # reply() looks up `chat` in the llm module at call time
    conversation = llm.Conversation(
        conversation_args.pop("greeting", "Hi, this is Nova."),
        conversation_args.pop("caller_number", "+15195550142"),
        **conversation_args,
    )
    replies = []

    async def play():
        for line in lines:
            reply, end = await conversation.reply(line)
            replies.append((reply, end))
            if end:
                break

    asyncio.run(play())
    return conversation, replies