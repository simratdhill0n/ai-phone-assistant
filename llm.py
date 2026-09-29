"""The conversation brain: talks to a local LLM through Ollama."""

from ollama import AsyncClient

from config import settings

# The model signals it has everything it needs by ending its reply with this.
END_CALL_TOKEN = "[END_CALL]"

SYSTEM_PROMPT = f"""You are {settings.assistant_name}, the AI phone assistant for {settings.owner_name}.
{settings.owner_name} is unavailable, and you are taking a message on a live phone call.

Your goal is to find out, one question at a time:
1. The caller's name
2. The reason for the call
3. How urgent it is
4. The best number or time to call them back

Rules:
- You are speaking on the phone. Reply in one or two short sentences, in plain spoken language.
- No lists, markdown, emojis or special characters. Your text is read aloud.
- Ask only one question at a time. If an answer is unclear, ask a short follow-up.
- If you already know something, don't ask for it again.
- Never share personal information about {settings.owner_name}, and never promise what he will do.
- If asked, say honestly that you are an AI assistant.
- Once you have all four details, briefly confirm them, say goodbye, and end your reply with {END_CALL_TOKEN}
"""

client = AsyncClient()  # connects to the Ollama server at http://localhost:11434


class Conversation:
    """One phone call's conversation. Holds the full message history."""

    def __init__(self, greeting: str):
        # The LLM remembers nothing between requests, so we keep the whole
        # conversation here and send all of it every time.
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": greeting},
        ]

    async def reply(self, caller_text: str) -> tuple[str, bool]:
        """Add what the caller said, get the assistant's reply.

        Returns (reply_text, end_call). end_call is True when the assistant
        has collected everything and said goodbye.
        """
        self.messages.append({"role": "user", "content": caller_text})

        # Ollama is a separate server, so this is network I/O, not CPU work
        # in our process. A plain await is enough, no asyncio.to_thread.
        response = await client.chat(
            model=settings.ollama_model,
            messages=self.messages,
            options={
                "temperature": 0.4,   # lower = more focused, less random
                "num_predict": 120,   # hard cap on reply length (tokens)
            },
            keep_alive="30m",         # keep the model loaded on the GPU
        )
        text = response["message"]["content"].strip()

        end_call = END_CALL_TOKEN in text
        text = text.replace(END_CALL_TOKEN, "").strip()

        self.messages.append({"role": "assistant", "content": text})
        return text, end_call