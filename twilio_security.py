"""Verify that incoming webhooks really come from Twilio.

Twilio signs every request with your Auth Token. We recompute the signature
and compare. Without this, anyone who learns your URL could fake calls or
texts to your server, including fake "notes" from you.

How Twilio's signature works:
1. Take the full URL Twilio called (https://your-host/sms)
2. Sort the POST form fields by name, append each name and value to the URL
3. HMAC-SHA1 that string, keyed with your Auth Token
4. Base64-encode the result. It must equal the X-Twilio-Signature header.
"""

import base64
import hashlib
import hmac

from fastapi import HTTPException, Request

from config import settings


def compute_signature(url: str, params: dict[str, str]) -> str:
    data = url + "".join(name + params[name] for name in sorted(params))
    digest = hmac.new(
        settings.twilio_auth_token.get_secret_value().encode(),
        data.encode(),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode()


async def verify_twilio(request: Request) -> None:
    """FastAPI dependency: rejects the request with 403 if the signature is wrong."""
    # Behind ngrok, request.url looks like http://127.0.0.1:8000/sms, but
    # Twilio signed the PUBLIC url. So rebuild the URL Twilio actually used.
    url = f"https://{settings.public_host}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    form = await request.form()
    params = {key: str(value) for key, value in form.items()}

    expected = compute_signature(url, params)
    received = request.headers.get("X-Twilio-Signature", "")

    # compare_digest takes the same time whether the strings match early or
    # late, so attackers can't guess the signature by timing our responses.
    if not hmac.compare_digest(expected, received):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")