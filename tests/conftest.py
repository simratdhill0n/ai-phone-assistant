"""Runs before any test module is imported.

The app reads its settings from .env files and environment variables. Tests
must never depend on your real .env (or need Twilio keys), so we point
APP_ENV at a file that doesn't exist and supply fake values here.
"""

import os

TEST_ENV = {
    "APP_ENV": "test",                       # -> .env.test, which doesn't exist
    "PUBLIC_HOST": "test.example.com",
    "TWILIO_ACCOUNT_SID": "ACtest",
    "TWILIO_AUTH_TOKEN": "test-token",
    "TWILIO_PHONE_NUMBER": "+12265550100",
    "OWNER_NAME": "Simrat",
    "OWNER_PHONE": "+15485550177",
    "ASSISTANT_NAME": "Nova",
    "OWNER_TIMEZONE": "America/Toronto",
    "TRANSFER_ENABLED": "true",
    "GOOGLE_CALENDAR_ID": "",                # calendar feature off unless a test turns it on
}
for key, value in TEST_ENV.items():
    os.environ[key] = value