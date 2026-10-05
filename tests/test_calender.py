"""Calendar logic with fake Google responses: no key, no network."""

from datetime import datetime, timezone

import calender_check as cc


def utc(hour, minute=0, day=5):
    return datetime(2026, 10, day, hour, minute, tzinfo=timezone.utc)


NOW = utc(18)   # 2 PM in Toronto


def test_free_at():
    cases = [
        ([], None),                                                          # nothing booked
        ([(utc(20), utc(21))], None),                                        # busy later, free now
        ([(utc(17), utc(19))], utc(19)),                                     # busy now
        ([(utc(17), utc(19)), (utc(19), utc(20))], utc(20)),                 # back-to-back
        ([(utc(17), utc(19)), (utc(18, 30), utc(19, 30))], utc(19, 30)),     # overlapping
        ([(utc(17), utc(19)), (utc(19, 15), utc(20))], utc(19)),             # a gap: free at 19:00
        ([(utc(17), utc(18))], None),                                        # ended exactly now
    ]
    for blocks, expected in cases:
        assert cc.free_at(blocks, NOW) == expected, blocks


def test_spoken_times_in_owner_timezone():
    assert cc.availability_sentence(utc(19, 30), NOW) == "Simrat is busy until 3:30 PM."
    assert cc.availability_sentence(utc(13, day=6), NOW) == "Simrat is busy until tomorrow at 9 AM."
    assert cc.availability_sentence(None, NOW) is None
    assert cc.appointment_sentence([utc(15, day=12)], NOW) == \
        "Your appointment with Simrat is on Monday, October 12 at 11 AM."


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_appointments_match_the_callers_number_only():
    events = {"items": [
        {"description": "nova: +1 (519) 555-0142", "start": {"dateTime": "2026-10-12T15:00:00Z"}},
        {"description": "nova: 226-555-0199", "start": {"dateTime": "2026-10-13T15:00:00Z"}},
        {"description": "nova: 5195550142", "start": {"date": "2026-10-14"}},   # all-day: skipped
        {"start": {"dateTime": "2026-10-15T15:00:00Z"}},                       # no description
    ]}
    original_get, original_token = cc.httpx.get, cc._token
    original_id = cc.settings.google_nova_calendar_id
    cc.httpx.get = lambda *a, **k: FakeResponse(events)
    cc._token = lambda: "fake-token"
    cc.settings.google_nova_calendar_id = "nova@group.calendar.google.com"
    try:
        assert cc.appointments_for("+15195550142", NOW) == [utc(15, day=12)]
        assert cc.appointments_for("anonymous", NOW) == []
    finally:
        cc.httpx.get, cc._token = original_get, original_token
        cc.settings.google_nova_calendar_id = original_id


def test_calendar_problems_never_break_a_call():
    assert cc.call_facts("+15195550142", True) == (None, None)   # not configured
    original_id, original_busy = cc.settings.google_calendar_id, cc.busy_until
    cc.settings.google_calendar_id = "me@gmail.com"
    cc.busy_until = lambda now: (_ for _ in ()).throw(RuntimeError("Google is down"))
    try:
        assert cc.call_facts("+15195550142", True) == (None, None)
    finally:
        cc.settings.google_calendar_id, cc.busy_until = original_id, original_busy