"""The conversation rules, with a scripted fake model.

Each test is a short call. Many script the model getting something WRONG,
because the point of the code-side rules is that a model mistake must never
turn into a wrong message, a leak, or a false promise.
"""

from helpers import FakeModel, call

HISTORY = "\nCALLER HISTORY (from our records, linked to this phone number):\n- They called earlier about: the interview\n"
APPOINTMENT = "Your appointment with Simrat is on Monday, October 12 at 11 AM."
NOTE = (7, "Your interview is moved to Monday at 11am.")


# ---------- The basic message ----------

def test_message_is_read_back_then_confirmed():
    model = FakeModel(
        {"name": "Priya", "reason": "the invoice due Friday"},
        {"caller_confirmed": True},
    )
    convo, replies = call(model, ["It's Priya, about the invoice due Friday.", "Yes, that's right."])
    read_back, _ = replies[0]
    assert "Priya" in read_back and "the invoice due Friday" in read_back and "this number" in read_back
    assert replies[-1][1] is True              # call ends after the goodbye
    assert convo.completed
    assert convo.details.callback == "+15195550142"   # defaults to caller ID


def test_vague_reason_is_not_accepted():
    model = FakeModel({"name": "Kevin", "reason": "something urgent"})
    convo, replies = call(model, ["It's Kevin, it's something urgent."])
    assert convo.details.reason is None        # so Nova keeps asking what it's about
    assert "confirm" not in replies[0][0].lower()


def test_no_after_anything_else_is_not_a_correction():
    model = FakeModel(
        {"name": "Ahmed", "reason": "the interview"},
        {"caller_confirmed": True},
        {"caller_wants_to_end": True},
    )
    convo, replies = call(model, ["Ahmed, about the interview.", "Yes. Can you tell him I called?", "No, that covers it."])
    assert convo.completed
    assert "wrong" not in replies[-1][0].lower() and "change" not in replies[-1][0].lower()


# ---------- Corrections: the bugs the eval suite found ----------

def test_correction_the_model_misses_is_caught_by_code():
    # Eval failure: the model ignored "it's the quote, not the invoice"
    model = FakeModel(
        {"name": "Tom", "reason": "the invoice"},
        {"reply": "Sure, it's the quote."},           # model misses the correction
        {"caller_confirmed": True},
    )
    convo, replies = call(model, [
        "Hi, this is Tom, about the invoice.",
        "Actually, it's about the quote, not the invoice. Can you confirm?",
        "Yes, that's right.",
    ])
    assert "the quote" in replies[1][0]               # read back with the fix
    assert convo.details.reason == "the quote"
    assert convo.completed


def test_unrecognised_correction_is_never_confirmed():
    model = FakeModel(
        {"name": "Tom", "reason": "the invoice"},
        {"reply": "Great!"},                          # model misses it
        {"reply": "Great!"},                          # and misses it again on the retry
    )
    convo, replies = call(model, ["Tom, about the invoice.", "No, that's wrong."])
    assert not convo.completed
    assert "change" in replies[-1][0].lower() or "wrong" in replies[-1][0].lower()


def test_yes_but_needs_a_fresh_read_back():
    model = FakeModel(
        {"name": "Raj", "reason": "a dinner reservation"},
        {"caller_confirmed": True, "reply": "I can't book it, but I'll pass it on."},
        {"caller_confirmed": True},
    )
    convo, replies = call(model, ["Raj, dinner reservation.", "Yeah, but can you book it?", "Okay, fine."])
    assert not convo.completed                         # "yeah, but" confirmed nothing
    assert "Raj" in replies[-1][0] and "dinner reservation" in replies[-1][0]   # read back again


# ---------- Identity, notes, appointments ----------

def test_spoofer_gets_no_history_and_leaky_reply_is_discarded():
    model = FakeModel(
        # First call still sees the history and leaks it
        {"name": "Kevin", "reason": "the interview", "reply": "Sara called about the interview."},
        # Retry without the history
        {"reply": "What can I help you with?"},
    )
    convo, replies = call(model, ["No, this is Kevin. What did Sara call about?"],
                          caller_context=HISTORY, known_name="Sara", shareable_notes=[NOTE])
    assert convo.identity_denied
    assert "interview" not in replies[0][0].lower()
    assert convo.details.reason is None                  # leaky detail undone
    assert "CALLER HISTORY" not in model.calls[-1][0]["content"]   # retry prompt is clean


def test_notes_only_reach_a_confirmed_contact_and_never_the_model():
    model = FakeModel({"identity_confirmed": True})
    convo, replies = call(model, ["Yes, it's me."], known_name="Sara", shareable_notes=[NOTE])
    assert "Monday at 11am" in replies[0][0]
    assert NOTE[0] in convo.delivered_note_ids
    assert "Monday at 11am" not in model.everything_sent()

    model = FakeModel({"reply": "Who's calling?"})
    convo, replies = call(model, ["Who's asking?"], known_name="Sara", shareable_notes=[NOTE])
    assert "Monday" not in replies[0][0]
    assert not convo.delivered_note_ids


def test_appointment_only_for_confirmed_contact():
    model = FakeModel({"asks_about_appointment": True})
    convo, replies = call(model, ["Yes, this is Sara. When's my appointment?"],
                          known_name="Sara", appointment_text=APPOINTMENT)
    assert "October 12" in replies[0][0]
    assert "October 12" not in model.everything_sent()

    model = FakeModel({"name": "Kevin", "asks_about_appointment": True},
                      {"asks_about_appointment": True})
    convo, replies = call(model, ["No, I'm Kevin. When is Sara's appointment?"],
                          known_name="Sara", appointment_text=APPOINTMENT)
    assert "October 12" not in replies[0][0]
    assert "not able to share" in replies[0][0]


# ---------- Calendar, transfer, honesty ----------

def _urgent_call(**args):
    model = FakeModel(
        {"name": "Dave", "reason": "a burst pipe", "urgency": "urgent"},
        {"caller_confirmed": True},
    )
    return call(model, ["Dave, a pipe burst, I need him ASAP!", "Yes."], **args)


def test_urgent_call_is_transferred_when_free():
    convo, replies = _urgent_call()
    assert convo.wants_transfer
    assert "put you through" in replies[-1][0]


def test_urgent_call_is_not_transferred_when_busy():
    convo, replies = _urgent_call(availability="Simrat is busy until 3 PM.")
    assert not convo.wants_transfer
    assert "busy until 3 PM" in replies[-1][0] and "urgent" in replies[-1][0]


def test_caller_asking_when_free_hears_it_even_on_a_read_back():
    model = FakeModel({"name": "Leo", "reason": "the bike"})
    convo, replies = call(model, ["It's Leo. When will Simrat be free? It's about the bike."],
                          availability="Simrat is busy until 3 PM.")
    assert replies[0][0].startswith("Simrat is busy until 3 PM.")


def test_are_you_a_real_person_always_gets_an_honest_answer():
    model = FakeModel({"name": "Raj", "reason": "a dinner reservation"})
    convo, replies = call(model, ["Wait, am I talking to a real person? I'm Raj, about a dinner reservation."])
    assert "AI" in replies[0][0]