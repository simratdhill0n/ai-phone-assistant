"""Small pure functions: easy to test exhaustively, and each one exists
because a real call or eval run went wrong without it."""

from llm import CallDetails, _clean_reason, _normalize_phone, code_correction, is_vague_reason


def test_clean_reason_reads_naturally_after_calling_about():
    assert _clean_reason("calling about the invoice") == "the invoice"
    assert _clean_reason("make a dinner reservation for Saturday") == "making a dinner reservation for Saturday"
    assert _clean_reason("an urgent business partnership") == "a business partnership"
    assert _clean_reason("the car he's selling") == "the car he's selling"


def test_phone_numbers_are_normalized_but_never_invented():
    assert _normalize_phone("519-555-0142") == "+15195550142"
    assert _normalize_phone("+1 (519) 555-0142") == "+15195550142"
    assert _normalize_phone("555-1234") == "555-1234"        # no area code: kept as said
    assert _normalize_phone("+15551234") == "555-1234"       # model's invented "+1"
    assert _normalize_phone("evenings") == "evenings"


def test_vague_reasons():
    for vague in ["something urgent", "an urgent matter", "personal", "a quick question"]:
        assert is_vague_reason(vague), vague
    for real in ["the interview", "something about the car", "the quote"]:
        assert not is_vague_reason(real), real


def test_code_correction_patterns():
    tom = CallDetails(name="Tom", reason="the invoice")
    assert code_correction("Actually, it's about the quote, not the invoice.", tom) == ("reason", "the quote")
    assert code_correction("Just change it to quote instead of invoice.", tom) == ("reason", "the quote")
    assert code_correction("It's Jon, not John.", CallDetails(name="John")) == ("name", "Jon")
    # Only changes something that's actually stored
    assert code_correction("I want it today instead of tomorrow", tom) is None
    assert code_correction("No, it's not about the invoice", tom) is None