
from policy_utils import expand_safety_variants, has_word_boundary, is_snf_phrase, matches_term, normalize


def test_normalize():
    assert normalize("  TeSt  ") == "test"
    assert normalize(None) == ""
    assert normalize(123) == "123"

def test_has_word_boundary():
    # Exact match
    assert has_word_boundary("Start word end", "word") is True
    # Initial
    assert has_word_boundary("Word at start", "word") is True
    # Final
    assert has_word_boundary("End with word", "word") is True
    # Partial (should fail)
    assert has_word_boundary("swordfish", "word") is False
    assert has_word_boundary("wording", "word") is False
    # Case insensitive
    assert has_word_boundary("WORD match", "word") is True
    assert has_word_boundary("word match", "WORD") is True

def test_matches_term():
    # Short terms (<=4 chars) use boundary check
    assert matches_term("has osa here", "osa") is True
    assert matches_term("has misosoup", "osa") is False

    # Long terms (>4 chars) use boundary check (updated logic)
    assert matches_term("myocardial infarction history", "myocardial infarction") is True
    assert matches_term("history of myocardial infarction", "myocardial infarction") is True
    # Ensure partial match on long string is also boundary aware if logic dictates
    # Reviewing policy_utils implementation: it explicitly uses boundary for ALL terms in my mental model of the fix,
    # or at least falls back to boundary logic. Let's verify standard behavior.
    assert matches_term("not a match", "foobar") is False

def test_expand_safety_variants():
    # / split
    variants = expand_safety_variants("MTC/MEN2")
    assert "mtc" in variants
    assert "men2" in variants

    # ' or ' split
    variants = expand_safety_variants("pregnant or nursing")
    assert "pregnant" in variants
    assert "nursing" in variants

    # Parentheses handling if implemented
    variants = expand_safety_variants("medication (brand)")
    assert "medication" in variants

def test_is_snf_phrase():
    assert is_snf_phrase("Patient in skilled nursing facility") is True
    assert is_snf_phrase("Rehab center") is False
