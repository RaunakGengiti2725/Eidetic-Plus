"""Unit floor for `_clean_fact_form_credible`: a verified structured answer must be a
self-contained fact, not a raw dialogue turn shard. All fixtures are SYNTHETIC (no benchmark
strings) -- the rules are general answer-form properties, not memorized cases.

Two rejected shapes:
  1. a first-person pronoun+verb opening carrying no factual anchor;
  2. a comma-list that captured a '<Name>: <text>' dialogue turn-header.

Non-negotiable: real answers must PASS -- bare dates/counts, single proper nouns, first-person
clauses that carry an anchor (quoted title / proper noun / digit), and legitimate lists whose
items share a common noun.
"""
from eidetic.models import StructuredAnswerResult
from eidetic.smqe.verify import _clean_fact_form_credible


def _r(answer: str, op: str = "latest_value") -> StructuredAnswerResult:
    return StructuredAnswerResult(answer=answer, op=op, backend="claim", supports=[])


# --- REJECT: first-person conversational shard with no anchor -----------------------------
def test_rejects_bare_first_person_verb_opening():
    assert _clean_fact_form_credible("What did she photograph?", _r("I took")) is False


def test_rejects_pronoun_lead_filler_clause():
    assert _clean_fact_form_credible("Which tournament did he win?", _r("It's just like")) is False


def test_rejects_first_person_count_sentence_for_a_names_question():
    # A count sentence does not answer a 'what are the names' question; no proper-noun anchor.
    assert _clean_fact_form_credible(
        "What are the names of the dogs?", _r("I already have three at home")) is False


# --- PASS: first-person clauses that DO carry a factual anchor -----------------------------
def test_keeps_first_person_with_quoted_title_anchor():
    assert _clean_fact_form_credible(
        "What book is being read?", _r('I\'m currently reading "The Quiet Machine"')) is True


def test_keeps_first_person_with_proper_noun_anchor():
    assert _clean_fact_form_credible(
        "Where is the trip?", _r("I am going to Reykjavik next spring")) is True


def test_keeps_first_person_with_digit_anchor():
    assert _clean_fact_form_credible("How old is the car?", _r("It is 12 years old")) is True


# --- REJECT: comma-list that captured a dialogue turn-header -------------------------------
def test_rejects_list_with_captured_turn_header():
    assert _clean_fact_form_credible(
        "Which countries were visited?",
        _r("Norway, Iceland, Sam: Everything went great, and later")) is False


# --- PASS: legitimate answers that must never trip ----------------------------------------
def test_keeps_bare_date():
    assert _clean_fact_form_credible("When did it happen?", _r("2021")) is True


def test_keeps_single_proper_noun():
    assert _clean_fact_form_credible("Where did they go?", _r("Jasper")) is True


def test_keeps_list_with_shared_common_noun():
    # 'X shop and Y shop' repeats 'shop' by nature -- a real list, not garble.
    assert _clean_fact_form_credible(
        "Which shops?", _r("Cedar corner shop and Harbor corner shop")) is True


def test_keeps_multi_item_plain_list():
    assert _clean_fact_form_credible(
        "What are the goals?", _r("improve accuracy, win the league")) is True


# --- exemptions: computed ops and empty answers -------------------------------------------
def test_computed_op_answers_are_not_form_gated_here():
    # The call site exempts computed ops; the pure function still must not flag a bare value.
    assert _clean_fact_form_credible("How many?", _r("1", op="count_aggregate")) is True


def test_empty_answer_is_not_flagged_here():
    assert _clean_fact_form_credible("anything?", _r("")) is True
