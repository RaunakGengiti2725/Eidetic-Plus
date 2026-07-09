"""Numeric computed-op TYPE floor: a count/sum/elapsed-delta answer must carry a number, so
non-numeric garbage ("Apply the Dr" for a money sum) can no longer ship verified=True.
General verified-precision hardening (benchmark-agnostic); event_order/table_lookup exempt."""
from eidetic.models import NLILabel, StructuredAnswerResult, StructuredSupport
from eidetic.smqe.verify import answer_from_result


class _Rec:
    def __init__(self, text):
        self.text = text
        self.summary = text[:40]
        self.memory_id = "mem_x"
        self.content_hash = "c" * 64
        self.raw_uri = "raw://mem_x"
        self.source = "test"
        self.valid_at = 1_700_000_000.0


class _Retr:
    """Stub retriever: NLI always entails, extractive/anchor helpers absent -> forces the
    verify path to rely on the type floor, not on lexical shortcuts."""
    def __init__(self, text):
        self._rec = _Rec(text)

    class _store:
        pass

    @property
    def store(self):
        s = _Retr._store()
        s.get_record = lambda mid: self._rec
        return s

    def _ground_truth(self, rec):
        return rec.text

    def verify_citation(self, rec, hyp):
        return NLILabel.ENTAILMENT, 1.0

    def verify(self, premise, hypothesis):
        return NLILabel.ENTAILMENT, 1.0


def _result(answer, op):
    return StructuredAnswerResult(
        answer=answer, op=op, backend="record",
        supports=[StructuredSupport(memory_id="mem_x", proof_atom=answer, answer_atom=answer,
                                    score=1.0)],
        confidence=0.9, note=f"smqe:{op}:record")


def test_non_numeric_money_sum_is_refused():
    # "Apply the Dr" for a total-spent question: no number -> refused (was verified-wrong)
    out = answer_from_result(_Retr("I spent $1,300 in total on the designer handbag."),
                             "What is the total amount I spent?",
                             _result("Apply the Dr", "multi_session_sum"), verify=True)
    assert out is None


def test_non_numeric_count_is_refused():
    out = answer_from_result(_Retr("I completed 23 pieces of writing in total."),
                             "How many total pieces of writing have I completed?",
                             _result("It's just like", "count_aggregate"), verify=True)
    assert out is None


def test_valid_numeric_count_passes_type_floor():
    # a real numeric count is not blocked by the type floor (it proceeds to NLI, which entails)
    out = answer_from_result(_Retr("James has three dogs: Ned, Daisy, Max."),
                             "How many dogs does James have?",
                             _result("three", "count_aggregate"), verify=True)
    assert out is not None and out.verified


def test_digit_sum_passes_type_floor():
    out = answer_from_result(_Retr("Total spent was $1,300 across the sessions."),
                             "What is the total amount I spent?",
                             _result("$1,300", "multi_session_sum"), verify=True)
    assert out is not None and out.verified


def test_event_order_not_subject_to_numeric_floor():
    # event_order answers are sequences, not numbers -> must NOT be blocked by the numeric floor
    out = answer_from_result(_Retr("First the studio repaint, then the housewarming."),
                             "What order did these happen?",
                             _result("studio repaint, then housewarming", "event_order"),
                             verify=True)
    assert out is not None    # passed the numeric floor (op exempt)
