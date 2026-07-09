"""Multi-session non-money measure summation (weight/distance/volume) -- the LME-S
aggregation fix. A 'total weight I purchased' question must add pounds/kg across sessions,
never misroute to a money sum ('$28') or a count ('one'). Pure-function tests on synthetic
atoms (no store)."""
import pytest

from eidetic.smqe.record_ops import _measure_type_sum_answer


class _Item:
    def __init__(self, key):
        self.key = key
        self.source_memory_id = key


def _atoms(*pairs):
    # (score, item, atom_text)
    return [(1.0 - i * 0.05, _Item(f"s{i}"), t) for i, t in enumerate(pairs)]


def test_weight_sum_adds_pounds_and_ignores_dollars():
    ans, sel = _measure_type_sum_answer(
        "What is the total weight of the new feed I purchased?",
        _atoms("I bought 40 pounds of feed Monday.",
               "Picked up another 30 pounds of feed later.",
               "Also grabbed a $28 supplement."))
    assert ans == "70 pounds"          # 40+30, the $28 is NOT summed
    assert len(sel) == 2


def test_distance_sum():
    ans, _ = _measure_type_sum_answer(
        "total distance I ran this week?",
        _atoms("Ran 5 miles Tuesday.", "Ran 8 miles Thursday."))
    assert ans == "13 miles"


def test_volume_sum_liters():
    ans, _ = _measure_type_sum_answer(
        "how much volume of water did I use in total?",
        _atoms("Used 2 liters in the morning.", "Another 3 liters at night."))
    assert ans == "5 liters"


def test_singular_unit_label():
    ans, _ = _measure_type_sum_answer(
        "total weight?", _atoms("It weighed 1 pound."))
    assert ans == "1 pound"


def test_non_measure_question_does_not_fire():
    assert _measure_type_sum_answer("how many dogs do I have?",
                                    _atoms("I have 3 dogs."))[0] == ""


def test_no_units_in_atoms_returns_empty_so_caller_can_abstain():
    # weight question but atoms carry no mass units -> empty -> multi_session_sum abstains
    assert _measure_type_sum_answer("total weight of feed?",
                                    _atoms("I bought some feed."))[0] == ""


def test_subject_gate_excludes_unrelated_weights_when_subject_appears():
    # The 1226.3-vs-70 corruption: retrieved atoms mix in-scope feed weights with UNRELATED
    # weights (body weight, gym plates). When the subject ('feed') appears in some atoms, sum
    # ONLY the subject-relevant ones -- do not blindly add every pound on the page.
    ans, sel = _measure_type_sum_answer(
        "What is the total weight of the new feed I purchased in the past two months?",
        _atoms("I bought a 50-pound batch of layer feed for the hens.",
               "Picked up another 20 pounds of feed later that month.",
               "At the gym I deadlifted 315 pounds.",
               "I weigh 180 pounds these days.",
               "My dog is about 60 pounds now."))
    assert ans == "70 pounds"          # 50+20 feed only; 315+180+60 excluded
    assert len(sel) == 2


def test_subject_silent_atoms_still_sum_when_subject_appears_nowhere():
    # Fallback: if the subject term never appears in any atom (the amount sentences omit it),
    # the conditional gate must NOT over-filter -- sum all measured units as before.
    ans, _ = _measure_type_sum_answer(
        "how much volume of water did I use in total?",
        _atoms("Used 2 liters in the morning.", "Another 3 liters at night."))
    assert ans == "5 liters"           # neither atom says 'water' -> gate falls back to all


@pytest.mark.xfail(reason="known limitation (task #42): the subject gate drops a cross-session "
                          "subject-SILENT amount ('another 20 pounds' in a different memory that "
                          "does not name 'feed'), so 70 undercounts to 50. Group fallback only "
                          "rescues same-memory siblings. Needs live-failure-set measurement + a "
                          "cross-session subject-carry heuristic before widening.", strict=True)
def test_cross_session_subject_silent_amount_is_a_known_undercount():
    # Documents (not hides) the residual risk: the correct total is 70, the conservative gate
    # returns 50 because the 2nd session's amount sentence omits 'feed'. xfail => the day a
    # future fix makes this pass, strict=True flips it red so we update the record.
    ans, _ = _measure_type_sum_answer(
        "What is the total weight of the new feed I purchased?",
        [(1.0, _Item("m1"), "I bought a 50-pound batch of layer feed for the hens."),
         (0.9, _Item("m2"), "Picked up another 20 pounds later that month.")])
    assert ans == "70 pounds"          # ideal; currently returns "50 pounds" -> xfail
