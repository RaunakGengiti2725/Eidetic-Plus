"""Multi-session non-money measure summation (weight/distance/volume) -- the LME-S
aggregation fix. A 'total weight I purchased' question must add pounds/kg across sessions,
never misroute to a money sum ('$28') or a count ('one'). Pure-function tests on synthetic
atoms (no store)."""
from eidetic.smqe.record_ops import _measure_type_sum_answer


class _Item:
    def __init__(self, key):
        self.key = key


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
