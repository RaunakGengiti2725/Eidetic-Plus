"""Unit floors for the free-tier collection harness (bench/notebooklm_freetier_run.py).
Synthetic fixtures only. The live run is a separate, labeled artifact -- these tests pin
the deterministic pieces: the prelim heuristic's tolerance and the source packing caps."""
from bench.notebooklm_freetier_run import pack_record_sources, prelim_contains


class _Bridge:
    def __init__(self, sources):
        self._sources = sources

    def build_sources(self, namespace):
        return list(self._sources)


def test_prelim_contains_requires_every_gold_content_token():
    assert prelim_contains("Lisbon in March", "She moved to Lisbon in March 2024") is True
    assert prelim_contains("Lisbon and Porto", "She moved to Lisbon") is False


def test_prelim_contains_is_prefix_tolerant_for_inflection():
    assert prelim_contains("electric and intense",
                           "the arena was electric with extra intensity") is True


def test_prelim_contains_rejects_empty_gold():
    assert prelim_contains("", "anything") is False


def test_pack_is_lossless_within_source_cap():
    """The miss-forensics fleet traced 16/58 judged misses to the old positional truncation
    (`packed[:max_sources]` silently dropped every record past ~23x12KB): the reader answered
    'no information' about facts sitting in the store. The contract is now LOSSLESS-OR-LOUD:
    the per-source budget grows to fit the data, every record lands, never more than
    max_sources sources."""
    sources = [{"display_name": f"s{i}", "text_content": "x" * 5_000} for i in range(30)]
    packed = pack_record_sources(_Bridge(sources), "ns-abcdef",
                                 max_sources=10, max_chars=12_000)
    assert len(packed) <= 10
    total = sum(len(p["text_content"]) for p in packed)
    # EVERY record's bytes are present (separators add chars, so >=)
    assert total >= 5_000 * 30


def test_pack_lossless_with_oversized_records_and_heavy_store():
    """An LME-S-shaped store (long records, total >> max_sources x max_chars) must still pack
    every byte: the budget grows past max_chars instead of dropping the tail."""
    sources = [{"display_name": f"s{i}", "text_content": f"REC{i:02d}" + "y" * 20_000}
               for i in range(50)]
    packed = pack_record_sources(_Bridge(sources), "ns-heavy1",
                                 max_sources=24, max_chars=12_000)
    assert len(packed) <= 24
    blob = " ".join(p["text_content"] for p in packed)
    for i in range(50):
        assert f"REC{i:02d}" in blob  # the OLD code dropped the tail records silently


def test_find_notebook_id_strict_never_falls_back_to_first_id():
    """The contamination bug: non-strict lookup resolved a MISSING title to the first
    notebook in the listing, so ten namespaces' sources and questions silently landed in
    one notebook. strict=True must return None for a missing title -- forcing a create."""
    from eidetic.integrations.notebooklm import find_notebook_id
    listing = '[{"id":"nb_DEMO_12345","title":"My Memory"}]'
    assert find_notebook_id(listing, "eidetic-win-g3-r0") == "nb_DEMO_12345"  # legacy fallback
    assert find_notebook_id(listing, "eidetic-win-g3-r0", strict=True) is None  # the fix
    assert find_notebook_id(listing, "My Memory", strict=True) == "nb_DEMO_12345"


def test_judge_rows_scores_with_injected_judge_and_writes_sidecar(tmp_path):
    """--judge path: same pinned-judge interface as the scoreboard, durable sidecar,
    honest different-reader label. Injected fake judge -> no key, no network."""
    import json
    from bench.notebooklm_freetier_report import judge_rows

    p = tmp_path / "col.jsonl"
    rows = [
        {"sample_id": "s1", "question": "q1", "gold": "g1", "nb_answer": "right", "category": "single-hop"},
        {"sample_id": "s2", "question": "q2", "gold": "g2", "nb_answer": "wrong", "category": "temporal"},
        {"sample_id": "s3", "error": "boom"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))

    class _J:
        def judge_locomo(self, q, gold, hyp):
            return hyp == "right"

    out = judge_rows(p, judge=_J())
    assert out["judged"] is True
    assert out["n"] == 2 and out["correct"] == 1 and out["accuracy"] == 0.5
    assert "never merged" in out["label"]
    side = json.loads((tmp_path / "col.judged.json").read_text())
    assert side["accuracy"] == 0.5


def test_judge_rows_refuses_plainly_without_key(tmp_path, monkeypatch):
    import json
    from bench.notebooklm_freetier_report import judge_rows

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    p = tmp_path / "col.jsonl"
    p.write_text(json.dumps({"sample_id": "s1", "question": "q", "gold": "g",
                             "nb_answer": "a"}))
    out = judge_rows(p)
    assert out["judged"] is False
    assert "DASHSCOPE_API_KEY" in out["reason"]


def test_pack_preserves_record_text_verbatim():
    sources = [{"display_name": "s0",
                "text_content": "--- EIDETIC VERIFIED MEMORY (provenance) ---\nbody text"}]
    packed = pack_record_sources(_Bridge(sources), "ns-abcdef")
    assert len(packed) == 1
    assert "EIDETIC VERIFIED MEMORY" in packed[0]["text_content"]
    assert "body text" in packed[0]["text_content"]


def test_judge_v2_gold_containment_flips_superset_answers_only():
    """Judge v2 pre-check: the fleet's confirmed false-negative shape (superset/paraphrase
    list answers) short-circuits to correct; the two measured false-positive shapes --
    insufficiency answers that merely mention the gold's tokens, and temporal golds token-
    contained by an answer asserting a DIFFERENT date -- must NEVER short-circuit."""
    from bench.notebooklm_freetier_report import _gold_containment_correct

    # the c7_q41 shape: every gold item present, extras don't penalize
    assert _gold_containment_correct(
        "candles, music, essential oils",
        "They use **candles and essential oils** to set the mood, plus a Music section "
        "with calming playlists.") is True
    # missing item -> no shortcut
    assert _gold_containment_correct(
        "candles, music, essential oils", "They use candles and soft music.") is False
    # insufficiency answer mentioning the tokens (gpt4_85da3956 shape) -> no shortcut
    assert _gold_containment_correct(
        "the summer festival",
        "I could not find any mention of the summer festival in the sources.") is False
    # temporal gold (c6_q29 shape): date tokens contained but the asserted date differs
    assert _gold_containment_correct(
        "July 11, 2022",
        "He left on July 12, 2022, after buying tickets on July 11 that week in 2022.") is False


def test_judge_v2_quarantines_defects_and_shortcircuits_before_llm(tmp_path):
    """Quarantined dataset-defect rows leave the denominator (reported, never counted);
    containment-correct rows never reach the LLM; the rest do."""
    import json
    from bench.notebooklm_freetier_report import judge_rows_v2

    p = tmp_path / "col.jsonl"
    rows = [
        # quarantined defect row -- must not reach the judge or the denominator
        {"sample_id": "c9_q137", "question": "q", "gold": "at an early age",
         "nb_answer": "whatever", "category": "single-hop"},
        # superset answer -- deterministic shortcut, no LLM call
        {"sample_id": "s_list", "question": "q", "gold": "apples, pears",
         "nb_answer": "You bought apples, pears, and a melon.", "category": "multi-hop"},
        # needs the LLM
        {"sample_id": "s_llm", "question": "q", "gold": "blue",
         "nb_answer": "It was azure.", "category": "single-hop"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))

    class _FakeJudge:
        def __init__(self):
            self.calls = []

        def judge_locomo(self, q, gold, answer):
            self.calls.append((q, gold, answer))
            return True

    judge = _FakeJudge()
    out = judge_rows_v2(p, judge=judge)
    assert out["judge_version"] == "v2"
    assert out["quarantined_gold_defects"] == ["c9_q137"]
    assert out["n"] == 2 and out["correct"] == 2
    assert out["shortcircuit_correct"] == 1
    assert [c[1] for c in judge.calls] == ["blue"]          # ONLY the non-shortcut row
    side = json.loads((tmp_path / "col.judged_v2.json").read_text())
    assert side["judge_version"] == "v2"                    # own sidecar, v1 untouched
