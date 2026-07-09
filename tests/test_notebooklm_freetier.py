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


def test_pack_respects_source_and_char_caps():
    sources = [{"display_name": f"s{i}", "text_content": "x" * 5_000} for i in range(30)]
    packed = pack_record_sources(_Bridge(sources), "ns-abcdef",
                                 max_sources=10, max_chars=12_000)
    assert len(packed) <= 10
    # nothing dropped silently until the cap: all text present across packed sources
    total = sum(len(p["text_content"]) for p in packed)
    assert total >= 5_000 * 18  # 2 records per 12k-char pack x 9 packs + final pack


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
