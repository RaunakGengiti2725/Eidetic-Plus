"""The autoresearch ratchet: failure taxonomy, frontier queue + integrity wall,
proof-DNA constitution, operator compilation, proposer priors, trial ledger +
promotion discipline, replay honesty checks, MCP surfaces. All offline."""
import json

import pytest

from eidetic.autoresearch.agenda import ResearchAgenda
from eidetic.autoresearch.operators import PipelineError, compile_pipeline, validate_pipeline
from eidetic.autoresearch.proposer import propose, propose_one
from eidetic.autoresearch.registry import ChampionRegistry, ResearchMemory
from eidetic.autoresearch.space import (ALLOWED_KNOBS, PROOF_DNA,
                                        assert_hypothesis_env_legal, is_proof_dna)
from eidetic.autoresearch.trials import append_trial, load_trials, run_trial
from eidetic.autoresearch.types import (FailureClass, ResearchHypothesis, ResearchTask,
                                        ResearchTrial, classify_failure,
                                        failure_class_for_diagnosis, validate_trial_row)


# ---- failure taxonomy (mined note shapes must classify deterministically) ----------

@pytest.mark.parametrize("note,expected", [
    ("abstained: no source entails the answer", FailureClass.ENTAIL_FAILURE),
    ("abstained: active memory contradicts the answer", FailureClass.CONTRADICTION_VETO),
    ("abstained: a sentence-level claim was not grounded", FailureClass.SPAN_UNGROUNDED),
    ("abstained: insufficient evidence (coverage 0.12)", FailureClass.LOW_COVERAGE),
    ("abstained: confidence 0.31 < tau 0.55", FailureClass.LOW_COVERAGE),
    ("abstained: answer form is non-responsive (verbatim echo)", FailureClass.FORM_FLOOR),
    ("abstained: the number is not stated in any retrieved source", FailureClass.NUMERIC_FLOOR),
    ("abstained: false-premise (disconnected_entities)", FailureClass.FALSE_PREMISE),
    ("abstained: CoVe verification question failed grounding", FailureClass.COVE_UNGROUNDED),
    ("smqe:relative_temporal mention_selected", FailureClass.TEMPORAL_SELECTION),
    ("", FailureClass.UNKNOWN),
])
def test_classify_failure(note, expected):
    assert classify_failure(note) == expected


def test_classify_verified_wrong_needs_extra():
    assert classify_failure("smqe:latest_value",
                            {"verified": True, "correct": False}) == FailureClass.LATEST_VALUE_SELECTION
    assert classify_failure("anything",
                            {"verified": True, "correct": False}) == FailureClass.VERIFIED_WRONG


def test_diagnosis_mapping():
    assert failure_class_for_diagnosis("missing") == FailureClass.MISSING_KNOWLEDGE
    assert failure_class_for_diagnosis("hard_to_retrieve") == FailureClass.HARD_TO_RETRIEVE
    assert failure_class_for_diagnosis("contradicted") == FailureClass.CONTESTED_CONFLICT


# ---- the frontier queue ---------------------------------------------------------

def test_priority_order_contested_first():
    contested = ResearchTask(query="a", origin="contested_cell")
    unknown = ResearchTask(query="b", origin="unknown_cell", priority_hint=1.5)
    ask = ResearchTask(query="c", origin="ask_fail")
    repair = ResearchTask(query="d", origin="repair")
    assert contested.priority > unknown.priority > ask.priority > repair.priority


def test_agenda_dedup_bumps_priority(tmp_path):
    agenda = ResearchAgenda(tmp_path / "a.sqlite")
    t = ResearchTask(query="Where does Ada work?", namespace="u")
    k1 = agenda.enqueue(t)
    k2 = agenda.enqueue(ResearchTask(query="Where does Ada work?", namespace="u"))
    assert k1 == k2
    assert agenda.stats()["queued"] == 1
    top = agenda.peek(1)[0]
    assert top["priority"] > t.priority          # recurrence bumped


def test_agenda_integrity_wall(tmp_path):
    agenda = ResearchAgenda(tmp_path / "a.sqlite")
    assert agenda.enqueue(ResearchTask(query="q", namespace="locomo-g3-r0")) is None
    assert agenda.enqueue(ResearchTask(query="q", namespace="longmemeval")) is None
    ok = agenda.enqueue(ResearchTask(query="q", namespace="eidetic-plus-full-locomo-g0-r0",
                                     source="dev_lab"))
    assert ok is not None                        # the lab runner may steer research
    assert agenda.enqueue(ResearchTask(query="q", namespace="userns")) is not None


def test_agenda_pop_marks_running_and_statuses(tmp_path):
    agenda = ResearchAgenda(tmp_path / "a.sqlite")
    agenda.enqueue(ResearchTask(query="low", origin="repair"))
    agenda.enqueue(ResearchTask(query="high", origin="contested_cell"))
    key, task = agenda.pop_highest_priority()
    assert task.query == "high"
    assert agenda.stats()["running"] == 1
    agenda.mark(key, "done")
    assert agenda.stats()["done"] == 1
    with pytest.raises(ValueError):
        agenda.mark(key, "bogus")


# ---- the constitution ------------------------------------------------------------

def test_proof_dna_disjoint_from_search_space():
    assert not {k for k in ALLOWED_KNOBS if is_proof_dna(k)}


def test_wall_refuses_dna_and_unknown_knobs():
    with pytest.raises(ValueError, match="proof DNA"):
        assert_hypothesis_env_legal({"ABSTENTION_THRESHOLD": "0.0"})
    with pytest.raises(ValueError, match="proof DNA"):
        assert_hypothesis_env_legal({"VERIFY_MODEL": "weak"})   # prefix family
    with pytest.raises(ValueError, match="proof DNA"):
        assert_hypothesis_env_legal({"RRF_W_RECENCY": "0.5"})   # age-neutrality claim
    with pytest.raises(ValueError, match="outside the allowed space"):
        assert_hypothesis_env_legal({"SOME_RANDOM_KNOB": "1"})
    assert_hypothesis_env_legal({"READER_COT": "1"})            # legal


# ---- operator DSL -----------------------------------------------------------------

def test_compile_pipeline_lowers_to_real_knobs():
    env = compile_pipeline({
        "retrieve": {"channels": ["dense", "graph"], "weights": {"graph": 1.2},
                     "fusion": "dbsf", "final_topk": 15},
        "read": ["rerank", "temporal_rerank", "compress:0.75", "claim_select"],
    })
    assert env["RRF_W_BM25"] == "0.0"            # bm25 compiled out
    assert env["RRF_W_GRAPH"] == "1.2"
    assert env["FUSION_METHOD"] == "dbsf"
    assert env["FINAL_TOPK"] == "15"
    assert env["RERANK_ENABLED"] == "1" and env["TEMPORAL_RERANK"] == "1"
    assert env["MMR_ENABLED"] == "0"
    assert env["COMPRESSION_RATIO"] == "0.75" and env["CONTEXT_COMPRESS"] == "1"
    assert env["READ_CLAIM_SELECT"] == "1"
    assert json.loads(env["EXPECT_STAGES"]) == ["dense", "graph"]


@pytest.mark.parametrize("bad,msg", [
    ({"retrieve": {"channels": ["bm25"]}}, "dense"),
    ({"read": ["prove"]}, "prove is not an op"),
    ({"read": ["unknown_stage"]}, "unknown read op"),
    ({"retrieve": {"channels": ["dense"], "weights": {"bm25": 9.0}}}, "outside"),
    ({"retrieve": {"fusion": "magic"}}, "unknown fusion"),
    ({"prove": {"skip": True}}, "unknown pipeline sections"),
    ({"read": ["compress:0.1"]}, "outside"),
])
def test_pipeline_whitelist_refusals(bad, msg):
    with pytest.raises(PipelineError, match=msg):
        validate_pipeline(bad)


# ---- proposer ---------------------------------------------------------------------

def test_proposer_prior_order_and_block(tmp_path):
    memory = ResearchMemory(tmp_path / "m.jsonl")
    task = ResearchTask(query="q", failure_class=FailureClass.ENTAIL_FAILURE)
    first = propose_one(task, memory)
    assert first.tier == "B" and "claim_select" in json.dumps(first.pipeline)
    memory.record(hypothesis_key=first.key, decision="REJECT", delta_pp=-0.5,
                  mcnemar_p=0.9, failure_class="entail_failure", tier="B")
    second = propose_one(task, memory)
    assert second.key != first.key
    assert second.tier == "A" and second.knob == "READER_COT"


def test_proposer_falls_back_to_generic_ladder(tmp_path):
    memory = ResearchMemory(tmp_path / "m.jsonl")
    task = ResearchTask(query="q", failure_class=FailureClass.KNOB_IMBALANCE)
    hyp = propose_one(task, memory)
    assert hyp is not None and hyp.tier == "A"
    assert hyp.knob in ALLOWED_KNOBS


def test_proposer_exhausts_to_none(tmp_path):
    memory = ResearchMemory(tmp_path / "m.jsonl")
    task = ResearchTask(query="q", failure_class=FailureClass.SUGGESTION_SYNTH)
    seen = []
    while True:
        hyp = propose_one(task, memory)
        if hyp is None:
            break
        seen.append(hyp.key)
        memory.record(hypothesis_key=hyp.key, decision="REJECT", delta_pp=0.0,
                      mcnemar_p=None, failure_class="x", tier=hyp.tier)
    assert len(seen) == len(set(seen)) and len(seen) >= 5   # priors + ladder, no repeats


def test_proposer_never_names_dna(tmp_path):
    memory = ResearchMemory(tmp_path / "m.jsonl")
    for fc in FailureClass:
        task = ResearchTask(query="q", failure_class=fc)
        for hyp in propose(task, memory):
            env = hyp.env_overlay()
            assert_hypothesis_env_legal(
                {k: v for k, v in env.items() if k != "EXPECT_STAGES"})


# ---- registry + memory --------------------------------------------------------------

def test_registry_promote_and_refuse_dna(tmp_path):
    reg = ChampionRegistry(tmp_path)
    assert reg.champion_id == "baseline"
    champ = reg.promote(trial_id="tr_1", env={"READER_COT": "1"}, dev_acc=0.8,
                        paired_n=24, tier="A", describe={"knob": "READER_COT"})
    assert champ["champion_id"] == "tr_1"
    assert reg.champion_id == "tr_1"
    assert len(reg.promotions()) == 1
    with pytest.raises(ValueError, match="proof DNA"):
        reg.promote(trial_id="tr_2", env={"FAST_ABSTAIN": "0"}, dev_acc=0.9,
                    paired_n=24, tier="A", describe={})
    assert reg.champion_id == "tr_1"             # failed promote changed nothing


def test_research_memory_block_repeat(tmp_path):
    m = ResearchMemory(tmp_path / "m.jsonl")
    assert m.block_repeat("k1") is False
    m.record(hypothesis_key="k1", decision="ACCEPT", delta_pp=2.0, mcnemar_p=0.01,
             failure_class="entail_failure", tier="A")
    assert m.block_repeat("k1") is True
    assert m.rejected_keys() == set()


# ---- trial ledger -------------------------------------------------------------------

def _hyp():
    return ResearchHypothesis(tier="A", failure_class=FailureClass.ENTAIL_FAILURE,
                              rationale="test", knob="READER_COT", value="1")


def _trial(decision="REJECT"):
    return ResearchTrial(
        trial_id="tr_x", hypothesis=_hyp(), champion_id="baseline",
        challenger_env={"READER_COT": "1"}, dev_score=0.7, champion_score=0.68,
        delta_pp=2.0, mcnemar_p=0.04, paired_n=24, decision=decision,
        reason="ok", artifact_dir="/tmp/x")


def test_trial_row_schema_and_append(tmp_path):
    row = _trial().to_row()
    assert validate_trial_row(row) == []
    bad = dict(row)
    bad["decision"] = "MAYBE"
    assert any("decision" in p for p in validate_trial_row(bad))
    path = tmp_path / "trials.jsonl"
    append_trial(path, _trial())
    assert len(load_trials(path)) == 1
    broken = _trial()
    broken.decision = "MAYBE"
    with pytest.raises(RuntimeError, match="invalid trial row"):
        append_trial(path, broken)


class _FakeLab:
    """eval_config double: returns per-label dirs; never touches a store."""
    def __init__(self, root):
        self.root = root
        self.calls = []

    def eval_config(self, env, label, judge=None, overwrite=False):
        self.calls.append((label, dict(env)))
        d = self.root / label
        d.mkdir(parents=True, exist_ok=True)
        return d


def test_run_trial_accept_promotes_and_records(tmp_path, monkeypatch):
    import eidetic.autoresearch.trials as trials_mod
    verdicts = {"accept": True, "challenger_acc": 0.75, "champion_acc": 0.70,
                "delta_pp": 5.0, "mcnemar_p": 0.03, "paired_n": 24, "reason": "sig win"}
    import bench.guard as guard_mod
    monkeypatch.setattr(guard_mod, "run_guard",
                        lambda *a, **k: dict(verdicts))
    lab = _FakeLab(tmp_path / "lab")
    reg = ChampionRegistry(tmp_path / "reg")
    mem = ResearchMemory(tmp_path / "mem.jsonl")
    trial = run_trial(_hyp(), lab=lab, registry=reg, memory=mem,
                      trials_path=tmp_path / "trials.jsonl")
    assert trial.decision == "ACCEPT"
    assert reg.champion_id == trial.trial_id
    assert reg.load()["env"]["READER_COT"] == "1"
    assert mem.block_repeat(_hyp().key)
    rows = load_trials(tmp_path / "trials.jsonl")
    assert len(rows) == 1 and validate_trial_row(rows[0]) == []
    labels = [c[0] for c in lab.calls]
    assert any(l.startswith("champion__") for l in labels)
    assert any(l.startswith("trial__A__") for l in labels)


def test_run_trial_reject_keeps_champion(tmp_path, monkeypatch):
    import bench.guard as guard_mod
    monkeypatch.setattr(guard_mod, "run_guard", lambda *a, **k: {
        "accept": False, "challenger_acc": 0.66, "champion_acc": 0.70,
        "delta_pp": -4.0, "mcnemar_p": 0.5, "paired_n": 24, "reason": "worse"})
    lab = _FakeLab(tmp_path / "lab")
    reg = ChampionRegistry(tmp_path / "reg")
    mem = ResearchMemory(tmp_path / "mem.jsonl")
    trial = run_trial(_hyp(), lab=lab, registry=reg, memory=mem,
                      trials_path=tmp_path / "trials.jsonl")
    assert trial.decision == "REJECT" and reg.champion_id == "baseline"


# ---- replay honesty ------------------------------------------------------------------

def test_replay_offline_checks(tmp_path, monkeypatch):
    from eidetic.autoresearch.replay import replay_offline
    from eidetic.models import Answer, AnswerStatus, Citation, NLILabel, Scope, now

    class _Rec:
        def __init__(self, mid, h, created):
            self.memory_id, self.content_hash, self.created_at = mid, h, created

    class _Store:
        def __init__(self):
            self.recs = [_Rec("m1", "h1", 100.0)]

        def all_records(self, scope=None):
            return list(self.recs)

        def get_record(self, mid):
            return next((r for r in self.recs if r.memory_id == mid), None)

    class _Eng:
        def __init__(self):
            self.store = _Store()

        def ask(self, q, **kw):
            cit = Citation(memory_id="m1", content_hash="h1", raw_uri="u", source="s",
                           valid_at=1.0, nli_label=NLILabel.ENTAILMENT)
            return Answer(question=q, answer="a", status=AnswerStatus.VERIFIED,
                          verified=True, citations=[cit])

    rep = replay_offline(_Eng(), [{"query": "q", "namespace": "u"}],
                         promotion_ts=200.0, out_path=tmp_path / "r.json")
    assert rep["flipped_to_verified"] == 1
    assert rep["all_same_witness"] and rep["all_no_new_ingest"]
    assert rep["all_citations_preexist"]         # created 100 < promoted 200
    rep2 = replay_offline(_Eng(), [{"query": "q", "namespace": "u"}], promotion_ts=50.0)
    assert rep2["all_citations_preexist"] is False   # citation postdates promotion


# ---- leaderboard ----------------------------------------------------------------------

def test_leaderboard_render(tmp_path):
    from eidetic.autoresearch.leaderboard import render
    append_trial(tmp_path / "trials.jsonl", _trial())
    out = render(tmp_path)
    assert "READER_COT=1" in out and "REJECT" in out
