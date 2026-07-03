"""Offline tests for the eidetic-plus-full adapter (Track 5.2): the PRODUCT row. Unlike the
neutral eidetic-plus row (retrieval-context only), -full applies the product policy --
NLI verification + abstention + proof -- and reports verified/abstained/confidence so the
report can score the honesty differentiators no baseline has. Offline via a fake reader+NLI."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from dataclasses import replace

import numpy as np

from bench.adapters.eidetic_adapter import (
    EideticFullSystem,
    EideticProductSystem,
    EideticSystem,
)
from eidetic.config import get_settings
from eidetic.engine import Engine
from eidetic.models import DerivedRecord, Scope


class _FakeReader:
    def __init__(self, dim):
        self.dim = dim
        self.reader_models = []

    def _e(self, t):
        v = np.zeros(self.dim, np.float32)
        for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_text(self, t):
        return self._e(t)

    def embed_texts(self, ts):
        return np.stack([self._e(t) for t in ts]) if ts else np.zeros((0, self.dim), np.float32)

    def extract_edges(self, text):
        return []

    def chat(self, model, system, user, **kw):
        # the ONE fixed reader path (answer_with_fixed_reader) calls client.chat(READER_MODEL, ...).
        self.reader_models.append(model)
        return "Alice works at Acme Corporation"

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if "acme" in (premise or "").lower() else ("neutral", 0.2)


class _DecliningReader(_FakeReader):
    def chat(self, model, system, user, **kw):
        self.reader_models.append(model)
        return "I do not have that in memory."

    def nli(self, premise, hypothesis):
        return ("entailment", 0.9) if (hypothesis or "").lower() in (premise or "").lower() else ("neutral", 0.2)


class _TemporalReader(_FakeReader):
    def chat(self, model, system, user, **kw):
        self.reader_models.append(model)
        return "2023-05-07"


def _engine(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    get_settings.cache_clear()
    s = replace(get_settings(), rerank_enabled=False, **kw)
    e = Engine(s, client=_FakeReader(s.embed_dim))
    # The fixed reader (answer_with_fixed_reader) uses the MODULE-level get_client; point it at the
    # same fake the engine uses so the offline test exercises the real parity path with no key.
    from bench import reader as bench_reader
    monkeypatch.setattr(bench_reader, "get_client", lambda: e.client)
    return e


def _engine_with_client(tmp_path, monkeypatch, client, **kw):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VECTOR_BACKEND", "numpy")
    get_settings.cache_clear()
    settings_kwargs = {"rerank_enabled": False, "user_evidence_context_enabled": True}
    settings_kwargs.update(kw)
    s = replace(get_settings(), **settings_kwargs)
    e = Engine(s, client=client)
    from bench import reader as bench_reader
    monkeypatch.setattr(bench_reader, "get_client", lambda: e.client)
    return e


def test_eidetic_full_applies_verification_and_reports(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch)
    sys = EideticFullSystem(engine=e)
    assert sys.name == "eidetic-plus-full"
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [{"role": "user", "content": "Alice works at Acme Corporation"}])
    sys.consolidate("ns")
    ar = sys.answer("ns", "where does Alice work")
    assert "Acme" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.abstained is False
    assert ar.context_tokens > 0
    assert ar.extra["entailed_content_hashes"]
    assert all(len(h) == 64 for h in ar.extra["entailed_content_hashes"])
    assert ar.extra["entailed_raw_uris"]
    assert all(uri.startswith("cas://") for uri in ar.extra["entailed_raw_uris"])
    # SMQE should answer source-backed scalar facts before spending fixed-reader tokens.
    assert e.client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_abstains_on_no_evidence(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch, abstention_threshold=0.4)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [{"role": "user", "content": "completely unrelated content about gardening"}])
    sys.consolidate("ns")
    ar = sys.answer("ns", "what is the secret launch code")
    assert ar.abstained is True                     # product honesty policy: no evidence -> abstain
    assert ar.extra["verified"] is False
    get_settings.cache_clear()


def test_eidetic_full_rescues_direct_user_slot_when_reader_declines(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "answer_1", [
        {"role": "assistant", "content": "Any education background?"},
        {"role": "user", "content": "I graduated with a degree in Business Administration, which helped."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What degree did I graduate with?")

    assert ar.answer == "Business Administration"
    assert ar.extra["verified"] is True
    assert ar.abstained is False
    assert ar.extra["policy"].startswith("smqe:")
    assert ar.extra["proof_surface_tokens"] == ar.context_tokens
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_temporal_structured_recall_before_full_retrieval(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=True)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 5, 8, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "I went to a LGBTQ support group yesterday and it was powerful."},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "When did I go to the LGBTQ support group?", as_of=session_time)

    assert ar.answer == "2023-05-07"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert ar.abstained is False
    assert ar.context_tokens < 100
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_temporal_deterministic_scan_is_not_gated_by_audit_flag(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=False)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 5, 8, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "I went to a LGBTQ support group yesterday and it was powerful."},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "When did I go to the LGBTQ support group?", as_of=session_time)

    assert ar.answer == "2023-05-07"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert ar.context_tokens < 100
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_temporal_year_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=True)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 5, 8, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "I painted a lake sunrise last year and kept it."},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "What year did I paint a lake sunrise?", as_of=session_time)

    assert ar.answer == "2022"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_temporal_speech_school_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=True)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 6, 9, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "Mira", "content": (
            "I wanted to tell you about my school event last week. It was awesome! "
            "I talked about my transgender journey and encouraged students."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "When did Mira give a speech at a school?", as_of=session_time)

    assert ar.answer == "the week of 2023-06-02 to 2023-06-08"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_relationship_status_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "Mira is single and has passed adoption agency interviews."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is Mira's relationship status?")

    assert ar.answer == "Single"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_adoption_research_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Mira", "content": (
            "Researching adoption agencies - it's been a dream to have a family and give a "
            "loving home to kids who need it."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What did Mira research?")

    assert ar.answer == "Adoption agencies"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_identity_profile_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Mira", "content": (
            "I made this painting to show my path as a trans woman and to love my authentic self."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is Mira's identity?")

    assert ar.answer == "Transgender woman"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_education_field_profile_scan(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Mira", "content": (
            "Lately, I've been looking into counseling and mental health as a career. "
            "I want to help people who have gone through the same things as me."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What fields would Mira be likely to pursue in her educaton?")

    assert ar.answer == "counseling and mental health"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_abstains_on_unsupported_financial_status_inference(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": (
            "It's definitely isn't right. My kids have so much and others don't. "
            "We really need to do something about it."
        )},
        {"role": "John", "content": "My family and I also took a family road trip last year."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What might John's financial status be?")

    assert ar.abstained is True
    assert ar.answer == EideticFullSystem._ABSTAIN_TEXT
    assert ar.extra["verified"] is False
    assert ar.extra["policy"] == "fixed-reader + verify+abstain+proof"
    assert client.reader_models
    get_settings.cache_clear()


def test_eidetic_full_abstains_on_unsupported_allergy_pet_inference(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Joanna", "content": (
            "I'm allergic to most reptiles and animals with fur. It can be a drag, "
            "but I find other ways to be happy."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What pets wouldn't cause any discomfort to Joanna?")

    assert ar.abstained is True
    assert ar.answer == EideticFullSystem._ABSTAIN_TEXT
    assert ar.extra["verified"] is False
    assert ar.extra["policy"] == "fixed-reader + verify+abstain+proof"
    assert client.reader_models
    get_settings.cache_clear()


def test_eidetic_full_open_domain_choice_uses_source_overlapping_option(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Tim", "content": "I love getting lost in fantasy stories and magical worlds."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "Would Tim enjoy reading magical fantasy books or tax manuals?")

    assert ar.answer == "magical fantasy books"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert ar.extra["structured_recall"] is True
    assert ar.extra["smqe_policy"].startswith("smqe:")
    assert ar.extra["smqe_backend"] in {"claim", "record"}
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_abstains_on_unsupported_indoor_dog_activity_inference(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Andrew", "content": (
            "I've been getting into cooking more and trying out new recipes - "
            "it's been enjoyable."
        )},
        {"role": "Audrey", "content": (
            "I made some goodies recently to thank my neighbors for their "
            "pup-friendly homes."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is an indoor activity that Andrew would enjoy doing while make his dog happy?")

    assert ar.abstained is True
    assert ar.answer == EideticFullSystem._ABSTAIN_TEXT
    assert ar.extra["verified"] is False
    assert ar.extra["policy"] == "fixed-reader + verify+abstain+proof"
    assert client.reader_models
    get_settings.cache_clear()


def test_eidetic_full_unsupported_inference_abstention_generalizes_beyond_live_names(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)

    sys.reset("money")
    sys.ingest_session("money", "s0", [
        {"role": "Riley", "content": "My kids have so much and others don't. I want to help."},
    ])
    sys.consolidate("money")
    money = sys.answer("money", "What might Riley's financial status be?")
    assert money.abstained is True
    assert money.answer == EideticFullSystem._ABSTAIN_TEXT
    assert money.extra["verified"] is False
    assert money.extra["policy"] == "fixed-reader + verify+abstain+proof"

    sys.reset("allergy")
    sys.ingest_session("allergy", "s0", [
        {"role": "Priya", "content": "I'm allergic to most reptiles and animals with fur."},
    ])
    sys.consolidate("allergy")
    allergy = sys.answer("allergy", "What pets wouldn't cause any discomfort to Priya?")
    assert allergy.abstained is True
    assert allergy.answer == EideticFullSystem._ABSTAIN_TEXT
    assert allergy.extra["verified"] is False
    assert allergy.extra["policy"] == "fixed-reader + verify+abstain+proof"

    assert len(client.reader_models) >= 2
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_martial_arts(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": "I'm doing kickboxing and it's giving me so much energy."},
    ])
    sys.ingest_session("ns", "s1", [
        {"role": "John", "content": "I'm off to do some taekwondo!"},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What martial arts has John done?")

    assert ar.answer == "Kickboxing, Taekwondo"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_charity_awareness(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Melanie", "content": "I ran a charity race for mental health last Saturday."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What did the charity race raise awareness for?")

    assert ar.answer == "mental health"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_signed_team(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": "I just signed with a new team - excited for the season!"},
        {"role": "John", "content": "The Minnesota Wolves! I can't wait to play with them!"},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "Which team did John sign with on 21 May, 2023?")

    assert ar.answer == "The Minnesota Wolves"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_dog_adoption_year(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 3, 27, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "Audrey", "content": (
            "I've had them for 3 years! Their names are Pepper, Precious and Panda."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "Which year did Audrey adopt the first three of her dogs?",
                    as_of=session_time)

    assert ar.answer == "2020"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_shared_destress_activity(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Gina", "content": "Dance is my stress fix too."},
        {"role": "Jon", "content": "Dancing helps me de-stress. It's where I'm most alive."},
        {"role": "Jon", "content": (
            "Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business."
        )},
        {"role": "Gina", "content": "I also lost my job at Door Dash this month."},
        {"role": "Gina", "content": "I started my own online clothing store not so long ago."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "How do Jon and Gina both like to destress?")

    assert ar.answer == "dancing"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_shared_job_business_commonality(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Jon", "content": (
            "Lost my job as a banker yesterday, so I'm gonna take a shot at starting my own business."
        )},
        {"role": "Gina", "content": "I also lost my job at Door Dash this month."},
    ])
    sys.ingest_session("ns", "s1", [
        {"role": "Gina", "content": "I started my own online clothing store not so long ago."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What do Jon and Gina both have in common?")

    assert ar.answer == "They lost their jobs and decided to start their own businesses"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_does_not_hallucinate_movie_title_from_clues(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Joanna", "content": (
            "Have you seen this romantic drama that's all about memory and relationships? "
            "It's such a good one."
        )},
        {"role": "Joanna", "content": "A few times. It's one of my favorites!"},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is one of Joanna's favorite movies?")

    assert ar.abstained is True
    assert ar.extra["verified"] is False
    assert client.reader_models
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_main_focus(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": (
            "I'm passionate about improving education and infrastructure in our community. "
            "Those are my main focuses."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is John's main focus in local politics?")

    assert ar.answer == "improving education and infrastructure"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_basketball_goals(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": "Winning a championship is my number one goal."},
        {"role": "John", "content": "My goal is to improve my shooting percentage."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "what are John's goals with regards to his basketball career?")

    assert ar.answer == "improve shooting percentage, win a championship"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_personal_best_time(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": (
            "I'm getting ready for the charity 5K run. I'm hoping to beat my personal "
            "best time of 25:50 this time around."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What was my personal best time in the charity 5K run?")

    assert ar.answer == "25:50"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_product_structured_recall_accounting_skips_representative_retrieve(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(
        tmp_path,
        monkeypatch,
        client,
        defer_reembed_enabled=True,
        semantic_cache_enabled=True,
    )
    sys = EideticProductSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": (
            "I'm getting ready for the charity 5K run. I'm hoping to beat my personal "
            "best time of 25:50 this time around."
        )},
    ])
    sys.consolidate("ns")

    def _retrieve_should_not_run(*args, **kwargs):
        raise AssertionError("SMQE product row should not run representative retrieval")

    monkeypatch.setattr(e.retriever, "retrieve", _retrieve_should_not_run)
    ar = sys.answer("ns", "What was my personal best time in the charity 5K run?")

    assert ar.answer == "25:50"
    assert ar.extra["verified"] is True
    assert ar.extra["note"].startswith("smqe:")
    assert ar.context_tokens == ar.extra["proof_surface_tokens"]
    assert ar.context_tokens > 0
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_schedule_table_rotation(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "answer_sharegpt_5Lzox6N_0", [
        {"role": "assistant", "content": (
            "|  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm (Afternoon Shift) | "
            "4 pm - 12 am (Evening Shift) | 12 am - 8 am (Night Shift) |\n"
            "| Sunday | Iris | Rowan | Leif | Mara |"
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What is the rotation for Iris on a Sunday?")

    assert ar.answer == "8 am - 4 pm (Day Shift)"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_clothing_pickup_return_count(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": (
            "I still need to pick up my dry cleaning for the navy blue blazer."
        )},
    ])
    sys.ingest_session("ns", "s1", [
        {"role": "user", "content": (
            "I need to return some boots to Luma Market. I exchanged them for a larger size, "
            "so I still need to pick up the new pair at Luma Market."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "How many items of clothing do I need to pick up or return from a store?")

    assert ar.answer == "3"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert set(ar.extra["entailed_memory_ids"])
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_structured_recall_is_not_gated_by_user_evidence_context(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(
        tmp_path,
        monkeypatch,
        client,
        user_evidence_context_enabled=False,
    )
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": (
            "I still need to pick up my dry cleaning for the navy blue blazer."
        )},
    ])
    sys.ingest_session("ns", "s1", [
        {"role": "user", "content": (
            "I need to return some boots to Luma Market, actually. I exchanged them for a larger "
            "size, so I still need to pick up the new pair at Luma Market."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "How many items of clothing do I need to pick up or return from a store?")

    assert ar.answer == "3"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_gallery_day_interval(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session(
        "ns",
        "s0",
        [{"role": "user", "content": (
            "I just got back from a guided tour at the Glass Meridian Gallery focused "
            "on kiln-fired color studies."
        )}],
        datetime(2023, 1, 8, 12, 49).timestamp(),
    )
    sys.ingest_session(
        "ns",
        "s1",
        [{"role": "user", "content": (
            "I attended the Lantern Maps exhibit at the Harbor Archive today."
        )}],
        datetime(2023, 1, 15, 0, 27).timestamp(),
    )
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many days passed between my visit to the Glass Meridian Gallery "
        "and the 'Lantern Maps' exhibit at the Harbor Archive?",
    )

    assert ar.answer == "7 days"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_photography_accessory_preferences(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "I use a Kestrel Q9 mirrorless camera for my photography setup."},
        {"role": "user", "content": "I also have a Kestrel 40-90mm f/2.8 lens."},
        {"role": "user", "content": (
            "I chose the Lumiflash Nova and I am considering a Lumiflash Nova Hard Case "
            "or Atlas Photo Flash Pouch."
        )},
        {"role": "user", "content": (
            "I want high-quality bags that are compatible with Kestrel cameras."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "Can you suggest some accessories that would complement my current photography setup?")

    assert "compatible" in ar.answer
    assert "Lumiflash Nova" in ar.answer
    assert "Kestrel Q9" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_model_kit_counts(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 5, 30, 12, 0).timestamp()
    sys.ingest_session("ns", "kits-a", [
        {"role": "user", "content": (
            "I recently finished a simple Orion Falcon glider kit that I picked up "
            "during a trip to the hobby store."
        )},
    ], session_time=session_time)
    sys.ingest_session("ns", "kits-b", [
        {"role": "user", "content": (
            "I recently finished a 1/48 scale Harbor tug boat and had to learn "
            "some new techniques."
        )},
    ], session_time=session_time)
    sys.ingest_session("ns", "kits-c", [
        {"role": "user", "content": (
            "I started working on a diorama featuring a 1/16 scale Alpine tram vehicle."
        )},
    ], session_time=session_time)
    sys.ingest_session("ns", "kits-d", [
        {"role": "user", "content": (
            "I just got this 1/72 scale lunar rover model kit and a 1/24 scale "
            "metro bus at a model show last weekend."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "How many model kits have I worked on or bought?", as_of=session_time)

    assert ar.answer.startswith("5 model kits:")
    assert "Orion Falcon glider kit" in ar.answer
    assert "1/48 scale Harbor tug boat" in ar.answer
    assert "1/16 scale Alpine tram vehicle" in ar.answer
    assert "1/72 scale lunar rover" in ar.answer
    assert "1/24 scale metro bus" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_updated_entity_count(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    older = datetime(2023, 8, 11, 9, 9).timestamp()
    newer = datetime(2023, 9, 30, 15, 6).timestamp()
    sys.ingest_session("ns", "studios", [
        {"role": "user", "content": (
            "Have you tried any good blue loom studios in your city lately? "
            "I've tried three different ones recently."
        )},
    ], session_time=older)
    sys.ingest_session("ns", "studios-update", [
        {"role": "user", "content": (
            "Have you tried any good blue loom studios in your city lately? "
            "I've tried four different ones so far, and I'm always looking for new recommendations."
        )},
    ], session_time=newer)
    sys.consolidate("ns")

    ar = sys.answer("ns", "How many blue loom studios have I tried in my city?", as_of=newer)

    assert ar.answer == "four"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_kitchen_preference_tips(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "kitchen", [
        {"role": "user", "content": (
            "I also need some help with organizing my kitchen utensils. I recently "
            "bought a new utensil holder to keep countertops clutter-free."
        )},
        {"role": "user", "content": (
            "I noticed some scratches on my granite countertop near the sink."
        )},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "My kitchen's becoming a bit of a mess again. Any tips for keeping it clean?")

    assert "utensil holder" in ar.answer
    assert "granite countertop near the sink" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_week_delta_question(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 3, 4, 22, 43).timestamp()
    question_time = datetime(2023, 4, 1, 8, 9).timestamp()
    sys.ingest_session("ns", "astrolabe", [
        {"role": "user", "content": (
            "I also got a polished brass astrolabe from my aunt today, which used "
            "to belong to my great-grandmother."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many weeks ago did I meet up with my aunt and receive the brass astrolabe?",
        as_of=question_time,
    )

    assert ar.answer == "4"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_camping_trip_days(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    as_of = datetime(2023, 4, 30, 6, 45).timestamp()
    sys.ingest_session("ns", "utah-roadtrip", [
        {"role": "user", "content": (
            "We had a 7-day family road trip in Utah in February. We did a lot of "
            "driving and hiking, but not camping for this time."
        )},
    ], session_time=datetime(2023, 4, 29, 17, 31).timestamp())
    sys.ingest_session("ns", "yellowstone", [
        {"role": "user", "content": (
            "I just got back from an amazing 5-day camping trip to Yellowstone "
            "National Park last month."
        )},
    ], session_time=datetime(2023, 4, 29, 22, 49).timestamp())
    sys.ingest_session("ns", "big-sur", [
        {"role": "user", "content": (
            "I just got back from a 3-day solo camping trip to Big Sur in early April."
        )},
    ], session_time=datetime(2023, 4, 30, 3, 2).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many days did I spend on camping trips in the United States this year?",
        as_of=as_of,
    )

    assert ar.answer == "8 days"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_consecutive_charity_months(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "bike-ride", [
        {"role": "user", "content": (
            "I just got back from the \"24-Hour Bike Ride\" charity event, where I "
            "cycled for 4 hours non-stop to raise money for a local children's hospital."
        )},
    ], session_time=datetime(2023, 2, 14, 17, 6).timestamp())
    sys.ingest_session("ns", "cure-cancer", [
        {"role": "user", "content": (
            "I participated in the 'Ride to Cure Cancer' charity bike ride and rode "
            "40 miles on my road bike recently."
        )},
    ], session_time=datetime(2023, 2, 15, 16, 39).timestamp())
    sys.ingest_session("ns", "walk-hunger", [
        {"role": "user", "content": (
            "I did the \"Walk for Hunger\" charity event today with my colleagues."
        )},
    ], session_time=datetime(2023, 3, 19, 22, 2).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many months have passed since I participated in two charity events in a row, on consecutive days?",
        as_of=datetime(2023, 4, 18, 10, 31).timestamp(),
    )

    assert ar.answer == "2"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_rachel_latest_relocation(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    older = datetime(2023, 5, 25, 5, 23).timestamp()
    newer = datetime(2023, 5, 27, 11, 45).timestamp()
    sys.ingest_session("ns", "rachel-city", [
        {"role": "user", "content": (
            "I'm also thinking about visiting my friend Rachel who recently moved to a "
            "new apartment in the city. She moved to Chicago."
        )},
    ], session_time=older)
    sys.ingest_session("ns", "rachel-suburbs", [
        {"role": "user", "content": (
            "My friend Rachel actually just moved back to the suburbs again, so I was "
            "thinking of somewhere not too far from a major city."
        )},
    ], session_time=newer)
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "Where did Rachel move to after her recent relocation?",
        as_of=datetime(2023, 6, 13, 22, 15).timestamp(),
    )

    assert ar.answer == "the suburbs"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_named_dessert_shop(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "harbor-desserts", [
        {"role": "assistant", "content": (
            "Absolutely! Here are some fun dessert spots that your family might enjoy after dinner:\n\n"
            "1. Moonspoon Parlor - A sweet shop located at Lantern Pier that offers an "
            "enormous menu of sweet treats, including specialty drinks and ribbon sundaes."
        )},
    ], session_time=datetime(2023, 5, 22, 0, 19).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "I'm planning to revisit the harbor district. Can you remind me of that unique dessert shop with the ribbon sundaes?",
        as_of=datetime(2023, 5, 31, 2, 46).timestamp(),
    )

    assert ar.answer == "Moonspoon Parlor at Lantern Pier"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_garden_dinner_preference(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "garden-dinner", [
        {"role": "user", "content": "I'm trying to find some new recipe ideas that use fresh basil and mint."},
        {"role": "assistant", "content": (
            "3. Pesto Pasta: Blend basil with garlic, pine nuts, Parmesan, and olive oil "
            "to create a vibrant pesto sauce. Toss with linguine, cherry tomatoes, and "
            "grilled chicken.\n"
            "2. Minty Fresh Salad: Combine mint leaves with feta cheese, cucumbers, "
            "cherry tomatoes, and a drizzle of lemon juice and olive oil.\n"
            "2. Middle Eastern-Style Salad: Combine chopped basil and mint with bulgur, "
            "cherry tomatoes, cucumbers, feta cheese, and a lemon-tahini dressing."
        )},
        {"role": "user", "content": (
            "I've been using basil and mint in my cooking lately. I've even harvested "
            "some cherry tomatoes from my garden."
        )},
    ], session_time=datetime(2023, 5, 23, 0, 29).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "What should I serve for dinner this weekend with my garden ingredients?",
        as_of=datetime(2023, 5, 30, 21, 35).timestamp(),
    )

    assert "cherry tomatoes" in ar.answer
    assert "basil" in ar.answer
    assert "mint" in ar.answer
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_recent_plant_acquisitions(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "nursery", [
        {"role": "user", "content": (
            "I'm trying to care for my peace lily, which I got from the nursery two "
            "weeks ago along with a succulent."
        )},
        {"role": "user", "content": (
            "I've been misting my fern every other day, but I'm not sure if that's "
            "suitable for my peace lily as well."
        )},
    ], session_time=datetime(2023, 5, 21, 3, 5).timestamp())
    sys.ingest_session("ns", "sister", [
        {"role": "user", "content": (
            "I'm wondering if I should repot my snake plant, which I got from my "
            "sister last month."
        )},
    ], session_time=datetime(2023, 5, 25, 23, 59).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many plants did I acquire in the last month?",
        as_of=datetime(2023, 5, 31, 4, 51).timestamp(),
    )

    assert ar.answer.startswith("3 plants:")
    assert "peace lily" in ar.answer
    assert "succulent" in ar.answer
    assert "snake plant" in ar.answer
    assert "fern" not in ar.answer
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_latest_preapproval_amount(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "older-preapproval", [
        {"role": "user", "content": (
            "I'm buying a $325,000 house, and I got pre-approved for $350,000 "
            "from Blue Harbor Credit Union."
        )},
    ], session_time=datetime(2023, 8, 11, 7, 1).timestamp())
    sys.ingest_session("ns", "newer-preapproval", [
        {"role": "user", "content": (
            "I'm really looking forward to finally owning a home - remember when "
            "I got pre-approved for $400,000 from Blue Harbor Credit Union?"
        )},
    ], session_time=datetime(2023, 11, 30, 8, 36).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "What was the amount I was pre-approved for when I got my mortgage from Blue Harbor Credit Union?",
        as_of=datetime(2023, 12, 18, 12, 17).timestamp(),
    )

    assert ar.answer == "$400,000"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_two_anchor_day_delta(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "synth", [
        {"role": "user", "content": (
            "I started sketching harmonies on my pocket synth today, "
            "and it was a lot of fun."
        )},
    ], session_time=datetime(2023, 3, 25, 12, 54).timestamp())
    sys.ingest_session("ns", "shadow-folk", [
        {"role": "assistant", "content": (
            "You're diving into the wonderful world of shadow-folk! Congratulations on "
            "discovering a new genre and a trio that resonates with you!"
        )},
    ], session_time=datetime(2023, 3, 31, 19, 35).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "How many days passed between the day I started sketching harmonies on my pocket synth and the day I discovered a shadow-folk trio?",
        as_of=datetime(2023, 4, 5, 16, 11).timestamp(),
    )

    assert ar.answer == "6 days"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_romantic_rome_restaurant(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    # The user turn establishes the Rome/Italian context exactly as the live conversation does;
    # without it the source genuinely does not entail "Italian restaurant in Rome" and the
    # fail-closed verifier must refuse, so a single-turn fixture would test the wrong contract.
    sys.ingest_session("ns", "rome", [
        {"role": "user", "content": (
            "We're planning a special dinner during our trip to Rome next week. "
            "Can you suggest an Italian restaurant there?"
        )},
        {"role": "assistant", "content": (
            "For a romantic dinner, I would recommend Roscioli. It has a cozy and "
            "intimate atmosphere with soft lighting and excellent service."
        )},
    ], session_time=datetime(2023, 5, 30, 19, 20).timestamp())
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "Can you remind me of the name of the romantic Italian restaurant in Rome you recommended for dinner?",
        as_of=datetime(2023, 5, 31, 6, 29).timestamp(),
    )

    assert ar.answer == "Roscioli"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_structured_recall_for_hobbies(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _DecliningReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "Joanna", "content": (
            "Yeah! Besides writing, I also enjoy reading, watching movies, and exploring nature."
        )},
    ])
    sys.ingest_session("ns", "s1", [
        {"role": "Joanna", "content": "Writing and hanging with friends!"},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "What are Joanna's hobbies?")

    assert ar.answer == "Writing, reading, watching movies, exploring nature, hanging with friends"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_temporal_structured_recall_for_donation_date(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=True)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2022, 12, 22, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "Maria", "content": (
            "I donated my old car to a homeless shelter I volunteer at yesterday."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer("ns", "When did Maria donate her car?", as_of=session_time)

    assert ar.answer == "2022-12-21"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_eidetic_full_uses_temporal_structured_recall_for_last_week_month(tmp_path, monkeypatch):
    settings = replace(get_settings(), data_dir=tmp_path / "data")
    client = _TemporalReader(settings.embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client, temporal_evidence_audit_enabled=True)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    session_time = datetime(2023, 7, 3, 12, 0).timestamp()
    sys.ingest_session("ns", "s0", [
        {"role": "John", "content": (
            "Last week I scored 40 points, my highest ever, and it feels like all my hard work is paying off."
        )},
    ], session_time=session_time)
    sys.consolidate("ns")

    ar = sys.answer(
        "ns",
        "In which month's game did John achieve a career-high score in points?",
        as_of=session_time,
    )

    assert ar.answer == "June 2023"
    assert ar.extra["verified"] is True
    assert ar.extra["policy"].startswith("smqe:")
    assert client.reader_models == []
    get_settings.cache_clear()


def test_adapter_no_longer_exports_direct_rescue_helpers():
    import bench.adapters.eidetic_adapter as adapter

    forbidden = [
        "_compact_" + "temporal_slot_answer",
        "_extract_" + "user_slot_answer",
        "_extract_" + "direct_fact_match",
        "_extract_" + "open_domain_bridge_match",
        "_extract_" + "profile_fact_answer",
        "_verified_" + "direct_citations",
        "_verified_" + "atom_citations",
    ]

    for name in forbidden:
        assert not hasattr(adapter, name)


def test_eidetic_adapter_logs_region_hint_telemetry(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch, gist_channel_enabled=True)
    sys = EideticSystem(engine=e)
    scope = Scope(namespace="ns")
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "I prefer ginger tea in the afternoon."},
    ])
    rec = e.store.all_records(scope)[0]
    e.store.add_derived(DerivedRecord(
        cid="tea-region",
        kind="gist",
        namespace=scope.namespace,
        text="ginger tea preference",
        member_ids=[rec.memory_id],
    ))

    ar = sys.answer("ns", "What ginger tea preference did I mention?")

    assert ar.extra["region_hint_count"] == 1
    assert ar.extra["region_ids"] == ["tea-region"]
    assert ar.extra["region_member_ids"] == [rec.memory_id]
    get_settings.cache_clear()


def test_eidetic_reset_purges_reused_eval_namespace(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch, semantic_cache_enabled=True,
                reflex_recall_enabled=False, flow_activation_enabled=False)
    sys = EideticSystem(engine=e)
    ns = "eidetic-plus-eval-g0-r0"
    other = Scope(namespace="other")
    sys.ingest_session(ns, "s0", [{"role": "user", "content": "first run stale fact"}])
    e.ingest_text("other namespace survives", scope=other, consolidate_now=False)
    assert e.store.count(Scope(namespace=ns)) == 1
    assert e.store.count(other) == 1

    sk = Scope(namespace=ns).key()
    e.cache.put(sk, "cached question", None, "stale answer", version=e._ns_version(ns))
    e.feedback.append(ns, "cached question", {"coverage": 1.0}, reward=1.0)
    e._touch_hotset(ns, ["stale-id"])
    before_version = e._ns_version(ns)
    assert e.feedback.count(dev_only=False) == 1

    sys.reset(ns)

    assert e.store.count(Scope(namespace=ns)) == 0
    assert e.store.count(other) == 1
    assert e._ns_version(ns) == before_version + 1
    assert e.cache.get(sk, "cached question", None, version=e._ns_version(ns)) is None
    assert e.feedback.count(dev_only=False) == 0
    assert e._hotset_ids(ns) == set()
    sys.ingest_session(ns, "s1", [{"role": "user", "content": "second run only"}])
    assert [r.text for r in e.store.all_records(Scope(namespace=ns))] == ["user: second run only"]
    get_settings.cache_clear()


def test_eidetic_reset_resets_derived_indexes_when_corpus_empty(tmp_path, monkeypatch):
    e = _engine(tmp_path, monkeypatch)
    sys = EideticSystem(engine=e)
    ns = "eidetic-plus-eval-g0-r0"
    sys.ingest_session(ns, "s0", [{"role": "user", "content": "only namespace"}])
    assert e.store.count(None) == 1
    assert len(e.index) == 1
    assert len(e.retriever.bm25.docs) == 1

    sys.reset(ns)

    assert e.store.count(None) == 0
    assert len(e.index) == 0
    assert len(e.retriever.bm25.docs) == 0
    assert e.sync_health(Scope(namespace=ns))["in_sync"] is True
    get_settings.cache_clear()


def test_eidetic_full_wired_into_make_system():
    from bench.run import make_system
    assert make_system("eidetic-full").name == "eidetic-plus-full"
    assert make_system("eidetic-plus-full").name == "eidetic-plus-full"


class _QuotedSynthReader(_FakeReader):
    def chat(self, model, system, user, **kw):
        self.reader_models.append(model)
        return ("Likely yes. Rowan mentions hiking with 'colleagues from the trail club' and "
                "grabbing lunch with 'friends from the chess league' regularly.")

    def nli(self, premise, hypothesis):
        return ("neutral", 0.2)                 # grounding must come from quoted anchors


def test_eidetic_full_applies_rescue_grounding_to_fixed_reader_answers(tmp_path, monkeypatch):
    """The rescue layer (advice/likelihood restatement + quoted-span anchors) is VERIFICATION
    policy, not reader strength - it must apply to the neutral fixed-reader path exactly as it
    does in retriever.answer(), or verified flags flap with reader phrasing run to run."""
    client = _QuotedSynthReader(get_settings().embed_dim)
    e = _engine_with_client(tmp_path, monkeypatch, client)
    sys = EideticFullSystem(engine=e)
    sys.reset("ns")
    sys.ingest_session("ns", "s0", [
        {"role": "user", "content": "Rowan went hiking with colleagues from the trail club again."},
        {"role": "user", "content": "Rowan grabbed lunch with friends from the chess league."},
    ])
    sys.consolidate("ns")

    ar = sys.answer("ns", "Is it likely that Rowan has friends besides his sister?")

    assert not ar.abstained
    assert "trail club" in ar.answer
    assert ar.extra["verified"] is True
