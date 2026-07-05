"""Offline tests for deterministic extractive proof before model NLI."""
from __future__ import annotations

from datetime import datetime
from dataclasses import replace

from eidetic.graph import KnowledgeGraph
from eidetic.models import MemoryRecord, Modality, NLILabel, RetrievalCandidate, Scope
from eidetic.retrieval import Retriever, _extractive_entailment
from eidetic.store import RecordStore


def test_extractive_entailment_strips_source_tags():
    premise = "user: I attended The Glass Menagerie at the local community theater."
    assert _extractive_entailment(premise, "The Glass Menagerie [S10]")
    assert _extractive_entailment(premise, "The answer is The Glass Menagerie [source 10].")


def test_extractive_entailment_rejects_tiny_or_unsupported_fragments():
    premise = "user: I need to pick up 3 items."
    assert not _extractive_entailment(premise, "3 [S0]")
    assert not _extractive_entailment(premise, "The Glass Menagerie [S0]")
    assert not _extractive_entailment(premise, "I don't have enough verified evidence.")


def test_extractive_entailment_normalizes_duration_number_words():
    premise = "Priya has had her current circle of pen pals for four years."
    assert _extractive_entailment(premise, "4 years [S2]")
    assert _extractive_entailment("The job lasted a couple of months.", "2 months [S3]")
    assert not _extractive_entailment("Priya met her pen pals for a picnic.", "4 years [S2]")


def test_extractive_entailment_canonicalizes_user_preferences():
    assert _extractive_entailment("user: I prefer window seats.", "User prefers window seats [S0]")
    assert _extractive_entailment("user: I am allergic to peanuts.", "User is allergic to peanuts [S1]")
    assert not _extractive_entailment("user: I dislike coffee.", "User likes coffee [S2]")


def test_extractive_entailment_proves_session_relative_yesterday_date():
    premise = "Noor: I went to a chess club meetup yesterday and it was energizing."
    valid_at = datetime(2023, 5, 8, 12, 0, 0).timestamp()
    assert _extractive_entailment(premise, "2023-05-07 (Sunday) [S20]", valid_at)
    assert _extractive_entailment(premise, "7 May 2023 [S20]", valid_at)
    assert _extractive_entailment(
        premise,
        "Noor went to a chess club meetup on 2023-05-07 [S20]",
        valid_at,
    )
    assert not _extractive_entailment(premise, "2023-05-07 (Monday) [S20]", valid_at)
    assert not _extractive_entailment(premise, "2023-05-06 (Saturday) [S20]", valid_at)
    assert not _extractive_entailment(
        premise,
        "Noor went to the farmers market on 2023-05-07 [S20]",
        valid_at,
    )


def test_extractive_entailment_proves_strict_qa_temporal_answer_only_when_topic_supported():
    premise = "Ravi: I just signed up for a fencing class yesterday."
    valid_at = datetime(2023, 7, 3, 12, 0, 0).timestamp()
    assert _extractive_entailment(
        premise,
        "Question: When did Ravi sign up for a fencing class?\nAnswer: 2023-07-02",
        valid_at,
    )
    assert not _extractive_entailment(
        premise,
        "Question: When did Ravi adopt a parrot?\nAnswer: 2023-07-02",
        valid_at,
    )
    assert not _extractive_entailment(
        "Ravi: Absolutely!",
        "Question: When did Ravi buy the lanterns?\nAnswer: Absolutely",
        valid_at,
    )


def test_extractive_entailment_proves_strict_qa_duration_answer_only_when_topic_supported():
    premise = "Ari: How long have you been married?\nBlair: 5 years already!"
    assert _extractive_entailment(
        premise,
        "Question: How long has Blair been married?\nAnswer: 5 years",
    )
    assert _extractive_entailment(
        "Marco: They've stood by me through it all, I've known these teammates for 4 years.",
        "Question: How long has Marco had his current group of teammates for?\nAnswer: 4 years",
    )
    assert not _extractive_entailment(
        premise,
        "Question: How long has Blair lived in Lisbon?\nAnswer: 5 years",
    )


def test_extractive_entailment_proves_last_weekday_from_session_date():
    premise = "Wei: I went to a jazz concert last Friday."
    valid_at = datetime(2023, 8, 14, 12, 0, 0).timestamp()
    assert _extractive_entailment(premise, "August 11, 2023 (Friday) [S14]", valid_at)
    assert not _extractive_entailment(premise, "August 4, 2023 (Friday) [S14]", valid_at)


def test_extractive_entailment_bridges_trans_woman_alias():
    premise = "Priya: I painted this mural to celebrate my journey as a trans woman."
    assert _extractive_entailment(premise, "Priya is a transgender woman [S0]")
    assert _extractive_entailment(premise, "transgender woman [S0]")
    assert not _extractive_entailment(
        "Priya: I tutor a transgender teen in my art class.",
        "Priya is a transgender woman [S0]",
    )


def test_extractive_entailment_requires_title_to_appear_in_premise():
    premise = "Farid: The Lighthouse Bell is one of my favorite movies."
    assert _extractive_entailment(premise, "The Lighthouse Bell [S0]")
    assert not _extractive_entailment(
        "Farid: I like old coastal thrillers with foghorns.",
        "The Lighthouse Bell [S0]",
    )


def test_extractive_entailment_proves_markdown_schedule_table_cell():
    premise = (
        "assistant: |  | 8 am - 4 pm (Day Shift) | 12 pm - 8 pm (Afternoon Shift) | "
        "4 pm - 12 am (Evening Shift) | 12 am - 8 am (Night Shift) |\n"
        "| Sunday | Admon | Magdy | Ehab | Sara |"
    )
    assert _extractive_entailment(
        premise,
        "Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays.",
    )
    assert not _extractive_entailment(
        premise,
        "Admon was assigned to the 12 pm - 8 pm (Afternoon Shift) on Sundays.",
    )


def test_extractive_entailment_proves_next_month_named_month():
    premise = "Wei: We're thinking about going kayaking next month."
    valid_at = datetime(2023, 5, 25, 12, 0, 0).timestamp()
    assert _extractive_entailment(
        premise,
        "Wei planned on going kayaking in June 2023 [S1]",
        valid_at,
    )
    assert not _extractive_entailment(
        premise,
        "Wei planned on going kayaking in July 2023 [S1]",
        valid_at,
    )


def test_extractive_entailment_proves_last_week_session_anchor():
    premise = (
        "Noor: My cousins, neighbors and coaches are my anchors. "
        "Here's a photo from when we met up last week!"
    )
    valid_at = datetime(2023, 6, 9, 12, 0, 0).timestamp()
    assert _extractive_entailment(
        premise,
        "Noor met up with her cousins, neighbors, and coaches the week before 9 June 2023 [S2]",
        valid_at,
    )
    assert not _extractive_entailment(
        premise,
        "Noor met up with her coworkers the week before 9 June 2023 [S2]",
        valid_at,
    )


class _Substrate:
    def __init__(self, text):
        self.text = text

    def get(self, content_hash):
        return self.text.encode("utf-8")


class _Client:
    def __init__(self):
        self.nli_calls = []
        self.batch_calls = []

    def nli(self, premise, hypothesis):
        self.nli_calls.append((premise, hypothesis))
        return "neutral", 0.1

    def nli_batch(self, pairs):
        self.batch_calls.append(list(pairs))
        return [("neutral", 0.1) for _ in pairs]


def _retriever(settings, text, client):
    store = RecordStore(settings.sqlite_path)
    return Retriever(store, object(), KnowledgeGraph(store), _Substrate(text), client, settings)


def test_verify_citation_uses_extractive_entailment_before_nli(fresh_settings):
    client = _Client()
    r = _retriever(
        fresh_settings,
        "assistant: The playlist I created on Spotify is Summer Vibes.",
        client,
    )
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="fallback", scope=Scope())
    label, conf = r.verify_citation(rec, "Summer Vibes [S4]")
    assert (label, conf) == (NLILabel.ENTAILMENT, 1.0)
    assert client.nli_calls == []


def test_verify_citation_falls_back_to_nli_for_unsupported_claim(fresh_settings):
    client = _Client()
    r = _retriever(fresh_settings, "assistant: The playlist is Summer Vibes.", client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="fallback", scope=Scope())
    label, conf = r.verify_citation(rec, "Winter Vibes [S4]")
    assert (label, conf) == (NLILabel.NEUTRAL, 0.1)
    assert len(client.nli_calls) == 1


def test_verify_citation_bounds_long_premise_before_nli(fresh_settings):
    settings = replace(fresh_settings, raw_span_min_chars=500)
    client = _Client()
    text = (
        "\n".join(f"user: prefix filler {i} about routine status." for i in range(120))
        + "\nassistant: The playlist I created on Spotify is Summer Vibes."
        + "\n" + "\n".join(f"assistant: tail filler {i}." for i in range(80))
    )
    r = _retriever(settings, text, client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text=text, scope=Scope())

    label, conf = r.verify_citation(rec, "Winter Vibes [S4]")

    assert (label, conf) == (NLILabel.NEUTRAL, 0.1)
    assert len(client.nli_calls) == 1
    premise, _hypothesis = client.nli_calls[0]
    assert len(premise) < len(text)
    assert len(premise) <= 3200
    assert "Summer Vibes" in premise


def test_verify_candidates_cites_buried_long_span_not_prefix(fresh_settings):
    settings = replace(fresh_settings, raw_span_min_chars=500)
    client = _Client()
    text = (
        "\n".join(f"user: prefix filler {i} about routine status." for i in range(120))
        + "\nuser: The launch code for Project Helios is BLUE-17."
        + "\n" + "\n".join(f"assistant: tail filler {i}." for i in range(80))
    )
    r = _retriever(settings, text, client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text=text, scope=Scope(),
                       modality=Modality.TEXT)
    cands = [RetrievalCandidate(record=rec, dense_score=0.9, fused_score=1.0)]

    citations, entailed = r._verify_candidates(
        cands,
        "BLUE-17 [S0]",
        True,
        query="What is the launch code for Project Helios?",
    )

    assert entailed == 1
    assert citations[0].nli_label == NLILabel.ENTAILMENT
    assert "BLUE-17" in citations[0].snippet
    assert "prefix filler 0" not in citations[0].snippet
    assert client.nli_calls == []


def test_batch_verify_skips_local_extractively_proven_pairs(fresh_settings):
    settings = replace(fresh_settings, batch_nli_enabled=True)
    client = _Client()
    r = _retriever(settings, "assistant: I attended The Glass Menagerie.", client)
    rec = MemoryRecord(memory_id="m1", content_hash="h1", text="fallback", scope=Scope(),
                       modality=Modality.TEXT)
    cands = [RetrievalCandidate(record=rec, dense_score=0.9, fused_score=1.0)]
    citations, entailed = r._verify_candidates(cands, "The Glass Menagerie [S0]", True)
    assert entailed == 1
    assert citations[0].nli_label == NLILabel.ENTAILMENT
    assert client.batch_calls == []
