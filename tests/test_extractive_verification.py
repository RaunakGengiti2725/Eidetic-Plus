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
    premise = "Caroline has had her current group of friends for four years."
    assert _extractive_entailment(premise, "4 years [S2]")
    assert _extractive_entailment("The job lasted a couple of months.", "2 months [S3]")
    assert not _extractive_entailment("Caroline met her friends for a picnic.", "4 years [S2]")


def test_extractive_entailment_canonicalizes_user_preferences():
    assert _extractive_entailment("user: I prefer window seats.", "User prefers window seats [S0]")
    assert _extractive_entailment("user: I am allergic to peanuts.", "User is allergic to peanuts [S1]")
    assert not _extractive_entailment("user: I dislike coffee.", "User likes coffee [S2]")


def test_extractive_entailment_proves_session_relative_yesterday_date():
    premise = "Caroline: I went to a LGBTQ support group yesterday and it was powerful."
    valid_at = datetime(2023, 5, 8, 12, 0, 0).timestamp()
    assert _extractive_entailment(premise, "2023-05-07 (Sunday) [S20]", valid_at)
    assert _extractive_entailment(premise, "7 May 2023 [S20]", valid_at)
    assert _extractive_entailment(
        premise,
        "Caroline went to a LGBTQ support group on 2023-05-07 [S20]",
        valid_at,
    )
    assert not _extractive_entailment(premise, "2023-05-07 (Monday) [S20]", valid_at)
    assert not _extractive_entailment(premise, "2023-05-06 (Saturday) [S20]", valid_at)
    assert not _extractive_entailment(
        premise,
        "Caroline went to the adoption agency on 2023-05-07 [S20]",
        valid_at,
    )


def test_extractive_entailment_proves_last_weekday_from_session_date():
    premise = "Caroline: I went to a pride parade last Friday."
    valid_at = datetime(2023, 8, 14, 12, 0, 0).timestamp()
    assert _extractive_entailment(premise, "August 11, 2023 (Friday) [S14]", valid_at)
    assert not _extractive_entailment(premise, "August 4, 2023 (Friday) [S14]", valid_at)


def test_extractive_entailment_bridges_trans_woman_alias():
    premise = "Caroline: I made this painting to show my path as a trans woman."
    assert _extractive_entailment(premise, "Caroline is a transgender woman [S0]")
    assert _extractive_entailment(premise, "transgender woman [S0]")
    assert not _extractive_entailment(
        "Caroline: I mentor a transgender teen just like me.",
        "Caroline is a transgender woman [S0]",
    )


def test_extractive_entailment_requires_title_to_appear_in_premise():
    premise = "Joanna: The Lighthouse Bell is one of my favorite movies."
    assert _extractive_entailment(premise, "The Lighthouse Bell [S0]")
    assert not _extractive_entailment(
        "Joanna: I like old coastal thrillers with foghorns.",
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
    premise = "Melanie: We're thinking about going camping next month."
    valid_at = datetime(2023, 5, 25, 12, 0, 0).timestamp()
    assert _extractive_entailment(
        premise,
        "Melanie planned on going camping in June 2023 [S1]",
        valid_at,
    )
    assert not _extractive_entailment(
        premise,
        "Melanie planned on going camping in July 2023 [S1]",
        valid_at,
    )


def test_extractive_entailment_proves_last_week_session_anchor():
    premise = (
        "Caroline: My friends, family and mentors are my rocks. "
        "Here's a pic from when we met up last week!"
    )
    valid_at = datetime(2023, 6, 9, 12, 0, 0).timestamp()
    assert _extractive_entailment(
        premise,
        "Caroline met up with her friends, family, and mentors the week before 9 June 2023 [S2]",
        valid_at,
    )
    assert not _extractive_entailment(
        premise,
        "Caroline met up with her coworkers the week before 9 June 2023 [S2]",
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
