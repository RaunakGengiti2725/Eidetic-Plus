"""Rotating synthetic invariant eval for SMQE.

This is deliberately dataset-neutral: each run invents fresh names, objects, schedules, dates, and
preferences from a seed, then requires the real SMQE path to return verified answers with clean
proofs. It is cheap enough to run often and gives us a moving target without touching held-out
benchmark questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

from bench.seed_policy import resolve_seed
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from eidetic.models import ClaimRecord, MemoryRecord, NLILabel, Scope
from eidetic.smqe import structured_answer
from eidetic.store import RecordStore


class _Retriever:
    def __init__(self, store: RecordStore):
        self.store = store

    def verify_citation(self, rec, atom):
        premise = " ".join((rec.text or "").lower().split())
        hyp = " ".join((atom or "").lower().split())
        return (NLILabel.ENTAILMENT, 1.0) if hyp and hyp in premise else (NLILabel.NEUTRAL, 0.0)


@dataclass
class SyntheticCase:
    case_id: str
    op: str
    question: str
    expected: str
    rows: list[tuple[str, float]]
    forbidden_in_proof: list[str] = field(default_factory=list)
    claim_row: Optional[int] = None
    # P0 fail-closed (2026-07-09): a count/sum DERIVED by enumerating across atoms no longer
    # verifies (eidetic/smqe/verify.py aggregate citation floor); it abstains. Such cases assert
    # abstention -- the derivation may still compute the value, but the verify-or-abstain surface
    # returns None rather than an uncorroborated "verified" aggregate.
    expect_abstain: bool = False


_NAMES = ["Ari", "Nila", "Mika", "Sana", "Theo", "Rowan", "Lina", "Owen", "Tessa", "Ira"]
_OBJECTS = [
    "backup badge", "kiln token", "garden permit", "field notebook", "studio key",
    "travel charger", "linen receipt", "camera strap", "climbing pass", "museum ticket",
]
_LOCATIONS = [
    "Quartz Loft", "North Pier Studio", "Cedar Annex", "Blue Finch Lab", "Orchid Room",
    "Silver Gate Gym", "Harbor Desk", "Maple Archive", "River Gate", "Juniper Shelf",
]
_COUNT_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}
_COUNT_TARGETS = ["ceramic studios", "tea shops", "library workshops", "bike routes", "garden plots"]
_COUNT_DECOYS = ["museum exhibits", "hiking trails", "recipe cards", "repair notes", "train stops"]
_ITEMIZED_COUNT_TARGETS = {
    "fabric swatches": [
        "crimson linen fabric swatch", "blue wool fabric swatch", "moss cotton fabric swatch",
        "amber silk fabric swatch", "slate hemp fabric swatch",
    ],
    "sensor modules": [
        "north ridge sensor module", "harbor relay sensor module", "cedar gate sensor module",
        "quartz loft sensor module", "silver bay sensor module",
    ],
    "recipe cards": [
        "mint tart recipe card", "berry salad recipe card", "cedar soup recipe card",
        "harbor bread recipe card", "juniper tea recipe card",
    ],
    "field samples": [
        "orchid room field sample", "blue clay field sample", "maple dust field sample",
        "river silt field sample", "quartz shard field sample",
    ],
}
_ACQUIRED_ITEM_COUNT_TARGETS = {
    "workshop supplies": [
        "blue awl", "copper clamp", "brass workshop supply bin", "linen mallet", "opal bench supply tray",
    ],
    "field kit pieces": [
        "ridge compass", "silver flag", "cedar field kit pouch", "quartz tape", "maple sample loop",
    ],
    "studio materials": [
        "indigo wax block", "linen stencil", "amber studio material roll", "opal pigment jar", "copper frame",
    ],
    "repair parts": [
        "hinge collar", "brass washer", "cedar repair part sleeve", "silver latch", "opal cable clip",
    ],
}
_PREF_GOOD = ["mint tea", "fantasy novels", "graphite pens", "berry salad", "quiet playlists"]
_PREF_BAD = ["cedar tea", "tax manuals", "brass pens", "kale salad", "alarm playlists"]
_OPEN_GOOD = ["quiet murals", "warm notebooks", "linen tea", "moon gardens", "harbor maps"]
_OPEN_BAD = ["cedar candles", "alarm drills", "crowded tours", "tax ledgers", "brass whistles"]
_TABLE_VALUES = ["7 AM", "late", "north desk", "2 PM", "midday", "west desk"]
_PROJECTS = ["mural ledger", "orchid catalog", "harbor map", "kiln checklist", "field guide"]
_SUGGESTIONS = [
    "Virtual Coffee Breaks", "Online Team Activities", "Collaborative Projects",
    "Social Channels", "Interest-Based Groups", "Weekly Planning Notes",
]
_RESOURCE_SUGGESTION_SETS = [
    ("craft materials", "copper wire", "indigo paper", "amber beads", "velvet ribbon"),
    ("repair supplies", "brass hinge", "cedar shim", "opal washers", "linen tape"),
    ("studio components", "quartz lens", "silver clip", "maple spacer", "cotton cord"),
    ("field resources", "ridge twine", "blue chalk", "cedar flags", "museum ticket"),
]
_ORGANIZATION_SUGGESTION_SETS = [
    ("workspace", "drafting desk", "brass tray", "sketch pencils", "oak desktop", "lamp", "blue scarf"),
    ("studio", "paint shelf", "copper bin", "wax brushes", "maple counter", "easel", "travel charger"),
    ("office", "invoice drawer", "linen folder", "receipt slips", "slate desk", "monitor", "garden permit"),
    ("bench", "tool rail", "amber cup", "tiny clamps", "cedar surface", "vise", "museum ticket"),
]
_SUPPORT_SUGGESTION_SETS = [
    ("printer", "feed", "silicone roller", "paper guide kit", "picnic blanket"),
    ("camera rig", "balance", "counterweight clamp", "rail spacer", "garden permit"),
    ("field recorder", "wind noise", "foam sleeve", "cable clip", "museum ticket"),
    ("tablet stand", "wobble", "hinge shim", "rubber foot kit", "linen receipt"),
]
_INSPIRATION_SUGGESTION_SETS = [
    ("sound sketches", "modular synth demos on Signal Garden", "patching tutorials", "live looping forums", "10-day sound sketch challenge", "tax binder"),
    ("collage studies", "layered paper studies on Maker Atlas", "folding tutorials", "zine forums", "12-day collage challenge", "invoice drawer"),
    ("dance notes", "mirror step examples on Harbor Stage", "movement tutorials", "choreography boards", "8-day phrase challenge", "receipt slips"),
    ("story drafts", "microfiction prompts on Lantern Desk", "revision tutorials", "writer circles", "14-day scene challenge", "garden permit"),
]
_BEVERAGE_SUGGESTION_SETS = [
    ("studio gathering", "Aurora Spritz", "garden class", "Ruby Citrus", "prism cup", "cedar binder"),
    ("gallery party", "Orchid Spark", "fermentation class", "Lumen Citrus", "opal coupe", "invoice drawer"),
    ("workshop social", "Harbor Tonic", "botanical class", "Silver Lime", "tall flute", "tax ledger"),
    ("porch meetup", "Maple Fizz", "herbal class", "Amber Peel", "blue tumbler", "field permit"),
]
_COMPAT_SETUPS = [
    ("audio", "Novum S3", "Novum recorders", "shock mount", "ArcSound cable kit", "weather sleeves"),
    ("field", "Orchid X2", "Orchid systems", "stabilizer clip", "Harbor battery sled", "rain covers"),
    ("studio", "Quartz M7", "Quartz rigs", "desk clamp", "Northlight adapter rail", "dust covers"),
    ("travel", "Cedar P5", "Cedar packs", "strap latch", "Silvergate organizer insert", "compression sleeves"),
]
_ACTIVITY_SETS = [
    ("creative workshops", "cyanotype printing", "ceramic glazing", "copper etching"),
    ("movement classes", "silk balancing", "mirror stepping", "drift running"),
    ("repair drills", "hinge tuning", "cable tracing", "bracket folding"),
    ("field exercises", "ridge mapping", "signal flagging", "sample tagging"),
]
_HOBBY_SETS = [
    ("interests", "copper sketching", "quiet rowing", "prism baking", "moon gardening"),
    ("hobbies", "linen weaving", "harbor jogging", "cedar cooking", "map sorting"),
    ("interests", "opal carving", "signal painting", "drift baking", "blue quilting"),
    ("hobbies", "ridge journaling", "amber stitching", "mirror dancing", "field drawing"),
]
_TRAVEL_DURATION_TARGETS = ["delivery stops", "field locations", "survey sites", "route checkpoints"]
_CONSECUTIVE_EVENT_SETS = [
    ("cleanup drills", "harbor cleanup drill", "river cleanup drill", "invoice review"),
    ("safety checks", "north safety check", "cedar safety check", "recipe filing"),
    ("studio surveys", "kiln studio survey", "loft studio survey", "budget meeting"),
    ("field walks", "ridge field walk", "orchid field walk", "ticket sorting"),
]
_AFFILIATION_FOLLOWUP_SETS = [
    ("team", "signed with", "sign with", "a new team", "The Harbor Signal Club", "The Brass Ledger Group"),
    ("research cohort", "joined", "join", "a new research cohort", "The Silver Orchard Lab", "The Copper Lantern Cohort"),
    ("studio group", "accepted into", "get accepted into", "a new studio group", "The Prism Workroom", "The Cedar Archive Studio"),
    ("club", "committed to", "commit to", "a new club", "The Maple Signal Club", "The Amber Ridge Circle"),
    ("lab", "transferred to", "transfer to", "a new lab", "The Blue Finch Lab", "The Opal Quarry Lab"),
]


def _record(text: str, *, scope: Scope, valid_at: float) -> MemoryRecord:
    digest = hashlib.sha256(f"{scope.namespace}\0{text}\0{valid_at}".encode("utf-8")).hexdigest()
    return MemoryRecord(
        text=text,
        source="user",
        scope=scope,
        valid_at=valid_at,
        content_hash=f"h-{digest}",
        raw_uri="mem://synthetic-smqe",
    )


def _pick(rng: random.Random, values: list[str], suffix: str = "") -> str:
    return rng.choice(values) + suffix


def _titleish_for_expected(value: str) -> str:
    return " ".join(part[:1].upper() + part[1:] for part in value.split())


def _sentence_for_expected(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _latest_case(rng: random.Random, idx: int) -> SyntheticCase:
    name = _pick(rng, _NAMES)
    obj = _pick(rng, _OBJECTS, f" {idx}")
    loc = _pick(rng, _LOCATIONS)
    other = _pick(rng, [x for x in _NAMES if x != name])
    decoy_loc = _pick(rng, [x for x in _LOCATIONS if x != loc])
    t = 1_700_100_000 + idx * 100
    return SyntheticCase(
        case_id=f"latest-{idx}",
        op="latest_value",
        question=f"Where does {name} keep the {obj}?",
        expected=loc,
        rows=[
            (f"{other}: I keep the {obj} at {decoy_loc}.", t),
            (f"{name}: I keep the {obj} at {loc}.", t + 1),
        ],
        forbidden_in_proof=[other, decoy_loc],
        claim_row=1,
    )


def _count_case(rng: random.Random, idx: int) -> SyntheticCase:
    target = _pick(rng, _COUNT_TARGETS)
    singular = target[:-1] if target.endswith("s") else target
    decoy = _pick(rng, _COUNT_DECOYS, f" {idx}")
    n = rng.randint(2, 5)
    t = 1_700_200_000 + idx * 100
    labels = rng.sample(_LOCATIONS, k=n)
    rows = [
        (f"User: I visited the {label} {singular} this month.", t + j)
        for j, label in enumerate(labels, start=1)
    ]
    rows.extend([
        (f"User: I bookmarked a directory of {target}.", t + 20),
        (f"User: I visited {n + 3} new {decoy} this month.", t + 21),
    ])
    return SyntheticCase(
        case_id=f"count-{idx}",
        op="count_aggregate",
        question=f"How many {target} did I visit this month?",
        expected=str(n),
        rows=rows,
        forbidden_in_proof=[decoy, f"{n + 3} new"],
        expect_abstain=True,
    )


def _itemized_count_case(rng: random.Random, idx: int) -> SyntheticCase:
    target = rng.choice(sorted(_ITEMIZED_COUNT_TARGETS))
    items = rng.sample(_ITEMIZED_COUNT_TARGETS[target], k=rng.randint(2, 4))
    decoy_target = _pick(rng, [x for x in _COUNT_DECOYS if x != target], f" {idx}")
    decoy_item = _pick(rng, _OBJECTS, f" {idx}")
    t = 1_700_225_000 + idx * 100
    if len(items) == 2:
        item_text = f"a {items[0]} and a {items[1]}"
    else:
        item_text = ", ".join(f"a {item}" for item in items[:-1]) + f", and a {items[-1]}"
    return SyntheticCase(
        case_id=f"itemized-count-{idx}",
        op="count_aggregate",
        question=f"How many {target} did I buy?",
        expected=f"{len(items)} {target}: " + "; ".join(items),
        rows=[
            (f"User: I bought {item_text} at the {_pick(rng, _LOCATIONS)}.", t),
            (f"User: I bookmarked a directory of {target}, but I did not buy from it.", t + 1),
            (f"User: I bought a {decoy_item} while checking {decoy_target}.", t + 2),
        ],
        forbidden_in_proof=["directory", decoy_item, decoy_target],
        expect_abstain=True,
    )


def _acquired_item_count_case(rng: random.Random, idx: int) -> SyntheticCase:
    target = rng.choice(sorted(_ACQUIRED_ITEM_COUNT_TARGETS))
    target_items = _ACQUIRED_ITEM_COUNT_TARGETS[target]
    target_words = {word.rstrip("s") for word in target.split() if len(word) > 3}
    anchored = [item for item in target_items if any(word in item for word in target_words)]
    third = rng.choice(anchored)
    first, second = rng.sample([item for item in target_items if item != third], k=2)
    items = [first, second, third]
    decoy = _pick(rng, [x for x in _OBJECTS if x not in items], f" {idx}")
    t = 1_700_235_000 + idx * 100
    return SyntheticCase(
        case_id=f"acquired-count-{idx}",
        op="count_aggregate",
        question=f"How many {target} did I acquire last month?",
        expected=f"3 {target}: " + "; ".join(item.lower() for item in items),
        rows=[
            (f"User: My {items[0]}, which I got from the {_pick(rng, _LOCATIONS)} last month along with a {items[1]}.", t),
            (f"User: My {items[2]}, which I got from the {_pick(rng, _LOCATIONS)} last month, is labeled.", t + 1),
            (f"User: My {decoy}, which I got last month, is unrelated to the {target} shelf.", t + 2),
        ],
        forbidden_in_proof=[decoy],
        expect_abstain=True,
    )


def _relative_case(rng: random.Random, idx: int) -> SyntheticCase:
    item = _pick(rng, _OBJECTS, f" {idx}")
    ref = datetime(2024, rng.randint(1, 10), rng.randint(10, 24), 12, 0)
    expected = (ref - timedelta(days=1)).date().isoformat()
    decoy = _pick(rng, [x for x in _OBJECTS if x not in item], f" {idx}")
    t = ref.timestamp()
    return SyntheticCase(
        case_id=f"relative-{idx}",
        op="relative_temporal",
        question=f"When did I pick up the {item}?",
        expected=expected,
        rows=[
            (f"User: Yesterday I picked up the {item}.", t),
            (f"User: Yesterday I cleaned the {decoy}.", t + 1),
        ],
        forbidden_in_proof=[decoy],
    )


def _table_case(rng: random.Random, idx: int) -> SyntheticCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    day = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])
    other_day = rng.choice([d for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"] if d != day])
    value = _pick(rng, _TABLE_VALUES)
    decoy = _pick(rng, [x for x in _TABLE_VALUES if x != value])
    t = 1_700_300_000 + idx * 100
    return SyntheticCase(
        case_id=f"table-{idx}",
        op="table_lookup",
        question=f"What shift does {person} have on {day} in the schedule?",
        expected=value,
        rows=[
            (
                f"| Name | {day} | {other_day} |\n"
                f"| {other} | {decoy} | off |\n"
                f"| {person} | {value} | standby |",
                t,
            )
        ],
        forbidden_in_proof=[other],
    )


def _preference_case(rng: random.Random, idx: int) -> SyntheticCase:
    good = _pick(rng, _PREF_GOOD, f" {idx}")
    bad = _pick(rng, _PREF_BAD, f" {idx}")
    t = 1_700_400_000 + idx * 100
    return SyntheticCase(
        case_id=f"preference-{idx}",
        op="preference_synth",
        question=f"Would I prefer {bad} or {good}?",
        expected=good,
        rows=[
            (f"User: The shopping note mentions {bad} as a label.", t),
            (f"User: I avoid {bad} before meetings.", t + 1),
            (f"User: I enjoy {good} after work.", t + 2),
        ],
        forbidden_in_proof=[bad],
    )


def _open_inference_case(rng: random.Random, idx: int) -> SyntheticCase:
    good = _pick(rng, _OPEN_GOOD, f" {idx}")
    bad = _pick(rng, _OPEN_BAD, f" {idx}")
    decoy = _pick(rng, [x for x in _OPEN_GOOD if x not in good], f" {idx}")
    t = 1_700_450_000 + idx * 100
    return SyntheticCase(
        case_id=f"open-{idx}",
        op="open_inference",
        question=f"Would I enjoy {bad} or {good}?",
        expected=good,
        rows=[
            (f"User: I saw {decoy} listed in a newsletter.", t),
            (f"User: {bad} makes me uncomfortable during planning sessions.", t + 1),
            (f"User: I usually enjoy {good} when I need to focus.", t + 2),
        ],
        forbidden_in_proof=[bad, decoy],
    )


def _suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    group = _pick(rng, ["remote planning circle", "studio cohort", "weekend build team"], f" {idx}")
    items = rng.sample(_SUGGESTIONS, k=4)
    t = 1_700_475_000 + idx * 100
    expected = ", ".join(items[:-1]) + f", and {items[-1]}"
    numbered = " ".join(f"{n}. {item}." for n, item in enumerate(items, start=1))
    return SyntheticCase(
        case_id=f"suggestion-{idx}",
        op="preference_synth",
        question=f"I want to stay connected with my {group}. Any suggestions?",
        expected=expected,
        rows=[
            (f"Assistant: Suggestions to stay connected with the {group}: {numbered}", t),
            (f"User: I filed unrelated notes about the {group} agenda.", t + 1),
        ],
    )


def _resource_suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    target, first, second, third, decoy = rng.choice(_RESOURCE_SUGGESTION_SETS)
    t = 1_700_480_000 + idx * 100
    expected = f"{first}, {second}, and {third}"
    return SyntheticCase(
        case_id=f"resource-suggestion-{idx}",
        op="preference_synth",
        question=f"What should I use for my project with my {target}?",
        expected=expected,
        rows=[
            (f"User: I've been using {first} and {second} lately.", t),
            (f"User: I collected {third} from the {_pick(rng, _LOCATIONS)} for the project.", t + 1),
            (f"User: I stored {decoy} for a different label.", t + 2),
        ],
        forbidden_in_proof=[decoy],
    )


def _organization_suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    space, area, storage, items, surface, landmark, decoy = rng.choice(_ORGANIZATION_SUGGESTION_SETS)
    t = 1_700_482_000 + idx * 100
    expected = f"{area}, {storage}, {items} clutter-free, and {surface} near the {landmark}"
    return SyntheticCase(
        case_id=f"organization-suggestion-{idx}",
        op="preference_synth",
        question=f"My {space} is getting messy again. Any tips for keeping it tidy?",
        expected=expected,
        rows=[
            (f"User: I need help organizing my {area}. I recently bought a {storage} to keep {items} clutter-free.", t),
            (f"User: I noticed some marks on the {surface} near the {landmark}.", t + 1),
            (f"User: I keep a {decoy} by the hallway hooks.", t + 2),
        ],
        forbidden_in_proof=[decoy],
    )


def _support_suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    device, issue, first, second, decoy = rng.choice(_SUPPORT_SUGGESTION_SETS)
    t = 1_700_483_000 + idx * 100
    expected = f"{first}, and {second}"
    return SyntheticCase(
        case_id=f"support-suggestion-{idx}",
        op="preference_synth",
        question=f"I've been having trouble with {issue} on my {device}. Any tips?",
        expected=expected,
        rows=[
            (f"User: I am looking for advice on organizing {device} accessories, like my {first} and {second}.", t),
            (f"Assistant: Keep frequently used tools, like the {device} and {first}, near the work area.", t + 1),
            (f"User: I keep a {decoy} in the hallway bin.", t + 2),
        ],
        forbidden_in_proof=[decoy],
    )


def _inspiration_suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    domain, first, second, third, challenge, decoy = rng.choice(_INSPIRATION_SUGGESTION_SETS)
    t = 1_700_484_000 + idx * 100
    expected = f"{first}, {second}, {third}, and {challenge}"
    return SyntheticCase(
        case_id=f"inspiration-suggestion-{idx}",
        op="preference_synth",
        question=f"I've been feeling stuck with my {domain} lately. Any ideas on how I can find new inspiration?",
        expected=expected,
        rows=[
            (f"User: I've been looking at {first} for inspiration.", t),
            (f"User: I've been looking at some {second}, but I'm not sure where to start.", t + 1),
            (f"User: I have been getting inspiration from {third} and recently started a {challenge}.", t + 2),
            (f"User: I filed unrelated notes about a {decoy}.", t + 3),
        ],
        forbidden_in_proof=[decoy],
    )


def _beverage_suggestion_case(rng: random.Random, idx: int) -> SyntheticCase:
    event, title, class_source, ingredient, vessel, decoy = rng.choice(_BEVERAGE_SUGGESTION_SETS)
    t = 1_700_484_500 + idx * 100
    return SyntheticCase(
        case_id=f"beverage-suggestion-{idx}",
        op="preference_synth",
        question=f"I'm choosing a beverage for a {event}. Any suggestions?",
        expected=f"{title}, {class_source}, {vessel}, and {ingredient}",
        rows=[
            (
                f"User: I was thinking of making a mocktail for a {event}.\n"
                f"Assistant: {title}: A bright citrus beverage gets an herb syrup finish.\n"
                f"User: I liked the {title}, and I took notes after a {class_source}.\n"
                f"User: I think I'll try {ingredient} for the simple syrup.\n"
                f"User: I think I'll try serving the {title} in a {vessel}.\n"
                f"User: I filed unrelated notes about a {decoy}.",
                t,
            ),
        ],
        forbidden_in_proof=[decoy],
    )


def _compatibility_case(rng: random.Random, idx: int) -> SyntheticCase:
    domain, device, system, first_item, second_item, third_item = rng.choice(_COMPAT_SETUPS)
    t = 1_700_485_000 + idx * 100
    expected = (
        f"Options compatible with your {device} setup: high-quality {third_item}, "
        f"{device} {first_item}, and {second_item}"
    )
    return SyntheticCase(
        case_id=f"compatibility-{idx}",
        op="preference_synth",
        question=f"Can you suggest accessories that would complement my current {domain} setup?",
        expected=expected,
        rows=[
            (f"User: I use a {device} for my {domain} setup.", t),
            (f"Assistant: Consider a {device} {first_item} or {second_item}.", t + 1),
            (f"User: I want accessories that are compatible with {system}.", t + 2),
            (f"Assistant: FieldKit makes high-quality {third_item} that are compatible with {system}.", t + 3),
            (f"Assistant: I also filed unrelated shelf notes for the {domain} setup.", t + 4),
        ],
        forbidden_in_proof=["unrelated shelf"],
    )


def _activity_case(rng: random.Random, idx: int) -> SyntheticCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    label, first, second, decoy = rng.choice(_ACTIVITY_SETS)
    t = 1_700_492_000 + idx * 100
    expected = ", ".join(sorted(
        [_titleish_for_expected(first), _titleish_for_expected(second)],
        key=lambda value: value.lower(),
    ))
    return SyntheticCase(
        case_id=f"activity-{idx}",
        op="open_inference",
        question=f"What {label} has {person} done?",
        expected=expected,
        rows=[
            (f"{person}: I'm doing {first} and it helps me focus.", t),
            (f"{person}: I'm off to do some {second}!", t + 1),
            (f"{other}: I'm doing {decoy} this week.", t + 2),
        ],
        forbidden_in_proof=[other, decoy],
    )


def _hobby_case(rng: random.Random, idx: int) -> SyntheticCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    label, first, second, third, fourth = rng.choice(_HOBBY_SETS)
    t = 1_700_496_000 + idx * 100
    return SyntheticCase(
        case_id=f"hobby-{idx}",
        op="open_inference",
        question=f"What are {person}'s {label}?",
        expected=f"{_sentence_for_expected(first)}, {second}, {third}, {fourth}",
        rows=[
            (f"{person}: My {label} include {first}, {second}, and {third}.", t),
            (f"{other}: My {label} include tax ledgers and alarm drills.", t + 1),
            (f"{person}: I also enjoy {fourth}.", t + 2),
        ],
        forbidden_in_proof=[other, "tax ledgers", "alarm drills"],
    )


def _affiliation_followup_case(rng: random.Random, idx: int) -> SyntheticCase:
    person = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != person])
    label, memory_action, question_action, vague, title, decoy_title = rng.choice(_AFFILIATION_FOLLOWUP_SETS)
    t = 1_700_498_000 + idx * 100
    return SyntheticCase(
        case_id=f"affiliation-followup-{idx}",
        op="latest_value",
        question=f"Which {label} did {person} {question_action}?",
        expected=title,
        rows=[
            (
                f"{person}: I just {memory_action} {vague} - excited for the next rotation!\n"
                f"{person}: {title}! I start next week.",
                t,
            ),
            (
                f"{other}: I just {memory_action} {vague} too.\n"
                f"{other}: {decoy_title}! They start later.",
                t + 1,
            ),
        ],
        forbidden_in_proof=[other, decoy_title],
    )


def _speaker_case(rng: random.Random, idx: int) -> SyntheticCase:
    speaker = _pick(rng, _NAMES)
    other = _pick(rng, [x for x in _NAMES if x != speaker])
    topic = _pick(rng, _OBJECTS, f" {idx}")
    other_topic = _pick(rng, [x for x in _OBJECTS if x not in topic], f" {idx}")
    wanted = _pick(rng, _LOCATIONS).lower()
    distractor = _pick(rng, [x for x in _LOCATIONS if x.lower() != wanted]).lower()
    t = 1_700_500_000 + idx * 100
    expected = f"the {topic} stays in the {wanted}"
    return SyntheticCase(
        case_id=f"speaker-{idx}",
        op="speaker_fact",
        question=f"What did {speaker} say about the {topic}?",
        expected=expected,
        rows=[
            (f"{speaker}: I said the {other_topic} stays in the {wanted}.", t),
            (f"{other}: I said the {topic} stays in the {distractor}.", t + 1),
            (f"{speaker}: I said the {topic} stays in the {wanted}.", t + 2),
        ],
        forbidden_in_proof=[other, distractor, other_topic],
    )


def _temporal_delta_case(rng: random.Random, idx: int) -> SyntheticCase:
    start_item = _pick(rng, _OBJECTS, f" {idx}")
    finish_item = _pick(rng, [x for x in _OBJECTS if x not in start_item], f" {idx}")
    decoy_start = _pick(rng, [x for x in _OBJECTS if x not in start_item and x not in finish_item], f" {idx}")
    decoy_finish = _pick(rng, [x for x in _OBJECTS if x not in start_item and x not in finish_item and x not in decoy_start], f" {idx}")
    start = datetime(2024, rng.randint(1, 9), rng.randint(2, 12), 12, 0)
    days = rng.randint(3, 11)
    finish = start + timedelta(days=days)
    t0 = start.timestamp()
    return SyntheticCase(
        case_id=f"delta-{idx}",
        op="temporal_delta",
        question=(
            f"How many days passed between the day I started calibrating the {start_item} "
            f"and the day I finished installing the {finish_item}?"
        ),
        expected=f"{days} days",
        rows=[
            (f"User: I started calibrating the {start_item} today.", t0),
            (f"User: I started calibrating the {decoy_start} today.", (start + timedelta(days=1)).timestamp()),
            (f"User: I finished installing the {decoy_finish} today.", (finish - timedelta(days=1)).timestamp()),
            (f"User: I finished installing the {finish_item} today.", finish.timestamp()),
        ],
        forbidden_in_proof=[decoy_start, decoy_finish],
    )


def _sum_case(rng: random.Random, idx: int) -> SyntheticCase:
    project = _pick(rng, _PROJECTS, f" {idx}")
    decoy = _pick(rng, [x for x in _PROJECTS if x not in project], f" {idx}")
    first = rng.randint(1, 4)
    second = rng.randint(2, 5)
    total = first + second
    t = 1_700_600_000 + idx * 100
    if rng.choice([False, True]):
        currency_word = rng.choice(["dollars", "bucks"])
        question_template = rng.choice([
            "How much did I spend on the {project} altogether?",
            "How much have I paid for the {project} overall?",
            "What did I spend on the {project} in total?",
            "How much did the {project} cost me overall?",
        ])
        verb = rng.choice(["spent", "paid"])
        prep = "on" if verb == "spent" else "for"

        def amount(value: int) -> str:
            return f"{_COUNT_WORDS[value]} {currency_word}"

        return SyntheticCase(
            case_id=f"sum-money-words-{idx}",
            op="multi_session_sum",
            question=question_template.format(project=project),
            expected=f"${total}",
            rows=[
                (f"User: I {verb} {amount(first)} {prep} the {project}.", t),
                (f"User: I {verb} {amount(second)} {prep} the {project}.", t + 10),
                (f"User: I {verb} {amount(total + 3)} {prep} the {decoy}.", t + 20),
            ],
            forbidden_in_proof=[decoy, amount(total + 3)],
            expect_abstain=True,
        )
    return SyntheticCase(
        case_id=f"sum-{idx}",
        op="multi_session_sum",
        question=f"How many total hours did I spend on the {project}?",
        expected=f"{total} hours",
        rows=[
            (f"User: I spent {first} hours on the {project}.", t),
            (f"User: I spent {second} hours on the {project}.", t + 10),
            (f"User: I spent {total + 3} hours on the {decoy}.", t + 20),
        ],
        forbidden_in_proof=[decoy, f"{total + 3} hours"],
        expect_abstain=True,
    )


def _travel_duration_case(rng: random.Random, idx: int) -> SyntheticCase:
    target = _pick(rng, _TRAVEL_DURATION_TARGETS)
    stops = rng.sample(_LOCATIONS, k=3)
    durations = rng.sample([2, 3, 4, 5, 6, 7], k=3)
    total = sum(durations)
    t = 1_700_650_000 + idx * 100
    return SyntheticCase(
        case_id=f"travel-duration-{idx}",
        op="multi_session_sum",
        question=f"How many hours in total did I spend driving to my three {target} combined?",
        expected=f"{total:g} hours for getting to the three {target} (or {total * 2:g} hours for the round trip)",
        rows=[
            (f"User: The trip to {stops[0]} took {durations[0]} hours to drive from home.", t),
            (f"Assistant: {stops[1]} is about {durations[1]} hours from home by route.", t + 1),
            (f"User: I drove for {durations[2]} hours to {stops[2]} recently.", t + 2),
            (f"User: I spent {total + 1} hours packing maps for the route.", t + 3),
            (f"User: The museum visit lasted {total + 2} hours.", t + 4),
        ],
        forbidden_in_proof=["packing maps", "museum visit"],
        expect_abstain=True,
    )


def _consecutive_event_delta_case(rng: random.Random, idx: int) -> SyntheticCase:
    label, first_event, second_event, decoy_event = rng.choice(_CONSECUTIVE_EVENT_SETS)
    start = datetime(2024, rng.randint(1, 8), rng.randint(2, 18), 12, 0)
    second = start + timedelta(days=1)
    decoy = start + timedelta(days=4)
    question_time = datetime.fromtimestamp(1_800_000_000)
    expected_months = max(0, (question_time.date() - second.date()).days // 30)
    t = start.timestamp()
    return SyntheticCase(
        case_id=f"consecutive-event-{idx}",
        op="temporal_delta",
        question=f"How many months have passed since I did two {label} in a row, on consecutive days?",
        expected=str(expected_months),
        rows=[
            (f"User: I completed the {first_event} today.", t),
            (f"User: I completed the {second_event} today.", second.timestamp()),
            (f"User: I completed the {decoy_event} today.", decoy.timestamp()),
        ],
        forbidden_in_proof=[decoy_event],
    )


_ENUM_HOBBY_PAIRS = [
    ("pottery", "astronomy"), ("archery", "calligraphy"), ("birdwatching", "origami"),
    ("beekeeping", "woodturning"), ("kayaking", "printmaking"),
]


def _claim_enumeration_case(rng: random.Random, idx: int) -> SyntheticCase:
    a, b = _ENUM_HOBBY_PAIRS[idx % len(_ENUM_HOBBY_PAIRS)]
    name = _NAMES[idx % len(_NAMES)]
    decoy = "the linen receipt is in the drawer"
    t = 1_700_500_000 + idx * 100
    return SyntheticCase(
        case_id=f"claim-enum-{idx}",
        op="open_inference",
        question=f"What hobbies does {name} enjoy?",
        expected=f"{b} and {a}",
        rows=[
            (f"{name}: I really enjoy {a}.", t),
            (f"{name}: I also enjoy {b}.", t + 100),
            (f"{name}: The {decoy}.", t + 200),
        ],
        forbidden_in_proof=["linen receipt"],
    )


_GENERATORS: list[Callable[[random.Random, int], SyntheticCase]] = [
    _latest_case,
    _claim_enumeration_case,
    _count_case,
    _itemized_count_case,
    _acquired_item_count_case,
    _relative_case,
    _table_case,
    _preference_case,
    _open_inference_case,
    _suggestion_case,
    _resource_suggestion_case,
    _organization_suggestion_case,
    _support_suggestion_case,
    _inspiration_suggestion_case,
    _beverage_suggestion_case,
    _compatibility_case,
    _activity_case,
    _hobby_case,
    _affiliation_followup_case,
    _speaker_case,
    _temporal_delta_case,
    _sum_case,
    _travel_duration_case,
    _consecutive_event_delta_case,
]


def generate_cases(seed: int, cases: int) -> list[SyntheticCase]:
    rng = random.Random(seed)
    generators = list(_GENERATORS)
    out: list[SyntheticCase] = []
    for idx in range(cases):
        gen = generators[idx % len(generators)]
        out.append(gen(rng, idx))
    rng.shuffle(out)
    return out


def _add_case(store: RecordStore, scope: Scope, case: SyntheticCase) -> None:
    records: list[MemoryRecord] = []
    for text, valid_at in case.rows:
        rec = _record(text, scope=scope, valid_at=valid_at)
        store.upsert_record(rec)
        records.append(rec)
    if case.claim_row is not None:
        rec = records[case.claim_row]
        store.add_claim(ClaimRecord(
            claim_type="state",
            scope=scope,
            subject="synthetic",
            predicate=case.question,
            object=case.expected,
            value=case.expected,
            valid_at=rec.valid_at,
            source_memory_id=rec.memory_id,
            proof_atom=rec.text,
            confidence=1.0,
        ))


def _answer_matches(actual: str, expected: str) -> bool:
    norm = lambda s: " ".join((s or "").lower().split())
    if norm(actual) == norm(expected):
        return True
    split_items = lambda s: {i.strip() for i in re.split(r",\s*(?:and\s+)?|\s+and\s+", norm(s)) if i.strip()}
    exp_items = split_items(expected)
    return len(exp_items) > 1 and exp_items == split_items(actual)


def _proof_contains_term(proof: str, term: str) -> bool:
    normalized_proof = " ".join((proof or "").lower().split())
    normalized_term = " ".join((term or "").lower().split())
    if not normalized_term:
        return False
    pattern = re.escape(normalized_term).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", normalized_proof) is not None


def _proof_excludes_terms(proof: str, terms: list[str]) -> bool:
    return not any(_proof_contains_term(proof, term) for term in terms)


def run_eval(*, seed: Optional[int] = None, cases: int = 24) -> dict:
    if cases <= 0:
        raise ValueError("cases must be positive")
    seed, seed_mode = resolve_seed(seed)
    generated = generate_cases(seed, cases)
    failures = []
    op_counts: dict[str, int] = {}
    backend_counts: dict[str, int] = {}
    proof_tokens = 0
    with tempfile.TemporaryDirectory(prefix="smqe-synth-") as tmp:
        store = RecordStore(Path(tmp) / "mem.sqlite")
        retriever = _Retriever(store)
        for case in generated:
            op_counts[case.op] = op_counts.get(case.op, 0) + 1
            scope = Scope(namespace=f"smqe-synth-{case.case_id}")
            _add_case(store, scope, case)
            ans = structured_answer(retriever, case.question, at=1_800_000_000, verify=True, scope=scope)
            proof = " ".join(c.snippet for c in (ans.citations if ans else []))
            if case.expect_abstain:
                # Fail-closed contract: a DERIVED count/sum must abstain, never ship a verified
                # aggregate. Passing means the verify-or-abstain surface returned None; a verified
                # answer here would be exactly the leak the citation floor closes.
                ok = ans is None
            else:
                ok = (
                    ans is not None
                    and ans.verified
                    and _answer_matches(ans.answer, case.expected)
                    and _proof_excludes_terms(proof, case.forbidden_in_proof)
                )
            note = ans.note if ans else ""
            if note.startswith("smqe:"):
                backend = (note.split(":") + ["", "", ""])[2]
                backend_counts[backend] = backend_counts.get(backend, 0) + 1
            if ans is not None:
                proof_tokens += sum(max(0, len(c.snippet or "") // 4) for c in ans.citations)
            if not ok:
                failures.append({
                    "case_id": case.case_id,
                    "op": case.op,
                    "question": case.question,
                    "expected": case.expected,
                    "actual": ans.answer if ans else "",
                    "note": note,
                    "verified": bool(ans and ans.verified),
                    "proof": proof[:500],
                })
    return {
        "pass": not failures,
        "seed": seed,
        "seed_mode": seed_mode,
        "cases": cases,
        "correct": cases - len(failures),
        "failures": failures,
        "operator_counts": dict(sorted(op_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "avg_proof_tokens": round(proof_tokens / cases, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="repro seed; omitted means random")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    report = run_eval(seed=args.seed, cases=args.cases)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
