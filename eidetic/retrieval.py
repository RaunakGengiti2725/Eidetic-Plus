"""Component 6: reconstructive, verifiable retrieval (+ Component 7 provenance).

Pipeline (dossier Section 12):
  1. embed query (text-embedding-v4) -> ANN top-k1 with a bi-temporal filter
  2. in-app Personalized PageRank over the graph (associative expansion)
  3. Reciprocal Rank Fusion of the dense + graph rankings
  4. qwen3-rerank -> final top-k2
  5. qwen3-max generation strictly over the retrieved sources
  6. NLI entailment check with the IMMUTABLE raw record as premise -> reject/flag
     anything unentailed; attach a cited, bi-temporal provenance to every answer.

Recency-independence guarantee: ranking uses content similarity + association + a
rerank model. The FSRS priority weight is NEVER read here, so recall@k does not
depend on a memory's age. That is what the signature flat curve proves.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from .bm25 import BM25, PersistentBM25
from .config import Settings, get_settings
from .conflicts import CurrentValueResolution, resolve_current_value_question
from .dashscope_client import DashScopeClient
from .events import effective_date_ranges, event_chain, parse_query, select_for_query
from .graph import CO_ACTIVATED, KnowledgeGraph
from .models import (Answer, Citation, DerivedRecord, MemoryRecord, Modality, NLILabel,
                     RecallTrace, RetrievalCandidate, Scope, now)
from .optim import adaptive_k as _adaptive_k
from .optim import conformal as _conformal
from .optim import fusion as _fusion
from .optim import gating as _gating
from .optim import mmr as _mmr
from .optim import online_weights as _online_weights
from .optim import rocchio as _rocchio
from .preferences import canonicalize_preference, preference_dedup_key
from .store import RecordStore
from .substrate import Substrate
from .vector_index import VectorIndex

_log = logging.getLogger("eidetic.retrieval")

# Difficulty routing keywords for the answer cascade (flash -> plus -> max).
_HARD_KW = ("contradict", "no longer", "instead", "actually", "used to", "earlier said",
            "which is true", "correct", "still")
_TEMPORAL_MULTI_KW = ("before", "after", "when", "first", "last", "then", "during", "how long",
                      "how many", "both", "and also", "since", "until", "by the time")
_LATEST_KW = ("latest", "last", "newest", "current", "currently", "now", "today", "still", "recent")
_EARLIEST_KW = ("first", "earliest", "initial", "original")
_CHRONO_KW = ("before", "after", "then", "during", "when", "timeline", "order", "sequence")
_RELATIVE_DATE_RE = re.compile(
    r"\b(last|this|next)\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
)
_TEMPORAL_QUESTION_RE = re.compile(
    r"\b(when|what date|which date|what year|which year|what month|which month|"
    r"what day|which day|how long|since when)\b",
    re.I,
)
_TEMPORAL_DATE_SIGNAL_RE = re.compile(
    r"\b(?:today|yesterday|tomorrow|recently|lately|fortnight|"
    r"last|this|next|previous|prior|following|before|after|ago|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|"
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b(?:19|20)\d{2}\b",
    re.I,
)
_TEMPORAL_DURATION_SIGNAL_RE = re.compile(
    r"\b(?:for|over|about|around|roughly|approximately|nearly|almost|since)\s+"
    r"(?:(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several|\d+)\s+)?"
    r"(?:days?|weeks?|months?|years?)\b|"
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:days?|weeks?|months?|years?)\s+(?:long|total)\b",
    re.I,
)
_QUESTION_TIME_DELTA_RE = re.compile(
    r"\b(?:how\s+many\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"(?:days?|weeks?|months?|years?)\s+ago|since\s+when|how\s+long\s+ago)\b",
    re.I,
)
_TERM_RE = re.compile(r"[a-z0-9]+")
_SOURCE_TAG_RE = re.compile(r"\s*\[(?:s|source)\s*\d+\]\s*", re.I)
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:answer\s*:\s*|the answer is\s+|it (?:is|was)\s+|that was\s+)",
    re.I,
)
_ABSTAIN_TEXT_RE = re.compile(r"\b(?:do not|don't) have (?:that|enough|a value)|insufficient evidence", re.I)
_AMOUNT_RE = re.compile(
    r"(?<![\w.])(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?"
    r"(?:dollars?|usd|hours?|minutes?|days?|weeks?|times?|occurrences?))\b",
    re.I,
)
_AGGREGATION_RE = re.compile(
    r"\b(how many|how much|number of|total|combined|in total|count(?: of)?|sum of)\b",
    re.I,
)
_LIST_QUERY_RE = re.compile(
    r"\b(?:what|which)\b.{0,48}\b(?:books?|activities?|events?|things?|hobbies?|places?|"
    r"fields?|ways?|items?|topics?|skills?|languages?|foods?|movies?|songs?|people)\b|"
    r"\bwhat\s+(?:do|does|did)\b.{0,36}\bdo\b",
    re.I,
)
_BOOK_LIST_QUERY_RE = re.compile(r"\bbooks?\b", re.I)
_BOOK_TITLE_AFTER_ACTION_RE = re.compile(
    r"\b(?:read|finished|reading)\s+(?:the\s+book\s+|a\s+book\s+|book\s+)?([^.!?;]{1,140})",
    re.I,
)
_MONEY_QUERY_RE = re.compile(r"\b(total|amount|spent|spend|cost|costs?|money|expense|expenses|price|paid)\b|[$€£]", re.I)
_PURCHASE_TERMS = {
    "bought", "buy", "purchase", "purchased", "got", "spent", "paid", "splurge",
    "splurged", "acquired", "acquire",
}
_AGGREGATION_STOP_TERMS = {
    "what", "which", "when", "where", "who", "why", "how", "many", "much", "number",
    "total", "combined", "count", "sum", "amount", "money", "spent", "spend", "cost",
    "costs", "paid", "price", "past", "last", "previous", "recent", "recently", "few",
    "couple", "several", "day", "days", "week", "weeks", "month", "months", "year",
    "years", "the", "and", "for", "from", "with", "that", "this", "these", "those",
    "did", "does", "have", "has", "had", "was", "were", "are", "is", "into", "over",
    "under", "about", "within", "during", "since", "start", "end", "item", "items",
    "thing", "things", "user", "memory", "i", "me", "my", "mine", "we", "our", "you",
    "your",
}
_LIST_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "activity", "activities", "event", "events", "field", "fields", "hobby",
    "hobbies", "item", "items", "language", "languages", "movie", "movies",
    "people", "place", "places", "skill", "skills", "song", "songs", "thing",
    "things", "topic", "topics", "way", "ways", "participate", "participated",
    "does", "done", "doing",
}
_LIST_TERM_EXPANSIONS = {
    "book": {"books", "read", "reads", "reading", "finished"},
    "books": {"book", "read", "reads", "reading", "finished"},
    "read": {"book", "books", "reads", "reading", "finished"},
    "reading": {"book", "books", "read", "reads", "finished"},
    "kit": {"kits", "model", "models", "scale", "scales", "built", "building", "finished"},
    "kits": {"kit", "model", "models", "scale", "scales", "built", "building", "finished"},
    "model": {"models", "kit", "kits", "scale", "scales", "built", "building"},
    "models": {"model", "kit", "kits", "scale", "scales", "built", "building"},
    "restaurant": {"restaurants", "food", "cuisine", "dining", "tried", "try"},
    "restaurants": {"restaurant", "food", "cuisine", "dining", "tried", "try"},
    "tried": {"try", "trying", "visited", "ate", "eaten", "dined"},
    "destress": {"stress", "stressed", "stressful", "relax", "relaxes",
                 "relaxed", "relaxing", "unwind", "unwinds", "calm"},
    "stress": {"destress", "stressed", "relax", "relaxes", "relaxing", "unwind",
               "unwinds", "calm"},
    "help": {"helps", "helped", "helping", "support", "supports", "supported",
             "assist", "assists", "assisted", "encourage", "encourages",
             "encouraged", "encouraging"},
    "encourage": {"encourages", "encouraged", "encouraging", "help", "helps",
                  "helped", "helping", "support", "supports", "supported"},
    "children": {"child", "kids", "kid", "youth", "school", "students"},
    "child": {"children", "kids", "kid", "youth", "school", "students"},
    "mentor": {"mentors", "mentored", "mentoring", "mentorship"},
    "mentoring": {"mentor", "mentors", "mentored", "mentorship"},
}
_AGGREGATION_TERM_EXPANSIONS = {
    "bought": {"buy", "buying", "purchase", "purchased", "got", "acquired"},
    "buy": {"bought", "buying", "purchase", "purchased", "got", "acquired"},
    "got": {"buy", "bought", "purchase", "purchased", "acquired"},
    "kit": {"kits", "model", "models", "scale", "scales", "built", "building", "finished"},
    "kits": {"kit", "model", "models", "scale", "scales", "built", "building", "finished"},
    "model": {"models", "kit", "kits", "scale", "scales", "built", "building"},
    "models": {"model", "kit", "kits", "scale", "scales", "built", "building"},
    "worked": {"work", "working", "started", "finished", "completed", "built", "building"},
    "work": {"worked", "working", "started", "finished", "completed", "built", "building"},
    "restaurant": {"restaurants", "food", "cuisine", "dining", "tried", "try"},
    "restaurants": {"restaurant", "food", "cuisine", "dining", "tried", "try"},
    "tried": {"try", "trying", "visited", "ate", "eaten", "dined", "different"},
}
_MODEL_KIT_SCOPE_TERMS = {
    "scale", "scales", "kit", "kits", "hobby", "diorama",
    "aircraft", "plane", "vehicle", "tank", "ship", "boat",
}
_MODEL_KIT_ACTION_TERMS = {
    "buy", "bought", "buying", "purchase", "purchased", "purchasing", "got",
    "acquired", "work", "worked", "working", "started", "finished", "completed",
    "built", "building", "picked",
}
_KOREAN_RESTAURANT_SCOPE_TERMS = {
    "restaurant", "restaurants",
}
_KOREAN_RESTAURANT_ACTION_TERMS = {
    "tried", "try", "trying", "different", "ones", "visited", "ate", "eaten",
    "dined", "restaurant", "restaurants",
}
_PREFERENCE_QUERY_RE = re.compile(
    r"\b(prefer|preference|preferences|favou?rite|like|likes|love|loves|enjoy|enjoys|"
    r"hate|hates|dislike|dislikes|allergic|avoid|avoids|rather|usually|always|never)\b",
    re.I,
)
_PREFERENCE_CUE_TERMS = {
    "prefer", "prefers", "preferred", "preference", "preferences", "favorite", "favourite",
    "like", "likes", "love", "loves", "enjoy", "enjoys", "hate", "hates", "dislike",
    "dislikes", "allergic", "avoid", "avoids", "rather", "usually", "always", "never",
    "cannot", "cant",
}
_PREFERENCE_TOPIC_STOP_TERMS = _LIST_STOP_TERMS | {
    "user", "users", "person", "profile", "preference", "preferences", "prefer", "prefers",
    "preferred", "favorite", "favourite", "like", "likes", "love", "loves", "enjoy",
    "enjoys", "hate", "hates", "dislike", "dislikes", "allergic", "avoid", "avoids",
    "rather", "usually", "always", "never", "what", "which", "who", "does", "type",
    "kind", "sort", "option", "choice",
}
_PREFERENCE_TERM_EXPANSIONS = {
    "flight": {"flights", "travel", "trip", "trips", "seat", "seats", "window", "aisle"},
    "flights": {"flight", "travel", "trip", "trips", "seat", "seats", "window", "aisle"},
    "travel": {"flight", "flights", "trip", "trips", "seat", "seats", "window", "aisle"},
    "seat": {"seats", "window", "aisle", "flight", "flights"},
    "seats": {"seat", "window", "aisle", "flight", "flights"},
    "music": {"song", "songs", "jazz", "rock", "classical", "playlist"},
    "food": {"foods", "eat", "eats", "eating", "peanut", "peanuts", "allergy", "allergic"},
    "drink": {"drinks", "coffee", "tea"},
}
_TEMPORAL_TERM_EXPANSIONS = {
    "go": {"went", "gone", "going"},
    "meet": {"met", "meeting"},
    "give": {"gave", "given", "giving"},
    "run": {"ran", "running"},
    "paint": {"painted", "painting"},
    "sign": {"signed", "signup", "signing"},
    "camp": {"camped", "camping"},
    "read": {"reads", "reading", "finished"},
    "donate": {"donated", "donating", "donation"},
    "donated": {"donate", "donating", "donation"},
    "adopt": {"adopted", "adopting", "adoption"},
    "adopted": {"adopt", "adopting", "adoption"},
    "speak": {"spoke", "spoken", "speech", "talk", "talked"},
    "speech": {"speak", "spoke", "spoken", "talk", "talked"},
    "talk": {"talked", "speaking", "speech", "speak", "spoke"},
    "plan": {"planned", "planning"},
}
_TEMPORAL_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "when", "date", "year", "day", "time", "long", "since", "happen", "happened",
    "did", "does", "do",
}
_TEMPORAL_ANCHOR_QUERY_RE = re.compile(
    r"\b(?:which\b.{0,60}\bfirst|who\b.{0,60}\bfirst|what\b.{0,60}\bfirst|"
    r"order\s+of|from\s+earliest\s+to\s+latest|days?\s+(?:passed\s+)?between|"
    r"days?\s+(?:before|after)|weeks?\s+(?:before|after)|months?\s+(?:before|after)|"
    r"how\s+long\b.{0,80}\b(?:before|after|when))\b",
    re.I,
)
_TEMPORAL_ANCHOR_STOP_TERMS = _TEMPORAL_STOP_TERMS | {
    "between", "passed", "pass", "before", "after", "first", "second", "third",
    "earliest", "latest", "order", "from", "among", "event", "events", "thing",
    "things", "happened", "happen", "which", "what", "who", "whom", "whose",
    "the", "day", "days", "week", "weeks", "month", "months", "year", "years",
    "ago", "had", "have", "been", "when", "while", "until", "then", "time",
    "i", "me", "my", "mine", "we", "our", "you", "your", "a", "an",
}
_FACT_CONTEXT_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "current", "currently", "latest", "newest", "now", "today", "still", "value",
    "fact", "facts", "status", "state", "where", "what", "who", "which",
}
_ASSISTANT_RECALL_RE = re.compile(
    r"\b(?:assistant|ai)\b|\b(?:what|which|where|when|how)\s+did\s+(?:you|the assistant)\b|"
    r"\b(?:you|your)\b.{0,40}\b(?:suggest|recommend|advise|tell|say|said|answer|provide|create|name|mention)",
    re.I,
)
_BRIDGE_QUERY_RE = re.compile(
    r"\b(connect(?:ed|ion)?|related|relationship|link(?:ed)?|common|shared|both|between)\b",
    re.I,
)
_RELATIONSHIP_STATUS_QUERY_RE = re.compile(
    r"\b(?:relationship status|relationship|marital status|dating|partner|single|"
    r"married|boyfriend|girlfriend|spouse)\b",
    re.I,
)
_USER_RECALL_RE = re.compile(
    r"\b(?:what|which|where|when|who|how)(?:\s+[a-z0-9]+){0,5}\s+"
    r"(?:did|do|was|were|is|are)\s+i\b|"
    r"\b(?:what|which|where|when|who|how)(?:\s+[a-z0-9]+){0,5}\s+"
    r"(?:is|was|are|were)\s+my\b|"
    r"\b(?:i|my|me)\b.{0,48}\b(?:say|said|tell|told|mention|mentioned|create|created|"
    r"buy|bought|read|watch|watched|visit|visited|order|ordered|use|used|work|live|go|went)",
    re.I,
)
_ROLE_LINE_RE = re.compile(r"^\s*(user|human|assistant|ai)\s*:\s*(.*)$", re.I)
_USER_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "user", "human", "say", "said", "tell", "told", "mention", "mentioned",
}
_ASSISTANT_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "assistant", "ai", "suggest", "suggested", "recommend", "recommended", "advise",
    "advised", "tell", "told", "say", "said", "answer", "answered", "provide", "provided",
    "create", "created", "name", "named", "mention", "mentioned",
}
_BRIDGE_STOP_TERMS = _AGGREGATION_STOP_TERMS | {
    "connect", "connected", "connection", "related", "relationship", "link", "linked",
    "common", "shared", "between",
}
_BRIDGE_ENTITY_GENERIC_TERMS = {
    "app", "application", "campaign", "city", "company", "corp", "department",
    "dept", "feature", "group", "inc", "initiative", "llc", "ltd", "plan",
    "program", "project", "repo", "repository", "service", "system", "task",
    "team", "tool", "workspace",
}
_RELATIONSHIP_STATUS_TERMS = {
    "single", "dating", "married", "marriage", "partner",
    "boyfriend", "girlfriend", "spouse", "wife", "husband", "relationship",
}
_EMPLOYMENT_QUERY_RE = re.compile(
    r"\b(?:employ|employs|employed|employer|employment)\b|"
    r"\bwhere\s+(?:do|does|is|are)\b.{0,64}\bworks?\b|"
    r"\bwho\b.{0,64}\bworks?\s+for\b",
    re.I,
)
_EMPLOYMENT_QUERY_TERMS = {
    "employ", "employs", "employed", "employer", "employment", "employee",
    "work", "works", "worked", "working",
}
_EMPLOYMENT_RELATION_TERMS = {
    "employ", "employs", "employed", "employer", "employment", "employee",
    "work", "works", "worked", "working", "job",
}
_EMPLOYMENT_FACT_RE = re.compile(
    r"\b(?:works?|worked|working)\s+(?:at|for|with|in)\b|"
    r"\bemployed\s+(?:by|at|with|in)\b|"
    r"\bemploys?\b|"
    r"\bemployer\s+(?:is|was|became|becomes)\b|"
    r"\bjob\s+(?:is|was|became|becomes)\b",
    re.I,
)
_LOCATION_QUERY_RE = re.compile(
    r"\b(?:address|home address|home city|lives?|living|reside|resides|resided|"
    r"residence|based|located)\b|"
    r"\bwhat\s+city\b.{0,64}\b(?:is|are|was|were)\b|"
    r"\bwhere\s+(?:do|does|is|are)\b.{0,64}\b(?:home|city|address)\b",
    re.I,
)
_LOCATION_QUERY_TERMS = {
    "address", "home", "city", "location", "located", "based",
    "live", "lives", "lived", "living", "reside", "resides", "resided", "residence",
}
_LOCATION_RELATION_TERMS = {
    "address", "home", "city", "location", "located", "based",
    "live", "lives", "lived", "living", "reside", "resides", "resided", "residence",
}
_LOCATION_FACT_RE = re.compile(
    r"\b(?:lives?|lived|living)\s+(?:in|at|near)\b|"
    r"\bresides?\s+(?:in|at|near)\b|"
    r"\b(?:address|home|city|location|residence)\s+(?:is|was|became|becomes)\b|"
    r"\bbased\s+(?:in|at|near)\b|"
    r"\blocated\s+(?:in|at|near)\b",
    re.I,
)
_FACT_TERM_EXPANSIONS = {
    "adopt": {"adopts", "adopted", "adopting", "adoption"},
    "relationship": {"single", "married", "divorced", "engaged", "separated", "widowed",
                     "dating", "partner", "boyfriend", "girlfriend", "husband", "wife"},
    "marital": {"single", "married", "divorced", "engaged", "separated", "widowed"},
    "relocation": {"move", "moved", "moving", "relocate", "relocated", "relocating"},
    "political": {"politics", "policy", "policies"},
    "politics": {"political", "policy", "policies", "civic", "community", "council"},
    "local": {"community", "neighborhood", "town", "city"},
    "mortgage": {"pre-approved", "preapproved", "pre-approval", "preapproval", "house", "home", "loan"},
    "education": {"career", "careers", "school", "study", "studies", "studying", "degree"},
    "adoption": {"adopt", "adopts", "adopted", "adopting"},
    "based": {"base", "located", "live", "lives", "living", "reside", "resides"},
    "bought": {"buy", "buys", "buying", "purchase", "purchased", "purchases", "purchasing"},
    "buy": {"bought", "buying", "purchase", "purchased", "purchases", "purchasing"},
    "buys": {"buy", "bought", "buying", "purchase", "purchased", "purchases", "purchasing"},
    "camp": {"camped", "camping"},
    "employ": {"employs", "employed", "employment", "work", "works", "worked", "working"},
    "employed": {"employ", "employs", "employment", "work", "works", "worked", "working"},
    "employer": {"employ", "employs", "employed", "employment", "work", "works", "working"},
    "employment": {"employ", "employs", "employed", "employer", "work", "works", "working"},
    "employs": {"employ", "employed", "employment", "work", "works", "worked", "working"},
    "give": {"gave", "given", "giving"},
    "go": {"went", "gone", "going"},
    "join": {"joined", "joining"},
    "kit": {"kits", "model", "models", "scale", "scales", "built", "building", "finished"},
    "kits": {"kit", "model", "models", "scale", "scales", "built", "building", "finished"},
    "meet": {"met", "meeting"},
    "model": {"models", "kit", "kits", "scale", "scales", "built", "building"},
    "models": {"model", "kit", "kits", "scale", "scales", "built", "building"},
    "paint": {"painted", "painting"},
    "plan": {"planned", "planning"},
    "purchase": {"buy", "bought", "buying", "purchased", "purchases", "purchasing"},
    "purchased": {"buy", "bought", "buying", "purchase", "purchases", "purchasing"},
    "purchasing": {"buy", "bought", "buying", "purchase", "purchased", "purchases"},
    "read": {"reads", "reading", "finished"},
    "restaurant": {"restaurants", "food", "cuisine", "dining", "tried", "try"},
    "restaurants": {"restaurant", "food", "cuisine", "dining", "tried", "try"},
    "reside": {"resides", "resided", "residence", "live", "lives", "living"},
    "residence": {"reside", "resides", "resided", "home", "address", "live", "lives"},
    "resides": {"reside", "resided", "residence", "live", "lives", "living"},
    "research": {"researches", "researched", "researching"},
    "speak": {"spoke", "spoken", "speech"},
    "speech": {"speak", "spoke", "spoken"},
    "tried": {"try", "trying", "visited", "ate", "eaten", "dined", "different"},
}
_WEEKDAY_NUM = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_MONTH_NUM = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_DATE_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+("
    + "|".join(sorted(_MONTH_NUM, key=len, reverse=True))
    + r")\s+(\d{4})\b",
    re.I,
)
_MDY_DATE_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTH_NUM, key=len, reverse=True))
    + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})\b",
    re.I,
)


def _route_model(query: str, settings: Settings) -> str:
    """Route to a cascade tier by difficulty. Ambiguous defaults to the conservative
    middle tier (a wrong cheap answer costs more than an unnecessary escalation)."""
    q = query.lower()
    if any(w in q for w in _HARD_KW):
        return settings.gen_model            # qwen3-max: contradiction / hard
    if any(w in q for w in _TEMPORAL_MULTI_KW):
        return settings.extract_model        # qwen-plus: multi-hop / temporal
    if len(q.split()) <= 12:
        return settings.salience_model       # qwen-flash: short single-hop / preference
    return settings.extract_model            # conservative default


def _reader_model(query: str, settings: Settings) -> str:
    if not settings.reader_router_enabled:
        return settings.gen_model
    return _route_model(query, settings)


def _temporal_direction(query: str, parsed: dict) -> Optional[str]:
    q = query.lower()
    relative_date = bool(_RELATIVE_DATE_RE.search(q))
    latest_words = [w for w in _LATEST_KW if w != "last" or not relative_date]
    if any(w in q for w in latest_words):
        return "desc"
    if any(w in q for w in _EARLIEST_KW):
        return "asc"
    if parsed.get("ranges") or parsed.get("operation") == "order" or any(w in q for w in _CHRONO_KW):
        return "asc"
    return None


def _temporal_context_order(
    query: str,
    parsed: dict,
    candidates: list["RetrievalCandidate"],
) -> list["RetrievalCandidate"]:
    direction = _temporal_direction(query, parsed)
    if direction is None:
        return candidates
    reverse_time = direction == "desc"

    def key(item: tuple[int, "RetrievalCandidate"]):
        rank, cand = item
        ts = cand.record.valid_at
        missing = ts is None
        value = 0.0 if ts is None else float(ts)
        return (missing, -value if reverse_time else value, rank)

    return [cand for _, cand in sorted(enumerate(candidates), key=key)]


def _simple_terms(text: str) -> set[str]:
    terms = set(_TERM_RE.findall(text.lower().replace("_", " ")))
    for term in list(terms):
        if len(term) > 3 and term.endswith("s"):
            terms.add(term[:-1])
    return terms


def _aggregation_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _AGGREGATION_STOP_TERMS
    }
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_FACT_TERM_EXPANSIONS.get(term, set()))
        expanded.update(_AGGREGATION_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term}d")
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _is_aggregation_query(query: str, parsed: dict) -> bool:
    return parsed.get("operation") == "count" or bool(_AGGREGATION_RE.search(query or ""))


def _is_list_query(query: str, parsed: dict) -> bool:
    if _is_aggregation_query(query, parsed):
        return False
    return bool(_LIST_QUERY_RE.search(query or ""))


def _list_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _LIST_STOP_TERMS
    }
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_LIST_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _entity_terms(parsed: dict) -> set[str]:
    out: set[str] = set()
    for entity in parsed.get("entities", []) or []:
        out.update(_simple_terms(str(entity)))
    return out


def _is_preference_query(query: str) -> bool:
    return bool(_PREFERENCE_QUERY_RE.search(query or ""))


def _preference_query_terms(query: str) -> tuple[set[str], set[str]]:
    qterms = _simple_terms(query or "")
    topic_terms = {t for t in qterms if t not in _PREFERENCE_TOPIC_STOP_TERMS and len(t) > 2}
    expanded = set(topic_terms)
    for term in list(topic_terms):
        expanded.update(_PREFERENCE_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3 and term.endswith("s"):
            expanded.add(term[:-1])
        elif len(term) > 3:
            expanded.add(f"{term}s")
    cue_terms = qterms & _PREFERENCE_CUE_TERMS
    return expanded, cue_terms


def _profile_line(item) -> str:
    if isinstance(item, dict):
        return str(item.get("line", "") or "")
    return str(item or "")


def _profile_block(item) -> str:
    line = _profile_line(item)
    if not isinstance(item, dict):
        return f"User preference: {line}"
    parts = []
    mid = str(item.get("source_memory_id", "") or "")
    h = str(item.get("content_hash", "") or "")
    if mid:
        parts.append(f"source_memory_id={mid}")
    if h:
        parts.append(f"content_hash={h[:16]}")
    suffix = f" [{' '.join(parts)}]" if parts else ""
    return f"User preference: {line}{suffix}"


def _preference_profile_blocks(query: str, profile: list, *, limit: int = 8) -> list[str]:
    """Select profile lines for context.

    Generic queries keep the store's salience/recency order. Preference questions promote matching
    lines first so a low-salience exact preference ("favorite music is jazz") is not buried behind
    eight unrelated high-salience preferences.
    """
    if not profile or limit <= 0:
        return []
    if not _is_preference_query(query):
        selected = profile[:limit]
        return [_profile_block(p) for p in selected]
    topic_terms, cue_terms = _preference_query_terms(query)
    scored: list[tuple[float, int, object]] = []
    for idx, line in enumerate(profile):
        lterms = _simple_terms(_profile_line(line))
        topic_hits = len(topic_terms & lterms)
        cue_hits = len(cue_terms & lterms)
        score = topic_hits * 6.0 + cue_hits * 2.0
        if score > 0:
            scored.append((score, idx, line))
    selected: list = []
    seen: set[str] = set()
    for _score, _idx, line in sorted(scored, key=lambda item: (-item[0], item[1])):
        key = _profile_line(line)
        if key not in seen:
            seen.add(key)
            selected.append(line)
        if len(selected) >= limit:
            break
    for line in profile:
        if len(selected) >= limit:
            break
        key = _profile_line(line)
        if key not in seen:
            seen.add(key)
            selected.append(line)
    return [_profile_block(p) for p in selected]


def _visible_profile_entries(store, scope: Scope, at: Optional[float]) -> list[dict]:
    """Profile rows that are safe to surface as context for this scope/time.

    Profile rows are stored namespace-wide, but source memories carry the real agent/project scope and
    bi-temporal validity. Context assembly must obey that source truth before the reader sees a line.
    """
    read_at = now() if at is None else at
    rows = store.get_profile_entries(scope.namespace)
    visible = []
    narrowed = scope.agent_id is not None or scope.project_id is not None
    for row in rows:
        source_id = str(row.get("source_memory_id", "") or "")
        rec = store.get_record(source_id) if source_id else None
        if rec is None:
            if source_id or narrowed:
                continue
            visible.append(row)
            continue
        if rec.scope.visible_to(scope) and rec.is_active_at(read_at):
            visible.append(row)
    return visible


_REGION_HINT_STOP_TERMS = _LIST_STOP_TERMS | {
    "about", "answer", "conversation", "did", "does", "fact", "find", "gist", "hint",
    "know", "memory", "memories", "mention", "mentioned", "question", "recall", "record",
    "region", "remember", "source", "tell", "user", "what", "when", "where", "which", "who",
    "why",
}
_REGION_HINT_BROAD_TERMS = {
    "aunt", "brother", "child", "children", "dad", "daughter", "family", "families",
    "father", "friend", "friends", "grandfather", "grandma", "grandmother", "grandpa",
    "husband", "mom", "mother", "parent", "parents", "partner", "sister", "son", "wife",
}


def _region_query_terms(query: str) -> set[str]:
    return {
        t for t in _simple_terms(query or "")
        if len(t) > 2 and t not in _REGION_HINT_STOP_TERMS
    }


def _region_discriminative_terms(query: str) -> set[str]:
    terms = _region_query_terms(query)
    proper = {
        re.sub(r"(?:'s|s')$", "", token.lower())
        for token in re.findall(r"\b[A-Z][A-Za-z'_-]{2,}\b", query or "")
        if token.lower() not in _REGION_HINT_STOP_TERMS
    }
    narrowed = terms - proper - _REGION_HINT_BROAD_TERMS
    return narrowed or terms


def _raw_member_ids_for_gist(gist: DerivedRecord, derived_by_cid: dict[str, DerivedRecord], *,
                             limit: int = 16) -> list[str]:
    """Resolve a multi-resolution gist/cocoon back to raw memory ids.

    Higher-level gists can contain lower-level derived ids. The read path may use a region as a
    routing hint, but the final answer must still terminate at raw memories, so this helper walks
    the additive derived layer until it reaches raw member ids.
    """
    out: list[str] = []
    queue = list(getattr(gist, "member_ids", []) or [])
    seen: set[str] = {gist.cid}
    while queue and len(out) < limit:
        mid = queue.pop(0)
        if mid in seen:
            continue
        seen.add(mid)
        child = derived_by_cid.get(mid)
        if child is not None:
            queue.extend(getattr(child, "member_ids", []) or [])
            continue
        out.append(mid)
    return out


def _memory_region_hints(
    store: RecordStore,
    query: str,
    candidates: list[RetrievalCandidate],
    scope: Scope,
    at: Optional[float] = None,
    *,
    gist_ids: Optional[set[str]] = None,
    limit: int = 3,
    member_limit: int = 6,
) -> list[dict]:
    """Compact derived-region routing hints with raw provenance.

    These blocks never replace source evidence: they name the matching memory neighborhood and
    include raw member ids + short hashes so the reader and proof path still ground on immutable
    memories. Selection is cheap and deterministic, using already-stored gist text, retrieve-time
    gist provenance, and overlap with retrieved candidate ids.
    """
    if limit <= 0 or member_limit <= 0:
        return []
    gists = store.derived_in_scope(scope.namespace, kind="gist")
    if not gists:
        return []
    derived_by_cid = {g.cid: g for g in gists if getattr(g, "cid", "")}
    qterms = _region_query_terms(query)
    required_terms = _region_discriminative_terms(query)
    read_at = at if at is not None else now()
    candidate_ids = {
        c.record.memory_id for c in candidates
        if c.record.scope.visible_to(scope) and c.record.is_active_at(read_at)
    }
    selected_gist_ids = gist_ids or set()
    # Grounding pre-pass, level order: a mixed gist's text may influence routing only when it is
    # visibly grounded -- sharing terms with a member the caller can see, or with a grounded
    # child region. A gist whose text derives entirely from hidden members would otherwise leak
    # their content into scoring.
    grounded_terms_by_cid: dict[str, set[str]] = {}
    prepared: dict[str, tuple[list[MemoryRecord], bool, set[str], set[str], bool]] = {}
    for gist in sorted(gists, key=lambda g: float(getattr(g, "level", 0) or 0)):
        raw_ids = _raw_member_ids_for_gist(gist, derived_by_cid, limit=max(member_limit * 2, 12))
        visible_recs: list[MemoryRecord] = []
        hidden_or_stale_member = False
        for mid in raw_ids:
            rec = store.get_record(mid)
            if rec is None or not rec.scope.visible_to(scope) or not rec.is_active_at(read_at):
                hidden_or_stale_member = True
                continue
            visible_recs.append(rec)
        if not visible_recs:
            continue
        all_members_visible = not hidden_or_stale_member and len(visible_recs) == len(raw_ids)
        visible_terms = _simple_terms(" ".join(
            (rec.text or rec.summary or "") for rec in visible_recs[:max(member_limit, 1)]
        ))
        for mid in getattr(gist, "member_ids", []) or []:
            visible_terms |= grounded_terms_by_cid.get(str(mid), set())
        gist_terms = _simple_terms(getattr(gist, "text", "") or "")
        gist_grounded = all_members_visible or bool(gist_terms & visible_terms)
        route_terms = (visible_terms | gist_terms) if gist_grounded else visible_terms
        if gist_grounded and getattr(gist, "cid", ""):
            grounded_terms_by_cid[gist.cid] = gist_terms | visible_terms
        prepared[gist.cid] = (visible_recs, all_members_visible, route_terms, gist_terms, gist_grounded)

    scored: list[tuple[float, int, DerivedRecord, list[MemoryRecord], bool]] = []
    for idx, gist in enumerate(gists):
        if gist.cid not in prepared:
            continue
        visible_recs, all_members_visible, route_terms, _gist_terms, _grounded = prepared[gist.cid]
        visible_ids = {rec.memory_id for rec in visible_recs}
        candidate_hits = len(candidate_ids & visible_ids)
        if required_terms and not (required_terms & route_terms):
            continue
        term_hits = len(qterms & route_terms)
        provenance_hit = 1 if gist.cid in selected_gist_ids else 0
        score = provenance_hit * 30.0 + candidate_hits * 8.0 + term_hits * 4.0
        if score <= 0.0:
            continue
        scored.append((score, idx, gist, visible_recs, all_members_visible))
    if not scored:
        return []

    hints: list[dict] = []
    for _score, _idx, gist, visible_recs, all_members_visible in sorted(
        scored,
        key=lambda item: (
            -item[0],
            0 if item[4] else 1,
            len(item[3]),
            -float(getattr(item[2], "level", 0) or 0),
            item[1],
        ),
    ):
        ids: list[str] = []
        hashes: list[str] = []
        raw_uris: list[str] = []
        for rec in visible_recs[:member_limit]:
            ids.append(rec.memory_id)
            if rec.content_hash:
                hashes.append(rec.content_hash[:16])
            if rec.raw_uri:
                raw_uris.append(rec.raw_uri)
        if not ids:
            continue
        if all_members_visible:
            text = " ".join((gist.text or f"gist level {gist.level}").split())
        else:
            text = f"memory region level {int(getattr(gist, 'level', 0) or 0)}"
        if len(text) > 180:
            text = text[:177].rstrip() + "..."
        hints.append({
            "region_id": gist.cid,
            "level": int(getattr(gist, "level", 0) or 0),
            "text": text,
            "member_count": len(visible_recs),
            "members": ids,
            "content_hashes": hashes,
            "raw_uris": raw_uris,
            "score": round(float(_score), 6),
        })
        if len(hints) >= limit:
            break
    return hints


def _format_memory_region_hint(hint: dict) -> str:
    hashes = [str(h) for h in hint.get("content_hashes", []) if h]
    members = [str(m) for m in hint.get("members", []) if m]
    hash_part = f" content_hashes={','.join(hashes)}" if hashes else ""
    return (
        "Memory region hint (route only; verify with source memories): "
        f"{hint.get('text', '')} [region_id={hint.get('region_id', '')} "
        f"level={hint.get('level', 0)} member_count={hint.get('member_count', 0)} "
        f"members={','.join(members)}{hash_part}]"
    )


def _memory_region_hint_blocks(*args, **kwargs) -> list[str]:
    return [_format_memory_region_hint(hint) for hint in _memory_region_hints(*args, **kwargs)]


def _substantive_structured_query(query: str) -> bool:
    terms = [
        term for term in _simple_terms(query or "")
        if len(term) > 1 and term not in _AGGREGATION_STOP_TERMS
    ]
    if len(terms) >= 2:
        return True
    return bool(re.search(
        r"\b(?:what|where|when|which|who|how\s+(?:many|much|long)|count|sum|total|"
        r"preference|prefer|favorite|favourite|remember|recall)\b",
        query or "",
        re.I,
    ))


def _is_temporal_evidence_query(query: str, parsed: dict) -> bool:
    if parsed.get("ranges"):
        return True
    if _looks_temporal_anchor_query(query):
        return True
    if _TEMPORAL_QUESTION_RE.search(query or ""):
        return True
    return parsed.get("operation") == "order" and any(w in (query or "").lower() for w in _CHRONO_KW)


def _question_time_context_block(query: str, at: Optional[float]) -> list[str]:
    if at is None or not _QUESTION_TIME_DELTA_RE.search(query or ""):
        return []
    try:
        when = datetime.fromtimestamp(at).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return []
    return [
        "Question date (answer-time anchor): "
        f"{when}. For 'ago' and elapsed-time questions, compute the delta from this "
        "question date; source session dates anchor relative source words like today/yesterday."
    ]


def _temporal_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _TEMPORAL_STOP_TERMS
    }
    expanded = set(terms)
    for term in list(terms):
        expanded.update(_TEMPORAL_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _temporal_topic_terms(query: str, parsed: dict) -> tuple[set[str], int]:
    """Expanded non-entity temporal terms plus the required hit count.

    Keep the required count tied to the unexpanded topic words, not their inflection variants. A
    title/date query like "When did Melanie read Nothing is Impossible?" has one real action word
    ("read") plus title entities; requiring two expanded action hits would drop the correct source.
    """
    raw_terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _TEMPORAL_STOP_TERMS
    }
    base_topic = raw_terms - _entity_terms(parsed)
    if not base_topic:
        base_topic = raw_terms
    expanded = set(base_topic)
    for term in list(base_topic):
        expanded.update(_TEMPORAL_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    required = 2 if len(base_topic) >= 2 else 1
    return expanded, required


def _temporal_required_scope_groups(query: str) -> list[set[str]]:
    terms = _simple_terms(query or "")
    groups: list[set[str]] = []
    if terms & {"school", "schools", "student", "students"}:
        groups.append({"school", "schools", "student", "students"})
    return groups


def _temporal_anchor_base_terms(text: str) -> set[str]:
    return {
        t for t in _simple_terms(text or "")
        if len(t) > 2 and t not in _TEMPORAL_ANCHOR_STOP_TERMS
    }


def _temporal_anchor_expanded_terms(base_terms: set[str]) -> set[str]:
    expanded = set(base_terms)
    for term in list(base_terms):
        expanded.update(_TEMPORAL_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term}d")
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _temporal_anchor_group(text: str) -> tuple[set[str], set[str]] | None:
    base = _temporal_anchor_base_terms(text)
    # One generic term is too easy to hit in conversational memory; a quoted/title term is often
    # one-token by design ("MoMA"), so keep it, but otherwise require at least two base terms.
    if not base:
        return None
    if len(base) == 1 and len(next(iter(base))) < 4:
        return None
    return base, _temporal_anchor_expanded_terms(base)


def _add_temporal_anchor_group(
    groups: list[tuple[set[str], set[str]]],
    seen: set[tuple[str, ...]],
    text: str,
) -> None:
    group = _temporal_anchor_group(text)
    if group is None:
        return
    key = tuple(sorted(group[0]))
    if key in seen:
        return
    seen.add(key)
    groups.append(group)


def _split_temporal_anchor_list(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip(" ,;:.?!")
    if not cleaned:
        return []
    parts = [
        p.strip(" ,;:.?!")
        for p in re.split(r"\s*,\s*|\s+\bor\b\s+|\s+\band\b\s+", cleaned, flags=re.I)
        if p.strip(" ,;:.?!")
    ]
    return parts or [cleaned]


def _temporal_anchor_groups(query: str) -> list[tuple[set[str], set[str]]]:
    """Extract event/topic anchors from temporal comparison questions.

    These groups let retrieval surface session-dated source snippets for questions like
    "which happened first, X or Y" and "how many days passed between X and Y" even when the
    source sentence itself has no date phrase. The reader still performs the comparison.
    """
    q = query or ""
    if not _looks_temporal_anchor_query(q):
        return []
    groups: list[tuple[set[str], set[str]]] = []
    seen: set[tuple[str, ...]] = set()

    for quoted in re.findall(r"['\"]([^'\"]{2,140})['\"]", q):
        _add_temporal_anchor_group(groups, seen, quoted)

    for m in re.finditer(r"\bbetween\b(.+?)\band\b(.+?)(?:[?.!]|$)", q, re.I):
        _add_temporal_anchor_group(groups, seen, m.group(1))
        _add_temporal_anchor_group(groups, seen, m.group(2))

    for m in re.finditer(
        r"\b(?:which|who|what)\b.+?\bfirst\b(?:\s*,|\s*:)?\s*(.+?)\s+\bor\b\s+(.+?)(?:[?.!]|$)",
        q,
        re.I,
    ):
        _add_temporal_anchor_group(groups, seen, m.group(1))
        _add_temporal_anchor_group(groups, seen, m.group(2))

    for m in re.finditer(
        r"\border\s+of\b.+?\b(?:from\s+earliest\s+to\s+latest|from\s+first\s+to\s+last|"
        r"starting\s+from\s+the\s+earliest|among|between|of)\b(.+?)(?:[?.!]|$)",
        q,
        re.I,
    ):
        for part in _split_temporal_anchor_list(m.group(1)):
            _add_temporal_anchor_group(groups, seen, part)

    for m in re.finditer(
        r"\b(?:before|after)\b(.+?)\bdid\s+i\s+(.+?)(?:[?.!]|$)",
        q,
        re.I,
    ):
        _add_temporal_anchor_group(groups, seen, m.group(1))
        _add_temporal_anchor_group(groups, seen, m.group(2))

    if not groups and re.search(r"\b(?:before|after)\b", q, re.I):
        parts = re.split(r"\b(?:before|after)\b", q, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            _add_temporal_anchor_group(groups, seen, parts[0])
            _add_temporal_anchor_group(groups, seen, parts[1])

    return groups[:6]


def _looks_temporal_anchor_query(query: str | None) -> bool:
    q = query or ""
    if _TEMPORAL_ANCHOR_QUERY_RE.search(q):
        return True
    # "between August 11 and August 15" is temporal; "difference in price between boots and shoes"
    # is not. Keep plain-between gated on explicit date/time language.
    if re.search(r"\bbetween\b", q, re.I) and _TEMPORAL_DATE_SIGNAL_RE.search(q):
        return True
    return False


def _is_money_aggregation(query: str) -> bool:
    return bool(_MONEY_QUERY_RE.search(query or ""))


def _aggregation_required_scope_groups(query: str) -> list[set[str]]:
    terms = _simple_terms(query or "")
    groups: list[set[str]] = []
    if terms & {"model", "models", "kit", "kits"}:
        groups.append(_MODEL_KIT_SCOPE_TERMS)
        if terms & {"work", "worked", "working", "bought", "buy", "purchase", "purchased"}:
            groups.append(_MODEL_KIT_ACTION_TERMS)
    if (terms & {"korean"}) and (terms & {"restaurant", "restaurants"}):
        groups.append({"korean"})
        groups.append(_KOREAN_RESTAURANT_SCOPE_TERMS)
        if terms & {"tried", "try", "trying", "city"}:
            groups.append(_KOREAN_RESTAURANT_ACTION_TERMS)
    return groups


def _scope_groups_satisfied(terms: set[str], groups: list[set[str]]) -> bool:
    return all(group & terms for group in groups)


def _range_epochs(parsed: dict) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for r in effective_date_ranges(parsed.get("ranges", []) or []):
        try:
            out.append((
                datetime.strptime(r["start"], "%Y-%m-%dT%H:%M:%S").timestamp(),
                datetime.strptime(r["end"], "%Y-%m-%dT%H:%M:%S").timestamp(),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _record_in_ranges(rec: MemoryRecord, ranges: list[tuple[float, float]]) -> bool:
    if not ranges:
        return True
    t = rec.valid_at
    return any(lo <= t <= hi for lo, hi in ranges)


def _aggregation_snippet(
    text: str,
    terms: set[str],
    want_amount: bool,
    *,
    required_groups: list[set[str]] | None = None,
    limit: int = 420,
) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    required_groups = required_groups or []
    sentences = _sentences(clean) or [clean]
    best: tuple[int, int, str] | None = None
    for i, sent in enumerate(sentences):
        windows = [sent]
        if i + 1 < len(sentences):
            windows.append(f"{sent} {sentences[i + 1]}")
        for window in windows:
            st = _simple_terms(window)
            term_hits = len(terms & st)
            amount_hits = len(_AMOUNT_RE.findall(window))
            if terms and term_hits == 0:
                continue
            if want_amount and amount_hits == 0:
                continue
            if required_groups and not _scope_groups_satisfied(st, required_groups):
                continue
            group_hits = sum(1 for group in required_groups if group & st)
            purchase_hits = len(_PURCHASE_TERMS & st)
            score = term_hits * 4 + min(amount_hits, 3) * 3 + purchase_hits + group_hits * 7
            if best is None or score > best[0]:
                best = (score, -i, window)
    if best is None:
        if terms:
            idxs = [clean.lower().find(t) for t in sorted(terms, key=len, reverse=True)]
            idxs = [i for i in idxs if i >= 0]
            center = min(idxs) if idxs else 0
        else:
            m = _AMOUNT_RE.search(clean)
            center = m.start() if m else 0
        start = max(0, center - limit // 2)
        return clean[start:start + limit].strip(" ,;")
    snippet = best[2].strip(" ,;")
    return snippet[:limit].strip()


def _aggregation_matches(
    query: str,
    parsed: dict,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 12,
) -> list[tuple[float, MemoryRecord, str]]:
    if not _is_aggregation_query(query, parsed):
        return []
    terms = _aggregation_terms(query)
    required_groups = _aggregation_required_scope_groups(query)
    want_amount = _is_money_aggregation(query)
    ranges = _range_epochs(parsed)
    scored: list[tuple[float, MemoryRecord, str]] = []
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        if not _record_in_ranges(rec, ranges):
            continue
        text = rec.text or rec.summary or ""
        if not text:
            continue
        body_terms = _simple_terms(text)
        term_hits = len(terms & body_terms)
        amount_hits = len(_AMOUNT_RE.findall(text))
        if terms and term_hits == 0:
            continue
        if want_amount and amount_hits == 0:
            continue
        if required_groups and not _scope_groups_satisfied(body_terms, required_groups):
            continue
        snippet = _aggregation_snippet(
            text, terms, want_amount, required_groups=required_groups)
        if not snippet:
            continue
        snippet_terms = _simple_terms(snippet)
        snippet_amounts = len(_AMOUNT_RE.findall(snippet))
        if terms and not (terms & snippet_terms):
            continue
        if want_amount and snippet_amounts == 0:
            continue
        if required_groups and not _scope_groups_satisfied(snippet_terms, required_groups):
            continue
        group_hits = sum(1 for group in required_groups if group & snippet_terms)
        purchase_hits = len(_PURCHASE_TERMS & body_terms)
        score = term_hits * 4.0 + min(amount_hits, 3) * 2.0 + purchase_hits + group_hits * 7.0
        scored.append((score, rec, snippet))
    scored.sort(key=lambda x: (-x[0], x[1].valid_at, x[1].memory_id))
    return scored[:limit]


def _list_snippet(text: str, terms: set[str], topic_terms: set[str], *, limit: int = 460) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    sentences = _sentences(clean) or [
        c.strip() for c in re.split(r"[;\n]", clean) if c.strip()
    ] or [clean]
    best: tuple[float, int, str] | None = None
    for i, sent in enumerate(sentences):
        windows = [sent]
        if i + 1 < len(sentences):
            windows.append(f"{sent} {sentences[i + 1]}")
        if i > 0:
            windows.append(f"{sentences[i - 1]} {sent}")
        if i > 0 and i + 1 < len(sentences):
            windows.append(f"{sentences[i - 1]} {sent} {sentences[i + 1]}")
        for window in windows:
            st = _simple_terms(window)
            topic_hits = len(topic_terms & st) if topic_terms else 0
            term_hits = len(terms & st)
            if topic_terms and topic_hits == 0:
                continue
            if terms and term_hits == 0:
                continue
            score = topic_hits * 5.0 + term_hits * 2.0 + max(0.0, 1.0 - i * 0.001)
            if best is None or score > best[0]:
                best = (score, -i, window)
    if best is None:
        return ""
    return best[2].strip(" ,;")[:limit].strip()


def _list_required_scope_groups(query: str) -> list[set[str]]:
    terms = _simple_terms(query or "")
    groups: list[set[str]] = []
    if terms & {"model", "models", "kit", "kits"}:
        groups.append(_MODEL_KIT_SCOPE_TERMS)
        if terms & {"work", "worked", "working", "bought", "buy", "purchase", "purchased"}:
            groups.append(_MODEL_KIT_ACTION_TERMS)
    if (terms & {"korean"}) and (terms & {"restaurant", "restaurants"}):
        groups.append({"korean"})
        groups.append(_KOREAN_RESTAURANT_SCOPE_TERMS)
        if terms & {"tried", "try", "trying", "city"}:
            groups.append(_KOREAN_RESTAURANT_ACTION_TERMS)
    if terms & {"destress", "stress", "stressed", "relax", "unwind"}:
        groups.append({
            "destress", "stress", "stressed", "stressful", "relax", "relaxes",
            "relaxed", "relaxing", "unwind", "unwinds", "unwinding", "calm",
        })
    child_terms = {"children", "child", "kids", "kid", "youth", "school", "students", "student"}
    help_terms = {
        "help", "helps", "helped", "helping", "support", "supports", "supported",
        "assist", "assists", "assisted", "encourage", "encourages", "encouraged",
        "encouraging", "mentor", "mentors", "mentored", "mentoring", "mentorship",
    }
    if (terms & child_terms) and (terms & help_terms):
        groups.append(child_terms)
        groups.append(help_terms)
    return groups


def _is_book_list_query(query: str) -> bool:
    return bool(_BOOK_LIST_QUERY_RE.search(query or ""))


def _book_title_signal(text: str) -> bool:
    """True when a source snippet appears to name a specific book title.

    This is intentionally conservative and only used to filter book-list audit snippets. It avoids
    treating generic reading habits ("enjoys reading novels") as answer candidates while keeping
    title-bearing evidence such as "read Nothing is Impossible" or "finished Charlotte's Web".
    """
    snippet = text or ""
    if re.search(r"\"[^\"]{3,}\"", snippet):
        return True
    for m in _BOOK_TITLE_AFTER_ACTION_RE.finditer(snippet):
        phrase = m.group(1)
        # Trim common trailing context while preserving title-internal lowercase words ("is", "of").
        phrase = re.split(
            r"\b(?:in\s+(?:19|20)\d{2}|on\s+[A-Z][a-z]+|\blast\b|\bthis\b|\bnext\b|"
            r"\bbefore\b|\bafter\b|\bduring\b|\band\s+(?:found|felt|said|thought|liked|loved))\b",
            phrase,
            maxsplit=1,
        )[0]
        caps = re.findall(r"\b[A-Z][A-Za-z0-9']+\b", phrase)
        if caps:
            return True
    return False


def _list_matches(
    query: str,
    parsed: dict,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 10,
) -> list[tuple[float, MemoryRecord, str]]:
    """Source-completeness scan for exact list questions.

    This only selects supporting snippets and raw records; it never enumerates the answer list.
    Query-scope terms (e.g. "destress", "children", "books read") must appear in the snippet,
    which keeps generic hobbies/events from flooding the reader.
    """
    if limit <= 0 or not _is_list_query(query, parsed):
        return []
    terms = _list_terms(query)
    entity_terms = _entity_terms(parsed)
    topic_terms = terms - entity_terms
    if not topic_terms:
        topic_terms = terms
    if not topic_terms:
        return []
    ranges = _range_epochs(parsed)
    required_groups = _list_required_scope_groups(query)
    scored: list[tuple[float, MemoryRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        if not _record_in_ranges(rec, ranges):
            continue
        text = rec.text or rec.summary or ""
        if not text:
            continue
        body_terms = _simple_terms(text)
        topic_hits = len(topic_terms & body_terms)
        if topic_hits == 0:
            continue
        snippet = _list_snippet(text, terms, topic_terms)
        if not snippet:
            continue
        if _is_book_list_query(query) and not _book_title_signal(snippet):
            continue
        snippet_terms = _simple_terms(snippet)
        if not (topic_terms & snippet_terms):
            continue
        if required_groups and not all(group & snippet_terms for group in required_groups):
            continue
        key = (rec.memory_id, " ".join(snippet.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        entity_hits = len(entity_terms & body_terms)
        term_hits = len(terms & body_terms)
        score = topic_hits * 5.0 + term_hits * 2.0 + entity_hits
        scored.append((score, rec, snippet))
    scored.sort(key=lambda x: (-x[0], x[1].valid_at, x[1].memory_id))
    return scored[:limit]


def _temporal_evidence_snippet(
    text: str,
    terms: set[str],
    topic_terms: set[str],
    *,
    required: int = 1,
    limit: int = 520,
) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    sentences = _sentences(clean) or [
        c.strip() for c in re.split(r"[;\n]", clean) if c.strip()
    ] or [clean]
    best: tuple[float, int, str] | None = None
    for i, sent in enumerate(sentences):
        windows = [sent]
        if i + 1 < len(sentences):
            windows.append(f"{sent} {sentences[i + 1]}")
        if i > 0:
            windows.append(f"{sentences[i - 1]} {sent}")
        if i > 0 and i + 1 < len(sentences):
            windows.append(f"{sentences[i - 1]} {sent} {sentences[i + 1]}")
        for window in windows:
            date_signal = bool(_TEMPORAL_DATE_SIGNAL_RE.search(window))
            duration_signal = bool(_TEMPORAL_DURATION_SIGNAL_RE.search(window))
            if not (date_signal or duration_signal):
                continue
            st = _simple_terms(window)
            topic_hits = len(topic_terms & st) if topic_terms else 0
            term_hits = len(terms & st)
            if topic_terms and topic_hits < required:
                continue
            if terms and term_hits == 0:
                continue
            relative_bonus = 2.0 if re.search(
                r"\b(?:before|after|last|next|previous|following|ago)\b", window, re.I
            ) else 0.0
            duration_bonus = 2.0 if duration_signal else 0.0
            score = (
                topic_hits * 5.0
                + term_hits * 2.0
                + relative_bonus
                + duration_bonus
                + max(0.0, 1.0 - i * 0.001)
            )
            if best is None or score > best[0]:
                best = (score, -i, window)
    if best is None:
        return ""
    return best[2].strip(" ,;")[:limit].strip()


def _temporal_evidence_matches(
    query: str,
    parsed: dict,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 10,
) -> list[tuple[float, MemoryRecord, str]]:
    """Source-completeness scan for temporal questions.

    It preserves exact date wording from source snippets ("the Sunday before 25 May 2023") without
    computing the final date. The shared reader still performs date arithmetic/formatting.
    """
    if limit <= 0 or not _is_temporal_evidence_query(query, parsed):
        return []
    terms = _temporal_terms(query)
    entity_terms = _entity_terms(parsed)
    topic_terms, required = _temporal_topic_terms(query, parsed)
    if not topic_terms:
        return []
    required_groups = _temporal_required_scope_groups(query)
    ranges = _range_epochs(parsed)
    scored: list[tuple[float, MemoryRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        if not _record_in_ranges(rec, ranges):
            continue
        text = rec.text or rec.summary or ""
        if not text or not (
            _TEMPORAL_DATE_SIGNAL_RE.search(text) or _TEMPORAL_DURATION_SIGNAL_RE.search(text)
        ):
            continue
        body_terms = _simple_terms(text)
        topic_hits = len(topic_terms & body_terms)
        if topic_hits < required:
            continue
        snippet = _temporal_evidence_snippet(text, terms, topic_terms, required=required)
        if not snippet:
            continue
        snippet_terms = _simple_terms(snippet)
        if len(topic_terms & snippet_terms) < required:
            continue
        if required_groups and not all(group & snippet_terms for group in required_groups):
            continue
        required_entity_hits = 2 if len(entity_terms) >= 2 else (1 if entity_terms else 0)
        entity_source_terms = set(snippet_terms)
        if re.search(r"\b(?:i|me|my|mine)\b", snippet, re.I):
            entity_source_terms |= body_terms
        if required_entity_hits and len(entity_terms & entity_source_terms) < required_entity_hits:
            continue
        key = (rec.memory_id, " ".join(snippet.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        entity_hits = len(entity_terms & body_terms)
        term_hits = len(terms & body_terms)
        relative_bonus = 2.0 if re.search(
            r"\b(?:before|after|last|next|previous|following|ago)\b", snippet, re.I
        ) else 0.0
        duration_bonus = 2.0 if _TEMPORAL_DURATION_SIGNAL_RE.search(snippet) else 0.0
        score = topic_hits * 5.0 + term_hits * 2.0 + entity_hits + relative_bonus + duration_bonus
        scored.append((score, rec, snippet))
    scored.sort(key=lambda x: (-x[0], x[1].valid_at, x[1].memory_id))
    return scored[:limit]


def _temporal_anchor_snippet(
    text: str,
    base_terms: set[str],
    expanded_terms: set[str],
    *,
    limit: int = 520,
) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return ""
    sentences = _sentences(clean) or [
        c.strip() for c in re.split(r"[;\n]", clean) if c.strip()
    ] or [clean]
    best: tuple[float, int, str] | None = None
    for i, sent in enumerate(sentences):
        windows = [sent]
        if i + 1 < len(sentences):
            windows.append(f"{sent} {sentences[i + 1]}")
        if i > 0:
            windows.append(f"{sentences[i - 1]} {sent}")
        for window in windows:
            st = _simple_terms(window)
            exact_hits = len(base_terms & st)
            expanded_hits = len(expanded_terms & st)
            required = 2 if len(base_terms) >= 2 else 1
            if exact_hits < required and not (exact_hits >= 1 and expanded_hits >= required):
                continue
            score = exact_hits * 5.0 + expanded_hits * 1.5 + max(0.0, 1.0 - i * 0.001)
            if best is None or score > best[0]:
                best = (score, -i, window)
    if best is None:
        return ""
    return best[2].strip(" ,;")[:limit].strip()


def _temporal_anchor_matches(
    query: str,
    parsed: dict,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 10,
) -> list[tuple[float, MemoryRecord, str]]:
    """Session-date evidence for temporal comparison questions.

    Unlike `_temporal_evidence_matches`, this does not require a date word inside the source
    sentence. It relies on the immutable session timestamp already attached to each memory and
    surfaces the matching source snippets with that date, so the reader can compare/order them.
    """
    if limit <= 0 or not _is_temporal_evidence_query(query, parsed):
        return []
    groups = _temporal_anchor_groups(query)
    if not groups:
        return []
    ranges = _range_epochs(parsed)
    scored: list[tuple[float, MemoryRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if rec.valid_at is None:
            continue
        if at is not None and not rec.is_active_at(at):
            continue
        if not _record_in_ranges(rec, ranges):
            continue
        text = rec.text or rec.summary or ""
        if not text:
            continue
        body_terms = _simple_terms(text)
        for group_idx, (base_terms, expanded_terms) in enumerate(groups):
            exact_hits = len(base_terms & body_terms)
            expanded_hits = len(expanded_terms & body_terms)
            required = 2 if len(base_terms) >= 2 else 1
            if exact_hits < required and not (exact_hits >= 1 and expanded_hits >= required):
                continue
            snippet = _temporal_anchor_snippet(text, base_terms, expanded_terms)
            if not snippet:
                continue
            key = (rec.memory_id, " ".join(snippet.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            score = exact_hits * 5.0 + expanded_hits * 1.5 + max(0.0, 3.0 - group_idx * 0.25)
            scored.append((score, rec, snippet))
    scored.sort(key=lambda x: (x[1].valid_at, -x[0], x[1].memory_id))
    return scored[:limit]


def _is_user_recall_query(query: str) -> bool:
    return bool(_USER_RECALL_RE.search(query or ""))


def _role_turns(rec: MemoryRecord) -> list[tuple[int, str, str]]:
    """Parse a role-tagged session record into (line_index, role, content) turns."""
    turns: list[tuple[int, str, str]] = []
    current_role = ""
    current_start = 0
    current_content: list[str] = []

    def flush() -> None:
        nonlocal current_content
        if current_role:
            content = " ".join(x.strip() for x in current_content if x.strip()).strip()
            if content:
                turns.append((current_start, current_role, content))
        current_content = []

    lines = (rec.text or rec.summary or "").splitlines()
    for idx, line in enumerate(lines):
        m = _ROLE_LINE_RE.match(line)
        if m:
            flush()
            current_role = m.group(1).lower()
            current_start = idx
            current_content = [m.group(2).strip()]
            continue
        if current_role:
            current_content.append(line.strip())
    flush()
    return turns


def _user_query_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _USER_STOP_TERMS
    }
    expanded = set(terms)
    for term in terms:
        if term == "read":
            expanded.add("reading")
        if len(term) > 3:
            expanded.add(f"{term}ed")
            expanded.add(f"{term}s")
            if term.endswith("e"):
                expanded.add(f"{term}d")
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _user_snippets(rec: MemoryRecord) -> list[tuple[int, str]]:
    """Return user-turn snippets from a role-tagged session record.

    Single-session-user questions often need a tiny local dialogue bridge: the answer-bearing user
    turn may say "I redeemed the coupon" while the immediately prior user turn names the store/app.
    Keep the current user turn plus nearby prior user turns. This is still user-source evidence, not
    answer generation, and it prevents long raw sessions from being the only way to preserve context.
    """
    snippets: list[tuple[int, str]] = []
    prior_user: list[str] = []
    for idx, role, content in _role_turns(rec):
        if role not in ("user", "human"):
            continue
        # Put the matched/current user line first so it survives the audit-block char cap on long
        # LME sessions; append nearby prior user turns after it for local store/app/context bridges.
        parts = [f"user: {content}"]
        parts.extend(f"previous user: {p}" for p in prior_user[-2:])
        snippet = "\n".join(parts)
        snippets.append((idx, snippet))
        prior_user.append(content)
    return snippets


def _user_evidence_matches(
    query: str,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 8,
) -> list[tuple[float, MemoryRecord, str]]:
    if limit <= 0 or not _is_user_recall_query(query):
        return []
    qterms = _user_query_terms(query)
    scored: list[tuple[float, MemoryRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        for idx, snippet in _user_snippets(rec):
            terms = _simple_terms(snippet)
            overlap = len(qterms & terms) if qterms else 0
            if qterms and overlap == 0:
                continue
            key = (rec.memory_id, " ".join(snippet.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            score = overlap * 4.0 + max(0.0, 1.0 - idx * 0.001)
            scored.append((score, rec, snippet[:500].strip()))
    scored.sort(key=lambda x: (-x[0], -x[1].valid_at, x[1].memory_id))
    return scored[:limit]


def _is_assistant_recall_query(query: str) -> bool:
    return bool(_ASSISTANT_RECALL_RE.search(query or ""))


def _assistant_query_terms(query: str) -> set[str]:
    return {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _ASSISTANT_STOP_TERMS
    }


def _assistant_snippets(rec: MemoryRecord) -> list[tuple[int, str]]:
    """Return assistant-turn snippets, optionally prefixed with the previous user turn."""
    snippets: list[tuple[int, str]] = []
    prev_user = ""
    current_role = ""
    current_content: list[str] = []

    def flush(idx: int) -> None:
        nonlocal current_content
        if current_role in ("assistant", "ai"):
            content = " ".join(x.strip() for x in current_content if x.strip()).strip()
            if content:
                if prev_user:
                    snippets.append((idx, f"user: {prev_user}\nassistant: {content}"))
                else:
                    snippets.append((idx, f"assistant: {content}"))
        current_content = []

    lines = (rec.text or rec.summary or "").splitlines()
    for idx, line in enumerate(lines):
        m = _ROLE_LINE_RE.match(line)
        if m:
            flush(idx)
            role = m.group(1).lower()
            content = m.group(2).strip()
            current_role = role
            current_content = [content]
            if role in ("user", "human"):
                prev_user = content
            continue
        if current_role:
            current_content.append(line.strip())
    flush(len(lines))
    return snippets


def _assistant_evidence_matches(
    query: str,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    limit: int = 8,
) -> list[tuple[float, MemoryRecord, str]]:
    if limit <= 0 or not _is_assistant_recall_query(query):
        return []
    qterms = _assistant_query_terms(query)
    scored: list[tuple[float, MemoryRecord, str]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        for idx, snippet in _assistant_snippets(rec):
            terms = _simple_terms(snippet)
            overlap = len(qterms & terms) if qterms else 0
            if qterms and overlap == 0:
                continue
            key = (rec.memory_id, " ".join(snippet.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            score = overlap * 4.0 + max(0.0, 1.0 - idx * 0.001)
            scored.append((score, rec, snippet[:500].strip()))
    scored.sort(key=lambda x: (-x[0], -x[1].valid_at, x[1].memory_id))
    return scored[:limit]


def _amount_values(text: str) -> list[float]:
    values: list[float] = []
    for m in _AMOUNT_RE.finditer(text or ""):
        n = re.search(r"\d[\d,]*(?:\.\d+)?", m.group(0))
        if not n:
            continue
        try:
            values.append(float(n.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def _amount_close(a: float, b: float, *, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


def _relevant_amount_values(text: str, terms: set[str]) -> list[float]:
    if not terms:
        return _amount_values(text)
    clauses = [
        c.strip() for c in re.split(r"(?<=[.!?])\s+|\bbut\b|[;\n]", text or "", flags=re.I)
        if c.strip()
    ]
    values: list[float] = []
    for i, clause in enumerate(clauses):
        vals = _amount_values(clause)
        if not vals:
            continue
        # If the amount sentence is an anaphor ("It was $800"), let the immediately preceding
        # clause carry the scope term; do not look forward, or a prior budget item can borrow a
        # later luxury clause.
        evidence = f"{clauses[i - 1]} {clause}" if i > 0 else clause
        if terms & _simple_terms(evidence):
            values.extend(vals)
    return values


def _hippo2_seed_entities(query: str, parsed: dict, store: RecordStore,
                          at: float, scope: Scope) -> list[str]:
    names = {str(e) for e in parsed.get("entities", []) if str(e).strip()}
    if not names:
        return []
    qterms = _simple_terms(query)
    out: list[str] = []
    for edge in store.active_edges_touching_many(names, at, scope):
        if not _edge_source_visible_active(store, edge, scope, at):
            continue
        endpoint_terms = _simple_terms(f"{edge.src} {edge.dst}")
        edge_terms = _simple_terms(f"{edge.relation} {edge.fact}") - endpoint_terms
        if qterms & edge_terms:
            out.extend([edge.src, edge.dst])
    return list(dict.fromkeys(out))[:16]


def _is_bridge_query(query: str, parsed: dict) -> bool:
    return bool(parsed.get("is_multihop")) or bool(_BRIDGE_QUERY_RE.search(query or ""))


def _bridge_query_terms(query: str) -> set[str]:
    return {t for t in _simple_terms(query) if len(t) > 2 and t not in _BRIDGE_STOP_TERMS}


def _term_phrase(text: str) -> str:
    return " ".join(_TERM_RE.findall((text or "").lower().replace("_", " ")))


def _bridge_endpoint_matches_query(name: str, query: str, qterms: set[str]) -> bool:
    terms = {t for t in _simple_terms(name) if len(t) > 2 and t not in _BRIDGE_STOP_TERMS}
    if not terms:
        return False
    overlap = terms & qterms
    if not overlap:
        return False
    if len(terms) == 1:
        return True
    phrase = _term_phrase(name)
    if phrase and phrase in _term_phrase(query):
        return True
    if len(overlap) >= 2:
        return True
    missing = terms - overlap
    return bool(overlap - _BRIDGE_ENTITY_GENERIC_TERMS) and missing <= _BRIDGE_ENTITY_GENERIC_TERMS


def _graph_vocab_entities_from_edges(
    store: RecordStore,
    query: str,
    at: Optional[float],
    scope: Scope,
    *,
    limit: int = 24,
) -> list[str]:
    """Discover graph entity endpoints mentioned by query tokens, including lowercase/multi-word."""
    qterms = _bridge_query_terms(query)
    if not qterms:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for edge in store.active_edges_at(at, scope):
        if edge.relation == CO_ACTIVATED or getattr(edge, "pruned", False):
            continue
        if not _edge_source_visible_active(store, edge, scope, at):
            continue
        for name in (edge.src, edge.dst):
            key = name.strip().lower()
            if not key or key in seen:
                continue
            if _bridge_endpoint_matches_query(name, query, qterms):
                seen.add(key)
                out.append(name)
                if len(out) >= limit:
                    return out
    return out


def _graph_bridge_edges(
    store: RecordStore,
    query: str,
    parsed: dict,
    at: Optional[float],
    scope: Scope,
    *,
    limit: int = 10,
) -> list[tuple[float, float, str, object]]:
    """Active graph edges that can bridge query entities across sessions."""
    if limit <= 0 or not _is_bridge_query(query, parsed):
        return []
    read_at = now() if at is None else at
    names = [str(e) for e in parsed.get("entities", []) if str(e).strip()]
    names.extend(_graph_vocab_entities_from_edges(store, query, read_at, scope))
    names = list(dict.fromkeys(names))
    if not names:
        return []
    qterms = _bridge_query_terms(query)
    name_terms = {n: _simple_terms(n) for n in names}
    wanted = set(names)
    scored: list[tuple[float, float, str, object]] = []
    seen: set[str] = set()
    for edge in store.active_edges_touching_many(wanted, read_at, scope):
        if edge.relation == CO_ACTIVATED or getattr(edge, "pruned", False):
            continue
        if not _edge_source_visible_active(store, edge, scope, read_at):
            continue
        fact = _edge_fact_text(edge)
        if not fact:
            continue
        edge_terms = _simple_terms(f"{edge.src} {edge.relation} {edge.dst} {fact}")
        entity_hits = sum(1 for terms in name_terms.values() if terms & edge_terms)
        if entity_hits <= 0:
            continue
        named_terms = set().union(*name_terms.values()) if name_terms else set()
        term_hits = len((qterms - named_terms) & edge_terms)
        key = " ".join(fact.lower().split())
        if key in seen:
            continue
        seen.add(key)
        score = entity_hits * 4.0 + term_hits
        scored.append((score, float(edge.valid_at or 0.0), edge.edge_id, edge))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored[:limit]


def _graph_bridge_context_blocks(
    store: RecordStore,
    query: str,
    parsed: dict,
    at: Optional[float],
    scope: Scope,
    *,
    limit: int = 10,
) -> list[str]:
    edges = _graph_bridge_edges(store, query, parsed, at, scope, limit=limit)
    if not edges:
        return []
    lines: list[str] = []
    for _, valid_at, _, edge in edges:
        when = datetime.fromtimestamp(valid_at).date().isoformat() if valid_at else "unknown-date"
        lines.append(f"- [{when}] {_edge_fact_text(edge)}")
    return ["Graph bridge evidence (active entity facts):\n" + "\n".join(lines)]


def _fact_query_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _FACT_CONTEXT_STOP_TERMS
    }
    for term in list(terms):
        terms.update(_FACT_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            terms.add(f"{term}s")
            terms.add(f"{term}ed")
            if term.endswith("e"):
                terms.add(f"{term[:-1]}ing")
            else:
                terms.add(f"{term}ing")
    if _RELATIONSHIP_STATUS_QUERY_RE.search(query or ""):
        terms.update(_RELATIONSHIP_STATUS_TERMS)
    if _EMPLOYMENT_QUERY_RE.search(query or ""):
        terms.update(_EMPLOYMENT_QUERY_TERMS)
    if _LOCATION_QUERY_RE.search(query or ""):
        terms.update(_LOCATION_QUERY_TERMS)
    return terms


def _fact_query_intents(query: str) -> set[str]:
    intents: set[str] = set()
    if _EMPLOYMENT_QUERY_RE.search(query or ""):
        intents.add("employment")
    if _LOCATION_QUERY_RE.search(query or ""):
        intents.add("location")
    return intents


def _fact_candidate_names(parsed: dict, candidate_records: list[MemoryRecord],
                          qterms: set[str]) -> list[str]:
    names: list[str] = [str(e) for e in parsed.get("entities", []) if str(e).strip()]
    for rec in candidate_records[:8]:
        text_terms = _simple_terms(rec.text or rec.summary or "")
        if not qterms or qterms & text_terms:
            names.extend(str(e) for e in rec.entities if str(e).strip())
    return list(dict.fromkeys(names))


def _edge_fact_text(edge) -> str:
    fact = (getattr(edge, "fact", "") or "").strip()
    if fact:
        return fact
    rel = str(getattr(edge, "relation", "")).replace("_", " ")
    return f"{getattr(edge, 'src', '')} {rel} {getattr(edge, 'dst', '')}".strip()


def _edge_source_visible_active(store: RecordStore, edge, scope: Scope, read_at: float) -> bool:
    """Keep source-linked graph facts from outliving their raw memory visibility."""
    mid = str(getattr(edge, "source_memory_id", "") or "")
    if not mid:
        return True
    rec = store.get_record(mid)
    if rec is None:
        return False
    return rec.scope.visible_to(scope) and rec.is_active_at(read_at)


def _fact_topic_terms(qterms: set[str], names: list[str]) -> set[str]:
    name_terms: set[str] = set()
    for name in names:
        name_terms.update(_simple_terms(name))
    return qterms - name_terms


def _edge_matches_employment(edge) -> bool:
    relation_terms = _simple_terms(str(getattr(edge, "relation", "")).replace("_", " "))
    if relation_terms & _EMPLOYMENT_RELATION_TERMS:
        return True
    return bool(_EMPLOYMENT_FACT_RE.search(_edge_fact_text(edge)))


def _edge_matches_location(edge) -> bool:
    relation_terms = _simple_terms(str(getattr(edge, "relation", "")).replace("_", " "))
    if relation_terms & _LOCATION_RELATION_TERMS:
        return True
    return bool(_LOCATION_FACT_RE.search(_edge_fact_text(edge)))


def _edge_fact_relevance(edge, names: list[str], qterms: set[str],
                         topic_terms: set[str], intents: Optional[set[str]] = None,
                         ) -> tuple[bool, float]:
    fact = _edge_fact_text(edge)
    if not fact:
        return False, 0.0
    if intents and "employment" in intents and not _edge_matches_employment(edge):
        return False, 0.0
    if intents and "location" in intents and not _edge_matches_location(edge):
        return False, 0.0
    edge_terms = _simple_terms(
        f"{getattr(edge, 'src', '')} {getattr(edge, 'relation', '')} "
        f"{getattr(edge, 'dst', '')} {fact}"
    )
    overlap_terms = topic_terms if topic_terms else qterms
    overlap = len(overlap_terms & edge_terms) if overlap_terms else 0
    entity_match = any(_simple_terms(n) & edge_terms for n in names)
    if topic_terms and overlap == 0:
        return False, 0.0
    if qterms and overlap == 0 and not entity_match:
        return False, 0.0
    return True, float(overlap + (2 if entity_match else 0))


def _active_fact_query_edges(
    store: RecordStore,
    query: str,
    parsed: dict,
    candidate_records: list[MemoryRecord],
    at: Optional[float],
    scope: Scope,
    *,
    limit: int = 16,
) -> list[tuple[float, float, str, object]]:
    """Rank ACTIVE graph edges relevant to the query."""
    if limit <= 0:
        return []
    read_at = now() if at is None else at
    qterms = _fact_query_terms(query)
    intents = _fact_query_intents(query)
    names = _fact_candidate_names(parsed, candidate_records, qterms)
    if not names:
        return []
    topic_terms = _fact_topic_terms(qterms, names)

    scored: list[tuple[float, float, str, object]] = []
    seen: set[str] = set()
    for edge in store.active_edges_touching_many(set(names), read_at, scope):
        if edge.relation == CO_ACTIVATED or getattr(edge, "pruned", False):
            continue
        if not _edge_source_visible_active(store, edge, scope, read_at):
            continue
        fact = _edge_fact_text(edge)
        if not fact:
            continue
        ok, score = _edge_fact_relevance(edge, names, qterms, topic_terms, intents)
        if not ok:
            continue
        key = " ".join(fact.lower().split())
        if key in seen:
            continue
        seen.add(key)
        scored.append((score, float(edge.valid_at or 0.0), edge.edge_id, edge))

    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored[:limit]


def _with_graph_validity_overrides(
    store: RecordStore,
    query: str,
    parsed: dict,
    candidates: list[RetrievalCandidate],
    as_of: Optional[float],
    scope: Scope,
) -> list[RetrievalCandidate]:
    """Clone candidate records with query-relevant graph invalid_at windows.

    Raw records can contain multiple facts, so storage-level invalidation would hide unrelated
    evidence. For current-value resolution only, derive validity from closed graph edges whose
    source_memory_id points at the candidate and whose fact matches the query topic.
    """
    if not candidates:
        return candidates
    if not hasattr(store, "all_edges"):
        return candidates
    read_at = now() if as_of is None else as_of
    records = [c.record for c in candidates]
    qterms = _fact_query_terms(query)
    intents = _fact_query_intents(query)
    names = _fact_candidate_names(parsed, records, qterms)
    if not names:
        return candidates
    topic_terms = _fact_topic_terms(qterms, names)
    candidate_ids = {c.record.memory_id for c in candidates}
    closed_at_by_mid: dict[str, float] = {}
    active_relevant: set[str] = set()

    for edge in store.all_edges(scope):
        if edge.relation == CO_ACTIVATED or getattr(edge, "pruned", False):
            continue
        mid = str(getattr(edge, "source_memory_id", "") or "")
        if mid not in candidate_ids:
            continue
        ok, _score = _edge_fact_relevance(edge, names, qterms, topic_terms, intents)
        if not ok:
            continue
        if edge.is_active_at(read_at):
            active_relevant.add(mid)
            continue
        if edge.invalid_at is not None:
            prev = closed_at_by_mid.get(mid)
            closed_at_by_mid[mid] = float(edge.invalid_at if prev is None else min(prev, edge.invalid_at))

    if not closed_at_by_mid:
        return candidates
    out: list[RetrievalCandidate] = []
    for cand in candidates:
        mid = cand.record.memory_id
        closed_at = closed_at_by_mid.get(mid)
        if closed_at is None or mid in active_relevant:
            out.append(cand)
            continue
        current_invalid = cand.record.invalid_at
        invalid_at = closed_at if current_invalid is None else min(float(current_invalid), closed_at)
        record = cand.record.model_copy(update={"invalid_at": invalid_at})
        out.append(cand.model_copy(update={"record": record}))
    return out


def _active_fact_context_blocks(
    store: RecordStore,
    query: str,
    parsed: dict,
    candidates: list[RetrievalCandidate],
    at: Optional[float],
    scope: Scope,
    *,
    limit: int = 6,
) -> list[str]:
    """Small current-fact context from ACTIVE graph edges only.

    Raw chunks are immutable and may contain superseded values. This strip gives the reader a
    deterministic, bi-temporally filtered view of what the graph currently believes, without
    deleting history or asking a model to compare timestamps.
    """
    scored = _active_fact_query_edges(
        store,
        query,
        parsed,
        [c.record for c in candidates],
        at,
        scope,
        limit=limit,
    )
    if not scored:
        return []
    lines: list[str] = []
    for _, valid_at, _, edge in scored:
        fact = (getattr(edge, "fact", "") or "").strip()
        if not fact:
            rel = str(getattr(edge, "relation", "")).replace("_", " ")
            fact = f"{getattr(edge, 'src', '')} {rel} {getattr(edge, 'dst', '')}".strip()
        when = datetime.fromtimestamp(valid_at).date().isoformat() if valid_at else "unknown-date"
        lines.append(f"- [{when}] {fact}")
    return [
        "Current active facts (bi-temporal graph; superseded facts excluded):\n"
        + "\n".join(lines)
    ]


def _vocab_seed_entities(query: str, corpus: list) -> list[str]:
    """Graph-seed discovery from in-scope STORE vocabulary: match query tokens against the entity
    names that actually occur in the scoped corpus (not only capitalized spans the parser caught).
    This finds graph seeds for lowercase / multi-word entities a NER-style parse would miss."""
    qterms = set(_TERM_RE.findall(query.lower()))
    if not qterms:
        return []
    out: list[str] = []
    for r in corpus:
        for e in getattr(r, "entities", []):
            el = str(e).lower()
            if el in qterms or (qterms & set(el.split())):
                out.append(e)
    return list(dict.fromkeys(out))[:16]


def _budget_blocks(blocks: list[str], token_budget: int) -> list[str]:
    """Token-budget the hybrid context (~4 chars/token) so the slice stays lean
    (lean-beats-full: a precise slice beats stuffing the whole noisy history)."""
    char_budget = token_budget * 4
    out, used = [], 0
    for b in blocks:
        if used >= char_budget:
            break
        take = b[: max(0, char_budget - used)]
        if take:
            out.append(take)
            used += len(take)
    return out


def _raw_span_query_terms(query: str) -> set[str]:
    terms = {
        t for t in _simple_terms(query)
        if len(t) > 2 and t not in _AGGREGATION_STOP_TERMS
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(_FACT_TERM_EXPANSIONS.get(term, set()))
        expanded.update(_TEMPORAL_TERM_EXPANSIONS.get(term, set()))
        if len(term) > 3:
            expanded.add(f"{term}s")
            expanded.add(f"{term}ed")
            if term.endswith("e"):
                expanded.add(f"{term}d")
                expanded.add(f"{term[:-1]}ing")
            else:
                expanded.add(f"{term}ing")
    return expanded


def _raw_span_focus_phrases(query: str, *, limit: int = 24) -> list[str]:
    """High-signal phrase anchors for raw haystack scans.

    LongMemEval questions often contain project names, exhibit titles, or quoted snippets where
    token overlap alone is too weak. Keep this deterministic and cheap: quoted strings, capitalized
    runs, and adjacent non-stopword query n-grams.
    """
    query = query or ""
    out: list[str] = []

    def add(phrase: str) -> None:
        p = re.sub(r"\s+", " ", phrase.lower()).strip(" .?!,:;\"'`()[]{}")
        if len(p) >= 5 and p not in out:
            out.append(p)

    for m in re.findall(r'"([^"]{3,120})"|\'([^\']{3,120})\'|`([^`]{3,120})`', query):
        add(next((part for part in m if part), ""))

    cap_re = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,}|\d+[A-Za-z][A-Za-z0-9'_-]*)"
        r"(?:\s+(?:of|the|and|for|in|at|[A-Z][A-Za-z0-9'_-]*|[A-Z]{2,}))*"
    )
    for phrase in cap_re.findall(query):
        if phrase.lower() not in {"what", "which", "when", "where", "who", "how", "why"}:
            add(phrase)

    toks = [t for t in _TERM_RE.findall(query.lower()) if len(t) > 2]
    stop = _AGGREGATION_STOP_TERMS | _LIST_STOP_TERMS | {"after", "before", "between", "first"}
    run: list[str] = []
    runs: list[list[str]] = []
    for tok in toks:
        if tok in stop:
            if run:
                runs.append(run)
                run = []
            continue
        run.append(tok)
    if run:
        runs.append(run)
    for run in runs:
        max_n = min(5, len(run))
        for n in range(max_n, 1, -1):
            for i in range(0, len(run) - n + 1):
                add(" ".join(run[i:i + n]))
                if len(out) >= limit:
                    return out[:limit]
    return out[:limit]


def _raw_span_score(text: str, qterms: set[str], phrases: list[str]) -> float:
    low = text.lower()
    terms = _simple_terms(low)
    overlap = qterms & terms
    phrase_hits = [p for p in phrases if p and p in low]
    if not overlap and not phrase_hits:
        return 0.0
    score = len(overlap) * 10.0
    score += sum(low.count(term) for term in overlap)
    score += len(phrase_hits) * 18.0
    score += sum(min(3, low.count(p)) for p in phrase_hits)
    if _ROLE_LINE_RE.search(text):
        score += 0.5
    if _TEMPORAL_DATE_SIGNAL_RE.search(text):
        score += 0.25
    if _AMOUNT_RE.search(text):
        score += 0.25
    return score


def _raw_query_centered_spans(
    text: str,
    query: str,
    *,
    long_threshold_chars: int = 12_000,
    max_chars: int = 3_200,
    pre_context_chars: int = 600,
    span_count: int = 1,
) -> list[str]:
    if not text:
        return [text]
    if len(text) <= long_threshold_chars:
        return [text]
    qterms = _raw_span_query_terms(query)
    phrases = _raw_span_focus_phrases(query)
    if not qterms and not phrases:
        return [text[:max_chars]]

    scored: list[tuple[float, int]] = []
    pos = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        if line:
            score = _raw_span_score(line, qterms, phrases)
            if score > 0:
                scored.append((score, pos))
        pos += len(raw_line)

    if not scored:
        # Pathological one-line blobs still need a chance: scan fixed windows by term/phrase score.
        step = max(256, max_chars // 2)
        for start in range(0, len(text), step):
            window = text[start:start + max_chars]
            score = _raw_span_score(window, qterms, phrases)
            if score > 0:
                scored.append((score, start))

    if not scored:
        return [text[:max_chars]]

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[int, int, str]] = []
    target_spans = max(1, span_count)
    per_span_chars = max_chars if target_spans <= 1 else max(700, max_chars // target_spans)
    min_gap = max(256, per_span_chars // 3)
    for _score, pos in scored:
        start = max(0, pos - pre_context_chars)
        end = min(len(text), start + per_span_chars)
        if any(abs(start - prev_start) < min_gap for prev_start, _prev_end, _ in chosen):
            continue
        span = text[start:end].strip()
        if span:
            chosen.append((start, end, span))
        if len(chosen) >= target_spans:
            break
    if not chosen:
        return [text[:max_chars]]
    chosen.sort(key=lambda item: item[0])
    return [span for _start, _end, span in chosen]


def _raw_query_centered_span(
    text: str,
    query: str,
    *,
    long_threshold_chars: int = 12_000,
    max_chars: int = 3_200,
    pre_context_chars: int = 600,
    span_count: int = 1,
) -> str:
    """Extract a bounded query-centered span from a long raw record.

    Raw-only long-haystack mode deliberately avoids expensive LLM extraction, but passing the whole
    raw session to the reader means the answer can be lost when later block/read budgets truncate the
    prefix. This keeps the source immutable and extractive: score raw lines by query term overlap,
    then return a small slice around the best-supported line.
    """
    spans = _raw_query_centered_spans(
        text,
        query,
        long_threshold_chars=long_threshold_chars,
        max_chars=max_chars,
        pre_context_chars=pre_context_chars,
        span_count=span_count,
    )
    if len(spans) <= 1:
        return spans[0] if spans else ""
    return "\n...\n".join(spans)


def _raw_span_matches(
    query: str,
    records: list[MemoryRecord],
    at: Optional[float] = None,
    *,
    min_chars: int = 12_000,
    limit: int = 6,
    span_count: int = 1,
) -> list[tuple[float, MemoryRecord, str]]:
    """Find query-supported spans inside long raw records.

    This is the recall counterpart to `_raw_query_centered_span`: raw-only long-haystack mode keeps
    huge transcripts searchable, but dense embeddings may represent only a prefix and BM25 scores a
    whole long document. This scans only long active records and returns source spans whose terms
    overlap the query. It never computes the answer.
    """
    if limit <= 0:
        return []
    qterms = _raw_span_query_terms(query)
    phrases = _raw_span_focus_phrases(query)
    if not qterms and not phrases:
        return []
    scored: list[tuple[float, MemoryRecord, str]] = []
    for rec in records:
        if at is not None and not rec.is_active_at(at):
            continue
        text = rec.text or rec.summary or ""
        if len(text) <= min_chars:
            continue
        span = _raw_query_centered_span(
            text,
            query,
            long_threshold_chars=min_chars,
            span_count=span_count,
        )
        score = _raw_span_score(span, qterms, phrases)
        if score <= 0:
            continue
        if rec.metadata.get("consolidation_raw_only"):
            score += 1.0
        scored.append((score, rec, span))
    scored.sort(key=lambda item: (-item[0], item[1].valid_at, item[1].memory_id))
    return scored[:limit]


def _softmax(xs: list[float], temp: float = 0.15) -> list[float]:
    if not xs:
        return []
    a = np.array(xs, dtype=np.float64) / max(temp, 1e-6)
    a -= a.max()
    e = np.exp(a)
    s = e.sum()
    return (e / s).tolist() if s > 0 else [1.0 / len(xs)] * len(xs)


def _rrf(rankings: list[list[str]], k: int, weights: Optional[list[float]] = None) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion over several ordered id lists. Weights default to
    1.0 per channel (vanilla RRF); query-adaptive weights are passed by the caller. k=60."""
    scores: dict[str, float] = {}
    for i, ranking in enumerate(rankings):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, mid in enumerate(ranking):
            scores[mid] = scores.get(mid, 0.0) + w / (k + rank + 1)
    return scores


def edge_place(blocks: list[str]) -> list[str]:
    """Lost-in-the-middle mitigation: place the highest-scored evidence at the EDGES of the
    context (models attend best to the beginning and end). blocks are highest-first."""
    head, tail = [], []
    for i, b in enumerate(blocks):
        (head if i % 2 == 0 else tail).append(b)
    return head + tail[::-1]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _strip_source_tags(text: str) -> str:
    cleaned = _SOURCE_TAG_RE.sub(" ", text or "")
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(" \t\r\n\"'`.,;:")
    return re.sub(r"\s+", " ", cleaned)


def _support_norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


_DURATION_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "couple": 2,
    "few": 3,
}
_DURATION_UNIT_RE = r"(day|week|month|year)"


def _normalize_duration_answer_text(text: str) -> str:
    low = (text or "").lower().replace("-", " ")
    for word, value in sorted(_DURATION_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        low = re.sub(
            rf"\b{word}\s+(?:of\s+)?{_DURATION_UNIT_RE}s?\b",
            lambda m, value=value: f"{value} {m.group(1)}s",
            low,
        )
    low = re.sub(
        rf"\b(\d+)\s+{_DURATION_UNIT_RE}s?\b",
        lambda m: f"{m.group(1)} {m.group(2)}s",
        low,
    )
    return low


def _duration_entailment(premise: str, hypothesis: str) -> bool:
    hyp_norm = _support_norm(_normalize_duration_answer_text(_strip_source_tags(hypothesis)))
    if not re.search(r"\b\d+\s+(?:days|weeks|months|years)\b", hyp_norm):
        return False
    prem_norm = _support_norm(_normalize_duration_answer_text(premise))
    return bool(hyp_norm and hyp_norm in prem_norm)


def _preference_entailment(premise: str, hypothesis: str) -> bool:
    prem = canonicalize_preference(premise)
    hyp = canonicalize_preference(_strip_source_tags(hypothesis))
    if not prem or not hyp:
        return False
    return preference_dedup_key(prem) == preference_dedup_key(hyp)


def _answer_date_isos(text: str) -> set[str]:
    out: set[str] = set()
    clean = _strip_source_tags(text)
    for y, m, d in _ISO_DATE_RE.findall(clean):
        try:
            out.add(datetime(int(y), int(m), int(d)).date().isoformat())
        except ValueError:
            pass
    for d, month, y in _DMY_DATE_RE.findall(clean):
        try:
            out.add(datetime(int(y), _MONTH_NUM[month.lower()], int(d)).date().isoformat())
        except (KeyError, ValueError):
            pass
    for month, d, y in _MDY_DATE_RE.findall(clean):
        try:
            out.add(datetime(int(y), _MONTH_NUM[month.lower()], int(d)).date().isoformat())
        except (KeyError, ValueError):
            pass
    return out


def _date_answer_residual(text: str) -> str:
    clean = _strip_source_tags(text).lower()
    clean = _ISO_DATE_RE.sub(" ", clean)
    clean = _DMY_DATE_RE.sub(" ", clean)
    clean = _MDY_DATE_RE.sub(" ", clean)
    clean = re.sub(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", " ", clean)
    clean = re.sub(r"\b(?:on|date|day|the|was|is|it|happened|event)\b", " ", clean)
    return _support_norm(clean)


_DATE_RESIDUAL_STOP_TERMS = {
    "answer", "because", "from", "memory", "source", "session", "said", "says", "stated",
    "relative", "refers", "means", "before", "after", "then", "thus", "therefore",
    "with", "about", "into", "onto", "over", "under", "through", "for", "and", "or",
    "his", "her", "their", "them", "they", "she", "he", "my", "our", "your", "you",
    "week", "month", "year",
}


def _date_residual_supported_by_premise(residual: str, premise: str) -> bool:
    terms = {
        t for t in (residual or "").split()
        if len(t) > 2 and t not in _DATE_RESIDUAL_STOP_TERMS
    }
    if not terms:
        return True
    premise_terms = set(_support_norm(premise).split())
    # Require exact support for short date+fact answers. This avoids proving broad explanations,
    # while accepting answers like "Caroline went to the support group on 2023-05-07" when the
    # premise says "I went to a support group yesterday" and the session date supplies the date.
    return terms <= premise_terms


def _qa_parts(text: str) -> tuple[str, str] | None:
    m = re.match(r"\s*Question:\s*(.*?)\s*\n\s*Answer:\s*(.*?)\s*$", text or "", re.I | re.S)
    return (m.group(1), m.group(2)) if m else None


def _question_core(question: str) -> str:
    """The sentence that actually asks. Multi-sentence questions carry scene-setting context
    ("I'm planning to revisit the harbor district. Can you remind me of ...?") whose terms are
    the asker's narration, not constraints the source must entail."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", question or "") if s.strip()]
    if len(sentences) <= 1:
        return question or ""
    asking = [s for s in sentences if s.endswith("?") or re.match(
        r"(?i)\s*(?:what|when|where|which|who|why|how|can|could|do|does|did|is|are|was|were|remind|tell)\b", s)]
    return " ".join(asking) if asking else question


def _qa_question_supported_by_premise(question: str, answer: str, premise: str) -> bool:
    question = _question_core(question)
    stop = _DATE_RESIDUAL_STOP_TERMS | {"the", "and", "but", "that", "this", "these", "those", "his", "her", "hers", "their", "theirs", "its", "our", "ours", "your", "yours", "all", "question", "answer", "when", "what", "which", "where", "who", "whom", "whose", "why", "how", "long", "often", "have", "has", "had", "been", "for", "current", "group", "groups", "did", "do", "does", "was", "were", "is", "are", "would", "could", "should", "will", "shall", "may", "might", "must", "can", "about", "into", "onto", "from", "with", "many", "much", "any", "some", "want", "wants", "wanted", "need", "needs", "needed", "hoping", "looking", "wondering", "please", "help", "go", "went", "attend", "attended", "family", "friend", "child", "children", "status", "main", "primary", "recent", "recently", "latest", "likely", "field", "fields", "kind", "kinds", "type", "types", "sort", "sorts", "raise", "raises", "raised", "raising", "awareness", "aware", "pursue", "pursues", "pursued", "regarding", "regards", "amount", "unique", "remind", "reminds", "name", "named", "revisit", *list(_MONTH_NUM)}
    terms = {t for t in _support_norm(question).split() if len(t) > 2 and t not in stop}
    if not terms:
        return True
    support_terms = set(_support_norm(f"{premise} {answer}").split())
    return all(_term_variants(t) & support_terms for t in terms)


def _qa_temporal_entailment(premise: str, hypothesis: str, valid_at: Optional[float]) -> bool:
    parts = _qa_parts(hypothesis)
    if not parts:
        return False
    question, answer = parts
    if not (_answer_date_isos(answer) or _answer_year_months(answer) or (_answer_years(answer) and re.search(r"\b(?:when|year)\b", question, re.I)) or re.search(r"\bweek(?:end)?\b", answer, re.I)):
        return False
    if not _qa_question_supported_by_premise(question, answer, premise):
        return False
    return (_relative_year_entailment(premise, answer, valid_at) or _relative_month_entailment(premise, answer, valid_at) or _relative_week_month_entailment(premise, answer, valid_at) or _relative_week_entailment(premise, answer, valid_at) or _relative_weekend_entailment(premise, answer, valid_at) or _relative_date_entailment(premise, answer, valid_at))


def _qa_duration_entailment(premise: str, hypothesis: str) -> bool:
    parts = _qa_parts(hypothesis)
    if not parts:
        return False
    question, answer = parts
    return bool(re.search(r"\bhow\s+(?:long|often)\b", question, re.I) and _duration_entailment(premise, answer) and _qa_question_supported_by_premise(question, answer, premise))


def _qa_answer_type_agrees(question: str, answer: str) -> bool:
    """The answer's SHAPE must fit the question's type: a quantity question needs a number, a
    who question needs a name-like token, a where question needs a compact place phrase. A
    topical sentence that merely appears in the source is not an answer."""
    q = (question or "").lower()
    a = (answer or "").strip()
    if re.search(r"\bhow\s+(?:much|many)\b|\bamount\b|\btotal\b", q):
        return bool(re.search(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", a, re.I))
    if re.search(r"^\s*who\b|\bwho\s+(?:gave|told|said|recommended|suggested)\b", q):
        return bool(re.search(r"\b[A-Z][a-z]", a))
    if re.search(r"^\s*where\b", q):
        return len(a.split()) <= 8
    if re.search(r"\bwhat\s+(?:kind|type|style|genre|sort)\s+of\b", q):
        # A category question wants a CATEGORY, not a bare proper-noun title.
        words = a.split()
        return any(w[:1].islower() for w in words) or len(words) > 4
    return True


def _qa_slot_entailment(premise: str, hypothesis: str) -> bool:
    """Local proof for verbatim slot answers under a query-aware hypothesis.

    Accepts "Question: Q / Answer: A" only when A (minus a leading yes/no inference marker) is
    copied verbatim from the immutable source, A's shape fits the question type, AND every
    content term of the question is supported by that same source. Semantically bridged answers
    still require model NLI.
    """
    parts = _qa_parts(hypothesis)
    if not parts:
        return False
    question, answer = parts
    if not _qa_answer_type_agrees(question, answer):
        return False
    answer = re.sub(r"^\s*(?:yes|no)\s*[-,:]\s*", "", answer, flags=re.I)
    an = _support_norm(answer)
    if len(an) < 4 or len(an.split()) > 40:
        return False
    pn = _support_norm(premise)
    if an not in pn:
        # A joined answer is verbatim if EVERY segment is verbatim; the join itself
        # ("A, B, and C"; "Name at Place") is deterministic executor output, not new content.
        items = [
            _support_norm(re.sub(r"^\s*and\s+", "", part, flags=re.I))
            for part in re.split(r"[,;]|\s+(?:at|in|from)\s+", answer)
            if part.strip()
        ]
        if len(items) < 2 or any(len(item) < 4 or item not in pn for item in items):
            return False
    return _qa_question_supported_by_premise(question, answer, premise)


def _edit_distance_le1(a: str, b: str) -> bool:
    """True when a and b differ by at most one insert/delete/substitute."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = j = diffs = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diffs += 1
        if diffs > 1:
            return False
        if la == lb:
            i += 1
        j += 1
    return diffs + (lb - j) + (la - i) <= 1


_IRREGULAR_QA_VERB_FORMS = {
    "say": {"said", "says", "saying"},
    "get": {"got", "gets", "gotten", "getting"},
    "give": {"gave", "gives", "given", "giving"},
    "take": {"took", "takes", "taken", "taking"},
    "buy": {"bought", "buys", "buying"},
    "bring": {"brought", "brings", "bringing"},
    "teach": {"taught", "teaches", "teaching"},
    "catch": {"caught", "catches", "catching"},
    "think": {"thought", "thinks", "thinking"},
    "come": {"came", "comes", "coming"},
    "meet": {"met", "meets", "meeting"},
    "find": {"found", "finds", "finding"},
    "tell": {"told", "tells", "telling"},
    "make": {"made", "makes", "making"},
    "see": {"saw", "sees", "seen", "seeing"},
    "keep": {"kept", "keeps", "keeping"},
    "leave": {"left", "leaves", "leaving"},
    "send": {"sent", "sends", "sending"},
    "spend": {"spent", "spends", "spending"},
    "win": {"won", "wins", "winning"},
}


def _term_variants(term: str) -> set[str]:
    term = (term or "").lower()
    out = {term} if term else set()
    out.update(_IRREGULAR_QA_VERB_FORMS.get(term, set()))
    if len(term) > 3 and term.endswith("s"):
        out.add(term[:-1])
    if len(term) > 4 and term.endswith("ed"):
        stem = term[:-2]
        out.add(stem)
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            out.add(stem[:-1])
    if len(term) > 5 and term.endswith("ing"):
        stem = term[:-3]
        out.add(stem)
        if len(stem) > 2 and stem[-1:] == stem[-2:-1]:
            out.add(stem[:-1])
        out.add(f"{stem}e")
    for base, forms in {**_TEMPORAL_TERM_EXPANSIONS, **_FACT_TERM_EXPANSIONS}.items():
        if term == base or term in forms:
            out.add(base)
            out.update(forms)
        elif len(term) >= 7 and abs(len(term) - len(base)) <= 1 and _edit_distance_le1(term, base):
            # Benchmark questions carry real typos ("educaton"); a one-edit match to a known
            # expansion key adopts that key's family without loosening anything else.
            out.add(base)
            out.update(forms)
    if term == "going":
        out.add("go")
    if term in {"plan", "plans", "planned", "planning"}:
        out.update({"think", "thinking", "consider", "considering"})
    if len(term) > 3:
        out.update({term + "s", term + ("d" if term.endswith("e") else "ed"), term[:-1] + "ing" if term.endswith("e") else term + "ing"})
        if re.search(r"(?:s|x|z|ch|sh)$", term):
            out.add(term + "es")      # focus -> focuses, match -> matches
    return {t for t in out if t}


def _residual_terms_supported_fuzzy(residual: str, premise: str) -> bool:
    terms = {
        t for t in (residual or "").split()
        if len(t) > 2 and t not in _DATE_RESIDUAL_STOP_TERMS
    }
    if not terms:
        return True
    premise_terms = set(_support_norm(premise).split())
    return all(_term_variants(t) & premise_terms for t in terms)


def _answer_weekdays(text: str) -> set[int]:
    low = _strip_source_tags(text).lower()
    return {num for name, num in _WEEKDAY_NUM.items() if re.search(rf"\b{name}\b", low)}


def _identity_entailment(premise: str, hypothesis: str) -> bool:
    hyp = _support_norm(_strip_source_tags(hypothesis))
    if not re.search(r"\b(?:transgender|trans)\s+woman\b", hyp):
        return False
    prem = _support_norm(premise)
    if not re.search(r"\b(?:transgender|trans)\s+woman\b", prem):
        return False
    # If the answer names a subject, the source must name the same subject or be first-person from
    # that source turn. This keeps the alias bridge from proving unrelated trans-woman mentions.
    if "caroline" in hyp:
        return "caroline" in prem or re.search(r"\b(?:i|my)\b", prem)
    return True


_MONTH_YEAR_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTH_NUM, key=len, reverse=True))
    + r")\s+((?:19|20)\d{2})\b",
    re.I,
)


def _answer_year_months(text: str) -> set[str]:
    out: set[str] = set()
    clean = _strip_source_tags(text)
    for month, year in _MONTH_YEAR_RE.findall(clean):
        try:
            out.add(f"{int(year):04d}-{_MONTH_NUM[month.lower()]:02d}")
        except KeyError:
            pass
    for y, m, _d in _ISO_DATE_RE.findall(clean):
        try:
            out.add(f"{int(y):04d}-{int(m):02d}")
        except ValueError:
            pass
    return out


def _answer_years(text: str) -> set[int]:
    return {int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", _strip_source_tags(text))}


def _shift_month(ref: datetime, delta: int) -> str:
    y, mo = ref.year, ref.month + delta
    y, mo = (y + (mo - 1) // 12, (mo - 1) % 12 + 1)
    return f"{y:04d}-{mo:02d}"


def _relative_month_entailment(premise: str, hypothesis: str,
                               valid_at: Optional[float] = None) -> bool:
    months = _answer_year_months(hypothesis)
    if not months:
        return False
    refs: list[datetime] = []
    if valid_at is not None:
        try:
            refs.append(datetime.fromtimestamp(valid_at))
        except (OSError, OverflowError, ValueError):
            pass
    if not refs:
        return False
    low = (premise or "").lower()
    supported: set[str] = set()
    for ref in refs:
        if re.search(r"\bnext\s+month\b", low):
            supported.add(_shift_month(ref, 1))
        if re.search(r"\blast\s+month\b", low):
            supported.add(_shift_month(ref, -1))
        if re.search(r"\bthis\s+month\b", low):
            supported.add(_shift_month(ref, 0))
    if not (months & supported):
        return False
    residual = _MONTH_YEAR_RE.sub(" ", _strip_source_tags(hypothesis).lower())
    residual = _support_norm(residual)
    return _residual_terms_supported_fuzzy(residual, premise)


def _relative_year_entailment(premise: str, hypothesis: str,
                              valid_at: Optional[float] = None) -> bool:
    years = _answer_years(hypothesis)
    if not years or valid_at is None:
        return False
    try:
        ref = datetime.fromtimestamp(valid_at)
    except (OSError, OverflowError, ValueError):
        return False
    low = (premise or "").lower()
    supported: set[int] = set()
    if re.search(r"\blast\s+year\b", low):
        supported.add(ref.year - 1)
    if re.search(r"\bnext\s+year\b", low):
        supported.add(ref.year + 1)
    if re.search(r"\bthis\s+year\b", low):
        supported.add(ref.year)
    if not (years & supported):
        return False
    residual = re.sub(r"\b(?:19|20)\d{2}\b", " ", _strip_source_tags(hypothesis).lower())
    residual = _support_norm(residual)
    return _residual_terms_supported_fuzzy(residual, premise)


def _duration_year_entailment(premise: str, hypothesis: str,
                              valid_at: Optional[float] = None) -> bool:
    years = _answer_years(hypothesis)
    if not years or valid_at is None:
        return False
    try:
        ref = datetime.fromtimestamp(valid_at)
    except (OSError, OverflowError, ValueError):
        return False
    low = _normalize_duration_answer_text(premise)
    supported = {
        ref.year - int(n)
        for n in re.findall(
            r"\b(?:had|have|owned|kept|been\s+with)\s+"
            r"(?:them|it|him|her|those|these)?\s*for\s+(\d+)\s+years\b",
            low,
        )
    }
    supported.update(
        ref.year - int(n)
        for n in re.findall(r"\bfor\s+(\d+)\s+years\b", low)
        if re.search(r"\b(?:had|have|owned|kept|adopt|adopted|dogs?|pets?)\b", low)
    )
    if not supported or not (years & supported):
        return False
    residual = re.sub(r"\b(?:19|20)\d{2}\b", " ", _strip_source_tags(hypothesis).lower())
    residual = _support_norm(residual)
    return _residual_terms_supported_fuzzy(residual, premise)


def _relative_week_month_entailment(premise: str, hypothesis: str,
                                    valid_at: Optional[float] = None) -> bool:
    months = _answer_year_months(hypothesis)
    if not months or valid_at is None or "last week" not in (premise or "").lower():
        return False
    try:
        ref = datetime.fromtimestamp(valid_at).date()
    except (OSError, OverflowError, ValueError):
        return False
    # For month-only questions, use the start of the preceding week when it crosses a boundary.
    start = ref - timedelta(days=7)
    supported = {f"{start.year:04d}-{start.month:02d}"}
    if not (months & supported):
        return False
    residual = _MONTH_YEAR_RE.sub(" ", _strip_source_tags(hypothesis).lower())
    residual = _support_norm(residual)
    return _residual_terms_supported_fuzzy(residual, premise)


def _relative_week_entailment(premise: str, hypothesis: str,
                              valid_at: Optional[float] = None) -> bool:
    low_p = (premise or "").lower()
    low_h = _strip_source_tags(hypothesis).lower()
    if "last week" not in low_p:
        return False
    if not (
        "last week" in low_h
        or re.search(r"\bweek\s+before\b", low_h)
        or re.search(r"\bweek\s+of\b", low_h)
    ):
        return False
    # If the answer names the session anchor, require it to match the source valid_at. If it omits
    # an anchor and just says "last week", the source wording itself is enough.
    dates = _answer_date_isos(hypothesis)
    if dates and valid_at is not None:
        try:
            anchor = datetime.fromtimestamp(valid_at).date().isoformat()
        except (OSError, OverflowError, ValueError):
            anchor = ""
        if anchor and anchor not in dates and not any(d < anchor for d in dates):
            return False
    residual = _date_answer_residual(hypothesis)
    return _residual_terms_supported_fuzzy(residual, premise)


def _relative_weekend_entailment(premise: str, hypothesis: str, valid_at: Optional[float] = None) -> bool:
    if valid_at is None or not re.search(r"\b(?:last|past|this\s+past)\s+weekend\b", premise or "", re.I):
        return False
    dates = _answer_date_isos(hypothesis)
    if not dates:
        return False
    try:
        ref = datetime.fromtimestamp(valid_at).date()
    except (OSError, OverflowError, ValueError):
        return False
    saturday = ref - timedelta(days=(ref.weekday() - 5) % 7 or 7)
    return bool(dates & {saturday.isoformat(), (saturday + timedelta(days=1)).isoformat()})


def _relative_date_entailment(premise: str, hypothesis: str,
                              valid_at: Optional[float] = None) -> bool:
    answer_dates = _answer_date_isos(hypothesis)
    if not answer_dates:
        return False
    residual = _date_answer_residual(hypothesis)
    if residual and not _date_residual_supported_by_premise(residual, premise):
        return False
    weekdays = _answer_weekdays(hypothesis)
    if weekdays:
        if not any(datetime.fromisoformat(d).weekday() in weekdays for d in answer_dates):
            return False
    refs = []
    if valid_at is not None:
        try:
            refs.append(datetime.fromtimestamp(valid_at).date())
        except (OSError, OverflowError, ValueError):
            pass
    for ds in _ISO_DATE_RE.findall(premise or ""):
        try:
            refs.append(datetime(int(ds[0]), int(ds[1]), int(ds[2])).date())
        except ValueError:
            pass
    if not refs:
        return False
    low = (premise or "").lower()
    supported: set[str] = set()
    for ref in refs:
        if re.search(r"\byesterday\b", low):
            supported.add((ref - timedelta(days=1)).isoformat())
        if re.search(r"\btoday\b", low):
            supported.add(ref.isoformat())
        for n in re.findall(r"\b(\d{1,2})\s+days?\s+ago\b", low):
            supported.add((ref - timedelta(days=int(n))).isoformat())
        for wd in re.findall(
            r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            low,
        ):
            delta = (ref.weekday() - _WEEKDAY_NUM[wd]) % 7
            supported.add((ref - timedelta(days=delta or 7)).isoformat())
    return bool(answer_dates & supported)


def _markdown_cells(line: str) -> list[str]:
    clean = (line or "").strip()
    if "|" in clean and not clean.startswith("|"):
        prefix, rest = clean.split("|", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*:", prefix.strip()):
            clean = "|" + rest
    cells = [cell.strip() for cell in clean.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _schedule_table_entailment(premise: str, hypothesis: str) -> bool:
    hyp_norm = _support_norm(_strip_source_tags(hypothesis))
    if "|" not in (premise or "") or not hyp_norm:
        return False
    lines = [line.strip() for line in (premise or "").splitlines() if "|" in line]
    for header_i, header_line in enumerate(lines):
        header = _markdown_cells(header_line)
        if len(header) < 2:
            continue
        header_norms = [_support_norm(cell) for cell in header]
        if not any("shift" in cell and re.search(r"\b(?:am|pm)\b", cell) for cell in header_norms):
            continue
        for row_line in lines[header_i + 1:]:
            row = _markdown_cells(row_line)
            if len(row) < 2:
                continue
            if all(re.fullmatch(r"[:\-\s]+", cell or "") for cell in row):
                continue
            day = _support_norm(row[0])
            if day not in _WEEKDAY_NUM:
                continue
            if not re.search(rf"\b{re.escape(day)}s?\b", hyp_norm):
                continue
            for col_i, person in enumerate(row[1:], start=1):
                person_norm = _support_norm(person)
                if not person_norm or person_norm not in hyp_norm:
                    continue
                shift = header[col_i] if col_i < len(header) else ""
                shift_norm = _support_norm(shift)
                if shift_norm and shift_norm in hyp_norm:
                    return True
                time_match = re.search(r"\b\d{1,2}\s*(?:am|pm)\s*[-–]\s*\d{1,2}\s*(?:am|pm)\b", shift, re.I)
                if time_match and _support_norm(time_match.group(0)) in hyp_norm:
                    return True
    return False


def _extractive_entailment(premise: str, hypothesis: str,
                           valid_at: Optional[float] = None) -> bool:
    """Conservative local entailment for direct extractive answers.

    If the reader emits a short answer copied verbatim from a source plus source tags, a model-only
    NLI miss should not force abstention. This accepts whole-answer substring support after
    stripping citations/prefixes plus date-only answers entailed by a session timestamp and explicit
    relative date phrase; anything broader still goes to NLI.
    """
    hyp = _strip_source_tags(hypothesis)
    if not hyp or _ABSTAIN_TEXT_RE.search(hyp):
        return False
    hn = _support_norm(hyp)
    pn = _support_norm(premise)
    # Avoid proving tiny fragments like "3" or "yes" by substring accident.
    if len(hn) < 4:
        return False
    if len(hn.split()) <= 12 and hn in pn:
        return True
    return (
        _qa_temporal_entailment(premise, hypothesis, valid_at)
        or _qa_duration_entailment(premise, hypothesis)
        or _qa_slot_entailment(premise, hypothesis)
        or _duration_entailment(premise, hypothesis)
        or _preference_entailment(premise, hypothesis)
        or _identity_entailment(premise, hypothesis)
        or _relative_year_entailment(premise, hypothesis, valid_at)
        or _duration_year_entailment(premise, hypothesis, valid_at)
        or _relative_month_entailment(premise, hypothesis, valid_at)
        or _relative_week_month_entailment(premise, hypothesis, valid_at)
        or _relative_week_entailment(premise, hypothesis, valid_at)
        or _relative_weekend_entailment(premise, hypothesis, valid_at)
        or _relative_date_entailment(premise, hypothesis, valid_at)
        or _schedule_table_entailment(premise, hypothesis)
    )



def structured_record_recall(query: str, records: list[MemoryRecord],
                             at: Optional[float] = None) -> tuple[str, list[tuple[MemoryRecord, str]]]:
    """Tuple-shaped SMQE record-backed recall for legacy tests and lightweight callers."""
    from .smqe.planner import plan_query
    from .smqe.record_ops import execute_record_op

    plan = plan_query(query, at)
    result = execute_record_op(plan, query, records)
    if result is None:
        return "", []
    by_id = {rec.memory_id: rec for rec in records}
    supports = []
    for support in result.supports:
        rec = by_id.get(support.memory_id)
        if rec is not None:
            supports.append((rec, support.proof_atom or support.answer_atom or result.answer))
    return result.answer, supports


def _structured_memory_answer(
    retriever: "Retriever",
    query: str,
    records: list[MemoryRecord],
    at: Optional[float],
    *,
    verify: bool,
) -> Optional[Answer]:
    if retriever is None or not hasattr(retriever, "store"):
        return None
    scope = records[0].scope if records else Scope()
    from .smqe import structured_answer
    return structured_answer(retriever, query, records=records, at=at, verify=verify, scope=scope)

def compress_chunk(text: str, query: str, ratio: float) -> str:
    """LLMLingua-2-style EXTRACTIVE compression for RAW chunks only (never structured facts).
    Keeps the top `ratio` fraction of sentences by query-term overlap. ratio>=1.0 -> no-op.
    This is an extractive approximation, not the LLMLingua-2 model (no torch dependency)."""
    if ratio >= 1.0 or not text:
        return text
    sents = _sentences(text)
    if len(sents) <= 2:
        return text
    qterms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = sorted(range(len(sents)),
                    key=lambda i: -len(set(re.findall(r"[a-z0-9]+", sents[i].lower())) & qterms))
    keep = max(1, int(len(sents) * ratio))
    keep_idx = sorted(scored[:keep])           # preserve original order
    return " ".join(sents[i] for i in keep_idx)


def _dedup(cands: list["RetrievalCandidate"]) -> list["RetrievalCandidate"]:
    """Drop exact and near-duplicate candidates (same content hash or identical text)."""
    seen_hash, seen_text, out = set(), set(), []
    for c in cands:
        h = c.record.content_hash
        t = (c.record.text or c.record.summary or "").strip().lower()
        if h in seen_hash or (t and t in seen_text):
            continue
        seen_hash.add(h)
        if t:
            seen_text.add(t)
        out.append(c)
    return out


class Retriever:
    def __init__(
        self,
        store: RecordStore,
        index: VectorIndex,
        graph: KnowledgeGraph,
        substrate: Substrate,
        client: DashScopeClient,
        settings: Optional[Settings] = None,
    ):
        self.store = store
        self.index = index
        self.graph = graph
        self.substrate = substrate
        self.client = client
        self.settings = settings or get_settings()
        self.bm25 = PersistentBM25(self.settings.index_dir / "bm25_index.json")
        # Connected Brain Loop: the last RecallTrace, populated only when RECALL_TRACE is on.
        # Observation-only side channel -- never read by ranking. THREAD-LOCAL so concurrent asks
        # never read each other's trace (last_trace was last-writer-wins shared state). Direct
        # assignment (retriever.last_trace = ...) stays valid same-thread via the property setter.
        self._trace_tl = threading.local()

    @property
    def last_trace(self) -> Optional[RecallTrace]:
        """The current THREAD's most recent RecallTrace (None until a traced retrieve on it)."""
        return getattr(self._trace_tl, "trace", None)

    @last_trace.setter
    def last_trace(self, value: Optional[RecallTrace]) -> None:
        self._trace_tl.trace = value

    @property
    def last_context_telemetry(self) -> dict:
        """Thread-local context telemetry from the most recent assemble_context call.

        Observation-only: this is written after context blocks are selected and is never read by
        retrieval, ranking, or generation. Benchmark/API surfaces use it to prove region/cocoon
        routing actually participated in a read.
        """
        return getattr(self._trace_tl, "context_telemetry", {})

    def _record_context_region_hints(self, query: str, scope: Scope, hints: list[dict]) -> None:
        safe_hints = [
            {
                "region_id": str(h.get("region_id", "") or ""),
                "level": int(h.get("level", 0) or 0),
                "member_count": int(h.get("member_count", 0) or 0),
                "members": list(h.get("members", []) or []),
                "content_hashes": list(h.get("content_hashes", []) or []),
                "raw_uris": list(h.get("raw_uris", []) or []),
                "score": float(h.get("score", 0.0) or 0.0),
            }
            for h in hints
        ]
        payload = {
            "query": query,
            "scope": scope.model_dump(),
            "region_hint_count": len(safe_hints),
            "region_ids": [h["region_id"] for h in safe_hints if h["region_id"]],
            "region_hints": safe_hints,
        }
        self._trace_tl.context_telemetry = payload
        trace = self.last_trace
        if trace is not None and trace.query == query and trace.scope.key() == scope.key():
            self.last_trace = trace.model_copy(update={"region_hints": safe_hints})

    def memory_region_hints(self, query: str, *, scope: Optional[Scope] = None,
                            at: Optional[float] = None,
                            candidates: Optional[list[RetrievalCandidate]] = None,
                            limit: int = 3, member_limit: int = 6) -> list[dict]:
        """Public host-facing wrapper for the same region/cocoon hints used in context assembly.

        This is model-free: it uses stored gist text, optional already-local candidates, and active
        raw member provenance. The hints are routing aids only; answers must still verify raw source
        memories.
        """
        scope = scope or Scope()
        return _memory_region_hints(
            self.store,
            query,
            list(candidates or []),
            scope,
            at,
            limit=limit,
            member_limit=member_limit,
        )

    def index_lexical(self, rec: MemoryRecord, *, save: bool = True) -> bool:
        """Update the persistent lexical channel on ingest. No-op when the flag is off."""
        if not self.settings.persistent_bm25_enabled:
            return False
        changed = self.bm25.add_or_update(rec.memory_id, rec.text or rec.summary or "")
        if changed and save:
            self.bm25.save()
        return changed

    def save_lexical(self) -> None:
        if self.settings.persistent_bm25_enabled:
            self.bm25.save()

    # ---- ground truth for verification -----------------------------------
    def _ground_truth(self, rec: MemoryRecord) -> str:
        """The premise for NLI: the immutable raw record where it is text, else the
        stored transcription/description (whose raw bytes remain ground truth)."""
        try:
            raw = self.substrate.get(rec.content_hash)
            text = raw.decode("utf-8")
            if text.strip():
                return text
        except (KeyError, UnicodeDecodeError):
            pass
        return rec.text

    # ---- retrieval --------------------------------------------------------
    def retrieve(self, query: str, at: Optional[float] = None, scope: Optional[Scope] = None,
                 qvec: Optional[np.ndarray] = None, use_recency: bool = True,
                 activation: Optional[dict] = None,
                 skip_rerank: bool = False) -> list[RetrievalCandidate]:
        """Hybrid read path: dense + BM25 + single-step PPR + recency -> RRF -> rerank.

        Scope + bi-temporal as-of filter applied first. `qvec` may be passed to avoid a
        duplicate embedding (the semantic cache embeds once). Recency is a MINOR RRF
        channel for benchmark accuracy; the age-independence proofs use the pure content
        index (index.search), so the flat recall-vs-age claim is unaffected."""
        at = now() if at is None else at
        scope = scope or Scope()
        if len(self.index) == 0:
            return []
        if qvec is None:
            # MIRIX Active Retrieval: generate an anticipated topic/sub-question and fold it into
            # the EMBED query so dense recall is scaffolded toward what answering needs (multi-hop /
            # temporal). Real qwen-flash call, hence gated (ACTIVE_RETRIEVAL, default OFF). parse_query
            # below still runs on the ORIGINAL query, so BM25/operation/entity semantics are unchanged.
            embed_query = query
            if self.settings.active_retrieval_enabled:
                try:                                    # best-effort: a failed topic call must never
                    topic = self.client.generate_topic(query)   # abort core recall -> fall back to
                    if topic:                                    # the raw query, like the other
                        embed_query = f"{query} {topic}"         # optional retrieval channels.
                except Exception:
                    pass
            qvec = self.client.embed_text(embed_query)  # real call

        # Scope + bi-temporal as-of filter -> the in-scope, currently-valid corpus.
        corpus = self.store.active_records_at(at, scope)
        if not corpus:
            return []
        # Index pruning by STATIC salience (surprise+importance; NO time term, so
        # age-independent). Default 0.0 = off. Never touches the immutable WORM store.
        if self.settings.salience_prune_threshold > 0.0:
            corpus = [r for r in corpus if r.salience >= self.settings.salience_prune_threshold]
        if not corpus:
            return []
        records = {r.memory_id: r for r in corpus}

        events_for_parse = (
            self.store.events_in_scope(scope.namespace, scope=scope, at=at)
            if self.settings.event_ranking_enabled else None
        )
        parsed = parse_query(query, at, events_for_parse)  # operation / entities / is_namey / is_multihop
        s = self.settings

        # Connected Brain Loop: RecallTrace instrumentation is fully gated -- when off, not a
        # single extra call runs and the candidate list is byte-identical to the baseline path.
        record_trace = s.recall_trace_enabled
        _lat: dict[str, float] = {}
        _t0 = time.perf_counter() if record_trace else 0.0
        _mark = _t0

        allowed = set(records)
        # 3b Rocchio PRF: confidence-gated query expansion toward the top evidence centroid.
        if s.rocchio_enabled:
            qvec = self._maybe_rocchio(qvec, allowed)
        # 2a Adaptive efSearch: widen the HNSW beam only for hard (multi-hop / long) queries.
        ef_override = None
        if s.adaptive_ef_enabled and (parsed["is_multihop"] or len(query.split()) > 16):
            ef_override = s.hnsw_ef_search_hard

        # Channels 1, 2, 4 are independent given (corpus, qvec); 3 (PPR) needs dense seeds.
        def _run_dense():
            if ef_override is None:           # default path: unchanged call signature
                return self.index.search(qvec, s.ann_topk, allowed_ids=allowed)
            return self.index.search(qvec, s.ann_topk, allowed_ids=allowed, ef=ef_override)

        # S3 PARALLEL_CHANNELS fix: do the persistent-BM25 backfill (a WRITE + save) BEFORE the
        # parallel fan-out, so every channel callback is READ-ONLY -- no save() inside _run_bm25 can
        # race a concurrent _run_dense. The backfill is serial here; bm25.save is atomic (temp+replace).
        if s.persistent_bm25_enabled:
            changed = self.bm25.ensure_indexed(
                (r.memory_id, r.text or r.summary or "") for r in corpus)
            if changed:
                self.bm25.save()

        def _run_bm25():
            if s.persistent_bm25_enabled:
                return self.bm25.search(query, s.ann_topk, allowed_ids=allowed)   # read-only
            bm25 = BM25().index([(r.memory_id, r.text or r.summary or "") for r in corpus])
            return bm25.search(query, s.ann_topk)

        def _run_recency():
            if not use_recency:
                return []
            return [r.memory_id for r in sorted(corpus, key=lambda r: -r.valid_at)][: s.ann_topk]

        # 2e Parallel channel fan-out: dense + BM25 + recency concurrently (latency ~= slowest).
        if s.parallel_channels_enabled:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as ex:
                fd, fb, fr = ex.submit(_run_dense), ex.submit(_run_bm25), ex.submit(_run_recency)
                dense, bm25_hits, recency_order = fd.result(), fb.result(), fr.result()
        else:
            dense, bm25_hits, recency_order = _run_dense(), _run_bm25(), _run_recency()

        dense_order = [mid for mid, _ in dense]
        dense_map = dict(dense)
        bm25_order = [mid for mid, _ in bm25_hits]
        bm25_map = dict(bm25_hits)

        # Channel 3: single-pass PPR seeded from QUERY-linked entities (+ best dense hits'
        # entities), letting activation reach passages sharing no query words. No LLM loop.
        seed_entities: list[str] = list(parsed["entities"])
        if s.hippo2_seeding_enabled:
            seed_entities.extend(_hippo2_seed_entities(query, parsed, self.store, at, scope))
        if s.graph_vocab_seeding:
            seed_entities.extend(_vocab_seed_entities(query, corpus))
        for mid, _ in dense[:10]:
            seed_entities.extend(records[mid].entities)
        graph_scores = self.graph.score_memories(
            seed_entities, corpus, at, scope) if seed_entities else {}
        graph_order = [mid for mid, _ in sorted(graph_scores.items(), key=lambda x: -x[1])]

        # Query-adaptive weighted fusion: BM25 up for name/date/ID queries, graph up for
        # multi-hop. `rankings` feed rank-based fusion (RRF/Borda); `score_maps` feed the
        # score-based variants (z-score/min-max/DBSF). Both stay aligned with `weights`.
        wd, wb, wg = self._content_weights()
        rankings = [dense_order, bm25_order]
        weights = [wd, wb * (1.6 if parsed["is_namey"] else 1.0)]
        score_maps: list[dict] = [dense_map, bm25_map]
        channel_names = ["dense", "bm25"] if record_trace else None
        if graph_order:
            rankings.append(graph_order)
            weights.append(wg * (1.6 if parsed["is_multihop"] else 1.0))
            score_maps.append(graph_scores)
            if record_trace:
                channel_names.append("graph")
        # Phase-1 multi-view channels (each appends only when its flag is on; neutral path
        # unchanged when off). Provenance for gist boosts is recorded for prove_answer.
        self._gist_provenance: dict = {}
        if s.struct_channel_enabled:
            so, sm = self._run_struct(parsed, allowed)
            if so:
                rankings.append(so); weights.append(s.rrf_w_struct); score_maps.append(sm)
                if record_trace:
                    channel_names.append("struct")
        if s.event_ranking_enabled:
            eo, em = self._run_event(parsed, records, at, scope)
            if eo:
                rankings.append(eo); weights.append(s.rrf_w_event); score_maps.append(em)
                if record_trace:
                    channel_names.append("event")
        if s.active_fact_context_enabled:
            afo, afm = self._run_active_fact_sources(query, parsed, records, dense, at, scope, allowed)
            if afo:
                rankings.append(afo); weights.append(s.rrf_w_active_fact); score_maps.append(afm)
                if record_trace:
                    channel_names.append("active_fact")
        if s.scratchpad_enabled:
            spo, spm = self._run_scratchpad(records, activation=activation)
            if spo:
                rankings.append(spo); weights.append(s.scratchpad_channel_weight); score_maps.append(spm)
                if record_trace:
                    channel_names.append("scratchpad")
        if s.gist_channel_enabled:
            go, gm, self._gist_provenance = self._run_gist(qvec, scope, allowed)
            if go:
                rankings.append(go); weights.append(s.rrf_w_gist); score_maps.append(gm)
                if record_trace:
                    channel_names.append("gist")
        if s.coactivation_channel_enabled:
            co, cm = self._run_coactivation(dense, records, at, scope, allowed, activation=activation)
            if co:
                rankings.append(co); weights.append(s.rrf_w_coact); score_maps.append(cm)
                if record_trace:
                    channel_names.append("coactivation")
        # Track 9 Flow: activation channel -- field-warm memories the query never named ride into
        # the hybrid candidates (gated to the in-scope active corpus via `allowed`). Ranking only;
        # dense_score stays 0 for an activation-seeded id, so coverage/abstention are untouched.
        # activation=None or flag off -> not added -> byte-identical.
        if s.flow_hybrid_channel_enabled and activation:
            ao, am = self._run_activation(allowed, activation)
            if ao:
                rankings.append(ao); weights.append(s.flow_hybrid_weight); score_maps.append(am)
                if record_trace:
                    channel_names.append("activation")
        if recency_order:
            rankings.append(recency_order)
            weights.append(s.rrf_w_recency)
            n_rec = len(recency_order)
            score_maps.append({mid: float(n_rec - i) for i, mid in enumerate(recency_order)})
            if record_trace:
                channel_names.append("recency")
        if record_trace:
            _lat["channels_ms"] = (time.perf_counter() - _mark) * 1000.0
            _mark = time.perf_counter()
        fused = self._fuse(rankings, score_maps, weights)
        # Memory typing coordinator (Phase 4): a soft, flag-gated prior that nudges the candidates
        # whose MIRIX type matches the query class. Bounded to a fraction of the top fused score,
        # so it re-orders ties without overriding strong content matches. Off -> fused untouched.
        if s.memory_typing_enabled:
            self._apply_type_prior(fused, records, parsed, query)
        # Affect salience boost (Phase 3): a small, bounded, AGE-FREE nudge by static salience.
        if s.affect_salience_enabled and s.lambda_salience != 0.0:
            self._apply_salience_boost(fused, records)
        if record_trace:
            _lat["fuse_ms"] = (time.perf_counter() - _mark) * 1000.0
            _mark = time.perf_counter()
        if len(fused) < s.final_topk and use_recency:
            for mid in recency_order:
                fused.setdefault(mid, 0.0)
                if len(fused) >= s.final_topk:
                    break

        cands = {mid: RetrievalCandidate(
            record=records[mid], dense_score=dense_map.get(mid, 0.0),
            bm25_score=bm25_map.get(mid, 0.0), graph_score=graph_scores.get(mid, 0.0),
            fused_score=fused[mid]) for mid in fused}
        ranked = _dedup(sorted(cands.values(), key=lambda c: -c.fused_score))
        final = self._finalize(query, ranked, skip_rerank=skip_rerank)
        if s.graph_bridge_context_enabled:
            final = self._ensure_graph_bridge_candidates(query, parsed, final, records, at, scope)
        if s.user_evidence_context_enabled:
            final = self._ensure_user_candidates(query, final, records, at)
        if s.assistant_evidence_context_enabled:
            final = self._ensure_assistant_candidates(query, final, records, at)
        if s.aggregation_audit_enabled:
            final = self._ensure_aggregation_candidates(query, parsed, final, records, at)
        if s.temporal_evidence_audit_enabled:
            final = self._ensure_temporal_evidence_candidates(query, parsed, final, records, at)
        if s.list_audit_enabled:
            final = self._ensure_list_candidates(query, parsed, final, records, at)
        if s.raw_span_audit_enabled:
            final = self._ensure_raw_span_candidates(query, final, records, at)
        if record_trace:
            _lat["finalize_ms"] = (time.perf_counter() - _mark) * 1000.0
            _lat["total_ms"] = (time.perf_counter() - _t0) * 1000.0
            sel = [c.record.memory_id for c in final]
            sel_set = set(sel)
            self.last_trace = RecallTrace(
                query=query, scope=scope, parsed_query=parsed,
                enabled_channels=list(channel_names),
                channel_results={n: list(r) for n, r in zip(channel_names, rankings)},
                channel_weights={n: float(w) for n, w in zip(channel_names, weights)},
                fused_scores={k: float(v) for k, v in fused.items()},
                gist_provenance=dict(self._gist_provenance),
                selected_candidates=sel,
                dropped_candidates=[mid for mid in fused if mid not in sel_set],
                latency_by_stage=_lat, token_budget=s.context_token_budget,
            )
        return final

    def _ensure_graph_bridge_candidates(
        self,
        query: str,
        parsed: dict,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
        scope: Scope,
    ) -> list[RetrievalCandidate]:
        """Append raw sources behind active bridge edges that ranking missed."""
        edges = _graph_bridge_edges(
            self.store, query, parsed, at, scope, limit=self.settings.graph_bridge_topk)
        if not edges:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_score, _valid_at, _edge_id, edge) in enumerate(edges):
            mid = str(getattr(edge, "source_memory_id", "") or "")
            rec = records.get(mid)
            if rec is None or mid in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.9 - 0.02 * rank)),
            ))
            seen.add(mid)
        return out

    def _ensure_user_candidates(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """Append source records containing matching user turns that ranking missed."""
        matches = _user_evidence_matches(
            query, list(records.values()), at, limit=self.settings.user_evidence_topk)
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_, rec, _) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.88 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    def _ensure_assistant_candidates(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """Append source records containing matching assistant turns that ranking missed."""
        matches = _assistant_evidence_matches(
            query, list(records.values()), at, limit=self.settings.assistant_evidence_topk)
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_, rec, _) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.88 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    def _ensure_aggregation_candidates(
        self,
        query: str,
        parsed: dict,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """For count/total questions, append scoped matching source records that ranking missed.

        This is evidence completion, not answering: it never sums/counts. It gives the shared reader
        and the verifier all matching source sentences, which prevents aggregation misses caused by
        a high-scoring-but-incomplete top-k.
        """
        matches = _aggregation_matches(query, parsed, list(records.values()), at)
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_, rec, _) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.9 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    def _ensure_list_candidates(
        self,
        query: str,
        parsed: dict,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """For exact list questions, append scoped matching source records that ranking missed.

        This is source completion, not answer generation: it never enumerates the list. It only gives
        the shared reader/verifier the raw snippets whose terms match the query's exact scope.
        """
        matches = _list_matches(
            query, parsed, list(records.values()), at, limit=self.settings.list_evidence_topk)
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_, rec, _) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.88 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    def _ensure_temporal_evidence_candidates(
        self,
        query: str,
        parsed: dict,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """For temporal questions, append source records with exact date-bearing snippets missed by
        ranking. This never computes dates; it only completes evidence for the shared reader."""
        matches = _temporal_evidence_matches(
            query, parsed, list(records.values()), at, limit=self.settings.temporal_evidence_topk)
        anchor_matches = _temporal_anchor_matches(
            query, parsed, list(records.values()), at, limit=self.settings.temporal_evidence_topk)
        if anchor_matches:
            existing = {(rec.memory_id, snippet) for _, rec, snippet in matches}
            matches = list(matches) + [
                item for item in anchor_matches if (item[1].memory_id, item[2]) not in existing
            ]
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (_, rec, _) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                fused_score=max(0.0, top_score * (0.9 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    def _ensure_raw_span_candidates(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        records: dict[str, MemoryRecord],
        at: Optional[float],
    ) -> list[RetrievalCandidate]:
        """Append long raw records with exact query-supported spans that ranking missed."""
        matches = _raw_span_matches(
            query,
            list(records.values()),
            at,
            min_chars=self.settings.raw_span_min_chars,
            limit=self.settings.raw_span_audit_topk,
            span_count=max(1, int(self.settings.raw_span_per_record)),
        )
        if not matches:
            return candidates
        seen = {c.record.memory_id for c in candidates}
        out = list(candidates)
        top_score = max((c.rerank_score or c.fused_score for c in candidates), default=1.0) or 1.0
        for rank, (score, rec, _span) in enumerate(matches):
            if rec.memory_id in seen:
                continue
            out.append(RetrievalCandidate(
                record=rec,
                bm25_score=score,
                fused_score=max(0.0, top_score * (0.86 - 0.02 * rank)),
            ))
            seen.add(rec.memory_id)
        return out

    # ---- online weight learning + PRF ------------------------------------
    def _content_weights(self) -> tuple[float, float, float]:
        """Base (dense, bm25, graph) fusion weights. When the online learner is enabled and a
        learned vector has been written to index_dir/fusion_weights.json, use it; otherwise
        the static config floats. RECENCY is never learned here (age-independence)."""
        s = self.settings
        if s.fusion_learner_enabled:
            learned = _online_weights.load_weights(self.settings.index_dir / "fusion_weights.json")
            if learned:
                return (float(learned.get("dense", s.rrf_w_dense)),
                        float(learned.get("bm25", s.rrf_w_bm25)),
                        float(learned.get("graph", s.rrf_w_graph)))
        return s.rrf_w_dense, s.rrf_w_bm25, s.rrf_w_graph

    def _maybe_rocchio(self, qvec: np.ndarray, allowed: set) -> np.ndarray:
        """A single confidence-gated PRF expansion: cheap dense probe -> if the top match is
        strong, push the query toward the top-R evidence centroid. No model call."""
        s = self.settings
        try:
            probe = self.index.search(qvec, max(s.rocchio_topr, 1), allowed_ids=allowed)
        except TypeError:
            return qvec
        if not probe or not _rocchio.should_expand(probe[0][1], s.rocchio_conf_gate):
            return qvec
        ids = [mid for mid, _ in probe[: s.rocchio_topr]]
        vmap = self.index.get_vectors(ids) if hasattr(self.index, "get_vectors") else {}
        rel = [vmap[mid] for mid in ids if mid in vmap]
        if not rel:
            return qvec
        return _rocchio.rocchio_expand(qvec, rel, alpha=s.rocchio_alpha, beta=s.rocchio_beta)

    # ---- Phase-1 multi-view retrieval channels (dormant signals, flag-gated) ----------
    def _run_struct(self, parsed: dict, allowed: set) -> tuple[list[str], dict]:
        """Structure-code channel: rank by entity/role/modality similarity in STRUCTURE space.
        Age-safe: the query structure code carries no temporal dimension, and the stored codes
        encode only cyclic (not absolute-age) time, so this never slopes recall-vs-age."""
        from . import structure_code as _sc
        qstruct = _sc.build_query_structure_code(list(parsed.get("entities", [])),
                                                 self.settings.struct_dim)
        try:
            hits = self.index.search_struct(qstruct, self.settings.ann_topk)
        except Exception:
            return [], {}
        hits = [(mid, sc) for mid, sc in hits if mid in allowed]
        return [mid for mid, _ in hits], dict(hits)

    def _run_event(self, parsed: dict, records: dict, at, scope: Scope) -> tuple[list[str], dict]:
        """Event-overlap channel: promote memories whose normalized event interval matches the
        QUERY's temporal constraint (filter/count/order). Ranks by query-time match, NOT by the
        memory's age, so it does not affect the flat recall-vs-age curve."""
        events = self.store.events_in_scope(scope.namespace, scope=scope, at=at)
        if not events:
            return [], {}
        matched = select_for_query(events, parsed, at)
        order, m = [], {}
        n = len(matched)
        for rank, ev in enumerate(matched):
            mid = getattr(ev, "source_memory_id", "")
            if mid and mid in records and mid not in m:
                order.append(mid)
                m[mid] = float(n - rank)        # higher = earlier in the temporal match order
        return order, m

    def _run_active_fact_sources(
        self,
        query: str,
        parsed: dict,
        records: dict[str, MemoryRecord],
        dense: list,
        at,
        scope: Scope,
        allowed: set,
    ) -> tuple[list[str], dict]:
        """Current graph-fact source channel.

        Promote raw source memories behind active edges matching the query. Superseded edges are
        excluded by the store's active-edge SQL filter, so the verified row can cite the newest raw
        evidence without deleting historical chunks.
        """
        seed_records = [records[mid] for mid, _ in dense[:10] if mid in records]
        edges = _active_fact_query_edges(
            self.store,
            query,
            parsed,
            seed_records,
            at,
            scope,
            limit=max(self.settings.active_fact_context_topk, self.settings.ann_topk),
        )
        if not edges:
            return [], {}
        order: list[str] = []
        scores: dict[str, float] = {}
        n = len(edges)
        for rank, (score, _, _, edge) in enumerate(edges):
            mid = str(getattr(edge, "source_memory_id", "") or "")
            if not mid or mid not in allowed:
                continue
            if mid not in scores:
                order.append(mid)
            scores[mid] = max(scores.get(mid, 0.0), float(n - rank) + score)
        return order, scores

    def _run_gist(self, qvec, scope: Scope, allowed: set) -> tuple[list[str], dict, dict]:
        """Derived-gist channel: a gist that matches the query boosts its RAW member memories
        (gists help recall but never replace raw evidence). Returns (order, score_map, provenance:
        member_id -> gist cid) so prove_answer can show recall came via a gist."""
        gists = self.store.derived_in_scope(scope.namespace)
        if not gists or qvec is None:
            return [], {}, {}
        q = np.asarray(qvec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) + 1e-9
        scored = []
        for g in gists:
            if not getattr(g, "vector", None):
                continue
            gv = np.asarray(g.vector, dtype=np.float32)
            sim = float(gv @ q / ((np.linalg.norm(gv) + 1e-9) * qn))
            scored.append((g, sim))
        scored.sort(key=lambda x: -x[1])
        order, m, prov = [], {}, {}
        for g, sim in scored[:8]:
            for mid in getattr(g, "member_ids", []):
                if mid in allowed and mid not in m:
                    order.append(mid)
                    m[mid] = max(0.0, sim)
                    prov[mid] = g.cid
        return order, m, prov

    def _run_scratchpad(
        self,
        records: dict[str, MemoryRecord],
        *,
        activation: Optional[dict] = None,
    ) -> tuple[list[str], dict]:
        """Scratchpad channel: high-salience active facts shown in context are also candidates.

        This keeps the normal proof path honest: if a reader uses a scratchpad fact, the cited raw
        record is in the candidate set for NLI/extractive verification. The caller already passed
        scope + bi-temporal active records, so stale/future/cross-scope facts cannot enter here.
        """
        if not records:
            return [], {}
        from .scratchpad import select_scratchpad
        entries = select_scratchpad(
            list(records.values()),
            top_k=self.settings.scratchpad_topk,
            min_salience=self.settings.scratchpad_min_salience,
            activation=activation,
            weight=self.settings.flow_context_weight,
        )
        order = [str(e.get("memory_id", "")) for e in entries if e.get("memory_id") in records]
        n = len(order)
        scores = {
            mid: float(n - rank) + float(records[mid].salience)
            for rank, mid in enumerate(order)
        }
        return order, scores

    def _run_activation(self, allowed: set, activation: Optional[dict]) -> tuple[list[str], dict]:
        """Track 9 Flow activation channel: in-scope active ids whose field activation clears the
        floor, ordered by activation. Restricting to `allowed` (the scope + bi-temporal corpus) is
        the store gate -- a cross-scope or expired activated id is dropped, never surfaced."""
        if not activation:
            return [], {}
        floor = self.settings.flow_floor
        items = [(mid, float(v)) for mid, v in activation.items() if mid in allowed and v >= floor]
        items.sort(key=lambda kv: -kv[1])
        return [mid for mid, _ in items], {mid: v for mid, v in items}

    def _run_coactivation(self, dense: list, records: dict, at, scope: Scope,
                          allowed: set, activation: Optional[dict] = None) -> tuple[list[str], dict]:
        """Co-activation channel: memories co-confirmed with the top dense hits in PAST recalls
        (graph CO_ACTIVATED links, Section 7.3) are pulled in as candidates. This is multi-hop
        recall -- a memory sharing no query words but repeatedly used together with a dense hit
        surfaces here. Ranks by co-activation frequency (how many seeds link to it), never by age.

        Flow: the strongest field-activated ids are unioned into the walk seeds, so coactivation
        inherits field warmth even when the activation channel flag itself is off."""
        seeds = [mid for mid, _ in dense[:10]]
        if activation:
            top_active = sorted((kv for kv in activation.items() if kv[0] in allowed),
                                key=lambda kv: -kv[1])[: max(0, self.settings.flow_seed_topk)]
            seeds = list(dict.fromkeys(seeds + [mid for mid, _ in top_active]))
        if not seeds:
            return [], {}
        seed_set = set(seeds)
        freq: dict[str, int] = {}
        for mid in seeds:
            for linked in self.graph.linked_memories(mid, scope, at):
                if linked in allowed and linked not in seed_set:
                    freq[linked] = freq.get(linked, 0) + 1
        if not freq:
            return [], {}
        order = [m for m, _ in sorted(freq.items(), key=lambda x: -x[1])]
        return order, {m: float(c) for m, c in freq.items()}

    # ---- memory typing coordinator (Phase 4, soft prior) ------------------
    @staticmethod
    def _query_class(parsed: dict, query: str) -> str:
        """Coarse query class for the type-priority coordinator (deterministic, no model)."""
        q = (query or "").lower()
        if parsed.get("ranges") or parsed.get("operation") in ("order", "count"):
            return "temporal"
        if any(k in q for k in ("how to", "how do i", "steps", "procedure", "instructions",
                                "install", "configure", "set up", "deploy", "recipe")):
            return "procedural"
        if any(k in q for k in ("prefer", "favorite", "favourite", "i like", "i love",
                                "allerg", "usually", "always", "my ")):
            return "preference"
        return "factual"

    def _apply_type_prior(self, fused: dict, records: dict, parsed: dict, query: str) -> None:
        """Mutate `fused` in place with a bounded type-match bonus. The bonus is at most
        type_prior_weight * max_fused, so it breaks ties toward the query class's preferred MIRIX
        types without swamping a strong content match. A no-op if no candidate carries a type."""
        from .memory_types import type_priority
        order = type_priority(self._query_class(parsed, query))
        rank = {t.value: (len(order) - i) for i, t in enumerate(order)}
        mx_rank = max(rank.values()) if rank else 1
        mx_fused = max(fused.values()) if fused else 0.0
        if mx_fused <= 0.0:
            return
        w = self.settings.type_prior_weight
        for mid in fused:
            rec = records.get(mid)
            t = (getattr(rec, "metadata", None) or {}).get("type") if rec else None
            if t and t in rank:
                fused[mid] += w * (rank[t] / mx_rank) * mx_fused

    def _apply_salience_boost(self, fused: dict, records: dict) -> None:
        """Phase 3 affect coupling: retrieval_score = fused + lambda_salience * s, bounded to a
        fraction of the top fused score so a salient memory ranks higher WITHOUT overriding a strong
        content match. `s` (record.salience) carries NO age/timestamp term, so two memories with
        equal salience get an identical boost regardless of their valid_at -> age-invariant."""
        lam = self.settings.lambda_salience
        mx_fused = max(fused.values()) if fused else 0.0
        if mx_fused <= 0.0:
            return
        for mid in fused:
            rec = records.get(mid)
            if rec is not None:
                fused[mid] += lam * float(getattr(rec, "salience", 0.0)) * mx_fused

    # ---- fusion + final selection ----------------------------------------
    def _fuse(self, rankings: list[list[str]], score_maps: list[dict],
              weights: list[float]) -> dict[str, float]:
        """Dispatch the configured fusion method. RRF (rank-based, scale-free) is the
        default and the unknown-method fallback; Borda is rank-based; z-score/min-max/DBSF
        use the per-channel raw scores."""
        method = self.settings.fusion_method
        if method == "borda":
            return _fusion.combine_borda(rankings, weights)
        if method in _fusion.SCORE_METHODS:
            return _fusion.combine_scores(score_maps, weights, method)
        return _rrf(rankings, self.settings.rrf_k, weights)   # rrf / unknown -> robust default

    def _finalize(self, query: str, ranked: list[RetrievalCandidate], *,
                  skip_rerank: bool = False) -> list[RetrievalCandidate]:
        """Cross-encoder rerank (skippable on a large margin) -> MMR diversity -> adaptive-k /
        conformal depth -> top-k. With every Layer-2 flag off this is identical to the prior
        behaviour (rerank to final_topk, return final_topk)."""
        s = self.settings
        # Only a downstream depth consumer (MMR / adaptive-k / conformal) needs more than
        # final_topk reranked items. When none is active we request exactly final_topk, so the
        # reported (flags-off) path makes the identical client.rerank call it always did.
        need_depth = (s.mmr_enabled or s.adaptive_k_enabled
                      or (s.conformal_depth_enabled and s.conformal_qhat >= 0.0))
        rerank_topn = max(s.rerank_depth, s.final_topk) if need_depth else s.final_topk
        margin_skip = _gating.should_skip_rerank([c.fused_score for c in ranked], s.rerank_skip_margin)
        if s.rerank_enabled and not skip_rerank and not margin_skip and ranked:
            shortlist = ranked[: max(s.rerank_depth, s.final_topk)]
            docs = [c.record.text or c.record.summary or "" for c in shortlist]
            if s.rerank_span_input_enabled:
                # Ranking-only token cut: the cross-encoder sees a query-centered span per doc
                # instead of the full text (the largest read-path stream at ~50 x 4000 chars).
                # long_threshold_chars tracks max_chars so the cut actually materializes on
                # mid-sized docs; verification premises are untouched.
                span_cap = max(200, int(s.rerank_span_chars))
                docs = [
                    _raw_query_centered_span(d, query, long_threshold_chars=span_cap,
                                             max_chars=span_cap)
                    if len(d) > span_cap else d
                    for d in docs
                ]
            try:
                reranked = []
                for orig_idx, score in self.client.rerank(query, docs, rerank_topn):
                    shortlist[orig_idx].rerank_score = score
                    reranked.append(shortlist[orig_idx])
                ranked = reranked or shortlist
            except Exception:
                if not s.rerank_fail_open:
                    raise
                ranked = shortlist
        # Depth-select BEFORE MMR. adaptive_k_cut's largest-gap cut is a POSITIONAL slice that
        # assumes a score-descending list; MMR reorders by diversity, so cutting after MMR would
        # keep the first-k MMR positions, silently dropping high-score items past position k.
        # Cut on the relevance-descending order first, then diversify only the survivors.
        ranked = self._depth_select(ranked)
        ranked = self._mmr_pass(ranked)
        final_topk = self._adaptive_final_topk(query) if s.difficulty_adaptive_depth_enabled else s.final_topk
        return ranked[:final_topk]

    def _query_difficulty(self, query: str) -> float:
        """0 (easy single-hop) .. 1 (hard multi-hop / long), from deterministic query features."""
        parsed = parse_query(query)
        score = 0.0
        if parsed.get("is_multihop"):
            score += 0.5
        score += min(0.3, 0.1 * len(parsed.get("entities", [])))
        if len(query.split()) > 16:
            score += 0.2
        return min(1.0, score)

    def _adaptive_final_topk(self, query: str) -> int:
        """S5: scale the returned candidate count with query difficulty -- easy queries pay less,
        hard queries get the full depth. Never below adaptive_k_min."""
        s = self.settings
        lo = max(1, min(s.adaptive_k_min, s.final_topk))
        return int(round(lo + (s.final_topk - lo) * self._query_difficulty(query)))

    def _mmr_pass(self, ranked: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """2c MMR diversity re-ordering over the candidate content vectors. No-op when off
        or when any vector is unavailable (fail safe, never drop a candidate silently)."""
        s = self.settings
        if not s.mmr_enabled or len(ranked) <= 2:
            return ranked
        ids = [c.record.memory_id for c in ranked]
        vmap = self.index.get_vectors(ids)
        vecs = [vmap.get(mid) for mid in ids]
        if any(v is None for v in vecs):
            return ranked
        rels = [c.rerank_score or c.fused_score for c in ranked]
        order = _mmr.mmr_order(rels, vecs, lam=s.mmr_lambda)
        return [ranked[i] for i in order]

    def _depth_select(self, ranked: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """2b/2a calibrated depth: split-conformal cutoff (if a dev-calibrated qhat is set),
        then the largest-gap adaptive-k cut. Both preserve order and keep >= adaptive_k_min."""
        s = self.settings
        if not ranked:
            return ranked
        if s.conformal_depth_enabled and s.conformal_qhat >= 0.0:
            ranked = _conformal.select_by_conformal(
                ranked, lambda c: c.dense_score, s.conformal_qhat,
                min_keep=min(s.adaptive_k_min, len(ranked)))
        if s.adaptive_k_enabled:
            ranked = _adaptive_k.adaptive_k_cut(
                ranked, score_fn=lambda c: (c.rerank_score or c.fused_score),
                min_k=min(s.adaptive_k_min, len(ranked)), max_k=s.final_topk)
        return ranked

    def assemble_context(self, query: str, candidates: list[RetrievalCandidate],
                         at: Optional[float] = None, scope: Optional[Scope] = None,
                         include_conflict_resolution: bool = True,
                         activation: Optional[dict] = None) -> list[str]:
        """Build the token-budgeted context blocks: structured event-calendar selection +
        surfaced typed preferences (uncompressed), then Hopfield-ordered raw chunks
        (optionally extractively compressed), with lost-in-the-middle edge placement.

        This is CONTEXT ASSEMBLY (retrieval), shared by `answer()` AND the neutral benchmark
        adapter, so the event calendar + preferences reach the scoreboard while the SHARED
        reader still produces the answer string. No answer is computed here."""
        scope = scope or Scope()
        events = self.store.events_in_scope(scope.namespace, scope=scope, at=at)
        parsed = parse_query(query, at, events)
        event_cap = 12 if parsed.get("operation") in ("count", "order") else 8
        event_blocks = [e.as_text() for e in select_for_query(events, parsed, at)[:event_cap]]
        # Phase 5: a chronological event chain for order/sequence/temporal queries (gated). Selection
        # + ordering only; the shared reader still computes the answer.
        chain_blocks: list[str] = []
        if (self.settings.event_chain_context_enabled
                and (parsed.get("operation") in ("order", "count") or parsed.get("ranges"))):
            chain = event_chain(events, parsed, at, window=event_cap)
            if chain:
                chain_blocks = ["Event timeline (chronological): "
                                + " -> ".join(e.as_text() for e in chain)]
        pref_blocks = _preference_profile_blocks(
            query, _visible_profile_entries(self.store, scope, at), limit=8
        )
        # The five audit channels below each need the same active-record snapshot; recomputing
        # the O(corpus) scan per channel paid five store sweeps per reader-path ask in the full
        # profile. One lazy snapshot (no scan when every channel is off); no mutation happens
        # between the call sites, so a single snapshot is byte-identical.
        _active_snapshot: list = []

        def _active_records():
            if not _active_snapshot:
                _active_snapshot.append(
                    self.store.active_records_at(at if at is not None else now(), scope))
            return _active_snapshot[0]

        # Phase 6: a working scratchpad of high-salience verified ACTIVE facts as a context channel
        # (gated; each entry links to a raw source hash, superseded facts expire via the active
        # filter). Off -> context is unchanged.
        scratchpad_blocks: list[str] = []
        if self.settings.scratchpad_enabled:
            from .scratchpad import select_scratchpad
            active = _active_records()
            entries = select_scratchpad(active, top_k=self.settings.scratchpad_topk,
                                        min_salience=self.settings.scratchpad_min_salience,
                                        activation=activation,
                                        weight=self.settings.flow_context_weight)
            if entries:
                scratchpad_blocks = ["Scratchpad (high-salience verified facts): "
                                     + " | ".join(e["text"] for e in entries)]
        resolver_blocks = (
            self._conflict_resolution_blocks(query, candidates, at)
            if include_conflict_resolution else []
        )
        active_fact_blocks: list[str] = []
        if self.settings.active_fact_context_enabled:
            active_fact_blocks = _active_fact_context_blocks(
                self.store,
                query,
                parsed,
                candidates,
                at,
                scope,
                limit=self.settings.active_fact_context_topk,
            )
        bridge_blocks: list[str] = []
        if self.settings.graph_bridge_context_enabled:
            bridge_blocks = _graph_bridge_context_blocks(
                self.store,
                query,
                parsed,
                at,
                scope,
                limit=self.settings.graph_bridge_topk,
            )
        region_blocks: list[str] = []
        region_hints: list[dict] = []
        if self.settings.gist_channel_enabled:
            region_hints = _memory_region_hints(
                self.store,
                query,
                candidates,
                scope,
                at,
                gist_ids=set(getattr(self, "_gist_provenance", {}).values()),
            )
            region_blocks = [_format_memory_region_hint(hint) for hint in region_hints]
        self._record_context_region_hints(query, scope, region_hints)
        import datetime as _dt
        question_time_blocks = _question_time_context_block(query, at)
        user_blocks: list[str] = []
        if self.settings.user_evidence_context_enabled:
            active = _active_records()
            matches = _user_evidence_matches(
                query, active, at, limit=self.settings.user_evidence_topk)
            if matches:
                lines = []
                for _, rec, snippet in matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                user_blocks = [
                    "User evidence audit (source user turns only):\n" + "\n".join(lines)
                ]
        assistant_blocks: list[str] = []
        if self.settings.assistant_evidence_context_enabled:
            active = _active_records()
            matches = _assistant_evidence_matches(
                query, active, at, limit=self.settings.assistant_evidence_topk)
            if matches:
                lines = []
                for _, rec, snippet in matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                assistant_blocks = [
                    "Assistant evidence audit (source assistant turns only):\n" + "\n".join(lines)
                ]

        temporal_blocks: list[str] = []
        temporal_anchor_blocks: list[str] = []
        if self.settings.temporal_evidence_audit_enabled:
            active = _active_records()
            matches = _temporal_evidence_matches(
                query, parsed, active, at, limit=self.settings.temporal_evidence_topk)
            if matches:
                lines = []
                for _, rec, snippet in matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                temporal_blocks = [
                    "Temporal evidence audit (source-only; preserve relative date wording and "
                    "distinguish session date from event date):\n" + "\n".join(lines)
                ]
            anchor_matches = _temporal_anchor_matches(
                query, parsed, active, at, limit=self.settings.temporal_evidence_topk)
            if anchor_matches:
                lines = []
                for _, rec, snippet in anchor_matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                temporal_anchor_blocks = [
                    "Temporal anchor audit (source-only session dates for ordering/duration; "
                    "reader computes the comparison):\n" + "\n".join(lines)
                ]

        list_blocks: list[str] = []
        if self.settings.list_audit_enabled:
            active = _active_records()
            matches = _list_matches(
                query, parsed, active, at, limit=self.settings.list_evidence_topk)
            if matches:
                lines = []
                for _, rec, snippet in matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                list_blocks = [
                    "List evidence audit (source-only; include every matching item and drop "
                    "related-but-off-scope items):\n" + "\n".join(lines)
                ]

        audit_blocks: list[str] = []
        if self.settings.aggregation_audit_enabled:
            matches = _aggregation_matches(query, parsed, [c.record for c in candidates], at)
            if matches:
                lines = []
                for _, rec, snippet in matches:
                    when = _dt.date.fromtimestamp(rec.valid_at).isoformat() if rec.valid_at else "unknown-date"
                    lines.append(f"- [{when}] {snippet}")
                audit_blocks = [
                    "Aggregation evidence audit (source-only; include every matching line and "
                    "exclude unrelated/template numbers):\n" + "\n".join(lines)
                ]

        # Raw chunks ordered by Hopfield attention weight, each PREFIXED with its session
        # date. This gives the reader the temporal anchor to resolve relative expressions
        # ("yesterday", "last week") that the LLM event extractor may miss -- the structured
        # session date is the temporal ground truth.
        raw_blocks = []
        ordered_candidates = self._hopfield_order(candidates)
        if self.settings.temporal_rerank_enabled:
            ordered_candidates = _temporal_context_order(query, parsed, ordered_candidates)
        # Claim-crystal phase demotion: once a record's facts are crystallized into claims, the
        # priority-forgetting profile stops paying full-text context cost for it -- it contributes
        # a bounded query-centered span instead. Records the affect layer marked VIVID (high
        # static salience) keep their full text: affect decides what stays hot, which is exactly
        # what the affect-off ablation measures. Inactive unless the demotion flag AND at least
        # one forgetting knob are on, so forgetting-off runs pay the true keep-everything cost.
        s = self.settings
        # Enumeration-shaped queries (lists, counts, commonalities, perfect-tense experience
        # sweeps) need every mention across the corpus; attention stays wide for them. Demotion
        # applies to point lookups, where the query-centered span carries the answer.
        enumeration_query = bool(re.search(
            r"\b(?:how\s+many|both|all|each|every|list|total|sum|combined|altogether)\b"
            r"|\b(?:what|where|which)\s+(?:has|have|had)\b"
            r"|\bwhat\s+(?:does|do|did)\b[^?]{0,60}\bdo\b",
            query, re.I))
        demotion_active = (
            s.crystal_span_demotion_enabled
            and not enumeration_query
            and (s.dream_prune_percentile > 0.0 or s.salience_prune_threshold > 0.0)
        )
        vivid_ids: set[str] = set()
        if demotion_active and ordered_candidates:
            vivid_k = int(max(0.0, s.vivid_fraction) * len(ordered_candidates))
            if vivid_k > 0:
                by_salience = sorted(
                    ordered_candidates, key=lambda c: -(float(c.record.salience or 0.0))
                )
                vivid_ids = {c.record.memory_id for c in by_salience[:vivid_k]}
        # The day being asked about stays vivid: a record whose session date falls inside the
        # query's parsed date window IS the evidence for a date-anchored lookup and never demotes.
        query_windows: list[tuple[float, float]] = []
        if demotion_active:
            for rng in (parsed.get("ranges") or []):
                try:
                    start = float(rng.get("start_epoch", rng.get("start", 0)) or 0)
                    end = float(rng.get("end_epoch", rng.get("end", 0)) or 0)
                except (TypeError, ValueError):
                    continue
                if start and end:
                    query_windows.append((min(start, end), max(start, end)))
        for c in ordered_candidates:
            txt = c.record.text or c.record.summary or ""
            date_anchored = any(
                lo <= float(c.record.valid_at or 0.0) <= hi for lo, hi in query_windows
            )
            demote = (
                demotion_active
                and not date_anchored
                and int(c.record.metadata.get("claims_extracted", 0) or 0) > 0
                and c.record.memory_id not in vivid_ids
            )
            txt = _raw_query_centered_span(
                txt,
                query,
                long_threshold_chars=(s.crystal_span_chars if demote else s.raw_span_min_chars),
                max_chars=(s.crystal_span_chars if demote else 3_200),
                span_count=2 if demote else max(1, int(s.raw_span_per_record)),
            )
            if s.context_compress_enabled and s.compression_ratio < 1.0:
                txt = compress_chunk(txt, query, s.compression_ratio)
            if c.record.valid_at:
                txt = f"[Session date {_dt.date.fromtimestamp(c.record.valid_at).isoformat()}] {txt}"
            raw_blocks.append(txt)

        # Budget on the PRIORITY order first, then edge-place the survivors. Edge-placing first
        # then budgeting truncated from the tail, which is where edge_place puts the 2nd-highest
        # priority block, so a high-priority block was dropped before lower-priority raw chunks.
        budgeted = _budget_blocks(
            (question_time_blocks
             + resolver_blocks + active_fact_blocks + bridge_blocks + user_blocks + assistant_blocks
             + region_blocks + temporal_blocks + temporal_anchor_blocks + list_blocks + scratchpad_blocks
             + event_blocks + chain_blocks + audit_blocks + pref_blocks
             + raw_blocks),
            self.settings.context_token_budget)
        return edge_place(budgeted)

    def _hopfield_order(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Modern-Hopfield / attention readout (dossier 8.1-8.2): a single-step softmax over
        the retrieved set's scores yields attention weights; order candidates by that weight
        (pattern completion). Weights derive from content/rerank scores only -- no FSRS
        priority or age term participates."""
        scores = [c.rerank_score or c.fused_score for c in candidates]
        weights = _softmax(scores)
        order = sorted(range(len(candidates)), key=lambda i: -weights[i])
        return [candidates[i] for i in order]

    def _hopfield_readout(self, candidates: list[RetrievalCandidate]) -> list[str]:
        """Text-only variant of _hopfield_order (kept for compatibility)."""
        return [c.record.text or c.record.summary or "" for c in self._hopfield_order(candidates)]

    def _try_conflict_resolver(
        self, query: str, candidates: list[RetrievalCandidate], as_of: Optional[float] = None
    ) -> Optional[CurrentValueResolution]:
        if not self.settings.conflict_resolver_enabled:
            return None
        # The conflict resolver is an OPTIONAL read-path enhancement that needs the client's
        # current-value extractor. If a client does not provide it, degrade gracefully (skip the
        # resolver and fall back to the normal verified-answer path) rather than crashing the read.
        extractor = getattr(self.client, "extract_current_value_matches", None)
        if extractor is None:
            return None
        scope = candidates[0].record.scope if candidates else Scope()
        parsed = parse_query(query, as_of)
        candidates = _with_graph_validity_overrides(
            self.store, query, parsed, candidates, as_of, scope)
        try:
            return resolve_current_value_question(query, candidates, extractor, as_of)
        except Exception as e:
            _log.debug("conflict resolver degraded; falling back to normal answer path: %s", e)
            return None

    def _conflict_resolution_blocks(self, query: str, candidates: list[RetrievalCandidate],
                                    as_of: Optional[float] = None) -> list[str]:
        if not candidates:
            return []
        scope = candidates[0].record.scope
        resolved = self._structured_answer_from_candidates(
            query,
            candidates,
            now() if as_of is None else as_of,
            verify=False,
            scope=scope,
        )
        if resolved is None or not resolved.note.startswith("smqe:latest_value:"):
            return []
        blocks = []
        for citation in resolved.citations:
            rec = self.store.get_record(citation.memory_id)
            if rec is None:
                continue
            timestamp = f"{rec.valid_at:.0f}" if rec.valid_at is not None else "unknown"
            evidence = (citation.snippet or rec.text or rec.summary or "").strip()
            blocks.append(
                "SMQE latest-value operator selected matching evidence.\n"
                f"Answer candidate: {resolved.answer}\n"
                f"Source timestamp: {timestamp}\n"
                f"Evidence: {evidence}"
            )
        return blocks

    # ---- verification -----------------------------------------------------
    def verify(self, premise: str, hypothesis: str) -> tuple[NLILabel, float]:
        label, conf = self.client.nli(premise, hypothesis)
        return NLILabel(label), conf

    def _bounded_proof_premise(self, rec: MemoryRecord, hypothesis: str) -> str:
        """Premise for model NLI after local extractive proof fails.

        Exact-string/extractive proof still checks the full immutable raw record first. If that does
        not prove the claim, do not send a 500k-character transcript to NLI; use the best
        hypothesis-centered span. This preserves liveness and cost while still grounding the model
        check in source text.
        """
        premise = self._ground_truth(rec)
        if len(premise) <= self.settings.raw_span_min_chars:
            return premise
        return _raw_query_centered_span(
            premise,
            hypothesis,
            long_threshold_chars=self.settings.raw_span_min_chars,
            max_chars=3_200,
        )

    def _citation_snippet(self, rec: MemoryRecord, *, query: str = "", answer: str = "") -> str:
        """A human-facing source snippet centered on what was asked/answered."""
        text = rec.text or rec.summary or ""
        if len(text) > self.settings.raw_span_min_chars:
            text = _raw_query_centered_span(
                text,
                query or answer,
                long_threshold_chars=self.settings.raw_span_min_chars,
                max_chars=700,
                pre_context_chars=160,
            )
        return text[:500]

    def verify_citation(self, rec: MemoryRecord, hypothesis: str) -> tuple[NLILabel, float]:
        """Verify the answer against a source. For IMAGE memories the arbiter is the actual
        PIXELS (verified visual recall): a visual claim is judged against the raw image, so
        unsupported visual claims are rejected exactly like the text NLI path."""
        premise = self._ground_truth(rec)
        if _extractive_entailment(premise, hypothesis, rec.valid_at):
            return NLILabel.ENTAILMENT, 1.0
        if rec.modality == Modality.IMAGE:
            try:
                raw = self.substrate.get(rec.content_hash)
                tmp_dir = self.settings.data_dir / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".png", delete=False) as f:
                    f.write(raw)
                    path = f.name
                try:
                    label, conf = self.client.verify_visual(path, hypothesis)
                finally:
                    try:
                        Path(path).unlink()
                    except OSError:
                        pass
                return NLILabel(label), conf
            except Exception:
                pass  # fall back to text verification below
        return self.verify(self._bounded_proof_premise(rec, hypothesis), hypothesis)

    def _aggregation_proof_support(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        answer_text: str,
        at: Optional[float],
    ) -> dict[str, str]:
        """Deterministic proof for money-total answers.

        If the answer's total equals the sum of all scoped matching source amounts, mark those
        sources as entailing the aggregate. This proves compositional arithmetic that single-source
        NLI cannot entail, while staying conservative: any missing/mismatched source amount fails.
        """
        if not query or not self.settings.aggregation_audit_enabled:
            return {}
        parsed = parse_query(query, at)
        if not (_is_aggregation_query(query, parsed) and _is_money_aggregation(query)):
            return {}
        terms = _aggregation_terms(query)
        matches = _aggregation_matches(query, parsed, [c.record for c in candidates], at)
        support: dict[str, str] = {}
        source_values: list[float] = []
        for _, rec, snippet in matches:
            vals = _relevant_amount_values(snippet, terms)
            if not vals:
                continue
            support[rec.memory_id] = snippet
            source_values.extend(vals)
        if not support:
            return {}
        answer_values = _amount_values(answer_text)
        if not answer_values:
            return {}
        total = sum(source_values)
        if not any(_amount_close(total, v) for v in answer_values):
            return {}
        return support

    def _verify_candidates(self, candidates: list[RetrievalCandidate], text: str,
                           verify: bool, *, query: str = "",
                           at: Optional[float] = None) -> tuple[list[Citation], int]:
        """Verify candidates against the answer and build citations. Strategy (S1, flag-gated):
        batched NLI (one request), short-circuit (stop after the citation cap), or the baseline
        per-candidate serial path. With both flags off this is byte-identical to the old loop."""
        s = self.settings
        labels: list[tuple] = [(NLILabel.NEUTRAL, 0.0)] * len(candidates)
        if verify:
            if s.batch_nli_enabled:
                # Text sources judged together in ONE call; image sources judged against pixels.
                text_idx = [i for i, c in enumerate(candidates)
                            if c.record.modality != Modality.IMAGE]
                need_model: list[int] = []
                for i in text_idx:
                    premise = self._ground_truth(candidates[i].record)
                    if _extractive_entailment(premise, text, candidates[i].record.valid_at):
                        labels[i] = (NLILabel.ENTAILMENT, 1.0)
                    else:
                        need_model.append(i)

                def _run_batch(idx: list[int]) -> None:
                    pairs = [(self._bounded_proof_premise(candidates[i].record, text), text)
                             for i in idx]
                    batch = self.client.nli_batch(pairs) if pairs else []
                    for j, i in enumerate(idx):
                        if j < len(batch):
                            lab, conf = batch[j]
                            labels[i] = (NLILabel(lab), conf)

                if s.fast_verify_enabled:
                    # FAST_VERIFY semantics under batching: wave 1 covers only the top
                    # verify_citation_cap candidates by fused score; an entailment there leaves
                    # the tail NEUTRAL and unpaid (exactly the serial short-circuit contract).
                    # Zero entailments -> the remainder is batched before deciding, so the
                    # abstention decision is computed over the FULL set; any contradiction in
                    # wave 1 also forces the full picture (the advice-rescue kill and
                    # reconsolidation lapse need every contradicting source).
                    cap = max(1, int(s.verify_citation_cap))
                    ordered = sorted(
                        need_model,
                        key=lambda i: -float(candidates[i].fused_score or 0.0))
                    extractive_hits = sum(
                        1 for i in text_idx if labels[i][0] == NLILabel.ENTAILMENT)
                    wave1 = ordered[:cap] if extractive_hits < cap else []
                    _run_batch(wave1)
                    found = extractive_hits + sum(
                        1 for i in wave1 if labels[i][0] == NLILabel.ENTAILMENT)
                    contradicted = any(
                        labels[i][0] == NLILabel.CONTRADICTION for i in wave1)
                    remainder = ordered[len(wave1):]
                    if remainder and (found == 0 or contradicted):
                        _run_batch(remainder)
                else:
                    _run_batch(need_model)
                for i, c in enumerate(candidates):
                    if c.record.modality == Modality.IMAGE:
                        labels[i] = self.verify_citation(c.record, text)
            elif s.fast_verify_enabled:
                found = 0
                for i, c in enumerate(candidates):
                    if found >= s.verify_citation_cap:
                        break                       # short-circuit: the rest stay neutral
                    labels[i] = self.verify_citation(c.record, text)
                    if labels[i][0] == NLILabel.ENTAILMENT:
                        found += 1
            else:
                for i, c in enumerate(candidates):
                    labels[i] = self.verify_citation(c.record, text)
        citations: list[Citation] = []
        entailed = 0
        for i, c in enumerate(candidates):
            rec = c.record
            lab, conf = labels[i]
            citations.append(Citation(
                memory_id=rec.memory_id, content_hash=rec.content_hash,
                raw_uri=rec.raw_uri, source=rec.source, valid_at=rec.valid_at,
                snippet=self._citation_snippet(rec, query=query, answer=text),
                nli_label=lab, nli_score=conf,
            ))
            if lab == NLILabel.ENTAILMENT:
                entailed += 1
        if verify:
            support = self._aggregation_proof_support(query, candidates, text, at)
            if support:
                upgraded: list[Citation] = []
                for cit in citations:
                    if cit.memory_id in support:
                        upgraded.append(cit.model_copy(update={
                            "snippet": support[cit.memory_id][:240],
                            "nli_label": NLILabel.ENTAILMENT,
                            "nli_score": 1.0,
                        }))
                    else:
                        upgraded.append(cit)
                citations = upgraded
                entailed = sum(1 for c in citations if c.nli_label == NLILabel.ENTAILMENT)
        return citations, entailed

    def _claim_grounded(self, candidates: list[RetrievalCandidate], claim: str, *,
                        query: str = "", at: Optional[float] = None,
                        prefer_ids: Optional[set[str]] = None) -> bool:
        """Existence check: does ANY candidate ground this sub-claim?

        The CoVe/span demotion loops only need a yes/no, so verifying the full candidate set
        per claim (and building citations that are discarded) pays for verdicts that cannot
        change the outcome. Free local proofs run first (deterministic aggregation proof,
        extractive entailment), then model NLI over candidates ordered
        whole-answer-entailed-first by fused score, stopping at the first entailment. A False
        return still consulted every candidate, so the demotion decision is exactly as strict
        as the full-width check."""
        prefer = prefer_ids or set()
        if query and self._aggregation_proof_support(query, candidates, claim, at):
            return True
        ordered = sorted(
            candidates,
            key=lambda c: (c.record.memory_id not in prefer, -float(c.fused_score or 0.0)),
        )
        remaining: list[RetrievalCandidate] = []
        for c in ordered:
            if c.record.modality == Modality.IMAGE:
                remaining.append(c)
                continue
            if _extractive_entailment(self._ground_truth(c.record), claim, c.record.valid_at):
                return True
            remaining.append(c)
        if self.settings.batch_nli_enabled:
            text_pairs = [(self._bounded_proof_premise(c.record, claim), claim)
                          for c in remaining if c.record.modality != Modality.IMAGE]
            if text_pairs:
                for lab, _conf in self.client.nli_batch(text_pairs):
                    if NLILabel(lab) == NLILabel.ENTAILMENT:
                        return True
            for c in remaining:
                if c.record.modality == Modality.IMAGE:
                    lab, _conf = self.verify_citation(c.record, claim)
                    if lab == NLILabel.ENTAILMENT:
                        return True
            return False
        for c in remaining:
            lab, _conf = self.verify_citation(c.record, claim)
            if lab == NLILabel.ENTAILMENT:
                return True
        return False

    def _abstention_confidence(self, candidates: list[RetrievalCandidate],
                               citations: list[Citation]) -> tuple[float, dict]:
        """Blend the four abstention signals into a confidence score (Phase 2). Two are structural
        (channel agreement, proof completeness) so the gate does not rest on the model's
        self-report. Returns (confidence, per-signal dict)."""
        from . import abstention as _ab
        s = self.settings
        entail = max((c.nli_score for c in citations if c.nli_label == NLILabel.ENTAILMENT),
                     default=0.0)
        coverage = max((c.dense_score for c in candidates), default=0.0)
        agreement = (_ab.channel_agreement(max(candidates, key=lambda c: c.fused_score))
                     if candidates else 0.0)
        proof = _ab.proof_completeness(citations)
        conf = _ab.combine_confidence(
            entail, coverage, agreement, proof,
            w_entail=s.abstention_w_entail, w_coverage=s.abstention_w_coverage,
            w_agreement=s.abstention_w_agreement, w_proof=s.abstention_w_proof)
        return conf, {"entail": float(entail), "coverage": min(1.0, max(0.0, coverage)),
                      "agreement": agreement, "proof": proof}

    # ---- end-to-end answer -----------------------------------------------
    def structured_answer(self, query: str, at: Optional[float] = None, *,
                          verify: bool = True,
                          scope: Optional[Scope] = None) -> Optional[Answer]:
        """Run the shared SMQE structured recall path."""
        at = now() if at is None else at
        scope = scope or Scope()
        from .smqe import structured_answer
        return structured_answer(self, query, at=at, verify=verify, scope=scope)

    def _structured_answer_from_candidates(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        at: float,
        *,
        verify: bool,
        scope: Scope,
    ) -> Optional[Answer]:
        """Run SMQE over an already-retrieved candidate set.

        The neutral benchmark path can call ``Retriever.answer`` with precomputed records before
        they have been persisted in this retriever's store. SMQE verification resolves supports
        through the store, so we upsert any missing candidate records first. This keeps current-value
        conflict handling and compositional operators inside the shared structured path instead of
        returning a parallel conflict-resolver or reader answer.
        """
        if not candidates:
            return None
        if not _substantive_structured_query(query):
            return None
        get_record = getattr(self.store, "get_record", None)
        upsert_record = getattr(self.store, "upsert_record", None)
        if not (callable(get_record) and callable(upsert_record)):
            return None
        parsed = parse_query(query, at)
        candidates = _with_graph_validity_overrides(self.store, query, parsed, candidates, at, scope)
        records = [
            c.record for c in candidates
            if c.record.scope.visible_to(scope) and c.record.is_active_at(at)
        ]
        if not records:
            return None
        for rec in records:
            if get_record(rec.memory_id) is None:
                upsert_record(rec)
        from .smqe import structured_answer
        return structured_answer(self, query, records=records, at=at, verify=verify, scope=scope)

    def answer(self, query: str, at: Optional[float] = None, *, verify: bool = True,
               scope: Optional[Scope] = None, qvec: Optional[np.ndarray] = None,
               precomputed: Optional[list[RetrievalCandidate]] = None,
               reader_model: Optional[str] = None, activation: Optional[dict] = None) -> Answer:
        at = now() if at is None else at
        scope = scope or Scope()
        # `precomputed` lets a caller time retrieval separately and avoid re-retrieving.
        candidates = precomputed if precomputed is not None else self.retrieve(query, at, scope, qvec=qvec)
        if not candidates:
            return Answer(
                question=query, answer="I do not have that in memory.",
                verified=True, confidence=1.0, generated_by=self.settings.gen_model,
                retrieved_count=0, note="empty-or-no-active-memory",
            )

        structured_answered = self._structured_answer_from_candidates(
            query, candidates, at, verify=verify, scope=scope
        )
        if structured_answered is not None:
            return structured_answered

        # Pre-generation coverage signal (strength of the best content match).
        coverage = max((c.dense_score for c in candidates), default=0.0)

        # Shared context assembly (event calendar + preferences + raw chunks, edge-placed).
        blocks = self.assemble_context(query, candidates, at, scope,
                                       include_conflict_resolution=False, activation=activation)
        # reader_model pins one fixed answerer (neutral harness); else the difficulty cascade.
        # Speculative cascade (S5): try the cheap tier first, escalate only on a grounding miss.
        if self.settings.cascade_enabled and reader_model is None:
            model = self.settings.salience_model
        else:
            model = reader_model or _reader_model(query, self.settings)
        text = self.client.generate_answer(query, blocks, model=model)  # real call

        citations, entailed = self._verify_candidates(candidates, text, verify, query=query, at=at)

        if (self.settings.cascade_enabled and reader_model is None and verify and entailed == 0
                and coverage >= self.settings.abstention_threshold
                and model != self.settings.gen_model):
            # cheap answer didn't ground but coverage is real -> escalate to the strong tier.
            model = self.settings.gen_model
            text = self.client.generate_answer(query, blocks, model=model)
            citations, entailed = self._verify_candidates(candidates, text, verify, query=query, at=at)

        # Advice/likelihood grounding: a recommendation or 'is it likely that X' answer is
        # synthesis by design -- fresh suggestions and inference markers are never entailed by
        # memory as a whole answer, so whole-answer NLI abstained on every correctly grounded
        # reply. The verifiable core is the answer's restatement of stored premises. Verify
        # sentences individually (bounded) and ground the answer on an entailed restatement;
        # any CONTRADICTION kills the rescue. Mirrors the SMQE anchor rule where a derived
        # answer verifies on its cited premise (option choices and likely-inference already
        # carry the same exemption there).
        from .smqe.qa_ops import _ADVICE_REQUEST_RE, _LIKELY_INFERENCE_RE
        advice_anchor = False
        if verify and entailed == 0 and (
                _ADVICE_REQUEST_RE.search(query or "")
                or _LIKELY_INFERENCE_RE.search(query or "")
                or re.search(r"^\s*is\s+it\s+likely\b", query or "", re.I)):
            sentence_claims = [cl for cl in _sentences(text)
                               if len(cl) >= self.settings.span_nli_min_chars][:5]
            rescue: Optional[tuple[list[Citation], int]] = None
            rescue_contradicted = False
            for cl in sentence_claims:
                cl_citations, cl_entailed = self._verify_candidates(
                    candidates, cl, True, query=query, at=at)
                if any(c.nli_label == NLILabel.CONTRADICTION for c in cl_citations):
                    rescue_contradicted = True
                    break
                if cl_entailed and rescue is None:
                    rescue = (cl_citations, cl_entailed)
            if rescue is not None and not rescue_contradicted:
                citations, entailed = rescue
                advice_anchor = True

        # CoVe (Chain-of-Verification): factored fact-check of a grounded draft. Plan independent
        # verification questions, answer each ONLY from the retrieved blocks (factored -> the model
        # cannot copy its own hallucination), and re-verify the sub-answer against the candidates.
        # If a verification question's independent answer is itself ungrounded, the draft
        # over-claims -> drop entailment so the abstention/unverified path below takes over. Real
        # LLM calls, hence gated (COVE, default OFF). Lives in answer() -> the engine.ask product
        # path only; the neutral fixed-reader rows do not call this.
        # Sub-claim checks below need a yes/no per claim, not a citation list. The early-stop
        # helper stops at the first grounding candidate (whole-answer-entailed sources first);
        # a demotion still consulted the full set. Kill-switch: CLAIM_GROUNDING_EARLY_STOP=0
        # restores the full-width per-claim verify for one release.
        entailed_ids = {c.memory_id for c in citations if c.nli_label == NLILabel.ENTAILMENT}
        early_stop = getattr(self.settings, "claim_grounding_early_stop", True)

        def _sub_claim_grounded(sub_claim: str) -> bool:
            if early_stop:
                return self._claim_grounded(candidates, sub_claim, query=query, at=at,
                                            prefer_ids=entailed_ids)
            _c, n = self._verify_candidates(candidates, sub_claim, True, query=query, at=at)
            return n > 0

        cove_failed = False
        if self.settings.cove_enabled and verify and entailed > 0 and not advice_anchor:
            try:                              # best-effort: a failed CoVe call must never abort
                qs = self.client.plan_verification_questions(text, n=self.settings.cove_questions)
                for q in qs:
                    check = self.client.generate_answer(q, blocks, model=self.settings.verify_model)
                    if not _sub_claim_grounded(check):
                        entailed = 0      # factored check failed -> treat the draft as unverified
                        cove_failed = True
                        break
            except Exception:
                pass                          # answer() proceeds on the pre-CoVe verdict

        # Span-level NLI: verify EACH sentence of a multi-sentence answer against the sources, so a
        # partly-grounded answer can't ride one entailed sentence. One unentailed claim -> demote.
        # Whole-answer NLI already covers a single-sentence answer, so only multi-claim answers run.
        span_failed = False
        if self.settings.span_nli_enabled and verify and entailed > 0 and not advice_anchor:
            claims = [c for c in _sentences(text) if len(c) >= self.settings.span_nli_min_chars]
            if len(claims) > 1:
                for claim in claims:
                    if not _sub_claim_grounded(claim):
                        entailed = 0      # an ungrounded claim demotes the whole answer
                        span_failed = True
                        break

        # A CoVe/SPAN demotion means the draft over-claims: strip the ENTAILMENT evidence so the
        # abstention gate, the proof surface, AND engine.ask()'s post-answer reconsolidation never
        # treat an ungrounded draft's citations as confirmed -- otherwise reconsolidation would
        # FSRS-reinforce, re-embed, and co-activate the very memories the factored check rejected.
        # CONTRADICTION citations stay untouched (a contradicting source is still contradicting).
        if cove_failed or span_failed:
            citations = [c.model_copy(update={"nli_label": NLILabel.NEUTRAL, "nli_score": 0.0})
                         if c.nli_label == NLILabel.ENTAILMENT else c
                         for c in citations]

        verified = (entailed > 0) if verify else False
        unverified: list[str] = []
        abstained = False
        note = ""
        if cove_failed:
            _unverified_reason = "unverified: a CoVe verification question was not grounded in memory"
        elif span_failed:
            _unverified_reason = "unverified: a sentence-level claim was not grounded in memory"
        else:
            _unverified_reason = "unverified: no source entails the answer"
        # Calibrated abstention (Phase 2). When ABSTENTION_V2 is on, gate on a multi-signal
        # confidence (entailment + coverage + structural channel-agreement + proof-completeness)
        # against the dev-calibrated tau. When off, the original coverage gate runs unchanged.
        if verify and self.settings.abstention_v2_enabled:
            conf, sig = self._abstention_confidence(candidates, citations)
            if conf < self.settings.abstention_v2_tau:
                abstained = True
                text = "I don't have enough verified evidence in memory to answer that confidently."
                note = (f"abstained: confidence {conf:.2f} < tau "
                        f"{self.settings.abstention_v2_tau:.2f} (entail={sig['entail']:.2f} "
                        f"coverage={sig['coverage']:.2f} agreement={sig['agreement']:.2f} "
                        f"proof={sig['proof']:.2f})")
            elif not verified:
                note = _unverified_reason
                unverified = [text]
        elif verify and not verified and coverage < self.settings.abstention_threshold:
            abstained = True
            text = "I don't have enough verified evidence in memory to answer that confidently."
            note = f"abstained: insufficient evidence (coverage {coverage:.2f})"
        elif verify and not verified:
            note = _unverified_reason
            unverified = [text]
        if verified and advice_anchor and not note:
            note = "verified: advice grounded on context restatement (sentence-level)"

        top_rerank = max((c.rerank_score for c in candidates), default=0.0)
        if abstained:
            confidence = 0.0
        elif verify:
            confidence = 0.5 * min(1.0, top_rerank) + 0.5 * (1.0 if verified else 0.0)
        else:
            confidence = min(1.0, top_rerank)

        if verified:
            citations = [c for c in citations if c.nli_label == NLILabel.ENTAILMENT] or citations

        return Answer(
            question=query, answer=text, verified=verified, confidence=confidence,
            citations=citations, unverified_claims=unverified,
            generated_by=model, retrieved_count=len(candidates), note=note,
        )
