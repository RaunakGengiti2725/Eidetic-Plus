"""AnyBURL-style Horn-rule mining over the observed graph, token-free.

Mines length-2 rules  (X -r1-> Y) AND (Y -r2-> Z)  =>  (X -r3-> Z)  with confidence =
support(body & head) / support(body), then applies high-confidence rules to PROPOSE new
facts. Bounded 2-hop expansion with a per-node cap -- near-linear on sparse graphs, never
naive all-pairs O(N^2).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Rule:
    r1: str
    r2: str
    r3: str
    confidence: float
    support: int

    def text(self) -> str:
        return f"{self.r1} & {self.r2} => {self.r3} (conf={self.confidence:.2f}, n={self.support})"


def mine_rules(triples: list[tuple[str, str, str]], *, min_confidence: float = 0.5,
               min_support: int = 2, max_paths_per_node: int = 200) -> list[Rule]:
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    direct: dict[tuple[str, str], set[str]] = defaultdict(set)   # (X,Z) -> {relations}
    for h, r, t in triples:
        out[h].append((r, t))
        direct[(h, t)].add(r)

    body: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)   # (r1,r2) -> {(X,Z)}
    head: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)  # (r1,r2,r3)->{(X,Z)}
    for x, edges in out.items():
        seen = 0
        for r1, y in edges:
            for r2, z in out.get(y, ()):
                if x == z:
                    continue
                seen += 1
                if seen > max_paths_per_node:           # bound the 2-hop expansion
                    break
                body[(r1, r2)].add((x, z))
                for r3 in direct.get((x, z), ()):       # heads that actually hold
                    head[(r1, r2, r3)].add((x, z))
            if seen > max_paths_per_node:
                break

    rules: list[Rule] = []
    for (r1, r2, r3), pairs in head.items():
        b = len(body[(r1, r2)])
        if b < min_support:
            continue
        conf = len(pairs) / b
        if conf >= min_confidence:
            rules.append(Rule(r1, r2, r3, conf, b))
    rules.sort(key=lambda r: -r.confidence)
    return rules


def apply_rules(triples: list[tuple[str, str, str]], rules: list[Rule],
                max_facts: int = 500) -> list[dict]:
    """Apply rules to propose NEW facts (not already observed). Returns
    [{'src','relation','dst','confidence','provenance'}]."""
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    existing: set[tuple[str, str, str]] = set()
    for h, r, t in triples:
        out[h].append((r, t))
        existing.add((h, r, t))
    by_body: dict[tuple[str, str], list[Rule]] = defaultdict(list)
    for rule in rules:
        by_body[(rule.r1, rule.r2)].append(rule)

    proposed: dict[tuple[str, str, str], dict] = {}
    for x, edges in out.items():
        for r1, y in edges:
            for r2, z in out.get(y, ()):
                if x == z:
                    continue
                for rule in by_body.get((r1, r2), ()):
                    key = (x, rule.r3, z)
                    if key in existing:
                        continue
                    prev = proposed.get(key)
                    if prev is None or rule.confidence > prev["confidence"]:
                        proposed[key] = {
                            "src": x, "relation": rule.r3, "dst": z,
                            "confidence": rule.confidence,
                            "provenance": f"rule:{rule.r1}&{rule.r2}=>{rule.r3}",
                        }
                if len(proposed) >= max_facts:
                    break
    return list(proposed.values())
