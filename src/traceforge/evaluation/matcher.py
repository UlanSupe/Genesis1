"""Strict one-to-one matcher.

A prediction is a candidate TP for a truth issue iff ALL hold:
  1. type: finding.type.strip().lower() == truth.type
  2. affected_ids: equal as SETS after strip(), dropping empty strings
  3. amount: |claimed_amount_minor - expected_amount_minor| <= AMOUNT_TOLERANCE_MINOR
  4. currency: finding.currency.strip().upper() == truth.currency
  5. evidence: evidence_ids (stripped, as a set) is a superset of at least one allowed set

One prediction matches at most one truth issue and vice versa. Unmatched predictions
are FP, unmatched truth issues FN. The result is a MAXIMUM-cardinality bipartite
matching (Kuhn augmenting paths), made deterministic by a published rule:
predictions are processed in canonical order (type, affected_ids, amount, currency,
evidence, submission index) and each tries compatible truth issues in ascending issue_id.
Equal predictions are interchangeable, so TP/FP/FN never depend on submission order."""
from __future__ import annotations

from dataclasses import dataclass

from ..constants import AMOUNT_TOLERANCE_MINOR
from ..schemas import Finding
from .evidence_validator import evidence_satisfied


def norm_ids(ids) -> tuple:
    return tuple(sorted({s.strip() for s in ids if isinstance(s, str) and s.strip()}))


@dataclass
class MatchResult:
    pairs: list            # [(finding_index, issue_id)]
    false_positives: list  # finding indices
    false_negatives: list  # issue_ids

    @property
    def tp(self) -> int:
        return len(self.pairs)


def compatible(f: Finding, t: dict, tol: int = AMOUNT_TOLERANCE_MINOR) -> bool:
    return (f.type.strip().lower() == t["type"]
            and norm_ids(f.affected_ids) == tuple(sorted(set(t["affected_ids"])))
            and abs(f.claimed_amount_minor - t["expected_amount_minor"]) <= tol
            and f.currency.strip().upper() == t["currency"]
            and evidence_satisfied(f.evidence_ids, t["allowed_evidence_sets"]))


def match(findings: list, truth: list, tol: int = AMOUNT_TOLERANCE_MINOR) -> MatchResult:
    truths = sorted(truth, key=lambda t: t["issue_id"])
    order = sorted(range(len(findings)), key=lambda i: (
        findings[i].type.strip().lower(), norm_ids(findings[i].affected_ids),
        findings[i].claimed_amount_minor, findings[i].currency.strip().upper(),
        norm_ids(findings[i].evidence_ids), i))
    adj = {i: [j for j, t in enumerate(truths) if compatible(findings[i], t, tol)] for i in order}
    owner = {}                                  # truth index -> finding index

    def augment(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in order:
        augment(i, set())
    pairs = sorted((i, truths[j]["issue_id"]) for j, i in owner.items())
    matched = {i for i, _ in pairs}
    return MatchResult(pairs, [i for i in range(len(findings)) if i not in matched],
                       sorted(t["issue_id"] for j, t in enumerate(truths) if j not in owner))
