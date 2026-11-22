"""ABLATION policies: the reconciliation auditor with ONE capability removed or degraded.

They are not extra baselines for the leaderboard (`REGISTRY` stays at the three agent-side policies + the
evaluator-side oracle). They exist to show that the reward is SENSITIVE to each ingredient of a correct audit:
every error family, the amount, the evidence set, precision (false accusations) and the inspection budget - and
that the free-text explanation is NOT scored. Same observations only; no hidden data."""
from __future__ import annotations

from datetime import datetime, timezone

from ..constants import ALLOCATION, CUTOFF, DUPLICATE, FX
from .policy_base import finding
from .reconciliation_policy import ReconciliationPolicy


class _Variant(ReconciliationPolicy):
    name = "variant"

    def decide(self, observation: dict) -> list:
        return self.tweak(super().decide(observation))

    def tweak(self, findings: list) -> list:
        return findings


def _blind(kind: str):
    class Blind(_Variant):
        name = f"blind_{kind}"

        def tweak(self, findings):
            return [f for f in findings if f["type"] != kind]
    return Blind


class ThinEvidence(_Variant):
    """Right findings, but cites only the source document: the evidence set is incomplete."""
    name = "thin_evidence"

    def tweak(self, findings):
        return [{**f, "evidence_ids": f["evidence_ids"][:1]} for f in findings]


class WrongAmount(_Variant):
    """Right findings, amount off by 5 minor units (tolerance is +-1)."""
    name = "wrong_amount"

    def tweak(self, findings):
        return [{**f, "claimed_amount_minor": f["claimed_amount_minor"] + 5} for f in findings]


class NoExplanation(_Variant):
    """Identical findings with an empty explanation: the explanation is not scored, so nothing changes."""
    name = "no_explanation"

    def tweak(self, findings):
        return [{**f, "explanation": ""} for f in findings]


class OverAccuser(_Variant):
    """Correct findings PLUS three unfounded accusations on the largest documents (precision pressure)."""
    name = "over_accuser"

    def tweak(self, findings):
        used = {i for f in findings for i in f["affected_ids"]}
        extra = []
        for d in sorted(self.docs(), key=lambda d: -(d.get("total_minor") or abs(d.get("amount_minor", 0)))):
            key = d.get("invoice_id") or d.get("payment_id")
            if key not in used and len(extra) < 3:
                extra.append(finding(DUPLICATE, [key], 1, self.policy["base_currency"], [d["doc_id"]], "unfounded"))
        return findings + extra


class NaiveClock(ReconciliationPolicy):
    """Reads every timestamp as wall-clock time and ignores the UTC offset (a representation-fragile auditor).
    Right when all offsets happen to agree; wrong - and DIFFERENT on a re-timed twin - when they do not."""
    name = "naive_clock"

    @staticmethod
    def when(ts: str) -> datetime:
        return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


class Budgeted(ReconciliationPolicy):
    """Stops reading after `pages` list_records calls and decides on the partial books."""
    pages = 8

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self._reads = 0

    def _next(self, observation: dict) -> dict:
        if self.queue and self._reads >= self.pages:
            self.queue = []
        if self.queue:
            self._reads += 1
        return super()._next(observation)


class Budget8(Budgeted):
    name, pages = "budget_8", 8


class Budget4(Budgeted):
    name, pages = "budget_4", 4


ABLATIONS = {c.name: c for c in (*(_blind(k) for k in (DUPLICATE, ALLOCATION, FX, CUTOFF)), ThinEvidence, WrongAmount,
                                 NoExplanation, OverAccuser, NaiveClock, Budget8, Budget4)}
