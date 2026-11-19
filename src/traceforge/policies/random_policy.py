"""RANDOM EXPLORER: seeded random exploration, then a few random guesses."""
from __future__ import annotations

import random

from ..constants import ISSUE_TYPES, TABLES
from .policy_base import Policy, finding, list_action


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, observation: dict) -> None:
        self.rng = random.Random(self.seed)             # independent, never the global RNG
        self.budget = self.rng.randint(8, 25)
        self.ids, self.amounts, self.docs = [], [], []
        self.base = observation["policy"]["base_currency"]
        self.episode_id = observation["manifest"]["episode_id"]

    def _learn(self, lr) -> None:
        if not lr or not lr.get("ok"):
            return
        recs = lr.get("records") or [lr.get("record") or lr.get("document")]
        for r in filter(None, recs):
            for k in ("row_id", "invoice_id", "payment_id"):
                if r.get(k):
                    self.ids.append(r[k])
            if r.get("doc_id"):
                self.docs.append(r["doc_id"])
            self.amounts += [v for v in (r.get("debit_minor"), r.get("credit_minor")) if v]

    def act(self, observation: dict) -> dict:
        self._learn(observation.get("last_result"))
        if self.budget > 0:
            self.budget -= 1
            roll = self.rng.random()
            if roll < 0.35 or not self.ids:
                return list_action(self.rng.choice(TABLES))
            if roll < 0.6:
                return {"action": "get_record", "table": self.rng.choice(("ledger", "invoices", "payments")),
                        "id": self.rng.choice(self.ids)}
            if roll < 0.75 and self.docs:
                return {"action": "get_document", "doc_id": self.rng.choice(self.docs)}
            if roll < 0.9:
                return {"action": "calculator", "expression": f"{self.rng.randint(1, 999)} * 1.08"}
            return {"action": "get_record", "table": "ledger", "id": "NOPE"}   # deliberate invalid id
        found = [finding(self.rng.choice(ISSUE_TYPES),
                         [self.rng.choice(self.ids)] if self.ids else [],
                         self.rng.choice(self.amounts) if self.amounts else 0, self.base,
                         self.rng.sample(self.ids, min(2, len(self.ids))), "random guess")
                 for _ in range(self.rng.randint(0, 3))]
        return {"action": "submit", "findings": {"episode_id": self.episode_id, "findings": found}}
