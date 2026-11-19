"""MARK EVERYTHING: accuses the most 'suspicious-looking' documents of every error type
(up to the 30-finding limit). Exists to prove that blanket accusations score badly."""
from __future__ import annotations

from collections import Counter

from ..constants import ISSUE_TYPES
from .policy_base import FullScanPolicy, finding


class MarkAllPolicy(FullScanPolicy):
    name = "mark_all"

    def decide(self, observation: dict) -> list:
        base = self.policy["base_currency"]
        journals = self.journals()
        size = lambda d: abs(d.get("total_minor", d.get("amount_minor", 0)))
        freq = Counter(size(d) for d in self.docs())
        docs = sorted(self.docs(), key=lambda d: (-freq[size(d)], d["doc_id"]))
        out = []
        for d in docs:
            key = d.get("invoice_id") or d["payment_id"]
            rows = journals.get(key, [])
            if not rows:
                continue
            amt = max(rows[0]["debit_minor"], rows[0]["credit_minor"])
            for t in ISSUE_TYPES:
                aff = [r["row_id"] for r in rows] if t == "cutoff" else [key]
                out.append(finding(t, aff, amt, base, [d["doc_id"], rows[0]["row_id"]],
                                   "looks suspicious"))
                if len(out) >= 30:
                    return out
        return out
