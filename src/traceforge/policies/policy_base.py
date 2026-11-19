"""Policy interface. A policy sees ONLY JSON observations and returns JSON actions."""
from __future__ import annotations

from datetime import datetime

from ..constants import MAX_PAGE, TABLES


class Policy:
    name = "base"

    def reset(self, observation: dict) -> None:
        pass

    def act(self, observation: dict) -> dict:
        raise NotImplementedError


def list_action(table: str, cursor=None) -> dict:
    return {"action": "list_records", "table": table, "cursor": cursor, "limit": MAX_PAGE}


class FullScanPolicy(Policy):
    """Reads every table page by page, then calls `decide(obs)` once to produce a submit."""

    def reset(self, observation: dict) -> None:
        self.data = {t: [] for t in TABLES}
        self.queue = [list_action(t) for t in TABLES]
        self.policy = observation["policy"]
        self.manifest = observation["manifest"]

    def _ingest(self, lr) -> None:
        if lr and lr.get("ok") and lr.get("type") == "list_records":
            self.data[lr["table"]].extend(lr["records"])
            if lr["next_cursor"]:
                self.queue.insert(0, list_action(lr["table"], lr["next_cursor"]))

    def _submit(self, findings: list) -> dict:
        clean = [{k: v for k, v in f.items() if not k.startswith("_")} for f in findings[:30]]   # private notes stay local
        return {"action": "submit", "findings": {"episode_id": self.manifest["episode_id"], "findings": clean}}

    def act(self, observation: dict) -> dict:
        self._ingest(observation.get("last_result"))
        if self.queue:
            return self.queue.pop(0)
        return self._submit(self.decide(observation))

    def decide(self, observation: dict) -> list:
        raise NotImplementedError

    # helpers shared by heuristic policies
    @staticmethod
    def when(ts: str) -> datetime:
        return datetime.fromisoformat(ts)

    def docs(self) -> list:
        return self.data["invoices"] + self.data["payments"]

    def journals(self) -> dict:
        out = {}
        for r in self.data["ledger"]:
            out.setdefault(r["payment_id"] or r["invoice_id"], []).append(r)
        return out


def finding(typ, affected, amount, currency, evidence, text="") -> dict:
    return {"type": typ, "affected_ids": sorted(set(affected)), "claimed_amount_minor": int(amount),
            "currency": currency, "evidence_ids": list(evidence)[:10], "explanation": text[:1000]}
