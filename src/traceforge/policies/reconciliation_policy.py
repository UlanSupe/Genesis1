"""SIMPLE RECONCILIATION: a transparent rule-based auditor working only from the
observations (list_records over all four tables).

Deliberately simple, so it is NOT a perfect detector:
  * duplicates   : more than one anchor line for a document
  * fx           : recomputes the anchor with the fx-table rate of the document date
                   (company timezone, as policy.fx.rate_date_rule says)
  * allocation   : compares ledger allocations with invoice_refs using the payment's own
                   implied rate (tolerance 2 minor units)
  * cutoff       : compares instants of ledger rows vs document against the period

verify=True adds an INVESTIGATION phase before submitting, using the other public tools:
  get_document  to read the source document of every finding,
  get_record    to read the fx record a fx finding relies on,
  calculator    to recompute the expected converted amount of fx findings from the raw
                document amount and the fx rate; a finding whose recomputation disagrees by
                more than 2 minor units is dropped.
Every extra step costs 0.002 reward, which is why it is off by default (benchmarks use
verify=False)."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from ..constants import ACC_AR, ACC_CASH, ACC_UNAPPLIED, CUTOFF, DUPLICATE, ALLOCATION, FX
from ..money import convert_minor
from .policy_base import FullScanPolicy, finding

TOL = 2
MAX_VERIFIED = 8


class ReconciliationPolicy(FullScanPolicy):
    name = "reconciliation"

    def __init__(self, verify: bool = False):
        self.verify = verify

    def reset(self, observation: dict) -> None:
        super().reset(observation)
        self._findings = None
        self._calc = {}          # calculator expression -> (finding index, finding)
        self._last = None
        self._dropped = set()

    # ------------------------------------------------------------------ protocol
    def act(self, observation: dict) -> dict:
        lr = observation.get("last_result")
        self._ingest(lr)
        self._check_calculation(lr)
        action = self._next(observation)
        self._last = action
        return action

    def _next(self, observation: dict) -> dict:
        if self.queue:
            return self.queue.pop(0)
        if self._findings is None:
            self._findings = self.decide(observation)[:30]
            if self.verify:
                self.queue = self._verification_actions(self._findings)
                if self.queue:
                    return self.queue.pop(0)
        return self._submit([f for i, f in enumerate(self._findings) if i not in self._dropped])

    # ---------------------------------------------------------------- investigation
    def _verification_actions(self, findings: list) -> list:
        dec = self.policy["rounding"]["currency_decimals"]
        base = self.policy["base_currency"]
        docs = {d["doc_id"]: d for d in self.docs()}
        actions = []
        for i, f in enumerate(findings[:MAX_VERIFIED]):
            doc_id = f["evidence_ids"][0]
            actions.append({"action": "get_document", "doc_id": doc_id})
            if f["type"] == FX:
                actions.append({"action": "get_record", "table": "fx", "id": f["evidence_ids"][2]})
                d = docs[doc_id]
                ccy = d["currency"]
                raw = d["total_minor"] if "total_minor" in d else abs(d["amount_minor"])
                rate = next(r["rate"] for r in self.data["fx"]
                            if f"FX-{r['date']}-{r['source_currency']}-{r['target_currency']}" == f["evidence_ids"][2])
                expr = f"round({raw} / {10 ** dec[ccy]} * {rate} * {10 ** dec[base]}, 0)"
                self._calc[expr] = (i, f)
                actions.append({"action": "calculator", "expression": expr})
        return actions

    def _check_calculation(self, lr) -> None:
        """The calculator result must agree with the amount the finding was based on."""
        if not (self._last and self._last.get("action") == "calculator"):
            return
        i, f = self._calc.get(self._last["expression"], (None, None))
        if f is None:
            return
        if not (lr and lr.get("ok")):
            self._dropped.add(i)
            return
        expected = f.get("_expected_anchor")
        if expected is not None and abs(Decimal(lr["value"]) - Decimal(expected)) > TOL:
            self._dropped.add(i)

    # ------------------------------------------------------------------- detection
    def decide(self, observation: dict) -> list:
        base = self.policy["base_currency"]
        start, end = self.when(self.policy["period"]["start"]), self.when(self.policy["period"]["end"])
        cls = lambda t: -1 if t < start else (1 if t >= end else 0)
        fx = {(r["date"], r["source_currency"]): r["rate"] for r in self.data["fx"]}
        journals, out = self.journals(), []
        inv_doc = {i["invoice_id"]: i["doc_id"] for i in self.data["invoices"]}
        amt = lambda r: max(r["debit_minor"], r["credit_minor"])
        for doc in self.docs():
            inv = "total_minor" in doc
            key = doc["invoice_id"] if inv else doc["payment_id"]
            ts = self.when(doc["issued_at"] if inv else doc["timestamp"])
            rows = journals.get(key, [])
            anchors = [r for r in rows if (r["account"] == ACC_AR and r["payment_id"] is None)
                       ] if inv else [r for r in rows if r["account"] == ACC_CASH]
            if not anchors:
                continue
            if len(anchors) > 1:
                out.append(finding(DUPLICATE, [key], amt(anchors[1]), base,
                                   [doc["doc_id"], anchors[0]["row_id"], anchors[1]["row_id"]],
                                   f"{key} is posted {len(anchors)} times (anchor lines {anchors[0]['row_id']} and "
                                   f"{anchors[1]['row_id']} are identical) but only document {doc['doc_id']} exists."))
            a0 = anchors[0]
            ccy = doc["currency"]
            if ccy != base:
                day = ts.astimezone(start.tzinfo).date().isoformat()
                rate = fx.get((day, ccy))
                if rate:
                    cv = lambda v: convert_minor(v, ccy, base, rate)
                    if inv:
                        exp = cv(doc["total_minor"])
                    else:
                        rem = doc["amount_minor"] - sum(r["allocated_minor"] for r in doc["invoice_refs"])
                        exp = abs(sum(cv(r["allocated_minor"]) for r in doc["invoice_refs"]) + (cv(rem) if rem else 0))
                    if exp != amt(a0):
                        f = finding(FX, [key], abs(amt(a0) - exp), base,
                                    [doc["doc_id"], a0["row_id"], f"FX-{day}-{ccy}-{base}"],
                                    f"{key} ({ccy}) is booked at {amt(a0)} {base} minor units; the fx table gives rate "
                                    f"{rate} for {day}, i.e. {exp}; difference {abs(amt(a0) - exp)}.")
                        f["_expected_anchor"] = exp          # private note, stripped before submit
                        out.append(f)
            if not inv and doc["amount_minor"] > 0:
                out += self._allocation(doc, key, rows, a0, base, inv_doc)
            bad = [r for r in rows if cls(self.when(r["timestamp"])) != cls(ts)]
            if bad:
                out.append(finding(CUTOFF, [r["row_id"] for r in bad],
                                   max(sum(r["debit_minor"] for r in bad), sum(r["credit_minor"] for r in bad)),
                                   base, [doc["doc_id"], bad[0]["row_id"]],
                                   f"Document {doc['doc_id']} is dated {ts.isoformat()} but its journal is booked at "
                                   f"{bad[0]['timestamp']}, in another reporting period."))
        return sorted(out, key=lambda f: -f["claimed_amount_minor"])

    @staticmethod
    def _allocation(doc, key, rows, a0, base, inv_doc) -> list:
        cash, total = a0["debit_minor"], doc["amount_minor"]
        uniq = {}
        for r in rows:                          # collapse exact reposts (reported as duplicates)
            uniq.setdefault(tuple(v for k, v in sorted(r.items()) if k != "row_id"), r)
        rows = list(uniq.values())
        act, unact, ar_rows, un_rows = defaultdict(int), 0, defaultdict(list), []
        for r in rows:
            if r["account"] == ACC_AR and r["invoice_id"]:
                act[r["invoice_id"]] += r["credit_minor"] - r["debit_minor"]
                ar_rows[r["invoice_id"]].append(r["row_id"])
            elif r["account"] == ACC_UNAPPLIED:
                unact += r["credit_minor"] - r["debit_minor"]
                un_rows.append(r["row_id"])
        exp = {r["invoice_id"]: round(r["allocated_minor"] * cash / total) for r in doc["invoice_refs"]}
        rem = total - sum(r["allocated_minor"] for r in doc["invoice_refs"])
        un_exp = round(rem * cash / total)
        bad = {i for i in set(exp) | set(act) if abs(exp.get(i, 0) - act.get(i, 0)) > TOL}
        un_excess = unact - un_exp
        if not bad and un_excess <= TOL:
            return []
        inv_excess = sum(act.get(i, 0) - exp.get(i, 0) for i in bad if act.get(i, 0) - exp.get(i, 0) > TOL)
        amount = un_excess if un_excess > TOL else inv_excess
        ev = [doc["doc_id"]] + [inv_doc[i] for i in bad if i in inv_doc] \
            + [r for i in bad for r in ar_rows.get(i, [])] + un_rows
        lines = un_rows + [r for i in bad for r in ar_rows.get(i, [])]
        text = (f"Payment {key} ({doc['doc_id']}) allocates to {', '.join(sorted(bad)) or 'invoices'} but the ledger "
                f"credits differ: {amount} {base} minor units are missing from accounts receivable "
                f"(unapplied/misdirected lines {', '.join(lines)}).")
        return [finding(ALLOCATION, [key, *bad], amount, base, ev, text)]
