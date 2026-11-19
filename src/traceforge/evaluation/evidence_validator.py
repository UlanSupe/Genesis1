"""Evidence rules.

A finding's evidence is sufficient iff its evidence_ids contain (as a superset) at
least one complete allowed evidence set. Extra and unknown ids are harmless."""
from __future__ import annotations

from ..constants import MAX_EVIDENCE
from ..schemas import PublicEpisode, fx_id


def evidence_satisfied(evidence_ids, allowed_sets) -> bool:
    ev = {e.strip() for e in evidence_ids if isinstance(e, str)}
    return any(set(s) <= ev for s in allowed_sets)


def public_id_index(pub: PublicEpisode) -> set:
    ids = {r["row_id"] for r in pub.ledger}
    for d in pub.invoices:
        ids |= {d["doc_id"], d["invoice_id"], d["entity_id"]}
    for d in pub.payments:
        ids |= {d["doc_id"], d["payment_id"]}
    ids |= {fx_id(r) for r in pub.fx}
    return ids


def check_truth_evidence(truth: list, pub: PublicEpisode) -> list:
    """Problems with truth evidence: unavailable/unknown ids or oversized sets."""
    known, problems = public_id_index(pub), []
    for t in truth:
        sets = t["allowed_evidence_sets"]
        if not sets or any(not s for s in sets):
            problems.append(f"{t['issue_id']}: empty evidence set")
        for s in sets:
            if len(s) > MAX_EVIDENCE:
                problems.append(f"{t['issue_id']}: evidence set larger than {MAX_EVIDENCE}")
            missing = [i for i in s if i not in known]
            if missing:
                problems.append(f"{t['issue_id']}: evidence not public: {missing}")
    return problems
