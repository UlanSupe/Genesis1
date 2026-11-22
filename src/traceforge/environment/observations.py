"""Agent-facing observation: task, manifest, brief policy, last result, bounded error,
remaining steps. Nothing else."""
from __future__ import annotations

TASK = (
    "You are a financial auditor. Investigate the ledger and its source documents for the "
    "reporting period in `manifest` and report injected accounting errors. Finding types: "
    "duplicate, allocation, fx, cutoff (conventions in policy.finding_conventions). Some "
    "legitimate situations look suspicious (repeated amounts, refunds, partial and split "
    "payments, foreign currency, items near period edges); false accusations are penalised. "
    "Tools (one per step, max 100 steps including submit): "
    "list_records{table: ledger|invoices|payments|fx, cursor, limit<=50}, "
    "get_record{table, id}, get_document{doc_id}, calculator{expression}, "
    "submit{findings}. A finding is {type, affected_ids, claimed_amount_minor (int, base "
    "currency minor units), currency (the currency of claimed_amount_minor, i.e. the BASE "
    "currency from the manifest, not the source document's currency), evidence_ids (<=10), "
    "explanation (<=1000 chars)}; "
    "at most 30 findings. Score: F1 of exact matches minus a small cost per inspection step."
)


def build_observation(manifest: dict, policy: dict, last_result, error, steps_remaining: int) -> dict:
    return {"task": TASK, "manifest": manifest, "policy": policy,
            "last_result": last_result, "error": error, "steps_remaining": steps_remaining}
