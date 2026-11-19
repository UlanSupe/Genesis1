"""Data containers and the submit/findings schema."""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (MAX_AFFECTED, MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS,
                        MAX_ID_LEN)


@dataclass
class PublicEpisode:
    """Everything an agent may ever see."""
    manifest: dict
    policy: dict
    ledger: list
    invoices: list
    payments: list
    fx: list


@dataclass
class PrivateEpisode:
    """Evaluator-only data. Never passed to agent-facing code."""
    truth: list
    provenance: dict
    clean_baseline: dict


@dataclass
class Episode:
    public: PublicEpisode
    private: PrivateEpisode


@dataclass
class Finding:
    type: str
    affected_ids: list
    claimed_amount_minor: int
    currency: str
    evidence_ids: list
    explanation: str


def fx_id(row: dict) -> str:
    return f"FX-{row['date']}-{row['source_currency']}-{row['target_currency']}"


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_list(v, max_len: int):
    if not isinstance(v, list) or len(v) > max_len:
        return None
    if not all(isinstance(x, str) and len(x) <= MAX_ID_LEN for x in v):
        return None
    return v


def parse_submission(payload, episode_id: str):
    """Validate a submission. Returns (findings, error); exactly one is None.
    The error string is evaluator-internal and must not be shown to agents."""
    if not isinstance(payload, dict):
        return None, "submission is not an object"
    if payload.get("episode_id") != episode_id:
        return None, "episode_id mismatch"
    raw = payload.get("findings")
    if not isinstance(raw, list):
        return None, "findings is not a list"
    if len(raw) > MAX_FINDINGS:
        return None, "too many findings"
    out = []
    for i, f in enumerate(raw):
        if not isinstance(f, dict):
            return None, f"finding {i} is not an object"
        if not isinstance(f.get("type"), str):
            return None, f"finding {i}: bad type"
        aff = _str_list(f.get("affected_ids"), MAX_AFFECTED)
        if aff is None:
            return None, f"finding {i}: bad affected_ids"
        if not _is_int(f.get("claimed_amount_minor")):
            return None, f"finding {i}: bad claimed_amount_minor"
        if not isinstance(f.get("currency"), str):
            return None, f"finding {i}: bad currency"
        ev = _str_list(f.get("evidence_ids"), MAX_EVIDENCE)
        if ev is None:
            return None, f"finding {i}: bad evidence_ids"
        expl = f.get("explanation", "")
        if not isinstance(expl, str) or len(expl) > MAX_EXPLANATION:
            return None, f"finding {i}: bad explanation"
        out.append(Finding(f["type"], aff, f["claimed_amount_minor"], f["currency"], ev, expl))
    return out, None
