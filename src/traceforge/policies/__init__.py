"""Agent-side baseline policies. The oracle is deliberately NOT here (evaluator-only)."""
from __future__ import annotations

from .ablations import ABLATIONS
from .mark_all_policy import MarkAllPolicy
from .policy_base import Policy
from .random_policy import RandomPolicy
from .reconciliation_policy import ReconciliationPolicy

REGISTRY = {"random": RandomPolicy, "reconciliation": ReconciliationPolicy, "mark_all": MarkAllPolicy}


def make_policy(name: str, seed: int = 0, verify: bool = False) -> Policy:
    """verify=True (reconciliation only) enables the tool-based investigation phase."""
    if name in ABLATIONS:
        return ABLATIONS[name]()
    if name not in REGISTRY:
        raise KeyError(f"unknown agent policy {name!r}; available: {sorted(REGISTRY)}")
    if name == "random":
        return REGISTRY[name](seed)
    return REGISTRY[name](verify=True) if (verify and name == "reconciliation") else REGISTRY[name]()
