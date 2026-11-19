"""ORACLE: evaluator-side test policy. It is built from the private truth, is NOT
registered in `traceforge.policies`, and the agent worker refuses to load it. Its only
purpose is to check that the evaluator can award the maximum reward."""
from __future__ import annotations


class OraclePolicy:
    name = "oracle"

    def __init__(self, truth: list):
        self._truth = truth

    def reset(self, observation: dict) -> None:
        pass

    def act(self, observation: dict) -> dict:
        findings = [{"type": t["type"], "affected_ids": list(t["affected_ids"]),
                     "claimed_amount_minor": t["expected_amount_minor"], "currency": t["currency"],
                     "evidence_ids": list(t["allowed_evidence_sets"][0]),
                     "explanation": "oracle"} for t in self._truth]
        return {"action": "submit",
                "findings": {"episode_id": observation["manifest"]["episode_id"],
                             "findings": findings}}
