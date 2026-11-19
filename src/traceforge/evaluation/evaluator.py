"""Evaluator: holds the private truth, scores submissions. Lives outside the agent."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

from ..schemas import parse_submission
from .matcher import match
from .metrics import Metrics, make_metrics
from .reward import f1_fraction, invalid_reward, terminal_reward

log = logging.getLogger("traceforge.evaluator")


@dataclass
class EvaluationResult:
    metrics: Metrics
    valid: bool
    n_findings: int
    n_truth: int
    matches: list = field(default_factory=list)     # PRIVATE diagnostics
    internal_log: list = field(default_factory=list)  # PRIVATE

    @property
    def reward(self) -> float:
        return self.metrics.reward


class Evaluator:
    def __init__(self, truth: list, episode_id: str):
        self._truth = copy.deepcopy(truth)
        self.episode_id = episode_id

    def truth_copy(self) -> list:
        return copy.deepcopy(self._truth)

    def leak_tokens(self) -> list:
        return [t["issue_id"] for t in self._truth]

    def evaluate(self, payload, inspection_steps: int, elapsed: float = 0.0) -> EvaluationResult:
        findings, err = parse_submission(payload, self.episode_id)
        if err is not None:
            msg = f"invalid submission: {err}"
            log.info("%s (episode %s)", msg, self.episode_id)     # evaluator-internal journal
            m = make_metrics(0, 0, len(self._truth), invalid_reward(), inspection_steps,
                             elapsed, valid=False)
            return EvaluationResult(m, False, 0, len(self._truth), [], [msg])
        res = match(findings, self._truth)
        tp, fp, fn = res.tp, len(res.false_positives), len(res.false_negatives)
        reward = terminal_reward(f1_fraction(tp, fp, fn), inspection_steps)
        m = make_metrics(tp, fp, fn, reward, inspection_steps, elapsed)
        return EvaluationResult(m, True, len(findings), len(self._truth),
                                [{"finding": i, "issue_id": t} for i, t in res.pairs])
