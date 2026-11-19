"""AuditEnv: the evaluator-side environment implementing the JSON contract

    reset(seed, config)  -> observation, info
    step(action)         -> observation, reward, terminated, truncated, info

It holds the PUBLIC episode data plus an `Evaluator` (private truth). It runs in the
runner/evaluator process; an agent only ever receives the JSON observations it returns
(see `runner.agent_protocol` / `environment.isolation`). Do not hand this object itself
to untrusted agent code: it is the evaluator's view."""
from __future__ import annotations

import copy
import time

from ..constants import MAX_STEPS, TABLES
from ..evaluation.evaluator import EvaluationResult, Evaluator
from ..exceptions import CalculatorError, ConfigError, EnvError, EpisodeFinished, NotReset
from ..schemas import fx_id
from . import calculator
from .actions import ActionError, parse_action
from .observations import build_observation
from .pagination import paginate


class AuditEnv:
    """Gymnasium-style: reset(seed, options) -> (obs, info); step(a) -> (obs, r, terminated,
    truncated, info). Observations/actions are plain JSON, so no gymnasium dependency is needed."""

    metadata = {"render_modes": []}
    max_steps = MAX_STEPS

    def __init__(self, mode: str = "train"):
        if mode not in ("train", "eval"):
            raise EnvError("mode must be 'train' or 'eval'")
        self.mode = mode
        self._pub = None
        self._evaluator: Evaluator | None = None
        self._result: EvaluationResult | None = None
        self._done = False
        self._steps = 0

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, options: dict | None = None, *, config: dict | None = None):
        """options (alias: config): GeneratorConfig fields, or {"episode_dir": path} to load a
        stored episode. Returns (observation, info); info never contains the seed."""
        from ..episode_io import load_episode
        from ..generator.builder import generate_episode

        if options is not None and config is not None:
            raise ConfigError("pass either options or config, not both")
        config = options if options is not None else config
        if config is not None and not isinstance(config, dict):
            raise ConfigError("options/config must be an object")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, str))):
            raise ConfigError("seed must be an int or a string")
        config = dict(config or {})
        episode_dir = config.pop("episode_dir", None)
        ep = load_episode(episode_dir) if episode_dir else generate_episode(0 if seed is None else seed, config)
        pub = ep.public
        self._pub = pub
        self._evaluator = Evaluator(ep.private.truth, pub.manifest["episode_id"])
        fx_rows = [dict(r, id=fx_id(r)) for r in pub.fx]
        self._tables = {"ledger": pub.ledger, "invoices": pub.invoices,
                        "payments": pub.payments, "fx": fx_rows}
        self._index = {
            "ledger": {r["row_id"]: r for r in pub.ledger},
            "invoices": {r["invoice_id"]: r for r in pub.invoices},
            "payments": {r["payment_id"]: r for r in pub.payments},
            "fx": {r["id"]: r for r in fx_rows}}
        self._docs = {d["doc_id"]: d for d in pub.invoices + pub.payments}
        self._steps, self._done, self._result = 0, False, None
        self._t0 = time.monotonic()
        return self._obs(None, None), {"episode_id": pub.manifest["episode_id"], "mode": self.mode}

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self._pub is None:
            raise NotReset("call reset() first")
        if self._done:
            raise EpisodeFinished("episode is over; call reset()")
        self._steps += 1
        last, err, submission, name = None, None, None, None
        try:
            act = parse_action(action)
            name = act["action"]
            if name == "submit":
                submission = act["findings"]
                last = {"ok": True, "type": "submit", "status": "received"}
            else:
                last = self._execute(act)
        except ActionError as e:
            err = {"code": e.code, "message": e.message}
            last = {"ok": False, "type": name}
        except Exception:                       # never leak internals / paths
            err = {"code": "internal_error", "message": "action could not be processed"}
            last = {"ok": False, "type": name}

        terminated = submission is not None
        truncated = not terminated and self._steps >= MAX_STEPS
        reward, info = 0.0, {"step": self._steps}
        if terminated or truncated:
            if terminated:
                payload = submission if isinstance(submission, dict) else \
                    {"episode_id": self._evaluator.episode_id, "findings": submission}
                inspection = self._steps - 1
            else:
                payload = {"episode_id": self._evaluator.episode_id, "findings": []}
                inspection = self._steps
            self._result = self._evaluator.evaluate(payload, inspection, time.monotonic() - self._t0)
            self._done = True
            if self.mode == "train":
                reward = self._result.reward
                info["submission_valid"] = self._result.valid
        return self._obs(last, err), reward, terminated, truncated, info

    def close(self) -> None:
        """Drop the episode (public data and the evaluator's private reference)."""
        self._pub = self._evaluator = self._result = None
        self._done = True

    # ------------------------------------------------------- evaluator-side API
    @property
    def result(self) -> EvaluationResult | None:
        """Full (private) evaluation; available to the runner once the episode is over."""
        return self._result

    def evaluator_truth(self) -> list:
        return self._evaluator.truth_copy()

    def leak_tokens(self) -> list:
        return self._evaluator.leak_tokens() + ["provenance", "clean_baseline", "truth.json"]

    # ---------------------------------------------------------------- internals
    def _obs(self, last, err) -> dict:
        return build_observation(self._pub.manifest, self._pub.policy, last, err,
                                 MAX_STEPS - self._steps)

    def _execute(self, a: dict) -> dict:
        name = a["action"]
        if name == "list_records":
            page, nxt = paginate(self._tables[a["table"]], a["cursor"], a["limit"])
            return {"ok": True, "type": name, "table": a["table"], "records": copy.deepcopy(page),
                    "next_cursor": nxt, "total": len(self._tables[a["table"]])}
        if name == "get_record":
            rec = self._index[a["table"]].get(a["id"])
            if rec is None:
                raise ActionError("not_found", "no such record")
            return {"ok": True, "type": name, "table": a["table"], "record": copy.deepcopy(rec)}
        if name == "get_document":
            doc = self._docs.get(a["doc_id"])
            if doc is None:
                raise ActionError("not_found", "no such document")
            return {"ok": True, "type": name, "document": copy.deepcopy(doc)}
        try:
            return {"ok": True, "type": name, "value": calculator.evaluate(a["expression"])}
        except CalculatorError as e:
            raise ActionError("calculator_error", str(e)) from None
