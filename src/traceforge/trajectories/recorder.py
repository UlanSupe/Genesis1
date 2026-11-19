"""Trajectory recorder: one JSON line per step, agent-visible data only.

Line schema: episode_id, step, action, observation, reward, terminated, truncated. Step 0
holds the initial observation (action null). No seed, truth, provenance, issue ids,
evaluator diagnostics or filesystem paths are ever written.

The ACTION is agent-controlled, so it is size-capped before it is written (a hostile agent
could otherwise fill the disk with 100 huge messages). Every legitimate field is far below
the caps (ids <= 64 chars, expression <= 200, explanation <= 1000, <= 30 findings), so
capping never changes the outcome of a replay: an over-long value is rejected either way. Only a
pathological action above MAX_ACTION_BYTES (necessarily invalid) is replaced by a marker and is not
replayable."""
from __future__ import annotations

import json
from pathlib import Path

MAX_STR, MAX_ITEMS, MAX_DEPTH = 4096, 100, 6
MAX_ACTION_BYTES = 128 * 1024        # a legitimate action is <= ~94 KB (30 findings, every field at its limit)


def cap(obj, depth: int = 0):
    """Bounded copy of an agent-supplied JSON-like value."""
    if depth > MAX_DEPTH:
        return "<truncated: too deep>"
    if isinstance(obj, str):
        return obj if len(obj) <= MAX_STR else obj[:MAX_STR] + f"<truncated {len(obj) - MAX_STR} chars>"
    if isinstance(obj, (list, tuple)):
        out = [cap(x, depth + 1) for x in list(obj)[:MAX_ITEMS]]
        return out + [f"<truncated {len(obj) - MAX_ITEMS} items>"] if len(obj) > MAX_ITEMS else out
    if isinstance(obj, dict):
        items = list(obj.items())
        out = {str(k)[:MAX_STR]: cap(v, depth + 1) for k, v in items[:MAX_ITEMS]}
        if len(items) > MAX_ITEMS:
            out["<truncated>"] = f"{len(items) - MAX_ITEMS} keys"
        return out
    return obj


def bounded_action(action):
    """Per-field caps AND an overall budget (the per-field bounds multiply: 100 findings x 100 ids x 4 KB)."""
    a = cap(action)
    size = len(json.dumps(a, default=str))
    if size > MAX_ACTION_BYTES:
        name = action.get("action") if isinstance(action, dict) and isinstance(action.get("action"), str) else None
        return {"action": name[:64] if name else None, "<truncated>": f"action exceeded {MAX_ACTION_BYTES} bytes ({size} after caps)"}
    return a


class TrajectoryRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="\n")

    def record(self, episode_id, step, action, observation, reward, terminated, truncated) -> None:
        self._fh.write(json.dumps({
            "episode_id": episode_id, "step": step, "action": bounded_action(action), "observation": observation,
            "reward": reward, "terminated": terminated, "truncated": truncated},
            sort_keys=True, default=str) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
