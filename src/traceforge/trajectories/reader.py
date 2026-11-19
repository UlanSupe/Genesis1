"""Read and replay trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from ..environment.isolation import assert_no_leak


def read_trajectory(path: str | Path) -> list:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def replay(path: str | Path, env, seed=None, config=None) -> dict:
    """Re-execute the recorded actions on a freshly reset env (same seed/config/mode) and
    compare every observation, reward and flag with the recording."""
    steps = read_trajectory(path)
    obs, _ = env.reset(seed, config)
    mismatches = [] if obs == steps[0]["observation"] else [0]
    for rec in steps[1:]:
        obs, reward, term, trunc, _ = env.step(rec["action"])
        if (obs, reward, term, trunc) != (rec["observation"], rec["reward"],
                                          rec["terminated"], rec["truncated"]):
            mismatches.append(rec["step"])
    return {"ok": not mismatches, "steps": len(steps) - 1, "mismatches": mismatches}


def check_no_leak(path: str | Path, tokens) -> None:
    assert_no_leak(read_trajectory(path), tokens)
