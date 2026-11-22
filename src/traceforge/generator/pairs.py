"""Paired transformations and hidden evaluation sets (organizer tooling).

* `build_pairs`   : for every episode of a split, store its metamorphic twin (ids re-labelled,
                    records shuffled, neutral text renamed) in the same on-disk format. The twin has the
                    SAME economics, so a correct answer must survive; an agent that memorised ids or row
                    positions will not.
* `generate_eval` : N evaluation episodes whose seeds are derived from a SECRET with SHA-256 (64-bit),
                    so they cannot be guessed or enumerated, optionally together with their N pairs.

Layout (compatible with `evaluate` / `benchmark` / `paired-eval`):
    <root>/<split>/index.json                    {"episodes": [...], "originals": {twin: original}}
    <root>/<split>/<name>/{public,private}/...   private/transformation.json holds the id mapping"""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

from ..episode_io import load_episode, save_episode
from .builder import generate_episode
from .transformations import check_pair, transform_episode

EVAL_KINDS = [
    {"difficulty": "easy"}, {"difficulty": "medium"}, {"difficulty": "hard"}, {"clean": True, "difficulty": "medium"},
    {"difficulty": "easy", "scenarios": ["cross_border_partial"]}, {"difficulty": "medium", "scenarios": ["period_end_duplicate"]},
    {"difficulty": "hard", "scenarios": ["cross_border_partial", "period_end_duplicate"]},
    {"issue_counts": {"duplicate": 2, "allocation": 1}, "difficulty": "medium"},
    {"issue_counts": {"fx": 2, "cutoff": 1}, "difficulty": "hard"},
    {"min_issues": 3, "max_issues": 6, "clean_rate": 0, "compound_rate": 0.5, "difficulty": "hard"},
]
BASES = ("EUR", "USD", "GBP", "CHF", "EUR", "USD", "EUR", "GBP", "EUR", "CHF")


def _write_index(path: Path, names: list, originals: dict | None = None) -> None:
    body = {"episodes": names}
    if originals is not None:
        body["originals"] = originals
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def build_pairs(source_root: str | Path, split: str, out_root: str | Path, seed: int = 0) -> dict:
    """Write one transformed twin per source episode. Deterministic for a given `seed`."""
    src = Path(source_root) / split
    names = json.loads((src / "index.json").read_text(encoding="utf-8"))["episodes"]
    out = Path(out_root) / split
    twins, originals = [], {}
    for i, name in enumerate(names):
        ep = load_episode(src / name)
        new, mapping = transform_episode(ep, seed * 100_000 + i)
        checks = check_pair(ep, new, mapping)
        if not all(checks.values()):                                   # never ship a twin that lost a link
            raise RuntimeError(f"pair for {name} failed link verification: {[k for k, v in checks.items() if not v]}")
        twin = f"pair-{split}-{i:03d}"
        new.public.manifest["episode_id"] = twin
        save_episode(new, out / twin)
        (out / twin / "private" / "transformation.json").write_text(
            json.dumps({"original": name, "seed": seed * 100_000 + i, "id_mapping": mapping}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        twins.append(twin)
        originals[twin] = name
    _write_index(out, twins, originals)
    return {"split": split, "pairs": len(twins), "out": str(out)}


def eval_seed(secret: str, i: int) -> int:
    """64-bit episode seed derived from the organizer's secret (not guessable without it)."""
    return int(hashlib.sha256(f"traceforge-eval|{secret}|{i}".encode()).hexdigest()[:16], 16)


def generate_eval(out_root: str | Path, count: int = 20, secret: str | None = None, with_pairs: bool = True) -> dict:
    """Hidden evaluation set (+ pairs). Returns the secret that was used (store it privately)."""
    secret = secret or secrets.token_hex(16)
    out = Path(out_root)
    names = []
    for i in range(count):
        cfg = {**EVAL_KINDS[i % len(EVAL_KINDS)], "base_currency": BASES[i % len(BASES)], "episode_id": f"eval-{i:03d}"}
        save_episode(generate_episode(eval_seed(secret, i), cfg), out / "eval" / cfg["episode_id"])
        names.append(cfg["episode_id"])
    _write_index(out / "eval", names)
    res = {"episodes": count, "secret": secret, "out": str(out / "eval")}
    if with_pairs:
        res["pairs"] = build_pairs(out, "eval", out / "pairs", seed=int(secret[:8], 16) if all(c in "0123456789abcdef" for c in secret[:8]) else 1)
    return res
