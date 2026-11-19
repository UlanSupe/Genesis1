"""transformation_report.json: paired metamorphic checks over many original/transformed
episodes. Nothing is hard-coded: every flag is computed per pair."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..config import PRESETS
from ..constants import ISSUE_TYPES, SCENARIOS
from ..environment.audit_env import AuditEnv
from ..episode_io import save_episode
from ..generator.builder import generate_episode
from ..generator.transformations import check_pair, map_findings, transform_episode
from ..runner.episode_runner import make_agent, run_episode

VARIANTS = (dict(), dict(), dict(), dict(remap_ids=False), dict(shuffle=False), dict(rename=False))


def pair_specs(n: int) -> list:
    """Deterministic spread over presets, issue types, scenarios, currencies."""
    bases = ("EUR", "USD", "GBP", "CHF", "JPY", "SEK")
    out = []
    for i in range(n):
        base = {"base_currency": bases[i % len(bases)]}
        kind = i % 8
        if kind < 4:
            cfg = {**PRESETS[("easy", "medium", "hard", "compound")[kind]], **base}
        elif kind < 7:
            cfg = {"issue_counts": {ISSUE_TYPES[(i + j) % 4]: 1 + (j == 0) for j in range(2)}, **base}
        else:
            cfg = {"scenarios": [SCENARIOS[(i // 8) % 2]], **base}
        out.append((100 + i, cfg, VARIANTS[i % len(VARIANTS)]))
    return out


def _run(tmp: Path, ep, tag: str, policy: str, answer=None):
    save_episode(ep, tmp / tag)
    return run_episode(AuditEnv("eval"), make_agent(policy), config={"episode_dir": str(tmp / tag)})


def _score_answer(tmp: Path, ep, tag: str, findings: list) -> dict:
    save_episode(ep, tmp / tag)
    env = AuditEnv("eval")
    env.reset(config={"episode_dir": str(tmp / tag)})
    env.step({"action": "submit", "findings": {"episode_id": ep.public.manifest["episode_id"], "findings": findings}})
    m = env.result.metrics
    return {"tp": m.tp, "fp": m.fp, "fn": m.fn, "f1": m.f1}


def build_transformation_report(n_pairs: int = 24, out_path: str | Path | None = None) -> dict:
    pairs = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for i, (seed, cfg, variant) in enumerate(pair_specs(n_pairs)):
            ep = generate_episode(seed, cfg)
            new, mapping = transform_episode(ep, seed + 1000, **variant)
            links = check_pair(ep, new, mapping)
            truth = ep.private.truth
            answer = [{"type": x["type"], "affected_ids": x["affected_ids"], "claimed_amount_minor": x["expected_amount_minor"],
                       "currency": x["currency"], "evidence_ids": x["allowed_evidence_sets"][0], "explanation": ""}
                      for x in truth]
            on_orig = _score_answer(tmp, ep, "o_ans", answer)
            on_new = _score_answer(tmp, new, "n_ans", map_findings(answer, mapping))
            stale = _score_answer(tmp, new, "n_stale", answer)       # negative control: unmapped answer
            runs = {}
            for pol in ("oracle", "reconciliation"):
                for tag, e in (("orig", ep), ("new", new)):
                    r = _run(tmp, e, f"{pol}_{tag}", pol)
                    runs[f"{pol}_{tag}"] = {k: r[k] for k in ("tp", "fp", "fn", "f1", "reward", "steps")}
            checks = {
                **links,
                "mapped_answer_correct_on_transform": on_new["f1"] == 1.0 and on_new["tp"] == len(truth),
                "original_answer_correct_on_original": on_orig["f1"] == 1.0,
                # negative control: after remapping, the ORIGINAL ids must no longer score
                "unmapped_answer_fails_when_ids_changed": (stale["tp"] == 0) if (mapping and truth) else True,
                "oracle_same_result": runs["oracle_orig"] == runs["oracle_new"],
                "heuristic_same_result": runs["reconciliation_orig"] == runs["reconciliation_new"],
            }
            pairs.append({"pair": i, "seed": seed, "config": cfg, "transform": {"remap_ids": True, "shuffle": True,
                          "rename": True, **variant}, "issues": len(truth), "ids_remapped": len(mapping),
                          "checks": checks, "ok": all(checks.values()), "runs": runs,
                          "mapped_answer": on_new, "unmapped_answer_on_transform": stale})
    summary = {"pairs": len(pairs), "passed": sum(p["ok"] for p in pairs),
               "failed_pairs": [p["pair"] for p in pairs if not p["ok"]],
               "checks_run": sum(len(p["checks"]) for p in pairs),
               "pairs_with_issues": sum(p["issues"] > 0 for p in pairs),
               "total_ids_remapped": sum(p["ids_remapped"] for p in pairs)}
    report = {"summary": summary, "pairs": pairs}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
