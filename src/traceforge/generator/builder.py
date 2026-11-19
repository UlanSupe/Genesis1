"""Episode orchestration: plan -> clean world -> journals -> mutations -> files."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from .. import __version__
from ..config import GeneratorConfig
from ..constants import (ACCOUNTS, CURRENCY_DECIMALS, ISSUE_TYPES, LEDGER_FIELDS, SCENARIOS,
                         SCHEMA_VERSION, AMOUNT_TOLERANCE_MINOR)
from ..episode_io import public_files, save_episode
from ..exceptions import GenerationError
from ..schemas import Episode, PrivateEpisode, PublicEpisode
from . import compound_scenarios, mutations, validation
from .financial_world import build_world, iso, tz_string
from .ledger import build_journals
from .provenance import resolve

CONVENTIONS = {
    "duplicate": "affected_ids=[invoice_id|payment_id of a document whose journal is posted "
                 "more than once]; amount=base-currency amount of the extra posting(s) (anchor "
                 "line); evidence must include the document doc_id and the anchor line of the "
                 "original and of the repost.",
    "allocation": "affected_ids=[payment_id] plus every invoice_id whose ledger allocation "
                  "differs from the payment's invoice_refs; amount=total base-currency "
                  "shortfall; evidence must include the payment doc_id, the doc_id of EVERY "
                  "invoice listed in affected_ids, and the changed or added ledger row(s) of "
                  "that payment (e.g. the unapplied line).",
    "fx": "affected_ids=[invoice_id|payment_id]; amount=|booked - correct| of the anchor line "
          "in base currency; evidence must include the doc_id, one ledger row of the journal and the "
          "fx record id (FX-date-SRC-TGT) that policy prescribes.",
    "cutoff": "affected_ids=all ledger row_ids of the document's journal whose reporting "
              "period differs from the document's period; amount=larger of the debit/credit "
              "totals of those rows; evidence must include the doc_id and one such row.",
}


def make_plan(rng: random.Random, cfg: GeneratorConfig) -> dict:
    counts, scenarios = {t: 0 for t in ISSUE_TYPES}, []
    explicit = cfg.issue_counts is not None or cfg.scenarios is not None
    if cfg.clean or (not explicit and cfg.clean is None and rng.random() < cfg.clean_rate):
        return {"counts": counts, "scenarios": [], "explicit": explicit, "clean": True}
    if explicit:
        counts.update(cfg.issue_counts or {})
        scenarios = list(cfg.scenarios or [])
    else:
        n = rng.randint(max(cfg.min_issues, 1), max(cfg.max_issues, 1))
        if n >= 2 and rng.random() < cfg.compound_rate:
            scenarios, n = [rng.choice(SCENARIOS)], n - 2
        types, weights = zip(*sorted(cfg.mutation_weights.items()))
        for t in rng.choices(types, weights=weights, k=n):
            counts[t] += 1
    return {"counts": counts, "scenarios": scenarios, "explicit": explicit, "clean": False}


def _policy(world) -> dict:
    return {
        "base_currency": world.base,
        "period": {"start": iso(world.start), "end": iso(world.end), "end_inclusive": False,
                   "timezone": tz_string(world.tz_minutes)},
        "recognition_rule": "A journal belongs to the reporting period that contains its "
                            "document timestamp (invoice.issued_at / payment.timestamp) "
                            "compared as an instant; the booking timestamp must fall in the "
                            "same period.",
        "fx": {"table": "fx.csv", "semantics": "1 unit of source_currency = rate units of "
               "base currency", "rate_date_rule": "calendar date of the document timestamp "
               "converted to the company timezone; daily rates incl. weekends",
               "record_id": "FX-{date}-{source_currency}-{target_currency}"},
        "rounding": {"mode": "ROUND_HALF_UP", "scope": "per_ledger_line",
                     "currency_decimals": {c: d for c, d in CURRENCY_DECIMALS.items()}},
        "ledger_amounts": "debit_minor/credit_minor are base-currency minor units; the "
                          "currency column is the currency of the source document.",
        "journal_key": "payment_id if present else invoice_id",
        "postings": {"invoice": "Dr 1200 total | Cr 4000 total",
                     "payment": "Dr 1000 cash = sum of lines | Cr 1200 per invoice_ref | "
                                "Cr 2300 for amount not allocated",
                     "refund (negative payment)": "mirror image of a payment"},
        "accounts": dict(ACCOUNTS),
        "amount_tolerance_minor": AMOUNT_TOLERANCE_MINOR,
        "finding_types": list(ISSUE_TYPES),
        "finding_conventions": CONVENTIONS,
    }


def _generate(seed: int, cfg: GeneratorConfig, attempt: int) -> Episode:
    rng = lambda name: random.Random(f"traceforge|{seed}|{attempt}|{name}")
    plan = make_plan(rng("plan"), cfg)
    world = build_world(rng("world"), cfg, plan["scenarios"])
    journals = build_journals(world, rng("journal"))
    baseline_rows = [mutations.snap(r) for rows in journals.values() for r in rows]

    ctx = mutations.MutCtx(world, journals, rng("mutate"))
    for name in plan["scenarios"]:
        compound_scenarios.apply_scenario(ctx, name)
    for typ in ISSUE_TYPES:
        for _ in range(plan["counts"][typ]):
            if not mutations.apply_random(ctx, typ) and plan["explicit"]:
                raise GenerationError(f"not enough candidates for {typ}")

    rows = [r for rs in journals.values() for r in rs]
    rows.sort(key=lambda r: (datetime.fromisoformat(r["timestamp"]), r["_tb"]))
    mapping = {r["_uid"]: f"ROW-{i:05d}" for i, r in enumerate(rows, 1)}
    ledger = [{**{f: r[f] for f in LEDGER_FIELDS if f != "row_id"}, "row_id": mapping[r["_uid"]]}
              for r in rows]
    ledger = [{f: r[f] for f in LEDGER_FIELDS} for r in ledger]

    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    invoices, payments = [strip(d) for d in world.invoices], [strip(d) for d in world.payments]
    # SECURITY: a generated episode_id must NOT be a function of the seed (an agent sees it and could
    # otherwise recover a hidden seed by hashing, then regenerate the truth). It is a digest of the
    # episode's public CONTENT instead; inverting it means regenerating candidate episodes.
    eid = cfg.episode_id or ""
    manifest = {
        "schema_version": SCHEMA_VERSION, "episode_id": eid, "base_currency": world.base,
        "period_start": iso(world.start), "period_end": iso(world.end),
        "timezone": tz_string(world.tz_minutes), "record_count": len(ledger),
        "document_count": len(invoices) + len(payments),
        "rounding_rules": {"mode": "ROUND_HALF_UP", "scope": "per_ledger_line",
                           "currency_decimals": dict(CURRENCY_DECIMALS)}}
    public = PublicEpisode(manifest, _policy(world), ledger, invoices, payments, world.fx_rows)
    if not eid:
        h = hashlib.sha256()
        for name, text in sorted(public_files(public).items()):
            h.update(name.encode() + text.encode())
        manifest["episode_id"] = "ep-" + h.hexdigest()[:12]

    truth =[resolve(t, mapping) for t in ctx.issues]
    for t in truth:
        t["affected_ids"] = sorted(set(t["affected_ids"]))
        t["allowed_evidence_sets"] = [sorted(s) for s in t["allowed_evidence_sets"]]
    provenance = {
        "generator": {"traceforge_version": __version__, "seed": seed, "attempt": attempt,
                      "config": cfg.to_dict(), "plan": plan},
        "world": {"company": world.company, "entities": world.entities, "difficulty": cfg.difficulty,
                  "clean_decoys": sorted(k for k, d in ctx.docs.items()
                                         if d["_tag"].startswith("decoy") and k not in ctx.used),
                  "document_tags": {**{d["invoice_id"]: d["_tag"] for d in world.invoices},
                                    **{d["payment_id"]: d["_tag"] for d in world.payments}}},
        "mutations": resolve(ctx.mutations, mapping),
        "compound_scenarios": resolve(ctx.scenarios, mapping)}
    baseline = {"note": "ledger before any mutation; row_ids refer to the final episode",
                "ledger": sorted(resolve(baseline_rows, mapping), key=lambda r: r["row_id"])}
    episode = Episode(public, PrivateEpisode(truth, provenance, baseline))
    report = validation.validate_episode(episode, cfg.min_ledger_rows)
    if not report["ok"]:
        bad = [c for c in report["checks"] if not c["ok"]]
        raise GenerationError(f"validation failed: {bad}")
    provenance["validation"] = report
    return episode


def generate_episode(seed: int, config=None) -> Episode:
    """Deterministic: same (seed, config, code version) -> identical episode."""
    cfg = config if isinstance(config, GeneratorConfig) else GeneratorConfig.from_dict(config)
    last = None
    for attempt in range(6):
        try:
            return _generate(seed, cfg, attempt)
        except GenerationError as e:        # deterministic retry with a derived stream
            last = e
    raise GenerationError(f"could not generate episode for seed {seed}: {last}")


# ----------------------------------------------------------------------------- datasets
_BASES = {"train": ["EUR", "EUR", "USD", "EUR", "GBP", "EUR", "EUR", "CHF", "EUR", "USD"],
          "dev": ["GBP", "USD", "EUR", "CHF", "EUR", "GBP", "USD", "EUR", "JPY", "EUR"]}
_KINDS = [
    {"clean": True, "difficulty": "easy"},
    {"clean": True, "difficulty": "hard"},
    {"issue_counts": {"duplicate": 2}},
    {"issue_counts": {"allocation": 2}},
    {"issue_counts": {"fx": 2}},
    {"issue_counts": {"cutoff": 2}},
    {"scenarios": ["cross_border_partial"]},
    {"scenarios": ["period_end_duplicate"]},
    {"min_issues": 4, "max_issues": 6, "clean_rate": 0, "compound_rate": 0.5},
    {"issue_counts": {"duplicate": 3, "allocation": 3, "fx": 3, "cutoff": 3}, "difficulty": "hard"},
]


def dataset_specs(split: str) -> list:
    base_seed = {"train": 1000, "dev": 2000}[split]
    shift = {"train": 0, "dev": 1}[split]                  # dev rotates difficulty differently
    levels = ("easy", "medium", "hard")
    return [(f"{split}-{i:03d}", base_seed + i,
             {"difficulty": levels[(i + shift) % 3], **kind, "base_currency": _BASES[split][i],
              "episode_id": f"{split}-{i:03d}"})
            for i, kind in enumerate(_KINDS)]


def generate_dataset(out_dir: str | Path, splits=("train", "dev")) -> dict:
    out = Path(out_dir)
    index = {}
    for split in splits:
        names = []
        for name, seed, conf in dataset_specs(split):
            ep = generate_episode(seed, conf)
            save_episode(ep, out / split / name)
            names.append(name)
        (out / split / "index.json").write_text(json.dumps({"episodes": names}, indent=2) + "\n",
                                                encoding="utf-8", newline="\n")
        index[split] = names
    return index
