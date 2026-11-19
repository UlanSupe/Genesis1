"""Reading/writing episodes. Public and private data live in separate folders:

    episode/public/{manifest.json,ledger.csv,invoices.jsonl,payments.jsonl,fx.csv,policy.json}
    episode/private/{truth.json,provenance.json,clean_baseline.json}
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .constants import FX_FIELDS, LEDGER_FIELDS
from .schemas import Episode, PrivateEpisode, PublicEpisode

_NULL = "null"


def dumps(obj, indent=None) -> str:
    return json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=True)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _csv_text(rows: list, fields: tuple) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(fields)
    for r in rows:
        w.writerow([_NULL if r[f] is None else r[f] for f in fields])
    return buf.getvalue()


def public_files(pub: PublicEpisode) -> dict:
    """Canonical text of every public file (also used for hashing)."""
    return {
        "manifest.json": dumps(pub.manifest, 2) + "\n",
        "policy.json": dumps(pub.policy, 2) + "\n",
        "ledger.csv": _csv_text(pub.ledger, LEDGER_FIELDS),
        "fx.csv": _csv_text(pub.fx, FX_FIELDS),
        "invoices.jsonl": "".join(dumps(r) + "\n" for r in pub.invoices),
        "payments.jsonl": "".join(dumps(r) + "\n" for r in pub.payments),
    }


def private_files(priv: PrivateEpisode) -> dict:
    return {
        "truth.json": dumps(priv.truth, 2) + "\n",
        "provenance.json": dumps(priv.provenance, 2) + "\n",
        "clean_baseline.json": dumps(priv.clean_baseline, 2) + "\n",
    }


def save_episode(ep: Episode, root: str | Path) -> Path:
    root = Path(root)
    for sub, files in (("public", public_files(ep.public)), ("private", private_files(ep.private))):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            _write(root / sub / name, text)
    return root


def _read_csv(path: Path, int_fields=()) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            r = {k: (None if v == _NULL else v) for k, v in r.items()}
            for f in int_fields:
                r[f] = int(r[f])
            rows.append(r)
    return rows


def _read_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_public(root: str | Path) -> PublicEpisode:
    p = Path(root) / "public"
    rj = lambda n: json.loads((p / n).read_text(encoding="utf-8"))
    return PublicEpisode(
        manifest=rj("manifest.json"), policy=rj("policy.json"),
        ledger=_read_csv(p / "ledger.csv", ("debit_minor", "credit_minor")),
        invoices=_read_jsonl(p / "invoices.jsonl"),
        payments=_read_jsonl(p / "payments.jsonl"),
        fx=_read_csv(p / "fx.csv"))


def load_private(root: str | Path) -> PrivateEpisode:
    p = Path(root) / "private"
    rj = lambda n: json.loads((p / n).read_text(encoding="utf-8"))
    return PrivateEpisode(rj("truth.json"), rj("provenance.json"), rj("clean_baseline.json"))


def load_episode(root: str | Path, private: bool = True) -> Episode:
    return Episode(load_public(root), load_private(root) if private else None)
