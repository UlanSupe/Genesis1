"""Generator configuration (strictly validated, JSON-friendly).

Difficulty is defined by the NUMBER OF INTERRELATIONS the auditor must follow (split /
partial / two-part payments, foreign currencies, decoys, counterparties, issue count),
never by hiding or corrupting data: every episode stays fully solvable from public data
(this is enforced by the independent validator)."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .constants import CURRENCY_DECIMALS, ISSUE_TYPES, MAX_ISSUES, SCENARIOS
from .exceptions import ConfigError

DIFFICULTIES = ("easy", "medium", "hard")
_PERIOD = re.compile(r"^(\d{4})-(\d{2})-01$")

PRESETS = {
    "easy": {"difficulty": "easy", "clean_rate": 0.25, "min_issues": 1, "max_issues": 2, "compound_rate": 0.0},
    "medium": {"difficulty": "medium", "clean_rate": 0.15, "min_issues": 2, "max_issues": 5, "compound_rate": 0.1},
    "hard": {"difficulty": "hard", "clean_rate": 0.1, "min_issues": 4, "max_issues": 9, "compound_rate": 0.6},
    "compound": {"difficulty": "hard", "clean_rate": 0.0, "min_issues": 2, "max_issues": 6, "compound_rate": 1.0},
}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class GeneratorConfig:
    base_currency: str = "EUR"
    foreign_currencies: list | None = None      # None -> chosen by difficulty
    timezone_minutes: int | None = None         # company UTC offset; None -> random
    period_start: str | None = None             # "YYYY-MM-01"; None -> random month
    difficulty: str = "medium"
    n_invoices: int | None = None               # None -> random within the difficulty range
    n_entities: int | None = None               # None -> random within the difficulty range
    min_ledger_rows: int = 200
    clean_rate: float = 0.2                     # P(clean episode) in random planning
    min_issues: int = 1
    max_issues: int = 6
    mutation_weights: dict = field(default_factory=lambda: {t: 1 for t in ISSUE_TYPES})
    compound_rate: float = 0.15
    issue_counts: dict | None = None            # explicit per-type counts (overrides random)
    scenarios: list | None = None               # explicit compound scenarios
    clean: bool | None = None                   # True forces a clean episode
    episode_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "GeneratorConfig":
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError("config must be an object")
        known = {f.name for f in fields(cls)}
        extra = set(data) - known
        if extra:
            raise ConfigError(f"unknown config keys: {sorted(map(str, extra))}")
        cfg = cls(**data)
        cfg.validate()
        return cfg

    @classmethod
    def preset(cls, name: str, **overrides) -> "GeneratorConfig":
        if name not in PRESETS:
            raise ConfigError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
        return cls.from_dict({**PRESETS[name], **overrides})

    def validate(self) -> None:
        def need(cond, msg):
            if not cond:
                raise ConfigError(msg)

        need(isinstance(self.base_currency, str) and self.base_currency in CURRENCY_DECIMALS, "unsupported base_currency")
        need(self.foreign_currencies is None or isinstance(self.foreign_currencies, list), "foreign_currencies must be a list")
        for c in self.foreign_currencies or []:
            need(isinstance(c, str) and c in CURRENCY_DECIMALS and c != self.base_currency, f"bad foreign currency {c!r}")
        need(self.difficulty in DIFFICULTIES, "difficulty must be easy|medium|hard")
        for name in ("n_invoices", "n_entities"):
            v = getattr(self, name)
            need(v is None or (_is_int(v) and 4 <= v <= 5000), f"{name} must be an integer in 4..5000 or null")
        need(_is_int(self.min_ledger_rows) and 0 <= self.min_ledger_rows <= 20000, "min_ledger_rows must be an integer in 0..20000")
        need(self.timezone_minutes is None or (_is_int(self.timezone_minutes) and -720 <= self.timezone_minutes <= 840),
             "timezone_minutes must be an integer in -720..840")
        if self.period_start is not None:
            m = _PERIOD.match(self.period_start) if isinstance(self.period_start, str) else None
            need(m and 1900 <= int(m.group(1)) <= 2100 and 1 <= int(m.group(2)) <= 12, "period_start must look like YYYY-MM-01")
        need(_is_int(self.min_issues) and _is_int(self.max_issues) and 0 <= self.min_issues <= self.max_issues <= MAX_ISSUES,
             f"need integers 0 <= min_issues <= max_issues <= {MAX_ISSUES}")
        need(isinstance(self.mutation_weights, dict) and set(self.mutation_weights) <= set(ISSUE_TYPES)
             and all(_is_num(w) and w >= 0 for w in self.mutation_weights.values())
             and sum(self.mutation_weights.values()) > 0, "mutation_weights: known types, non-negative numbers, positive sum")
        need(self.issue_counts is None or (isinstance(self.issue_counts, dict) and set(self.issue_counts) <= set(ISSUE_TYPES)
             and all(_is_int(n) and n >= 0 for n in self.issue_counts.values())), "issue_counts: known types, non-negative integers")
        need(self.scenarios is None or (isinstance(self.scenarios, list) and all(s in SCENARIOS for s in self.scenarios)),
             f"scenarios must be a list of {SCENARIOS}")
        if self.issue_counts is not None or self.scenarios is not None:
            total = sum((self.issue_counts or {}).values()) + 2 * len(self.scenarios or [])
            need(total <= MAX_ISSUES, f"more than {MAX_ISSUES} issues requested")
        need(_is_num(self.clean_rate) and _is_num(self.compound_rate)
             and 0 <= self.clean_rate <= 1 and 0 <= self.compound_rate <= 1, "rates must be numbers within [0, 1]")
        need(self.clean is None or isinstance(self.clean, bool), "clean must be a boolean or null")
        need(self.episode_id is None or (isinstance(self.episode_id, str) and 0 < len(self.episode_id) <= 64),
             "episode_id must be a short string")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> GeneratorConfig:
    return GeneratorConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
