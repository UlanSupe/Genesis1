"""Symbolic-reference resolution for provenance, truth and the clean baseline.

During generation ledger rows are addressed by symbolic uids ("@u000123"). Final
ROW-ids are assigned only after all mutations and the final sort, so that row ids
carry no information about which rows were injected."""
from __future__ import annotations

import re

_UID = re.compile(r"@u\d{6}")


def resolve(obj, mapping: dict):
    if isinstance(obj, str):
        return _UID.sub(lambda m: mapping[m.group(0)], obj) if "@u" in obj else obj
    if isinstance(obj, list):
        return [resolve(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: resolve(v, mapping) for k, v in obj.items()}
    return obj
