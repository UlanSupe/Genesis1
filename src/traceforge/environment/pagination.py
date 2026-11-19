"""Stable offset pagination with opaque cursors ("c<offset>")."""
from __future__ import annotations

import re

from ..constants import MAX_PAGE
from .actions import ActionError

_CURSOR = re.compile(r"^c(\d{1,7})$")


def paginate(records: list, cursor, limit: int = MAX_PAGE):
    """Returns (page, next_cursor). The record order never changes within an episode."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE:
        raise ActionError("invalid_limit", f"limit must be an integer in 1..{MAX_PAGE}")
    if cursor is None:
        offset = 0
    else:
        m = _CURSOR.match(cursor) if isinstance(cursor, str) else None
        if not m or int(m.group(1)) > len(records):
            raise ActionError("invalid_cursor", "cursor is not valid for this table")
        offset = int(m.group(1))
    end = offset + limit
    return records[offset:end], (f"c{end}" if end < len(records) else None)
