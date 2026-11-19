"""Action parsing/validation. Errors carry fixed, bounded messages (never echo input,
never mention private data)."""
from __future__ import annotations

from ..constants import ACTIONS, MAX_ID_LEN, MAX_PAGE, TABLES


class ActionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _need(cond: bool, code: str, msg: str) -> None:
    if not cond:
        raise ActionError(code, msg)


def _is_id(v) -> bool:
    return isinstance(v, str) and 0 < len(v) <= MAX_ID_LEN


def parse_action(raw) -> dict:
    """Normalise a raw action dict; raises ActionError when malformed."""
    _need(isinstance(raw, dict), "invalid_action", "action must be an object")
    name = raw.get("action")
    _need(isinstance(name, str) and name in ACTIONS, "unknown_action",
          "unknown action; allowed: " + ", ".join(ACTIONS))
    if name == "list_records":
        _need(raw.get("table") in TABLES, "invalid_table", "table must be one of " + ", ".join(TABLES))
        return {"action": name, "table": raw["table"], "cursor": raw.get("cursor"),
                "limit": raw.get("limit", MAX_PAGE)}
    if name == "get_record":
        _need(raw.get("table") in TABLES, "invalid_table", "table must be one of " + ", ".join(TABLES))
        _need(_is_id(raw.get("id")), "invalid_id", "id must be a non-empty string")
        return {"action": name, "table": raw["table"], "id": raw["id"]}
    if name == "get_document":
        _need(_is_id(raw.get("doc_id")), "invalid_id", "doc_id must be a non-empty string")
        return {"action": name, "doc_id": raw["doc_id"]}
    if name == "calculator":
        _need(isinstance(raw.get("expression"), str), "invalid_expression",
              "expression must be a string")
        return {"action": name, "expression": raw["expression"]}
    _need("findings" in raw, "invalid_action", "submit requires findings")
    return {"action": name, "findings": raw["findings"]}
