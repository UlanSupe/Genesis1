"""Narrow runner<->agent protocol: newline-delimited JSON over a pipe.

    runner -> agent : {"type":"reset","observation":{...}} | {"type":"observation",...} | {"type":"close"}
    agent  -> runner: {"type":"ready"}                      | {"type":"action","action":{...}}

Only observations (public data) flow to the agent; only actions flow back."""
from __future__ import annotations

import json

from ..exceptions import AgentProtocolError

MAX_LINE = 16_000_000


def encode(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"), sort_keys=True, default=str) + "\n"


def decode(line: str) -> dict:
    if len(line) > MAX_LINE:
        raise AgentProtocolError("message too large")
    try:
        msg = json.loads(line)
    except ValueError:
        raise AgentProtocolError("malformed message") from None
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise AgentProtocolError("message must be an object with a type")
    return msg


def action_from(msg: dict) -> dict:
    if msg.get("type") != "action" or not isinstance(msg.get("action"), dict):
        raise AgentProtocolError("expected an action message")
    return msg["action"]
