"""A standalone agent in ANY language speaks this newline-delimited JSON protocol on
stdin/stdout. This one only lists the first ledger page and submits an empty answer.

Run it isolated against a stored episode:

    python examples/run_custom_agent.py
"""
import json
import sys


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


seen_ledger = False
for line in sys.stdin:
    msg = json.loads(line)
    if msg["type"] == "close":
        break
    if msg["type"] == "reset":
        send({"type": "ready"})
    elif msg["type"] == "observation":
        obs = msg["observation"]
        if not seen_ledger:
            seen_ledger = True
            send({"type": "action", "action": {"action": "list_records", "table": "ledger", "limit": 50}})
        else:
            send({"type": "action", "action": {"action": "submit", "findings": {
                "episode_id": obs["manifest"]["episode_id"], "findings": []}}})
