"""Agent process entry point: `python -m traceforge.runner.agent_worker --policy NAME`.

Imports only the agent-side policies (never the generator/evaluator), so private data
does not exist in this process. The oracle cannot be loaded here."""
from __future__ import annotations

import argparse
import sys

from ..policies import REGISTRY, make_policy
from .agent_protocol import decode, encode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sandbox", action="store_true", help="install the audit-hook sandbox before serving")
    args = ap.parse_args(argv)
    if args.policy not in REGISTRY:
        print("unknown policy", file=sys.stderr)
        return 2
    policy = make_policy(args.policy, args.seed, args.verify)
    if args.sandbox:
        from .sandbox import install
        install()
    out = sys.stdout
    sys.stdout = sys.stderr                  # stray prints must not corrupt the protocol
    for line in sys.stdin:
        msg = decode(line)
        if msg["type"] == "close":
            break
        if msg["type"] == "reset":
            policy.reset(msg["observation"])
            out.write(encode({"type": "ready"}))
        elif msg["type"] == "observation":
            out.write(encode({"type": "action", "action": policy.act(msg["observation"])}))
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
