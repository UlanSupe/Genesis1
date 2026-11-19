"""Minimal reset/step loop (evaluator-side API). Run from the repo root:

    python examples/gym_loop.py
"""
from traceforge.environment.audit_env import AuditEnv

env = AuditEnv(mode="train")                       # "eval" hides the terminal reward from the agent
obs, info = env.reset(seed=7, config={"issue_counts": {"fx": 1, "cutoff": 1}})
episode_id = info["episode_id"]
print("episode:", episode_id, "| steps left:", obs["steps_remaining"])

obs, reward, terminated, truncated, info = env.step({"action": "list_records", "table": "payments", "limit": 3})
for rec in obs["last_result"]["records"]:
    print(" ", rec["payment_id"], rec["currency"], rec["amount_minor"])

obs, *_ = env.step({"action": "calculator", "expression": "round(1234.56 * 1.0825, 2)"})
print("calculator:", obs["last_result"]["value"])

# an (almost certainly wrong) guess just to show the submit contract
findings = [{"type": "fx", "affected_ids": ["PAY-0001"], "claimed_amount_minor": 100, "currency": obs["manifest"]["base_currency"],
             "evidence_ids": [], "explanation": "demo"}]
obs, reward, terminated, truncated, info = env.step(
    {"action": "submit", "findings": {"episode_id": episode_id, "findings": findings}})
print("terminated:", terminated, "| reward:", reward)
print("evaluator-side metrics:", env.result.metrics)   # the agent never sees this in eval mode
