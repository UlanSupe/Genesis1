"""Throw-away audit probes (kept for reproducibility of the audit)."""
import json, os, random, subprocess, sys, hashlib
from collections import Counter
sys.path.insert(0, "src")
from traceforge.environment.audit_env import AuditEnv
from traceforge.generator.builder import generate_episode
from traceforge.config import GeneratorConfig
from traceforge.exceptions import ConfigError

# 1. cross-process determinism under different PYTHONHASHSEED
code = ("import sys;sys.path.insert(0,'src');from traceforge.generator.builder import generate_episode as g;"
        "from traceforge.episode_io import public_files as p, private_files as q;import hashlib;"
        "e=g(3,{'issue_counts':{'duplicate':2,'allocation':2,'fx':2,'cutoff':2}});h=hashlib.sha256();"
        "[h.update((k+v).encode()) for f in (p(e.public),q(e.private)) for k,v in sorted(f.items())];print(h.hexdigest())")
hs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                     env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip() for s in ("0", "1", "12345", "random")}
print("1. distinct digests across PYTHONHASHSEED:", len(hs))

# 2. fuzz env.step with garbage
rng = random.Random(1)
def junk(depth=0):
    c = rng.random()
    if c < .15: return None
    if c < .3: return rng.choice([0, 1, -1, 50, 51, 10**100, 3.5, True])
    if c < .5: return rng.choice(["", "ledger", "c50", "x" * 500, "ROW-00001", "\x00", "1+1", "__import__('os')"])
    if c < .7 and depth < 3: return [junk(depth + 1) for _ in range(rng.randint(0, 3))]
    if depth < 3: return {rng.choice(["action", "table", "id", "cursor", "limit", "doc_id", "expression", "findings", "type"]): junk(depth + 1) for _ in range(rng.randint(0, 4))}
    return "z"
env = AuditEnv("eval"); obs, _ = env.reset(1, {"issue_counts": {"fx": 1}})
tokens = env.leak_tokens(); bad = 0; crashes = 0
for i in range(3000):
    if env._done: env.reset(1, {"issue_counts": {"fx": 1}})
    a = junk()
    if rng.random() < .5: a = {"action": rng.choice(["list_records", "get_record", "get_document", "calculator", "submit"]), **{k: junk() for k in ("table", "id", "cursor", "limit", "doc_id", "expression", "findings") if rng.random() < .5}}
    try:
        o, r, t, tr, inf = env.step(a)
        if any(tok in json.dumps([o, inf]) for tok in tokens): bad += 1
    except Exception as e:
        crashes += 1; print("   CRASH", type(e).__name__, str(e)[:80], str(a)[:80])
print("2. fuzz: crashes", crashes, "leaks", bad)

# 3. generator tells: fractions of cross-border partial payments
fr = Counter()
for s in range(30):
    ep = generate_episode(s, {"scenarios": ["cross_border_partial"]})
    inv = {i["invoice_id"]: i for i in ep.public.invoices}
    for p in ep.public.payments:
        if len(p["invoice_refs"]) == 1 and p["amount_minor"] > 0:
            r = p["invoice_refs"][0]; fr[round(100 * r["allocated_minor"] / inv[r["invoice_id"]]["total_minor"])] += 1
print("3. allocation % histogram top:", fr.most_common(4))
# decoy immunity: are mutated docs ever decoys?
tags = Counter()
for s in range(40):
    ep = generate_episode(s, {"issue_counts": {"duplicate": 2, "fx": 2, "cutoff": 2, "allocation": 2}})
    dt = ep.private.provenance["world"]["document_tags"]
    for m in ep.private.provenance["mutations"]:
        d = m["document_ids"][0]
        key = next(i["invoice_id"] if "invoice_id" in i else i["payment_id"] for i in ep.public.invoices + ep.public.payments if i["doc_id"] == d)
        tags[dt[key]] += 1
print("   mutated doc tags:", dict(tags))
# 4. config edge cases
for cfg in ({"mutation_weights": {"fx": 0, "duplicate": 0, "allocation": 0, "cutoff": 0}}, {"n_invoices": "48"}, {"period_start": "banana"}, {"min_issues": 7, "max_issues": 3},
            {"mutation_weights": {"fx": -1}}, {"timezone_minutes": "x"}, {"clean_rate": "a"}):
    try:
        generate_episode(1, cfg); print("4. accepted:", cfg)
    except ConfigError: print("4. ConfigError ok:", cfg)
    except Exception as e: print("4. UNHANDLED", type(e).__name__, cfg)
try: AuditEnv().reset(1, [1, 2])
except Exception as e: print("   reset(config=list) ->", type(e).__name__)
