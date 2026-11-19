import sys, collections; sys.path.insert(0, "src")
from traceforge.generator.builder import generate_episode
bad = collections.Counter(); n = 0; att = collections.Counter(); tags = collections.Counter()
cfgs = [{"difficulty": "hard", "issue_counts": {"duplicate": 3, "allocation": 3, "fx": 3, "cutoff": 3}},
        {"scenarios": ["cross_border_partial", "period_end_duplicate"]}, {"clean": True, "difficulty": "easy"},
        {"difficulty": "medium", "base_currency": "JPY", "min_issues": 3, "max_issues": 6}, {"difficulty": "easy", "base_currency": "USD"}]
for s in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60):
    for cfg in cfgs:
        try:
            ep = generate_episode(s, cfg); n += 1; att[ep.private.provenance["generator"]["attempt"]] += 1
            dt = ep.private.provenance["world"]["document_tags"]
            for m in ep.private.provenance["mutations"]:
                d = m["document_ids"][0]
                key = next(i.get("invoice_id") or i["payment_id"] for i in ep.public.invoices + ep.public.payments if i["doc_id"] == d)
                tags[dt[key]] += 1
        except Exception as e: bad[str(e)[:300]] += 1
print("ok", n, "attempts", dict(att), "mutated-doc tags", dict(tags)); print(dict(bad))
