#!/usr/bin/env bash
# Clean-environment verification: copy ONLY the submission files to a scratch directory, create a brand-new
# virtual environment, install, then run the judge sequence (install -> tests -> benchmark -> demo).
# Usage: bash tools/clean_env_check.sh [scratch_dir]      (needs network access for `pip install`)
set -u
SRC="$(cd "$(dirname "$0")/.." && pwd)"
D="${1:-${TMPDIR:-/tmp}/traceforge_clean}"
rm -rf "$D" && mkdir -p "$D/repo/trajectories"
cd "$SRC"
cp -r src tests datasets configs examples tools "$D/repo/"
cp ./*.md ./*.toml ./*.txt LICENSE .gitignore FINAL_BENCHMARK* "$D/repo/"
find "$D/repo" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
rm -rf "$D/repo/src/traceforge.egg-info" "$D/repo/reports" "$D/repo/graphify-out"
echo "== copied files: $(find "$D/repo" -type f | wc -l)   (no reports/, no graphify-out/, no cached trajectories)"
python -m venv "$D/venv" || exit 1
PY="$D/venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$D/venv/bin/python"
echo "== python in the clean venv: $($PY --version)  site-packages before install:"; $PY -m pip list 2>/dev/null | tail -n +3 | awk '{printf "%s ", $1} END {print ""}'
echo "== STEP 1: install"; cd "$D/repo"
$PY -m pip install -q -r requirements.txt && $PY -m pip install -q -e . ; echo "install exit=$?"
echo "== runtime third-party packages (must be only traceforge, pytest and its deps):"; $PY -m pip list 2>/dev/null | tail -n +3 | awk '{printf "%s ", $1} END {print ""}'
echo "== STEP 2: tests"; $PY -m pytest -q -p no:cacheprovider > "$D/tests.log" 2>&1; echo "pytest exit=$?"; tail -3 "$D/tests.log"
echo "== STEP 3: benchmark"; $PY -m traceforge benchmark --suite all --seeds 8 --out "$D/out" --name CLEAN > "$D/bench.log" 2>&1; echo "benchmark exit=$?"; tail -2 "$D/bench.log"
echo "== STEP 4: demo"; $PY -m traceforge demo --no-color --report-dir "$D/out" --trajectories "$D/out" > "$D/demo.log" 2>&1; echo "demo exit=$?"; grep -E "reward =|Verdict" "$D/demo.log"
echo "== compare benchmark with the submitted FINAL_BENCHMARK.json (everything except timing):"
$PY - "$D" <<'PYEOF'
import json, sys
d = sys.argv[1]
a = json.load(open(d + "/out/CLEAN.json", encoding="utf-8"))
b = json.load(open(d + "/repo/FINAL_BENCHMARK.json", encoding="utf-8"))
key = lambda rows: sorted((r["suite"], r["episode"], r["policy"], r["tp"], r["fp"], r["fn"], r["reward"], r["inspection_steps"]) for r in rows)
print("identical results:", key(a["rows"]) == key(b["rows"]), "| runs:", len(a["rows"]), len(b["rows"]),
      "| dataset digest equal:", a["meta"]["stored_dataset_public_sha256"] == b["meta"]["stored_dataset_public_sha256"])
PYEOF
echo "== done"
