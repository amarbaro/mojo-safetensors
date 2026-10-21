#!/usr/bin/env bash
# Proves the gate rejects wrong code: each mutant is applied to a copy of the tree and
# scripts/check.sh must FAIL on it. Then the oracle as its own stand-in must pass the oracle part.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
mutants=(
  "src/safetensors/st.mojo|            if i and names[i].s == names[i - 1].s:|            if False:|duplicate names accepted"
  "src/safetensors/st.mojo|            if t.begin != at or t.end < t.begin:|            if t.begin < at or t.end < t.begin:|holes between tensors accepted"
  "src/safetensors/st.mojo|chr(83 - _rank(|chr(65 + _rank(|writer dtype order reversed"
  "src/safetensors/st.mojo|        elif c < 32:|        elif c < 0:|control characters written raw"
  "src/safetensors/st.mojo|                if self.shards[s].find(wm.key_at(i)) < 0:|                if False:|weight_map shard not checked"
  "src/safetensors/st.mojo|        while len(h) % 8:|        while len(h) % 4:|header padded to 4"
)
for m in "${mutants[@]}"; do
  IFS='|' read -r file from to why <<< "$m"
  d=$ROOT/.work/mutant; rm -rf "$d"; mkdir -p "$d"
  cp -r src tests tools scripts examples "$d"; cp -a lib "$d/"; ln -s "$ROOT/.venv" "$d/.venv"; ln -sfn "$(realpath lib/json)" "$d/lib/json"; mkdir -p "$d/.work"; ln -s "$ROOT/.work/fx" "$d/.work/fx"; ln -s "$ROOT/.work/big50k.safetensors" "$d/.work/"
  python3 -I -c 'import sys; p,a,b=sys.argv[1:]; s=open(p).read(); assert a in s, "mutant does not apply: "+a; open(p,"w").write(s.replace(a,b,1))' "$d/$file" "$from" "$to" \
    || { echo "FAIL selfcheck: mutant does not apply ($why)"; exit 1; }
  if "$d/scripts/check.sh" > "$d.log" 2>&1; then echo "FAIL selfcheck: gate passed a mutant ($why)"; exit 1; fi
  echo "selfcheck: rejected ($why): $(grep -m1 FAIL "$d.log")"
done
CASES="uv run --no-project -q --with safetensors==0.6.2 --with numpy python -I tools/oracle.py" scripts/check.sh > .work/oracle-self.log 2>&1 || { echo "FAIL selfcheck: oracle fails its own gate"; tail -3 .work/oracle-self.log; exit 1; }
echo "PASS selfcheck"
