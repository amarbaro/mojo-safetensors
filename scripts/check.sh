#!/usr/bin/env bash
# The gate. Fixtures from tools/fixtures_oracle.py (hand-built headers, valid and invalid, plus
# files the reference serializer wrote). For every file, $CASES read must print what the reference
# (huggingface/safetensors, tools/oracle.py) prints, "error" included; every readable file is
# rewritten by the Writer and must have the reference's layout (header length, tensor entries in
# order, data bytes), its metadata, and identical bytes when it has at most one metadata key.
# Checkpoint directories list the same tensors in the same order. Then the API test and example.
# CASES="$O tools/oracle.py" proves the gate with the oracle as its own stand-in.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .work
M=.venv/bin/mojo
O="uv run --no-project -q --with safetensors==0.6.2 --with numpy python -I"
CASES=${CASES:-.work/cases}
if [[ $CASES == .work/cases ]]; then
  $M build -I src -I lib tests/cases.mojo -o .work/cases 2>.work/build.err || { echo "FAIL build: tests/cases.mojo"; cat .work/build.err; exit 1; }
fi
F=.work/fx
[[ -f $F/s_many.safetensors ]] || $O tools/fixtures_oracle.py $F || { echo "FAIL setup: fixtures"; exit 1; }
nr=0; ne=0; nw=0; nb=0; ns=0
for f in $F/*.safetensors; do
  got=$($CASES read "$f" 2>&1) || got="crash (exit $?): ${got: -200}"; want=$($O tools/oracle.py read "$f")
  [[ $got == "$want" ]] || { echo "FAIL oracle: read $(basename "$f")"; diff <(echo "$want") <(echo "$got") | head -8; exit 1; }
  if [[ $got == error ]]; then ne=$((ne + 1)); continue; fi
  nr=$((nr + 1))
  rm -f .work/w.impl .work/w.ref
  wrote=$($CASES copy "$f" .work/w.impl)
  ref=$($O tools/oracle.py copy "$f" .work/w.ref)
  if [[ $wrote == error && $ref == error ]]; then  # only the oracle standing in for the library gets here
    echo "writer: $(basename "$f") not written by the reference itself"; ns=$((ns + 1)); continue
  fi
  [[ $($O tools/oracle.py read .work/w.impl) == "$got" ]] || { echo "FAIL writer: $(basename "$f") rewrite does not read back as the source"; exit 1; }
  if [[ $ref == error ]]; then
    echo "writer: $(basename "$f") read back identical; layout not compared (the reference cannot write $(grep -o 'F6_[A-Z0-9]*' <<< "$got" | head -1 || echo it))"
    ns=$((ns + 1)); continue
  fi
  for v in layout read; do
    [[ $($O tools/oracle.py $v .work/w.impl) == "$($O tools/oracle.py $v .work/w.ref)" ]] \
      || { echo "FAIL writer: $(basename "$f") $v differs from the reference"; diff <($O tools/oracle.py $v .work/w.ref) <($O tools/oracle.py $v .work/w.impl) | head -8; exit 1; }
  done
  nw=$((nw + 1))
  if (( $(grep -c '^meta ' <<< "$got") <= 1 )); then
    cmp -s .work/w.impl .work/w.ref || { echo "FAIL writer: $(basename "$f") bytes differ from the reference"; exit 1; }
    nb=$((nb + 1))
  fi
done
echo "oracle: $nr files read identically, $ne rejected as the reference rejects them"
echo "writer: $nw rewrites with the reference layout and metadata, $nb byte-identical, $ns read-back only"
nd=0
for d in $F/dir_*; do
  got=$($CASES dir "$d" 2>&1) || got="crash (exit $?): ${got: -200}"; want=$($O tools/oracle.py dir "$d")
  [[ $got == "$want" ]] || { echo "FAIL oracle: dir $(basename "$d")"; diff <(echo "$want") <(echo "$got") | head -8; exit 1; }
  nd=$((nd + 1))
done
echo "checkpoint: $nd directories identical (index, no index, bad index x2, empty)"
if [[ $CASES == .work/cases ]]; then
  [[ -f .work/big50k.safetensors ]] || $O -c "from safetensors import _safetensors_rust as r; open('.work/big50k.safetensors', 'wb').write(r.serialize({'layers.%d.experts.%d.w' % (i // 128, i % 128): {'dtype': 'bfloat16', 'shape': [2], 'data': bytes(4)} for i in range(50000)}, None))"
  t=$(python3 -I -c 'import subprocess, sys, time
t = time.monotonic(); subprocess.run(sys.argv[1:-1], stdout=open(sys.argv[-1], "w"), check=True)
print("%.2f" % (time.monotonic() - t))' $CASES read .work/big50k.safetensors .work/big.out) || { echo "FAIL perf: reader crashed on the 50k header"; exit 1; }
  [[ $(wc -l < .work/big.out) == 50001 ]] && awk -v t="$t" 'BEGIN { exit !(t < 1.0) }' \
    || { echo "FAIL perf: 50,000-tensor header read in ${t} s, want < 1 s"; exit 1; }
  echo "perf: 50,000-tensor header read and hashed in ${t} s"
  $M run -I src -I lib tests/api.mojo || { echo "FAIL api: tests/api.mojo"; exit 1; }
  $M run -I src -I lib examples/quickstart.mojo > /dev/null || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
fi
echo "PASS all"
