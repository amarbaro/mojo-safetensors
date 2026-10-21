"""Oracle: tests/cases.mojo's operations with the reference implementation (huggingface/safetensors).
Run with: uv run --no-project --with safetensors==0.6.2 python -I tools/oracle.py OP ARGS"""
import json, os, sys
from safetensors import _safetensors_rust as r, safe_open

TO_PY = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "F8_E5M2": "float8_e5m2", "F8_E4M3": "float8_e4m3fn",
         "F8_E8M0": "float8_e8m0fnu", "I16": "int16", "U16": "uint16", "F16": "float16", "BF16": "bfloat16",
         "I32": "int32", "U32": "uint32", "F32": "float32", "F64": "float64", "I64": "int64", "U64": "uint64",
         "F4": "float4_e2m1fn_x2"}

BITS = {"F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, **{k: 8 for k in "BOOL U8 I8 F8_E5M2 F8_E4M3 F8_E8M0".split()},
        **{k: 16 for k in "I16 U16 F16 BF16".split()}, **{k: 32 for k in "I32 U32 F32".split()}, **{k: 64 for k in "I64 U64 F64".split()}}

def fnv(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def tensors(path):
    return sorted(r.deserialize(open(path, "rb").read()), key=lambda t: t[0])

def shape(s):
    return "[" + ",".join(map(str, s)) + "]"

op, out = sys.argv[1], []
try:
    if op == "read":
        ts = tensors(sys.argv[2])
        with safe_open(sys.argv[2], "np") as f:
            meta = f.metadata() or {}
        out = ["ok"] + ["meta %s %s" % (k, meta[k]) for k in sorted(meta)]
        out += ["t %s %s %s %d %d" % (n, t["dtype"], shape(t["shape"]), len(t["data"]), fnv(t["data"])) for n, t in ts]
    elif op == "copy":
        ts = tensors(sys.argv[2])
        with safe_open(sys.argv[2], "np") as f:
            meta = f.metadata()
        # float4_e2m1fn_x2 counts packed pairs: the header's last dimension is twice the Python one
        d = {n: {"dtype": TO_PY[t["dtype"]], "shape": t["shape"][:-1] + [t["shape"][-1] // 2] if t["dtype"] == "F4" else t["shape"],
                 "data": bytes(t["data"])} for n, t in ts}
        open(sys.argv[3], "wb").write(r.serialize(d, meta))
        out = ["ok"]
    elif op == "layout":  # what a writer decides: header length, tensor entries in order, data bytes
        b = open(sys.argv[2], "rb").read()
        n = int.from_bytes(b[:8], "little")
        h = json.loads(b[8:8 + n], object_pairs_hook=lambda kv: kv)
        out = ["header %d" % n] + [json.dumps(kv, ensure_ascii=False) for kv in h if kv[0] != "__metadata__"]
        out.append("data %d %d" % (len(b) - 8 - n, fnv(b[8 + n:])))
    else:
        d = sys.argv[2]
        idx = os.path.join(d, "model.safetensors.index.json")
        wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        parts = sorted(set(wm.values())) if wm else sorted(f for f in os.listdir(d) if f.startswith("model") and f.endswith(".safetensors"))
        if not parts:
            raise ValueError("no shards")
        have = {}
        for p in parts:
            with safe_open(os.path.join(d, p), "np") as f:
                for n in sorted(f.keys()):
                    s = f.get_slice(n)
                    have[n] = p
                    numel = 1
                    for x in s.get_shape():
                        numel *= x
                    out.append("%s %s %s %s %d" % (p, n, s.get_dtype(), shape(s.get_shape()), numel * BITS[s.get_dtype()] // 8))
        if wm and (len(wm) != len(have) or any(have.get(n) != p for n, p in wm.items())):
            raise ValueError("weight_map disagrees with shards")
except Exception:
    out = ["error"]
print("\n".join(out))
