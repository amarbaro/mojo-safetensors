"""Writes the gate's input files into DIR: hand-built headers (valid and invalid, each judged by the
reference) and files the reference serializer writes. Deterministic (fixed seed)."""
import json, os, random, struct, sys
from safetensors import _safetensors_rust as r

D = sys.argv[1]
os.makedirs(D, exist_ok=True)
rnd = random.Random(11)

def raw(h, data=b""):
    hb = json.dumps(h, separators=(",", ":")).encode() if isinstance(h, dict) else h
    return struct.pack("<Q", len(hb)) + hb + data

cases = {
 "ok": raw({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, bytes(8)),
 "hole": raw({"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}, bytes(8)),
 "extra": raw({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, bytes(8)),
 "short": raw({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}, bytes(4)),
 "badsize": raw({"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}, bytes(8)),
 "dtype": raw({"a": {"dtype": "F99", "shape": [2], "data_offsets": [0, 8]}}, bytes(8)),
 "f6": raw({"a": {"dtype": "F6_E2M3", "shape": [4], "data_offsets": [0, 3]}}, bytes(3)),
 "f4odd": raw({"a": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}, bytes(2)),
 "e8m0": raw({"a": {"dtype": "F8_E8M0", "shape": [2], "data_offsets": [0, 2]}}, bytes(2)),
 "c64": raw({"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}}, bytes(8)),
 "meta_int": raw({"__metadata__": {"k": 1}, "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, bytes(1)),
 "noshape": raw({"a": {"dtype": "U8", "data_offsets": [0, 1]}}, bytes(1)),
 "extrafield": raw({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "x": 1}}, bytes(1)),
 "negshape": raw({"a": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}}, bytes(1)),
 "space_start": raw(b' {"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', bytes(1)),
 "trail_space": raw(b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}   ', bytes(1)),
 "trail_nul": raw(b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}\0\0', bytes(1)),
 "dup": raw(b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}', bytes(2)),
 "overlap": raw({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}, "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}}, bytes(3)),
 "zero": raw({"a": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}, "b": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, bytes(1)),
 "scalar": raw({"a": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}, bytes(4)),
 "empty": raw({}, b""),
 "meta_only": raw({"__metadata__": {"x": "y"}}, b""),
 "tiny": b"\x01\x00\x00",
 "huge": struct.pack("<Q", 200_000_000) + b"{}",
 "notjson": raw(b"{nope", b""),
 "rev": raw({"b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}, "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, bytes(2)),
}
cases["hdrpast"] = struct.pack("<Q", 64) + b"{}"
cases["nonobj"] = raw(b'[1,2]')
cases["entry_str"] = raw({"a": "x"})
cases["off3"] = raw({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1, 2]}}, bytes(1))
cases["offneg"] = raw({"a": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 1]}}, bytes(2))
cases["meta_arr"] = raw({"__metadata__": [1], "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, bytes(1))
cases["shape_float"] = raw({"a": {"dtype": "U8", "shape": [1.0], "data_offsets": [0, 1]}}, bytes(1))
for k, v in cases.items():
    open(os.path.join(D, "h_" + k + ".safetensors"), "wb").write(v)

SZ = {"bool": 1, "uint8": 1, "int8": 1, "float8_e5m2": 1, "float8_e4m3fn": 1, "float8_e8m0fnu": 1, "int16": 2,
      "uint16": 2, "float16": 2, "bfloat16": 2, "int32": 4, "uint32": 4, "float32": 4, "float64": 8, "int64": 8,
      "uint64": 8, "float4_e2m1fn_x2": 1}

def tensor(dt, shape):
    n = 1
    for d in shape:
        n *= d
    return {"dtype": dt, "shape": shape, "data": rnd.randbytes(n * SZ[dt])}

def ser(name, d, meta=None):
    open(os.path.join(D, "s_" + name + ".safetensors"), "wb").write(r.serialize(d, meta))

ser("all_dtypes", {"t_" + dt: tensor(dt, [3, 2]) for dt in SZ})
ser("meta", {"w": tensor("float32", [4])}, {"format": "pt", "z": "last", "a": "first", "uni": "caf\u00e9 \u65e5\u672c",
     "esc": "q\"b\\ n\n t\t c\x01 d\x7f", "empty": ""})
ser("names", {"caf\u00e9.weight": tensor("bfloat16", [2]), "a\"q": tensor("uint8", [1]), "ctl\x1f\n": tensor("int8", [1]),
     "sp ace": tensor("float16", [1])})
ser("shapes", {"scalar": tensor("float32", []), "zero": tensor("float32", [0, 5]), "big": tensor("bfloat16", [64, 33]),
     "deep": tensor("int16", [2, 1, 3, 1, 2])})
ser("empty", {})
ser("only_meta", {}, {"k": "v"})
ser("many", {"layer.%04d.w" % i: tensor(rnd.choice(["float32", "bfloat16", "uint8", "int64"]), [rnd.randint(0, 9)]) for i in range(1500)})
ser("onebyte_meta", {"x": tensor("uint8", [7])}, {"m%d" % i: str(i) for i in range(12)})

def shard_dir(name, shards, index=True, bad=None):
    d = os.path.join(D, "dir_" + name); os.makedirs(d, exist_ok=True)
    wm = {}
    for fn, ts in shards.items():
        open(os.path.join(d, fn), "wb").write(r.serialize(ts, None))
        wm.update({n: fn for n in ts})
    if bad == "missing":
        wm["ghost.weight"] = list(shards)[0]
    if bad == "wrong":
        k = sorted(wm)[0]; wm[k] = [f for f in shards if f != wm[k]][0]
    if index:
        json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
two = {"model-00002-of-00002.safetensors": {"b.w": tensor("float16", [2]), "a.w": tensor("float32", [1])},
       "model-00001-of-00002.safetensors": {"z.w": tensor("bfloat16", [3]), "c.w": tensor("uint8", [5])}}
shard_dir("index", two)
shard_dir("noindex", two, index=False)
shard_dir("missing", two, bad="missing")
shard_dir("wrong", two, bad="wrong")
os.makedirs(os.path.join(D, "dir_none"), exist_ok=True)
