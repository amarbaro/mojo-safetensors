# read FILE   -> "ok", "meta K V" (sorted), "t NAME DTYPE SHAPE NBYTES FNV" per tensor by name; or "error"
# copy IN OUT -> rewrites IN through Writer ("ok" or "error"); the gate compares OUT with the reference
# dir DIR     -> "SHARD NAME DTYPE SHAPE NBYTES" per tensor in Checkpoint order; or "error"
# tools/oracle.py prints the same with the reference implementation.
from std.sys import argv
from safetensors import SafeTensors, Writer, Checkpoint, TensorInfo


def fnv(data: Span[UInt8, _]) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for b in data:
        h = (h ^ UInt64(b)) * 0x100000001B3
    return h


def shape(t: TensorInfo) -> String:
    var s = String("[")
    for i in range(len(t.shape)):
        s += ("," if i else "") + String(t.shape[i])
    return s + "]"


def main() raises:
    var a = argv()
    var op = String(a[1])
    var out = List[String]()
    try:
        if op == "read":
            var f = SafeTensors(String(a[2]))
            out.append("ok")
            var ks = f.meta_keys.copy()
            sort(ks)
            for k in ks:
                out.append("meta " + k + " " + f.metadata(k).value())
            for i in range(len(f.tensors)):
                ref t = f.tensors[i]
                out.append("t " + t.name + " " + t.dtype + " " + shape(t) + " " + String(t.nbytes()) + " " + String(fnv(f.data(i))))
        elif op == "copy":
            var f = SafeTensors(String(a[2]))
            var w = Writer()
            for i in range(len(f.meta_keys)):
                w.set_metadata(f.meta_keys[i], f.meta_vals[i])
            for i in range(len(f.tensors)):
                w.add(f.tensors[i].name, f.tensors[i].dtype, f.tensors[i].shape, f.data(i))
            w.save(String(a[3]))
            out.append("ok")
        else:
            var c = Checkpoint(String(a[2]))
            for si in range(len(c.shards)):
                ref s = c.shards[si]
                var base = s.path[byte = s.path.rfind("/") + 1 :]
                for t in s.tensors:
                    out.append(String(base) + " " + t.name + " " + t.dtype + " " + shape(t) + " " + String(t.nbytes()))
    except e:
        out = ["error"]
    for l in out:
        print(l)
