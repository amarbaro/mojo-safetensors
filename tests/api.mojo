# Spans point into the mapping (no copy); find, metadata and Writer errors behave as documented.
from safetensors import SafeTensors, Writer


def main() raises:
    var f = SafeTensors(".work/fx/s_meta.safetensors")
    if f.find("w") != 0 or f.find("nope") != -1:
        raise Error("api: find")
    if f.metadata("format").value() != "pt" or f.metadata("missing"):
        raise Error("api: metadata")
    var d = f.data("w")
    if Int(d.unsafe_ptr()) != f.base + f.data_start + f.tensors[0].begin or len(d) != 16:
        raise Error("api: data is not a view into the mapping")
    var w = Writer()
    var b: List[UInt8] = [1, 2, 3, 4]
    w.add("x", "U8", [4], b)
    for bad in [("x", "U8", 4), ("y", "F32", 2), ("z", "F99", 4)]:
        var raised = False
        try:
            w.add(bad[0], bad[1], [bad[2]], b)
        except:
            raised = True
        if not raised:
            raise Error("api: Writer.add accepted ", bad[0])
    print("api: ok")
