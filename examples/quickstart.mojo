from safetensors import SafeTensors, Writer


def main() raises:
    var w = Writer()
    var data: List[UInt8] = [0, 0, 128, 63, 0, 0, 0, 64]  # float32 1.0, 2.0
    w.add("weight", "F32", [2], data)
    w.set_metadata("format", "pt")
    w.save(".work/quick.safetensors")

    var f = SafeTensors(".work/quick.safetensors")  # mmapped; data() is a view, no copy
    for t in f.tensors:
        print(t.name, t.dtype, len(t.shape), t.nbytes(), "bytes")
    print(len(f.data("weight")), f.metadata("format").value())
