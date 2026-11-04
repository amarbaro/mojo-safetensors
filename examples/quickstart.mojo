from std.os import remove
from safetensors import SafeTensors, Writer


def main() raises:
    var w = Writer()
    w.add("weight", "F32", [2], [0, 0, 128, 63, 0, 0, 0, 64])  # float32 1.0, 2.0
    w.save("quickstart.safetensors")
    var f = SafeTensors("quickstart.safetensors")  # mmapped; data() is a view, no copy
    print(f.tensors[0].name, f.tensors[0].dtype, len(f.data("weight")), "bytes")
    remove("quickstart.safetensors")
