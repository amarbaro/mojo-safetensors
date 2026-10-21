# Open argv[1] (a .safetensors) and list every tensor's name, dtype and shape (header work; data
# is not read: numpy, the reference's framework here, cannot slice BF16). Prints the count and ms. Same work as tools/bench_oracle.py.
from std.sys import argv
from std.time import perf_counter_ns
from safetensors import SafeTensors


def main() raises:
    var t = perf_counter_ns()
    var f = SafeTensors(String(argv()[1]))
    var s = 0
    for i in range(len(f.tensors)):
        s += len(f.tensors[i].shape) + f.tensors[i].dtype.byte_length()
    print(len(f.tensors), "tensors", (perf_counter_ns() - t) // 1_000_000, "ms", "check", s % 1000)
