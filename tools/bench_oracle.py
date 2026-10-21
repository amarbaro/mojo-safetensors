"""Same work as tests/bench.mojo with the reference (safe_open, framework np)."""
import sys, time
t = time.perf_counter()
from safetensors import safe_open
s = n = 0
with safe_open(sys.argv[1], "np") as f:
    for k in f.keys():
        sl = f.get_slice(k); sh = sl.get_shape(); s += len(sh) + len(sl.get_dtype()); n += 1
print(n, "tensors", int((time.perf_counter() - t) * 1000), "ms", "check", s % 1000)
