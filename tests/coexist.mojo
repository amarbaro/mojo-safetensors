# Builds only if the library's libc declarations do not clash with the standard library's own
# (FileHandle.seek declares lseek): a program that seeks a file and opens a safetensors file must link.
from safetensors import SafeTensors


def main() raises:
    with open(".work/fx/s_meta.safetensors", "r") as f:
        _ = f.seek(4)
        _ = f.read_bytes(4)
    var g = SafeTensors(".work/fx/s_meta.safetensors")
    print("coexist: ok", len(g.tensors))
