# mojo-safetensors

<!-- badges -->
[![ci](https://github.com/amarbaro/mojo-safetensors/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/amarbaro/mojo-safetensors/actions/workflows/ci.yml) [![latest release](https://img.shields.io/github/v/release/amarbaro/mojo-safetensors)](https://github.com/amarbaro/mojo-safetensors/releases/latest) [![license](https://img.shields.io/github/license/amarbaro/mojo-safetensors)](https://github.com/amarbaro/mojo-safetensors/blob/main/LICENSE)
<!-- /badges -->

<!-- install -->
Install: `pixi workspace channel add https://conda.modular.com/max https://prefix.dev/amarbaro/amarbaro`, then `pixi add mojo-safetensors` (import `safetensors`). From source: `git clone https://github.com/amarbaro/mojo-safetensors`, then `mojo run -I mojo-safetensors/src app.mojo`. It also needs mojo-json: put its `src` on another `-I`.
Toolchains: Mojo 1.2 stable, and the nightly 1.3.0.dev2026110805 in the weekly [mojo-libs](https://github.com/amarbaro/mojo-libs) matrix (36 libraries on stable, 35 on nightly).

```mojo
from std.os import remove
from safetensors import SafeTensors, Writer


def main() raises:
    var w = Writer()
    w.add("weight", "F32", [2], [0, 0, 128, 63, 0, 0, 0, 64])  # float32 1.0, 2.0
    w.save("quickstart.safetensors")
    var f = SafeTensors("quickstart.safetensors")  # mmapped; data() is a view, no copy
    print(f.tensors[0].name, f.tensors[0].dtype, len(f.data("weight")), "bytes")
    remove("quickstart.safetensors")
```
<!-- /install -->

Read and write [safetensors](https://github.com/huggingface/safetensors) files from Mojo. A
file is mmapped read-only and each tensor's data is a `Span` into the mapping, so nothing is
copied. The writer produces the reference's layout. The gate holds both sides to the reference
implementation (`huggingface/safetensors` 0.6.2):
- reading: 42 files, each accepted or rejected as the reference does
- writing: rewrites match the reference's bytes
- checkpoints: three real ones list identically
321 lines.

Requires Mojo 1.2 on Linux and [mojo-json](https://github.com/amarbaro/mojo-json): the conda package pulls it; a source checkout's `scripts/ci.sh` fetches v0.1.0.

## What you get

- `SafeTensors(path)`: `tensors` (`TensorInfo`: `name`, `dtype`, `shape`, `begin`, `end`,
  `numel()`, `nbytes()`), sorted by name; `find(name)`; `data(i)` or `data(name)` as a
  zero-copy span; `metadata(key)`, `meta_keys`; `bytes()` for the whole mapping.
- It accepts and rejects what the reference does: header size limit (100 MB), offsets that
  tile the data section exactly, size = elements × dtype bits, unknown dtypes, duplicate names,
  string-only `__metadata__`.
- All 19 dtypes, including F4, F6_E2M3, F6_E3M2 and F8_E8M0 (`dtype_bits`).
- `Writer`: `add(name, dtype, shape, data)`, `set_metadata(k, v)`, `save(path)`. Layout as the
  reference: dtype order (widest first), then name, header padded to 8. Metadata keys are
  written sorted; the reference writes them in hash order, so files with 2+ metadata keys
  match in content, not bytes.
- `Checkpoint(dir)`: `model.safetensors.index.json` with its shards (every `weight_map` entry
  checked against the shard it names), or every `model*.safetensors`. `shards` by file name.

API: [docs/api.md](docs/api.md). Design: [docs/design.md](docs/design.md).

## Performance

`scripts/bench.sh` (`tests/bench.mojo` vs `tools/bench_oracle.py`, open and list every tensor):

| | mojo-safetensors | reference (Python) |
|---|---|---|
| 1,500 tensors (gate fixture) | 1 ms, 12 MB | 163 ms, 31 MB |
| deepseek-ocr-2 shard, 2,707 tensors | 2 ms, 13 MB | 505 ms, 32 MB |
| 50,000 tensors | 51 ms, 47 MB | 164 s, 66 MB |

**Limits:** Read-only through mmap: a tensor's `Span` lives as long as its `SafeTensors` value. The writer sorts metadata keys; the reference writes them in HashMap order, so files with 2 or more metadata keys match it in content, not byte for byte.

## Checks

| Script | What |
|---|---|
| `scripts/check.sh` | 42 files read as the reference reads them (19 accepted, 23 rejected); 18 rewrites with the reference layout, 16 byte-identical, one read-back only (the reference cannot write F6); 5 checkpoint directories; the 50k-header time; API and example |
| `scripts/selfcheck.sh` | 6 mutants (duplicates, holes, dtype order, escaping, weight_map, padding) all rejected; the oracle passes its own gate |
| `scripts/bench.sh` | the speed numbers above, 3 runs each, peak RSS |
| `scripts/ci.sh` | install, fetch mojo-json v0.1.0, gate; same on GitHub and a local VM |

## License

Apache-2.0.
