# mojo-safetensors

[![CI](https://github.com/amarbaro/mojo-safetensors/actions/workflows/ci.yml/badge.svg)](https://github.com/amarbaro/mojo-safetensors/actions/workflows/ci.yml)

Read and write [safetensors](https://github.com/huggingface/safetensors) files from Mojo. A
file is mmapped read-only and each tensor's data is a `Span` into the mapping, so nothing is
copied. The writer produces the reference's layout. The gate holds both sides to the reference
implementation (`huggingface/safetensors` 0.6.2):
- reading: 42 files, each accepted or rejected as the reference does
- writing: rewrites match the reference's bytes
- checkpoints: three real ones list identically
321 lines.

```mojo
from safetensors import SafeTensors, Writer, Checkpoint

def main() raises:
    var f = SafeTensors("model.safetensors")          # header parsed, data mmapped
    for t in f.tensors:                                # sorted by name
        print(t.name, t.dtype, t.numel(), t.nbytes())
    var w = f.data("lm_head.weight")                   # Span[UInt8] into the file, no copy

    var ck = Checkpoint("Qwen3-8B")                    # index.json + shards, validated
    print(len(ck.shards))
```

## Install

Requires Mojo 1.1 on Linux and [mojo-json](https://github.com/amarbaro/mojo-json) v0.1.0.
Builds on Mojo 1.2 nightly too.

| Way | Command |
|---|---|
| Source path | `mojo run -I path/to/mojo-safetensors/src -I path/to/mojo-json/src app.mojo` |
| Vendor | copy `src/safetensors/` and mojo-json's `src/json/` into your project |

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

## Checks

| Script | What |
|---|---|
| `scripts/check.sh` | 42 files read as the reference reads them (19 accepted, 23 rejected); 18 rewrites with the reference layout, 16 byte-identical, one read-back only (the reference cannot write F6); 5 checkpoint directories; the 50k-header time; API and example |
| `scripts/selfcheck.sh` | 6 mutants (duplicates, holes, dtype order, escaping, weight_map, padding) all rejected; the oracle passes its own gate |
| `scripts/bench.sh` | the speed numbers above, 3 runs each, peak RSS |
| `scripts/ci.sh` | install, fetch mojo-json v0.1.0, gate; same on GitHub and a local VM |

## License

Apache-2.0.
