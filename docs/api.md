# API

`from safetensors import SafeTensors, TensorInfo, Writer, Checkpoint, dtype_bits, MAX_HEADER`

## SafeTensors(path) raises

Opens and mmaps the whole file read-only (unmapped when the value is destroyed), parses and
validates the header. Errors are `safetensors: <path>: <what>`: `header too small`,
`header too large`, `invalid header length`, `invalid JSON in header: ...`,
`header is not an object`, `bad entry for tensor X`, `bad shape for tensor X`,
`bad data_offsets for tensor X`, `unknown dtype D`, `invalid offset for tensor X`,
`invalid shape, data type, or offset for tensor X`, `incomplete metadata, file not fully covered`,
`duplicate tensor X`, `__metadata__ is not an object`, `metadata value is not a string`.

| Member | What |
|---|---|
| `tensors: List[TensorInfo]` | sorted by name |
| `find(name) -> Int` | index into `tensors`, -1 when absent; binary search |
| `data(i) -> Span[UInt8]`, `data(name) raises -> Span[UInt8]` | the tensor's bytes inside the mapping; valid while the `SafeTensors` lives |
| `metadata(key) -> Optional[String]`, `meta_keys`, `meta_vals` | `__metadata__`, in file order |
| `bytes() -> Span[UInt8]` | the whole file |
| `path`, `size`, `data_start` | file path, file size, offset of the data section (8 + header length) |

## TensorInfo

`name`, `dtype` (the header string, for example `BF16`), `shape: List[Int]`, `begin`, `end`
(offsets inside the data section), `numel()`, `nbytes()`.

## Writer

| Member | What |
|---|---|
| `Writer()` | empty |
| `add(name, dtype, shape, data) raises` | copies `data`; raises on an unknown dtype, a size that is not elements × bits / 8, or a duplicate name |
| `set_metadata(key, value)` | adds or replaces |
| `save(path) raises` | writes the file |

The layout is the reference's: tensors ordered by dtype from U64 down to BOOL, then by name,
contiguous; header as compact JSON, `__metadata__` first, padded with spaces to a multiple of 8;
strings keep UTF-8 and escape only `"`, `\` and control characters. Metadata keys are sorted.

## Checkpoint(dir) raises

`shards: List[SafeTensors]`, sorted by file name. With `model.safetensors.index.json`, the
shards are the files its `weight_map` names; the tensor count must match and every entry must
be in the shard it names. Without it, every `model*.safetensors` in `dir`.

## dtype_bits(dtype) -> Int

Bits per element: 4 for F4, 6 for F6_E2M3 and F6_E3M2, 8 to 64 for the others, 0 if unknown.
