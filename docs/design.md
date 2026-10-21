# Design

Rules: tensor data is never copied on read, the header is walked once, and every rule comes from
the reference implementation (huggingface/safetensors 0.6.2), checked by the gate.

## mmap, spans

`SafeTensors` maps the whole file read-only (`PROT_READ`, `MAP_PRIVATE`) and keeps only the
parsed header. `data(i)` is a `Span` into the mapping, tied to the `SafeTensors` value's lifetime;
pages are read when touched. Peak RSS is the header plus the pages you read: listing a
2,707-tensor shard takes 13 MB.

## One pass over the header

mojo-json parses the header into one node array in preorder, each node with its subtree size.
The reader walks that array once. The first version used `Json.at(i)`, which is O(i) and copies
a subtree, plus insertion sorts: 4.09 s for a 50,000-tensor header. The linear walk with
`sort` on small keys takes 0.10 s. The gate fails if it goes over 1 s again.

## Validation is the reference's

The reference sorts tensors by offsets and requires them to tile the data section exactly.
Sizes must equal elements times dtype bits / 8 (F4 is 4 bits, F6 is 6). Duplicate names are
rejected; serde keeps the last duplicate, and the offsets then fail. The reference accepts
leading whitespace and unknown fields, so this reader does too.

## Writer layout

The reference writes tensors ordered by its dtype enum from U64 down to BOOL, then by name,
contiguous, with the header padded to 8 with spaces. The Writer sorts on one key string,
a rank character followed by the name. Strings keep UTF-8 and escape only `"`, `\` and control
characters, as serde_json does. The reference writes metadata in HashMap order, which differs
from run to run (three runs gave three files). The Writer sorts the keys instead, so files with
2+ metadata keys match the reference in content, and the gate checks exactly that.

## Speed

`scripts/bench.sh` (open and list): 1,500 tensors in 1 ms against 163 ms, and a real 2,707-tensor
shard in 2 ms against 505 ms.
