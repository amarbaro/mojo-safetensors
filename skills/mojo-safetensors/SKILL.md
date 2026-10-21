---
name: mojo-safetensors
description: "Read (mmapped, zero copy) and write .safetensors files and HF checkpoint directories from Mojo, validated as the reference does. Use when Mojo code loads model weights from safetensors or writes them."
---

# mojo-safetensors

`mojo run -I <repo>/src -I <mojo-json>/src app.mojo`.

    from safetensors import SafeTensors, Writer, Checkpoint
    var f = SafeTensors("model.safetensors")    # f.tensors sorted by name
    var w = f.data("lm_head.weight")            # Span into the mmap; keep f alive while using it
    var ck = Checkpoint("model_dir")            # index.json + shards

Writer: `add(name, dtype, shape, bytes)`, `set_metadata`, `save`. Check: `scripts/check.sh` ends `PASS all`.
