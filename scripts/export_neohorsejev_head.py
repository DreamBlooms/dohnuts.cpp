#!/usr/bin/env python3
"""Exports the NeoHorse-Jev pointer head to the .f32 layout dohnuts reads.

The upstream head (pointer_head.safetensors) holds two nn.Linear layers, q and
k, each [pointer_dim, hidden] weight + [pointer_dim] bias. dohnuts reads one
file of 2 * pointer_dim * (hidden + 1) floats: the q rows first, then the k
rows, each row being [hidden, 1] with the bias as the last column. This matches
scripts/export_kev.py exactly.

Usage:
    python3 export_neohorsejev_head.py pointer_head.safetensors out.f32
"""
import struct
import sys


def main(src: str, dst: str) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError:
        sys.exit("pip install safetensors")
    tensors = load_file(src)
    # keys are "q.weight"/"q.bias"/"k.weight"/"k.bias" (nn.Linear naming).
    def get(name):
        for k, v in tensors.items():
            if k == name or k.endswith("." + name):
                return v
        raise KeyError(name)

    qw, qb = get("q.weight"), get("q.bias")
    kw, kb = get("k.weight"), get("k.bias")
    dp, hidden = qw.shape
    assert kw.shape == (dp, hidden), (qw.shape, kw.shape)
    assert qb.shape == (dp,) and kb.shape == (dp,)

    rows = []
    for w, b in ((qw, qb), (kw, kb)):
        for d in range(dp):
            rows.extend(w[d].tolist())
            rows.append(float(b[d]))
    with open(dst, "wb") as f:
        f.write(struct.pack("<%sf" % len(rows), *rows))
    print("wrote", dst, len(rows), "floats (dp=%d hidden=%d)" % (dp, hidden))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
