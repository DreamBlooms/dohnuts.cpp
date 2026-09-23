"""Streaming LoRA merge for kev checkpoints too large to hold in RAM.

Merges `adapter_model.safetensors` (PEFT, `base_model.model.*`) into a sharded
HF base (`model.language_model.*`) one tensor at a time and writes a single
safetensors file, so peak memory stays at one tensor. The vision tower is
dropped (the target is a text CausalLM).

Usage: export_kev_stream.py <base_dir> <checkpoint_dir> <out_dir>
"""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

BASE = Path(sys.argv[1])
CHECKPOINT = Path(sys.argv[2])
OUT = Path(sys.argv[3])

DTYPE_STR = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}


def raw_bytes(t: torch.Tensor) -> bytes:
    """Reinterpret a tensor as raw bytes (works for bf16 too)."""
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def main():
    shards = (BASE / "model.safetensors.index.json")
    if shards.exists():
        weight_map = json.loads(shards.read_text())["weight_map"]
    else:
        weight_map = {k: "model.safetensors" for k in safe_open(str(next(BASE.glob("*.safetensors"))), "pt").keys()}

    adapter = {}
    with safe_open(str(CHECKPOINT / "adapter_model.safetensors"), "pt") as handle:
        for name in handle.keys():
            adapter[name] = handle.get_tensor(name)
    cfg = json.loads((CHECKPOINT / "adapter_config.json").read_text())
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])
    print(f"adapter tensors: {len(adapter)}, scale {scale}", flush=True)

    # Language tensors only, in a stable order.
    names = [k for k in weight_map if not k.startswith("model.visual.")]
    print(f"language tensors: {len(names)} (dropped {len(weight_map) - len(names)} vision)", flush=True)

    # Pass 1: header from shape/dtype, no data loaded.
    header = {}
    offset = 0
    slices = {}
    for name in names:
        shard = weight_map[name]
        if shard not in slices:
            slices[shard] = safe_open(str(BASE / shard), "pt")
        sl = slices[shard].get_slice(name)
        shape = list(sl.get_shape())
        dtype = sl.get_dtype()
        nbytes = int(np.prod(shape)) * DTYPE_STR[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes

    payload = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(payload) % 8)) % 8
    payload += b" " * pad

    OUT.mkdir(parents=True, exist_ok=True)
    merged_count = 0
    with open(OUT / "model.safetensors", "wb") as out:
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)
        for name in names:
            shard = weight_map[name]
            tensor = slices[shard].get_tensor(name)
            tail = name[len("model.language_model."):] if name.startswith("model.language_model.") else None
            a = b = None
            if tail is not None:
                prefix = "base_model.model." + tail[: -len(".weight")]
                a, b = adapter.get(prefix + ".lora_A.weight"), adapter.get(prefix + ".lora_B.weight")
            if a is not None and b is not None:
                merged = tensor.to(torch.float32) + scale * (b.to(torch.float32) @ a.to(torch.float32))
                merged = merged.to(torch.bfloat16)
                merged_count += 1
            else:
                merged = tensor
            out.write(raw_bytes(merged))
        out.flush()
        os.fsync(out.fileno())
    print(f"merged {merged_count} LoRA tensors -> {OUT/'model.safetensors'}", flush=True)

    # Config: re-route the converter to the text CausalLM and keep base params.
    config = json.loads((BASE / "config.json").read_text())
    text = config.get("text_config", config)
    text = dict(text)
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    (OUT / "config.json").write_text(json.dumps(text, indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (BASE / f).exists():
            shutil.copy(BASE / f, OUT / f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
