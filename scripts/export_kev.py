"""Export a Kev checkpoint: merge its LoRA into the base and write the pointer head.

Two merge strategies, same output layout:

  default    PEFT merge (PeftModel.merge_and_unload). Loads the full base into
             RAM; the trusted, general path.
  --stream   one tensor at a time, never holding the model. For checkpoints too
             large for the machine (kev-4b on a 16 GB box). It reimplements the
             LoRA merge by hand, so it fails loudly if any adapter tensor is not
             applied.

Both write a merged HF model, the bilinear pointer head (`kev-head.f32`, q then
k with the bias as the last column) and `kev.json`.
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def export_head(checkpoint: Path, out_dir: Path, version: str):
    meta = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=False)
    dim = int(meta["head_dim"])
    merged = []
    for name in ("q", "k"):
        w = meta["head"][f"{name}.weight"].float()
        b = meta["head"][f"{name}.bias"].float().unsqueeze(1)
        merged.append(torch.cat([w, b], dim=1).contiguous())
    combined = torch.cat(merged, dim=0)  # [2 * dim, hidden + 1]
    combined.numpy().astype("<f4").tofile(str(out_dir / "kev-head.f32"))
    config = {
        "profile": "kev",
        "pointer_dim": dim,
        "temperature": float(meta["temperature"]),
        "version": version,
    }
    (out_dir / "kev.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"head dim {dim}, temperature {meta['temperature']:.6f} -> kev-head.f32", flush=True)


def finalize(out_dir: Path, base: Path):
    """Point the converter at the text CausalLM and carry over the tokenizer."""
    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config_path.write_text(json.dumps(config, indent=2))
    for name in TOKENIZER_FILES:
        if (base / name).exists():
            shutil.copy(base / name, out_dir / name)


def merge_peft(base: Path, checkpoint: Path, out: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print("loading base ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    print("applying adapter ...", flush=True)
    model.model = PeftModel.from_pretrained(model.model, str(checkpoint), torch_device="cpu")
    model.model = model.model.to(torch.float32)
    print("merging ...", flush=True)
    model.model = model.model.merge_and_unload(safe_merge=True)
    model = model.to(torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)


def _raw_bytes(t: torch.Tensor) -> bytes:
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def merge_stream(base: Path, checkpoint: Path, out: Path):
    index = base / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        single = next(base.glob("*.safetensors"))
        with safe_open(str(single), "pt") as handle:
            weight_map = {k: single.name for k in handle.keys()}

    adapter = {}
    with safe_open(str(checkpoint / "adapter_model.safetensors"), "pt") as handle:
        for name in handle.keys():
            adapter[name] = handle.get_tensor(name)
    config = json.loads((checkpoint / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    print(f"adapter tensors: {len(adapter)}, scale {scale}", flush=True)

    # Language tensors only, in a stable order.
    names = [k for k in weight_map if not k.startswith("model.visual.")]
    print(f"language tensors: {len(names)} (dropped {len(weight_map) - len(names)} vision)", flush=True)

    # Base prefix differs between the multimodal and text-only checkpoints.
    prefix = next((p for p in ("model.language_model.", "model.")
                   if any(n.startswith(p) for n in names)), "")
    if not prefix:
        raise SystemExit("cannot determine the base tensor prefix")

    # Pass 1: header from shape/dtype, no data loaded.
    header, offsets, slices = {}, 0, {}
    for name in names:
        shard = weight_map[name]
        if shard not in slices:
            slices[shard] = safe_open(str(base / shard), "pt")
        slice_ = slices[shard].get_slice(name)
        shape = list(slice_.get_shape())
        dtype = slice_.get_dtype()
        nbytes = int(np.prod(shape)) * DTYPE_BYTES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offsets, offsets + nbytes]}
        offsets += nbytes
    payload = json.dumps(header, separators=(",", ":")).encode()
    payload += b" " * ((8 - len(payload) % 8) % 8)

    out.mkdir(parents=True, exist_ok=True)
    consumed = set()
    merged_count = 0
    with open(out / "model.safetensors", "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for name in names:
            tensor = slices[weight_map[name]].get_tensor(name)
            a = b = None
            if name.startswith(prefix) and name.endswith(".weight"):
                tail = name[len(prefix): -len(".weight")]
                for adapter_name in ("", ".default"):
                    base_name = f"base_model.model.{tail}.lora_A{adapter_name}.weight"
                    if base_name in adapter:
                        a = adapter[base_name]
                        b = adapter[base_name.replace("lora_A", "lora_B")]
                        consumed.update((base_name, base_name.replace("lora_A", "lora_B")))
                        break
            if a is not None and b is not None:
                dtype = tensor.dtype
                tensor = (tensor.to(torch.float32) + scale * (b.to(torch.float32) @ a.to(torch.float32))).to(dtype)
                merged_count += 1
            handle.write(_raw_bytes(tensor))
        handle.flush()
        os.fsync(handle.fileno())

    # Fail loudly rather than ship a silently unmerged model.
    expected = {k for k in adapter if ".lora_A" in k or ".lora_B" in k}
    missing = expected - consumed
    if missing:
        raise SystemExit(f"{len(missing)} adapter tensors were not applied, e.g. {sorted(missing)[:3]}")
    print(f"merged {merged_count} LoRA tensors -> {out / 'model.safetensors'}", flush=True)

    # save_pretrained would write this; streaming writes it from the base config.
    config = json.loads((base / "config.json").read_text())
    config = dict(config.get("text_config", config))
    (out / "config.json").write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="base model directory")
    parser.add_argument("checkpoint", type=Path, help="kev checkpoint directory")
    parser.add_argument("out", type=Path, help="output merged model directory")
    parser.add_argument("--stream", action="store_true",
                        help="merge one tensor at a time (low memory; for large checkpoints)")
    parser.add_argument("--version", default=None, help="value for kev.json (default: from the checkpoint name)")
    args = parser.parse_args()

    version = args.version or args.checkpoint.name.removeprefix("kev-") or "unknown"
    if args.stream:
        merge_stream(args.base, args.checkpoint, args.out)
    else:
        merge_peft(args.base, args.checkpoint, args.out)
    finalize(args.out, args.base)
    # The head goes next to the merged model, like the published layout.
    export_head(args.checkpoint, args.out.parent, version)
    print("done", flush=True)


if __name__ == "__main__":
    main()
