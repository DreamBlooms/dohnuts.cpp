"""Export the Kev-0.8B LoRA and pointer head to files the side engine reads.

Merges the adapter into the pinned Qwen3.5-0.8B-Base text trunk in fp32, writes
a merged HF model, and exports the bilinear pointer head as two float32 matrices
with the bias appended as the last column plus a small JSON config.
"""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

BASE = Path(sys.argv[1])        # Qwen3.5-0.8B-Base snapshot
CHECKPOINT = Path(sys.argv[2])  # jaredpalmer/kev-0.8b snapshot (adapter + head.pt)
OUT = Path(sys.argv[3])         # merged HF model directory


def main():
    meta = torch.load(CHECKPOINT / "head.pt", map_location="cpu", weights_only=False)
    assert meta["head_dim"] == 256, meta["head_dim"]

    print("loading base ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()

    # Kev wraps the text trunk and applies the adapter with PeftModel, which
    # also handles the ".default" adapter-name handling in the saved keys.
    print("applying adapter ...", flush=True)
    model.model = PeftModel.from_pretrained(model.model, str(CHECKPOINT), torch_device="cpu")
    model.model = model.model.to(torch.float32)

    print("merging ...", flush=True)
    model.model = model.model.merge_and_unload(safe_merge=True)
    model = model.to(torch.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)

    # The converter routes on architectures[0]; keep the text CausalLM name.
    config_path = OUT / "config.json"
    out_cfg = json.loads(config_path.read_text())
    out_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    config_path.write_text(json.dumps(out_cfg, indent=2))

    # Pointer head: [dim, hidden + 1] rows (bias last), q then k, in one file.
    head = meta["head"]
    dim = meta["head_dim"]
    work = OUT.parent
    merged = []
    for name in ("q", "k"):
        w = head[f"{name}.weight"].float()                 # [dim, hidden]
        b = head[f"{name}.bias"].float().unsqueeze(1)      # [dim, 1]
        merged.append(torch.cat([w, b], dim=1).contiguous())  # [dim, hidden + 1]
    combined = torch.cat(merged, dim=0)                     # [2 * dim, hidden + 1]
    combined.numpy().astype("<f4").tofile(str(work / "kev-head.f32"))

    kev_cfg = {
        "profile": "kev",
        "pointer_dim": dim,
        "temperature": float(meta["temperature"]),
        "version": "0.8b",
    }
    (work / "kev.json").write_text(json.dumps(kev_cfg, indent=2))
    print("head dim", dim, "temperature", meta["temperature"], "-> kev-head.f32", flush=True)


if __name__ == "__main__":
    main()
