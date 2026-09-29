"""Merge the Linnaeus LoRA into Qwen3.5-2B and export a GGUF-ready HF model.

Same serving path as Dohnuts (see export_dohnuts.py): wrap language_model with
PEFT, load the adapter, merge, save a plain HF model with the multimodal
architecture name, and write the scalar scorer head as raw f32.

Linnaeus is a rank-8 LoRA on Qwen/Qwen3.5-2B. The merge runs in bf16 so the
~2.4B-parameter checkpoint fits in RAM on a 16 GB box; the dtype only affects
rounding before the GGUF f16 conversion.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModel

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
COPY_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "preprocessor_config.json", "chat_template.jinja",
]
META = "linnaeus.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Qwen3.5-2B directory")
    parser.add_argument("--adapter", required=True, help="Linnaeus checkpoint directory")
    parser.add_argument("--out", required=True, help="output merged model directory")
    parser.add_argument("--head", required=True, help="output path for head.f32")
    args = parser.parse_args()

    base = Path(args.base)
    adapter = Path(args.adapter)
    out = Path(args.out)
    cfg = json.loads((adapter / META).read_text())
    assert cfg["lora_rank"] == 8, cfg["lora_rank"]

    print("loading base (bf16) ...", flush=True)
    model = AutoModel.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()

    print("applying LoRA ...", flush=True)
    model.language_model = get_peft_model(
        model.language_model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                   target_modules=TARGET_MODULES),
    )
    model.language_model = model.language_model.to(torch.bfloat16)

    state = load_file(str(adapter / "adapter.safetensors"))
    head = state.pop("head.weight")
    hidden = model.config.text_config.hidden_size
    assert head.shape == (1, hidden), head.shape

    backbone_state = {k[len("backbone."):]: v for k, v in state.items()}
    params = {
        n: p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("language_model.")
    }
    if set(backbone_state) != set(params):
        missing = sorted(set(params) - set(backbone_state))
        extra = sorted(set(backbone_state) - set(params))
        raise SystemExit(f"key mismatch: missing={missing[:5]} extra={extra[:5]}")
    for name, value in backbone_state.items():
        if params[name].shape != value.shape:
            raise SystemExit(f"shape mismatch {name}")
        with torch.no_grad():
            params[name].copy_(value.to(torch.bfloat16))

    print("merging ...", flush=True)
    model.language_model = model.language_model.merge_and_unload(safe_merge=True)
    model = model.to(torch.bfloat16)

    out.mkdir(parents=True, exist_ok=True)
    print(f"saving {out} ...", flush=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="10GB")
    del model
    gc.collect()

    # AutoModel reports Qwen3_5Model; restore the multimodal arch and normalize
    # the key prefix to the Qwen3_5ForConditionalGeneration layout the llama.cpp
    # converter expects, and drop the unused MTP layer from the config.
    weights = out / "model.safetensors"
    tensors = load_file(str(weights))
    renamed = {
        k if k.startswith(("model.", "mtp.")) else f"model.{k}": v
        for k, v in tensors.items()
    }
    del tensors
    gc.collect()
    save_file(renamed, str(weights), metadata={"format": "pt"})

    config_path = out / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    config.get("text_config", {}).pop("mtp_num_hidden_layers", None)
    config_path.write_text(json.dumps(config, indent=2))

    for name in COPY_FILES:
        source = base / name
        if source.exists():
            (out / name).write_bytes(source.read_bytes())

    Path(args.head).write_bytes(head.float().numpy().astype("<f4").tobytes())
    print("wrote", args.head, tuple(head.shape))


if __name__ == "__main__":
    main()
