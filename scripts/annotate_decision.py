#!/usr/bin/env python3
"""Annotate an existing GGUF with the official llama.cpp decision metadata.

Writes "<arch>.decision.type", "<arch>.decision.temperature.<name>" and a named
"systemone" chat template, so the same GGUF can be served by both dohnuts
(--profile official) and upstream llama-server (/v1/systemone).

The prompt is not hard-coded here: pass a Jinja template with --template, or a
shipped one with --profile. See scripts/systemone_templates/.

Usage:
    annotate_decision.py in.gguf out.gguf --profile decider [--type openjev]
        [--template tpl.jinja] [--temperature choice=1.0]...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA = ROOT / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGUFValueType  # noqa: E402


def _load_new_metadata():
    spec = importlib.util.spec_from_file_location(
        "gnm", LLAMA / "gguf-py" / "gguf" / "scripts" / "gguf_new_metadata.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per-profile defaults: the official readout type and the shipped template.
# Add a profile here once its systemone template is authored and parity-checked
# (see scripts/compare/decision_parity.py). Others: pass --type and --template.
PROFILE_DEFAULTS = {
    "decider": ("openjev", "decider.jinja"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS))
    ap.add_argument("--type", dest="decision_type")
    ap.add_argument("--template", type=Path, help="systemone Jinja template file")
    ap.add_argument("--temperature", action="append", default=[], metavar="NAME=VALUE",
                    help="decision temperature, e.g. choice=1.0 (repeatable)")
    ap.add_argument("--version", help="informational general.version string")
    args = ap.parse_args()

    decision_type = args.decision_type
    template_path = args.template
    if args.profile:
        default_type, default_tpl = PROFILE_DEFAULTS[args.profile]
        decision_type = decision_type or default_type
        template_path = template_path or (ROOT / "scripts" / "systemone_templates" / default_tpl)
    if decision_type is None:
        ap.error("either --profile or --type is required")
    if template_path is None:
        ap.error("either --profile or --template is required")

    template = template_path.read_text(encoding="utf-8")

    reader = gguf.GGUFReader(str(args.input), "r")
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        sys.exit("input GGUF has no general.architecture")
    arch = arch_field.contents()

    gnm = _load_new_metadata()
    D = gnm.MetadataDetails
    new = {
        gguf.Keys.Tokenizer.CHAT_TEMPLATE: D(GGUFValueType.STRING, [{"name": "systemone", "template": template}]),
        f"{arch}.decision.type": D(GGUFValueType.STRING, decision_type),
    }
    for item in args.temperature:
        name, _, value = item.partition("=")
        new[f"{arch}.decision.temperature.{name}"] = D(GGUFValueType.FLOAT32, float(value))
    if args.version:
        new[gguf.Keys.General.VERSION] = D(GGUFValueType.STRING, args.version)

    out_path = args.output
    tmp_path = None
    if args.output.resolve() == args.input.resolve():
        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        out_path = tmp_path
    writer = gguf.GGUFWriter(str(out_path), arch=arch, endianess=reader.endianess)
    gnm.copy_with_new_metadata(reader, writer, new, [])
    if tmp_path is not None:
        os.replace(tmp_path, args.output)
    print(f"annotated {args.output}: {arch}.decision.type={decision_type}, template={template_path}")


if __name__ == "__main__":
    main()
