"""Reference runner for kev-0.8b (PyTorch), used to compare against dohnuts.cpp.

Reads the same JSONL the CLI consumes and prints one answer object per line.
The upstream `kev` package must be importable: point PYTHONPATH at a checkout of
github.com/jaredpalmer/kev (the directory containing its `kev/` folder).
"""

import json
import sys
from pathlib import Path

import torch

from kev.api import SystemOneRequest, to_answers, to_record   # noqa: E402
from kev.checkpoint import Checkpoint                          # noqa: E402


def main():
    checkpoint_dir = sys.argv[1]
    cases = sys.argv[2]
    checkpoint = Checkpoint(checkpoint_dir)
    tok, model = checkpoint.load("cpu")
    with open(cases, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            req = SystemOneRequest(**json.loads(line))
            record, meta = to_record(req)
            with torch.no_grad():
                enc = model.encode(tok, record)
                probs = [p.tolist() for p in model.probs(enc)]
            print(json.dumps(to_answers(probs, meta), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
