"""Reference runner for decider-0.8b (PyTorch), used to compare against dohnuts.cpp.

Reads the same JSONL the CLI consumes and prints one JSON answer object per
line, so the two can be diffed field by field.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(sys.argv[1])))          # model dir with decider/
from decider.infer import Decider                    # noqa: E402


def main():
    model_dir = sys.argv[1]
    cases = sys.argv[2]
    d = Decider(model_dir, device="cpu", dtype=torch.float32, use_graphs=False)
    with open(cases, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            out = d.system_one(request["state"], request["questions"])
            print(json.dumps(out["answers"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
