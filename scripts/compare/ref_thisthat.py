"""Reference runner for flock-io/this-that-model-1.2 (PyTorch), used to compare
against dohnuts.cpp.

Reads the same JSONL the CLI consumes, runs the release's own protocol
(thisthat.systemone_protocol), and prints one JSON answer object per line so the
two can be diffed field by field.

The `thisthat` package must be importable (pip install -e the repo). Pass its
path via THISTHAT_SRC if it is not installed.
"""

import json
import os
import sys
from pathlib import Path

import torch

src = os.environ.get("THISTHAT_SRC")
if src:
    sys.path.insert(0, src)
from thisthat import TypedDecider                      # noqa: E402
from thisthat.systemone_protocol import parse_request, render_answer  # noqa: E402


def main():
    model_dir = sys.argv[1]
    cases = sys.argv[2]
    decider = TypedDecider.from_pretrained(model_dir, device="cpu", dtype=torch.float32)
    with open(cases, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            state, typed = parse_request(request)
            decisions = decider.decide(state, [tq.question for tq in typed], temperature=1.0)
            answers = {tq.key: render_answer(tq, d) for tq, d in zip(typed, decisions)}
            print(json.dumps(answers, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
