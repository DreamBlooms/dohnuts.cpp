#!/usr/bin/env python3
"""Compare two /v1/systemone endpoints (e.g. dohnuts and upstream llama-server).

Each request in the JSONL is sent to both endpoints and the answers are compared
per question. Prints the per-question deltas and a summary.

    decision_parity.py requests.jsonl http://127.0.0.1:8080 http://127.0.0.1:8081
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def post(base: str, key: str | None, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/systemone", data=data,
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def answer_of(result: dict, qid: str) -> dict:
    if "answers" in result:
        return result["answers"].get(qid, {})
    if "results" in result and result["results"]:
        return result["results"][0].get("answers", {}).get(qid, {})
    return {}


def close(a, b, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def compare_field(ra: dict, rb: dict, field: str, tol: float) -> tuple[bool, str]:
    va, vb = ra.get(field), rb.get(field)
    if va is None and vb is None:
        return True, ""
    if va is None or vb is None:
        return False, f"{field}: {va!r} vs {vb!r}"
    if isinstance(va, dict) and isinstance(vb, dict):
        if set(va) != set(vb):
            return False, f"{field}: keys differ {sorted(va)} vs {sorted(vb)}"
        bad = [(k, va[k], vb[k]) for k in va if not close(va[k], vb[k], tol)]
        if bad:
            return False, f"{field}: " + ", ".join(f"{k} {x:.4f} vs {y:.4f}" for k, x, y in bad)
        return True, ""
    if not close(va, vb, tol):
        return False, f"{field}: {va!r} vs {vb!r}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("a", help="endpoint A (e.g. dohnuts)")
    ap.add_argument("b", help="endpoint B (e.g. upstream llama-server)")
    ap.add_argument("--key-a")
    ap.add_argument("--key-b")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    total = matched = 0
    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            body = json.loads(line)
            ra = post(args.a, args.key_a, body, args.timeout)
            rb = post(args.b, args.key_b, body, args.timeout)
            for qid, q in body.get("questions", {}).items():
                total += 1
                aa, ab = answer_of(ra, qid), answer_of(rb, qid)
                ok = True
                detail = ""
                for field in ("type", "choice", "noul", "score", "confidence"):
                    good, msg = compare_field(aa, ab, field, args.tol)
                    if not good:
                        ok = False
                        detail = msg
                        break
                if not ok:
                    good, msg = compare_field(aa, ab, "probabilities", args.tol)
                    ok, detail = good, msg
                matched += 1 if ok else 0
                print(f"{'OK  ' if ok else 'DIFF'} {qid}" + (f"  {detail}" if detail else ""))

    print(f"\n{matched}/{total} questions match (tol={args.tol})")
    sys.exit(0 if total and matched == total else 1)


if __name__ == "__main__":
    main()
