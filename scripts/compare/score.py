"""Compare dohnuts.cpp and PyTorch answers line by line.

Both files hold one JSON answer object per case in the same order. Choice
questions are compared exactly; noul and score are numeric, so a small
tolerance absorbs quantization and rounding (the choice, i.e. the decision, is
what matters). Probabilities are reported as the maximum absolute difference.
"""

import json
import sys

TOLERANCE = 0.05


def load(path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def decision(answer):
    """(kind, comparable value, tolerance)."""
    if "choice" in answer:
        return "choice", answer["choice"], 0.0
    if "noul" in answer:
        return "noul", answer["noul"], TOLERANCE
    if "score" in answer:
        return "score", answer["score"], TOLERANCE
    return "?", None, 0.0


def probs(answer):
    p = answer.get("probabilities")
    return {k: round(v, 4) for k, v in p.items()} if p else None


def main():
    cpp = load(sys.argv[1])
    ref = load(sys.argv[2])
    if len(cpp) != len(ref):
        print(f"line count differs: cpp {len(cpp)} vs py {len(ref)}")
    agree = total = 0
    for index, (c, r) in enumerate(zip(cpp, ref), 1):
        c_answers = c.get("answers", {})
        for key, r_ans in r.items():
            total += 1
            c_ans = c_answers.get(key, {})
            kind, c_val, tol = decision(c_ans)
            _, r_val, _ = decision(r_ans)
            if kind in ("noul", "score"):
                same = abs(c_val - r_val) <= tol
                shown = f"cpp={c_val} py={r_val} (tol {tol})"
            else:
                same = c_val == r_val
                shown = f"cpp={c_val} py={r_val}"
            agree += same
            print(f"case {index} {key:16s} {'ok  ' if same else 'DIFF'} {shown}")
            cp, rp = probs(c_ans), probs(r_ans)
            if cp and rp:
                shared = set(cp) & set(rp)
                delta = max(abs(cp[k] - rp[k]) for k in shared) if shared else 0.0
                print(f"                                  max prob diff {delta:.4f}")
                if cp != rp:
                    picks_c = max(cp, key=cp.get)
                    picks_r = max(rp, key=rp.get)
                    if picks_c != picks_r:
                        print(f"                                  argmax differs: {picks_c} vs {picks_r}")
    print(f"\nagreement {agree}/{total} = {agree / total:.1%}" if total else "no answers")


if __name__ == "__main__":
    main()
