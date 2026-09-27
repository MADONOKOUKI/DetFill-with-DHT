#!/usr/bin/env python3
"""Compare the Hint-AUC of a result.json with the numbers recorded in examples/expected_numbers.json.

    python examples/check_expected.py <result.json> <case> [--backend torch|pillow]

``case`` is ``demo``, ``quickstart_fast`` or ``quickstart``. The backend that produced the result is read from the
result's protocol record; ``--backend`` states which one is expected (CI passes it), so a result without a protocol
record, or from the wrong backend, fails instead of being compared with the fallback numbers. Every recorded metric
must be present, finite and within the tolerance of the file; the exit status is 1 otherwise.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def backend_of(result):
    b = result.get("protocol", {}).get("backend")
    if not isinstance(b, dict) or "resize" not in b or "ssim" not in b:
        return None
    return "torch" if (b["resize"] == "torchvision" and b["ssim"] == "torchmetrics") else "pillow"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("result")
    ap.add_argument("case")
    ap.add_argument("--backend", choices=["torch", "pillow"], default=None, help="the backend the result must come from")
    a = ap.parse_args()
    result = json.load(open(a.result))
    expected = json.load(open(os.path.join(HERE, "expected_numbers.json")))
    if a.case not in expected:
        sys.exit(f"unknown case {a.case!r}; known: {sorted(k for k in expected if not k.startswith('_'))}")
    spec = expected[a.case]
    key = backend_of(result)
    if key is None:
        sys.exit(f"FAIL {a.case}: the result has no protocol.backend record")
    if a.backend and key != a.backend:
        sys.exit(f"FAIL {a.case}: result comes from the {key} backend, expected {a.backend}")
    ref, tol, got = spec["hint_auc"][key], spec["tolerance"], result.get("hint_auc", {})
    failures = 0
    for metric, value in ref.items():
        v = got.get(metric)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(float(v) - float(value)) <= tol[metric]
        failures += not ok
        shown = f"{float(v):.6f}" if isinstance(v, (int, float)) else repr(v)
        print(f"{'ok  ' if ok else 'FAIL'} {a.case} [{key}] {metric}: got {shown} expected {float(value):.6f} (tolerance {tol[metric]})")
    if failures:
        sys.exit(1)
    print(f"{a.case}: all recorded values reproduced with the {key} backend")


if __name__ == "__main__":
    main()
