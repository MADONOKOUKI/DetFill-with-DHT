#!/usr/bin/env python3
"""Compare the Hint-AUC of a result.json with the numbers recorded in examples/expected_numbers.json.

    python examples/check_expected.py <result.json> <case>      # case: demo | quickstart_fast | quickstart

Both evaluation backends (torchvision + torchmetrics, or Pillow + scikit-image) must reproduce the recorded values
within the tolerances of the file; the backend in use is read from the result's protocol record.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    result = json.load(open(sys.argv[1]))
    case = sys.argv[2]
    expected = json.load(open(os.path.join(HERE, "expected_numbers.json")))
    if case not in expected:
        sys.exit(f"unknown case {case!r}; known: {sorted(k for k in expected if not k.startswith('_'))}")
    spec = expected[case]
    backend = result.get("protocol", {}).get("backend", {})
    key = "torch" if backend.get("resize") == "torchvision" and backend.get("ssim") == "torchmetrics" else "pillow"
    ref = spec["hint_auc"][key]
    tol = spec["tolerance"]
    got = result["hint_auc"]
    ok = True
    for metric, value in ref.items():
        diff = abs(float(got[metric]) - float(value))
        flag = "ok " if diff <= tol[metric] else "FAIL"
        if diff > tol[metric]:
            ok = False
        print(f"{flag} {case} [{key}] {metric}: got {float(got[metric]):.6f} expected {float(value):.6f} (|diff| {diff:.2e} <= {tol[metric]})")
    if not ok:
        sys.exit(1)
    print(f"{case}: all recorded values reproduced with the {key} backend")


if __name__ == "__main__":
    main()
