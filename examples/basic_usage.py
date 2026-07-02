"""Minimal end-to-end example of the hintauc library.

1. Generate deterministic hints for a color illustration.
2. Mask them to several hint ratios (the paper's front-loaded grid).
3. Evaluate colorizations against the ground truth and compute Hint-AUC.

Run: python examples/basic_usage.py <color_image.png>
"""

import sys

import hintauc


def main(image_path):
    # ---- 1. deterministic hint generation -------------------------------
    hints = hintauc.generate_hints(image_path, size=64, verbose=True)
    print(f"regions: {hints.n_regions()}  (FilFinder failures: {hints.failed_regions})")
    hints.save("example_out")  # example_out_region64.png, _scribble_col64.png, ...

    # ---- 2. hints at each hint ratio (deterministic size-sort) ----------
    for alpha in hintauc.DEFAULT_ALPHAS:
        color, mask = hints.at_ratio(alpha, hint_type="scribble")
        print(f"alpha={alpha:5.2f}  hint pixels: {(mask > 0).sum()}")

    # ---- 3. evaluation ----------------------------------------------------
    # Compare an image with itself just to demonstrate the API; in practice
    # `pred` is your model's colorization output.
    ev = hintauc.Evaluator(metrics=("mse", "psnr", "ssim"))
    print("self-comparison:", ev(image_path, image_path))

    # Hint-AUC from per-ratio scores (here: dummy numbers from the paper's
    # supp Table IV size-sort MSE row -> expected HAUC = 0.0119)
    scores = {0.00: 0.1003, 0.01: 0.0501, 0.03: 0.0310, 0.05: 0.0239,
              0.10: 0.0166, 0.25: 0.0113, 0.50: 0.0089, 1.00: 0.0075}
    print("Hint-AUC (MSE):", round(hintauc.hint_auc(scores), 4))


if __name__ == "__main__":
    main(sys.argv[1])
