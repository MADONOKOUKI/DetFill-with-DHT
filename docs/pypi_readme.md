# hintauc

Deterministic region-based hint generation (DHT) and Hint-AUC evaluation for line-art colorization, from
*Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation*
(IEEE Transactions on Visualization and Computer Graphics, 2026, DOI 10.1109/TVCG.2026.3738401).

```bash
pip install hintauc                 # deterministic hints, pixel metrics, Hint-AUC
pip install "hintauc[paper]"        # + torchvision / torchmetrics 1.4.0: the exact evaluation backend of the paper
pip install "hintauc[perceptual]"   # + LPIPS / OpenCLIP / DINOv2 / DreamSim
hintauc demo                        # self-contained example (synthetic image, ~10 s): hints -> curve -> Hint-AUC
```

```python
import hintauc
hints = hintauc.generate_hints("illustration.png", size=64)      # image -> deterministic scribble and dot hints
color, mask = hints.at_ratio(0.10, hint_type="scribble")         # hints of the largest 10 % of the regions
evaluator = hintauc.Evaluator(metrics=("psnr", "lpips", "dreamsim"))
print(evaluator("colorized.png", "ground_truth.png"))
```

Command line: `hintauc generate image.png --ratio 0.1`, `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`,
`hintauc curve pred_root gt_dir` (Hint-AUC from one directory of outputs per hint ratio; every ratio must hold the
same images). Your own model as a Python function: `hintauc.evaluate_colorizer(fn, samples)`. Every result carries a
protocol record (grid, metric settings, the resize / SSIM backend in use, library versions).

Documentation (setup, "evaluate your own model", API reference with input/output examples, model zoo,
reproduction guide): https://madonokouki.github.io/projects/hintauc/docs/ — project page:
https://madonokouki.github.io/projects/hintauc/ — repository (DetFill colorization model, checkpoints, reproduction
package): https://github.com/MADONOKOUKI/DetFill-with-DHT
