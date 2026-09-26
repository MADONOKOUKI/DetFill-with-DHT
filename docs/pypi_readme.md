# hintauc

Deterministic region-based hint generation (DHT) and Hint-AUC evaluation for line-art colorization, from
*Hint-AUC: Deterministic Region-based Hint Generation for Line Art Colorization Evaluation*
(IEEE Transactions on Visualization and Computer Graphics, 2026, DOI 10.1109/TVCG.2026.3738401).

```bash
pip install hintauc                 # deterministic hints, pixel metrics, Hint-AUC
pip install "hintauc[perceptual]"   # + LPIPS / OpenCLIP / DINOv2 / DreamSim
```

```python
import hintauc
hints = hintauc.generate_hints("illustration.png", size=64)      # image -> deterministic scribble and dot hints
color, mask = hints.at_ratio(0.10, hint_type="scribble")         # hints of the largest 10 % of the regions
evaluator = hintauc.Evaluator(metrics=("psnr", "lpips", "dreamsim"))
print(evaluator("colorized.png", "ground_truth.png"))
```

Command line: `hintauc generate image.png --ratio 0.1`, `hintauc eval pred_dir gt_dir --metrics mse psnr ssim`.

Documentation, the DetFill colorization model, all checkpoints and the reproduction package:
https://github.com/MADONOKOUKI/DetFill-with-DHT
