# Checkpoints

The DetFill checkpoints used for the paper results (200 epochs,
Danbooru2021 illustrations) are attached to the
[v1.0 GitHub Release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0).

| File | Size | SHA-256 | Description |
|---|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 1.08 GB | `fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4` | Scribble-hint model, **96 base channels**, retrained during the revision; exactly reproduces the paper's Table II scribble results. |
| `detfill_dot_illust_200ep.pth` | 463 MB | `fd872563e17cb1ac09957ed5ef993fac4dbeb2710f43a7ee29203444928d6fd4` | Dot-hint model, **64 base channels**. This is the original-submission checkpoint and exactly reproduces the paper's dot results. A 96-channel dot retrain (matching the scribble configuration) did not improve dot scores (PSNR-HAUC 16.83 vs. 17.32), so the original checkpoint is retained for exact reproducibility. |

## Download and placement

```bash
BASE=https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/download/v1.0

mkdir -p detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint
curl -L $BASE/detfill_scribble_illust_200ep.pth \
     -o detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth

mkdir -p detfill/results/dataset_name/BrownianBridge_dot_illust/checkpoint
curl -L $BASE/detfill_dot_illust_200ep.pth \
     -o detfill/results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth
```

These are the default paths used by `detfill/run_inference.sh`; any other location
works with `python main.py ... --resume_model <path>`.

## Notes

- The per-hint-type configuration (96 base channels for scribble, 64 for dot) is
  stated in Sec. VI-A of the paper; each checkpoint ships with the matching config
  (use the 96-channel config for scribble and the 64-channel config for dot).
- DetFill is a pixel-space model: no VQGAN / latent-diffusion weights are needed.
- Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically
  by their pip packages on first use; they are not part of this release.
- Each checkpoint contains the EMA weights used for all reported results
  (saved by the training loop at epoch 200).
