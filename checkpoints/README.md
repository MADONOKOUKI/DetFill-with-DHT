# Checkpoints

The DetFill scribble checkpoint used for the paper results (96 base channels,
200 epochs, Danbooru2021 illustrations) is attached to the
[v1.0 GitHub Release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0).

| File | Size | SHA-256 | Description |
|---|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 1.08 GB | `fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4` | Scribble-hint model, 96 base channels; reproduces the paper's Table II scribble results. |

The dot-hint checkpoint (64 base channels; see Sec. VI-A of the paper) is
available from the authors upon request.

## Download and placement

```bash
BASE=https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/download/v1.0

mkdir -p detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint
curl -L $BASE/detfill_scribble_illust_200ep.pth \
     -o detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth
```

This is the default path used by `detfill/run_inference.sh`; any other location
works with `python main.py ... --resume_model <path>`.

## Notes

- The per-hint-type configuration (96 base channels for scribble, 64 for dot) is
  described in the paper; use `configs/scribble_illust.yaml` (96 channels) with
  this checkpoint.
- DetFill is a pixel-space model: no VQGAN / latent-diffusion weights are needed.
- Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically
  by their pip packages on first use; they are not part of this release.
- The checkpoint contains the EMA weights used for all reported results
  (saved by the training loop at epoch 200).
