# Checkpoints

The DetFill checkpoints used for the paper results (200 epochs, Danbooru2021
illustrations) are attached to the
[v1.0 GitHub Release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0).

| File | Size | SHA-256 | Description |
|---|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 1.08 GB | `fa85d6838a28b853e82f16ce28e3f5d9e6452879685b1313d84be11431226ee4` | Scribble-hint model, 96 base channels; reproduces the paper's Table II/III scribble results. |
| `detfill_dot_illust_200ep.pth` | 0.48 GB | `fd872563e17cb1ac09957ed5ef993fac4dbeb2710f43a7ee29203444928d6fd4` | Dot-hint model, 64 base channels; the checkpoint behind the paper's Table II/III dot results. |

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

These are the default paths used by `detfill/run_inference.sh` (`scribble` / `dot`);
any other location works with `python main.py ... --resume_model <path>`.

## Notes

- Per-hint-type configuration as described in the paper: `configs/scribble_illust.yaml`
  (96 base channels) with the scribble checkpoint, `configs/dot_illust.yaml`
  (64 base channels) with the dot checkpoint.
- DetFill is a pixel-space model: no VQGAN / latent-diffusion weights are needed.
- Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically
  by their pip packages on first use; they are not part of this release.
- Both checkpoints contain the EMA weights used for all reported results
  (saved by the training loop at epoch 200).
- Inference uses the fixed global seed `1234` (`main.py --seed`, default); see the paper
  (Sec. VI-A) for the inference-seed policy. Equal-area region ties in the hint selection
  follow `np.argsort` of the pinned NumPy version (`environment.yml`, numpy==2.0.2).
