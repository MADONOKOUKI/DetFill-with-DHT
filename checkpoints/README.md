# Checkpoints

The DetFill checkpoints used for the paper results (96 base channels, 200 epochs,
Danbooru2021 illustrations) are attached to the
[v1.0 GitHub Release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.0).

| File | Size | SHA-256 | Description |
|---|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 463 MB | `249ac36b301a2eeac30f880786d3ab07f09de8517913719723cbbce5dff335dd` | Scribble-hint model (the paper's Table II scribble results) |
| `detfill_dot_illust_200ep.pth` | 463 MB | `fd872563e17cb1ac09957ed5ef993fac4dbeb2710f43a7ee29203444928d6fd4` | Dot-hint model |

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

These are the default paths used by `detfill/run_inference_mr.sh`; any other location
works with `python main.py ... --resume_model <path>`.

## Notes

- DetFill is a pixel-space model: no VQGAN / latent-diffusion weights are needed.
- Metric backbones (LPIPS, OpenCLIP, DINOv2, DreamSim) are downloaded automatically
  by their pip packages on first use; they are not part of this release.
- Each checkpoint contains the EMA weights used for all reported results
  (`latest_model_200.pth`, saved by the training loop at epoch 200).
