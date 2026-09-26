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

## Stored hint and label maps (test split)

`test_split_hint_maps_64.tar.gz` (60 MB, SHA-256
`01c0aa92ae976d9e0a5c48a02881c7ed9db9f0e98d77fd14e44c396471ac0dfd`), attached to the
same v1.0 release, contains the 64x64 hint maps used for all reported results on the
3,000 Danbooru2021 test images:

```
test_split_hint_maps_64/hint_from_regions_64_rev/<id>.image_{scribble,dot}_{mask,col}64.png
test_split_hint_maps_64/region64/<id>.image_region64.png
test_split_hint_maps_64/test_image_ids.txt
```

These stored maps are the reference evaluation inputs. Regenerating them from the
source images is not bit-exact: the medial-axis tie-breaking inside the skeleton
extraction and the region-label palette were not seeded when the dataset was built.

## v1.1 assets — segmenter-retrained scribble models (supplementary segmentation-dependency study)

Attached to the [v1.1 GitHub Release](https://github.com/MADONOKOUKI/DetFill-with-DHT/releases/tag/v1.1)
together with `SHA256SUMS.txt`. Both are DetFill scribble models (96 base channels, 200 epochs, same protocol
and training split as the v1.0 scribble model) trained on hints generated from a different region segmenter;
they are the "DanbooRegion" and "SLIC" *training* rows of the supplementary table reproduced by
`reproduce/scripts/A1_tables_from_released_metrics.py` [3].

| File | Size | SHA-256 | Training segmenter |
|---|---|---|---|
| `detfill_scribble_illust_danbooregion_200ep.pth` | 1.08 GB | `1770cd61bb496060b4ff9bd8c2ec2e72b549fa99187a0429912100628cad9a78` | DanbooRegion |
| `detfill_scribble_illust_slic_200ep.pth` | 1.08 GB | `eb25667c1955584f46f34abf98db10eef6d702e4c1078fe3cff388f646cf4217` | SLIC |

Use them with `configs/scribble_illust.yaml` and `--resume_model <file>` (or `CKPT=<file>` with
`reproduce/scripts/run_hauc_pipeline.sh`); the evaluation data root must contain hint maps produced from the
matching segmenter (`reproduce/paper_experiments/segmenter_dependency/`, `reproduce/README.md` B.3).
