# Checkpoints

The paper checkpoints are attached to the GitHub Release of this repository
(too large for git):

| File | Size | Description |
|---|---|---|
| `detfill_scribble_illust_200ep.pth` | 463 MB | DetFill (96ch), scribble hints, 200 epochs, Danbooru2021 illust. Used for the Table II scribble results. |
| `detfill_dot_illust_200ep.pth` | 463 MB | DetFill (96ch), dot hints, 200 epochs, Danbooru2021 illust. |

Place as:
```
detfill/results/dataset_name/BrownianBridge_scribble_illust/checkpoint/latest_model_200.pth
detfill/results/dataset_name/BrownianBridge_dot_illust/checkpoint/latest_model_200.pth
```
or pass the path via `--resume_model`.
