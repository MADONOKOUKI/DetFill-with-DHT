#!/bin/bash
# Train DetFill (200 epochs). Set dataset_path / scratch_root in the configs first
# (see README.md for the expected data layout), and adjust --gpu_ids to your machine.

python3 main.py --config configs/scribble_illust.yaml --train --sample_at_start --save_top --gpu_ids 0 --port 12356
python3 main.py --config configs/dot_illust.yaml      --train --sample_at_start --save_top --gpu_ids 0 --port 12356

# Natural-image (ImageNet) variants — prepare the corresponding dataset first:
# python3 main.py --config configs/scribble_real.yaml --train --sample_at_start --save_top --gpu_ids 0 --port 12356
# python3 main.py --config configs/dot_real.yaml      --train --sample_at_start --save_top --gpu_ids 0 --port 12356
