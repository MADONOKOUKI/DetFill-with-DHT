"""
Dataloader dedicated to the R1-1 seed-ranking-stability experiment.

Mirrors datasets.custom.HintColorizationDataset's inference (`sample_ratio is
not None`) branch line-for-line, with EXACTLY ONE substantive change: the
size-based sort of region ids

    cand_vals = cand_vals[np.argsort(-count)]      # custom.py:413

is removed.  Everything else — region-map load, `np.unique` over region ids,
`length_samples = int(len(cand_vals) * sample_ratio)`, `sample_area = np.isin(
id_maps, cand_vals[:length_samples])`, the `hint * sample_area` and
`mask * sample_area` multiplications, and all subsequent PIL / cv2 / transform
plumbing — is preserved verbatim.

At sample_ratio=1.0 (the seed-experiment setting) `length_samples ==
len(cand_vals)`, so `sample_area` collapses to all-ones regardless of order;
the loader therefore yields byte-identical CRimg / x_cond / hint_weight tensors
to the old loader, and the model output PNGs are unchanged.  We keep the
sample_area machinery so that:

  * the code mirrors custom.py except for the single removed line, making the
    "no size sort" claim auditable in a one-line diff, and
  * the loader still behaves correctly at sample_ratio < 1.0 (region-id-based
    sampling, just without size prioritisation).

Registered as `custom_seedexp_eval`.
"""

import os
import random
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from Register import Registers
from datasets.base import color_loader, sketch_loader, pil2opencv, opencv2pil


@Registers.datasets.register_with_name('custom_seedexp_eval')
class HintColorizationSeedExpDataset(Dataset):
    def __init__(self, dataset_config, stage='train', sample_ratio=None, sketch_type=None):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.sample_ratio = sample_ratio
        self.sketch_type = sketch_type
        self.domain = dataset_config.domain
        self.hint_type = dataset_config.hint_type

        scratch_root = getattr(dataset_config, 'scratch_root', None)
        if scratch_root is None:
            raise ValueError(
                "custom_seedexp_eval requires dataset_config.scratch_root "
                "(pointing at a wrapper like /scratch/madono/seedexp/detfill_roots/seedN_alpha_X)"
            )

        # Same layout as HintColorizationDataset's scratch_root branch.
        self.root_sketch = os.path.join(scratch_root, "sketch")
        self.root_scr = os.path.join(scratch_root, "hint_from_regions_64_rev")
        self.root_scr_region = os.path.join(scratch_root, "hint_from_regions_256")
        self.hint_subdir = '0016'
        self.root_region = os.path.join(scratch_root, "segmentations")

        self.image_paths = sorted(glob.glob(
            os.path.join(scratch_root, "segmentations/originals/*.image.png")
        ))
        if not self.image_paths:
            raise FileNotFoundError(
                f"no test images under {scratch_root}/segmentations/originals/*.image.png"
            )

        self.sketch_cands = ['pysimp', 'XDoG', 'sketchkeras']

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.stransform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        fname = (
            self.image_paths[index]
            .replace('\n', '')
            .split('/')[-1]
            .split('.png')[0]
            .split('.jpg')[0]
            .split('.image')[0]
        )
        dname = self.image_paths[index].replace('\n', '').split('/')[-2]

        if self.sketch_type is None:
            rnd = random.randint(0, 2)
        else:
            rnd = self.sketch_type

        names = self.image_paths[index].replace('\n', '').split('/')

        Cimg = color_loader(os.path.join(self.root_region, names[-2], names[-1]))

        if self.domain == "illust":
            Simg = sketch_loader(os.path.join(
                self.root_sketch, self.sketch_cands[rnd], fname + '.png'))
        else:
            Simg = sketch_loader(os.path.join(
                self.root_sketch, self.sketch_cands[rnd], fname + '.image.png'))

        hint_dname = self.hint_subdir
        region_root = self.root_scr_region

        mask = sketch_loader(os.path.join(
            self.root_scr, hint_dname,
            fname + '.image_' + str(self.hint_type) + '_mask64.png'))
        hint = color_loader(os.path.join(
            self.root_scr, hint_dname,
            fname + '.image_' + str(self.hint_type) + '_col64.png'))

        region = cv2.imread(os.path.join(region_root, fname + '.image_region64.png'))
        region = cv2.resize(region, self.image_size, interpolation=cv2.INTER_NEAREST)
        region_uint64 = region.astype(np.uint64)
        id_maps = region_uint64[:, :, 0] * 255 * 255 + region_uint64[:, :, 1] * 255 + region_uint64[:, :, 2]
        cand_vals, count = np.unique(id_maps.reshape(-1), axis=0, return_counts=True)

        # === The ONLY substantive deviation from custom.py inference branch: ===
        # custom.py:413   cand_vals = cand_vals[np.argsort(-count)]   # size desc sort
        # (deliberately omitted here — `count` is unused and cand_vals keeps the
        # `np.unique` default order, i.e. ascending region-id integer order.)
        # =====================================================================
        length_samples = int(len(cand_vals) * self.sample_ratio)

        sample_area = np.isin(id_maps, list(cand_vals[:length_samples]))[:, :, np.newaxis].astype(np.uint8)

        hint = pil2opencv(hint)
        mask = pil2opencv(mask)
        hint = cv2.resize(hint, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        hint = hint * sample_area
        mask = mask * sample_area[:, :, 0]

        hint_weight = torch.tensor([1.0 * sample_area.sum() / (sample_area.shape[0] * sample_area.shape[1])])

        Cimg = pil2opencv(Cimg)
        Simg = pil2opencv(Simg)

        hint = opencv2pil(hint.astype(np.uint8))
        mask = opencv2pil(mask.astype(np.uint8))
        Cimg = opencv2pil(Cimg.astype(np.uint8))
        Simg = opencv2pil(Simg.astype(np.uint8))

        Cimg = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(Cimg)
        Simg = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(Simg)
        mask = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(mask)
        hint = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(hint)

        # custom.py's inference branch (sample_ratio is not None) skips the
        # `random.random() < 0.5` horizontal flip — we mirror that.

        Cimg = self.transform(Cimg)
        Simg = self.stransform(Simg)

        totensor = transforms.ToTensor()
        mask = totensor(mask).float()
        hint = totensor(hint).float()
        hint = (hint - 0.5) / 0.5

        hint_c = torch.cat([mask, hint * mask], dim=0)
        x_cond = torch.cat([Simg, hint_c], dim=0)
        CRimg = torch.cat([Simg, mask, Cimg], dim=0)

        image_name = Path(os.path.join(self.root_region, names[-2], names[-1])).stem
        return (CRimg, image_name), (x_cond, image_name), (hint_weight, image_name)
