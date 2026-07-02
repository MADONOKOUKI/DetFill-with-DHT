"""
Design A — seed-aware, on-the-fly random region selection.

Mirrors datasets.custom.HintColorizationDataset's inference (`sample_ratio is
not None`) branch line-for-line, with EXACTLY ONE substantive change: the
size-based sort of region ids

    cand_vals = cand_vals[np.argsort(-count)]      # custom.py:413

is replaced by a SEEDED random permutation driven by
`(20260611, experiment_seed, index)` — matching the RNG signature used by
gen_seeded_random_hints.py so any future cross-check lines up.

Reads the SHARED full hint/mask/region files (not seed-baked copies); the
seed-specific sub-selection happens entirely at sample_area time via the
permuted `cand_vals[:length_samples]`.

Registered as `custom_seedexp_inline`.
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


@Registers.datasets.register_with_name('custom_seedexp_inline')
class HintColorizationSeededInlineDataset(Dataset):
    def __init__(self, dataset_config, stage='train', sample_ratio=None, sketch_type=None):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.stage = stage
        self.sample_ratio = sample_ratio
        self.sketch_type = sketch_type
        self.domain = dataset_config.domain
        self.hint_type = dataset_config.hint_type

        # Required experiment fields.
        self.experiment_seed = getattr(dataset_config, 'experiment_seed', None)
        self.experiment_alpha = getattr(dataset_config, 'experiment_alpha', None)

        scratch_root = getattr(dataset_config, 'scratch_root', None)
        if scratch_root is None:
            raise ValueError(
                "custom_seedexp_inline requires dataset_config.scratch_root "
                "(e.g., /scratch/madono/tvcg26_major_revision)"
            )

        # Same layout as the shared tvcg26_major_revision tree.
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

        limit = getattr(dataset_config, "limit", None)
        if limit is not None and limit > 0:
            self.image_paths = self.image_paths[:limit]

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

        # Load the SHARED full hint / mask. Seed-specific sub-selection happens
        # below at sample_area, NOT here.
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
        # Replaced with seed-aware random permutation when experiment_seed is set.
        # RNG signature [20260611, experiment_seed, index] MUST match
        # gen_seeded_random_hints.py so any future cross-check lines up.
        # =====================================================================
        if self.experiment_seed is not None:
            rng = np.random.default_rng([20260611, self.experiment_seed, index])
            cand_vals = rng.permutation(cand_vals)
            ratio = self.experiment_alpha if self.experiment_alpha is not None else self.sample_ratio
        else:
            cand_vals = cand_vals[np.argsort(-count)]   # default: size sort (matches custom.py)
            ratio = self.sample_ratio

        length_samples = int(len(cand_vals) * ratio)

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
