# R3-2 (TVCG-2026 rebuttal): deterministic-hint training loader for Diffusart.
# Class body is a verbatim port of Diffusion_v1/data/data_load.py::MyData_train_scrib
# (the approach='proposed' deterministic-hint loader), with only the hardcoded
# data root replaced by DATA_ROOT (env DIFFUSART_DET_DATA_ROOT).
#
# 2026-06-22: __getitem__ aligned to the PROPOSED (BBDM) HintColorizationDataset flow:
#   (1) hint channel order -> cat([mask, hint*mask]) (mask-FIRST, matches proposed)
#   (2) region sampling     -> np.random.shuffle (random subset, matches proposed train)
#   (3) GT/sketch transform -> PIL + Resize(NEAREST->size) + Resize(BICUBIC)+ToTensor+Normalize
#                              (proposed's exact chain, replacing skimage bilinear)
#   (4) augmentation         -> p=0.5 joint horizontal flip (matches proposed train)
# Hint/mask/region sources stay *_64.png (NN-upsampled), per the R3-2 64px requirement.
import torch
import numpy as np
from skimage import io
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
import os
from PIL import Image
from PIL import ImageFile
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = os.environ.get('DIFFUSART_DET_DATA_ROOT', '/scratch/madono/diffusart_det_R3-2')

def pil2opencv(in_image):
    out_image = np.array(in_image, dtype=np.uint8)
    if out_image.ndim == 2:
        pass
    elif out_image.shape[2] == 3:
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    return out_image

def opencv2pil(in_image):
    new_image = in_image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    else:
        new_image = new_image[:, :, 0]  # MASK_SQUEEZE_FIX: (H,W,1)->(H,W)
    new_image = Image.fromarray(new_image)
    return new_image

trans = Compose([
    transforms.ToTensor(),
    ])
trans_n = Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])

class MyData_train_scrib_det_64(Dataset):
    def __init__(self, root, size, approach='proposed', transform=trans, trans_norm=trans_n, hint=None, domain=None, dtype=None, dtype_region=None):
        self.root       = root
        self.approach = approach
        self.root_sketch = os.path.join(DATA_ROOT, domain, 'sketch')
        self.root_scr = os.path.join(DATA_ROOT, domain, "hint_from_regions", dtype)
        self.root_region = os.path.join(DATA_ROOT, domain, "segmentation_regions", dtype)

        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size
        self.hint = hint

        # PROPOSED-style GT/sketch transforms (Resize BICUBIC + ToTensor + Normalize -> [-1,1]).
        self.gt_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.sk_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __getitem__(self, index):
        rnd = random.randint(0, 2)
        fname = self.root[index].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.root[index].replace('\n', '').split('/')[-2]

        # --- GT color + sketch as PIL (proposed flow) ---
        original = self.load_image(os.path.join(self.root_region, dname, fname + '.image.png'))  # PIL RGB
        try:
            sketch_data = self.sketch_loader(os.path.join(self.root_sketch, self.sketch_cands[rnd], dname, fname + '.png'))  # PIL L
        except Exception:
            sketch_data = Image.fromarray(np.zeros((self.size, self.size), dtype=np.uint8))

        # --- region (64 -> 256 NEAREST), candidate enumeration ---
        region = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_region64.png'))
        region = cv2.resize(region, (256, 256), interpolation=cv2.INTER_NEAREST)
        r32 = region.astype(np.int32)
        _packed = (r32[:, :, 0] << 16) | (r32[:, :, 1] << 8) | r32[:, :, 2]
        cand_p, count = np.unique(_packed.ravel(), return_counts=True)
        np.random.shuffle(cand_p)                          # (2) PROPOSED train: random subset
        rand_val = np.random.rand()
        length_samples = int(len(cand_p) * rand_val)

        single_hint = np.zeros((region.shape[0], region.shape[1], 3))
        single_mask = np.zeros((region.shape[0], region.shape[1], 1))
        _top = cand_p[:length_samples]
        _lut = np.zeros(256, dtype=bool)
        if len(_top):
            _lut[np.concatenate([_top >> 16, (_top >> 8) & 255, _top & 255])] = True
        _sel = _lut[region[:, :, 0]] & _lut[region[:, :, 1]] & _lut[region[:, :, 2]]
        single_hint[_sel] = 1
        single_mask[_sel[:, :, None]] = 1

        # --- hint/mask from *_64 sources (NN upsample), masked by region selection ---
        if self.hint == 'dot':
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_dot_mask256.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_dot_col256.png'))
        else:  # strokes / scribble
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_scribble_mask256.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_scribble_col256.png'))

        hint = pil2opencv(hint)
        mask = pil2opencv(mask)
        hint = cv2.resize(hint, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        single_hint = single_hint * hint
        single_mask = single_mask * mask[:, :, np.newaxis]
        hint = opencv2pil(single_hint.astype(np.uint8))    # PIL RGB
        mask = opencv2pil(single_mask.astype(np.uint8))    # PIL L (squeezed)

        # --- resize all to (size,size) NEAREST, then joint flip (proposed flow) ---
        original = transforms.Resize((self.size, self.size), Image.NEAREST)(original)
        sketch_data = transforms.Resize((self.size, self.size), Image.NEAREST)(sketch_data)
        hint = transforms.Resize((self.size, self.size), Image.NEAREST)(hint)
        mask = transforms.Resize((self.size, self.size), Image.NEAREST)(mask)
        if random.random() < 0.5:                          # (4) PROPOSED train: joint H-flip
            original = original.transpose(Image.FLIP_LEFT_RIGHT)
            sketch_data = sketch_data.transpose(Image.FLIP_LEFT_RIGHT)
            hint = hint.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # --- tensorize (3): GT/sketch BICUBIC+Normalize; hint/mask ToTensor ---
        original = self.gt_transform(original)             # [-1,1], 3ch
        sketch_data = self.sk_transform(sketch_data)       # [-1,1], 1ch
        totensor = transforms.ToTensor()
        mask = totensor(mask).float()                      # [0,1], 1ch
        hint = totensor(hint).float()                      # [0,1], 3ch
        hint = (hint - 0.5) / 0.5                           # [-1,1]
        hint = torch.cat([mask, hint * mask], dim=0)       # (1) PROPOSED: mask-FIRST, 4ch

        return original, sketch_data, hint

    def __len__(self):
        return len(self.root)

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def sketch_loader(self, path):
        return Image.open(path).convert('L')
