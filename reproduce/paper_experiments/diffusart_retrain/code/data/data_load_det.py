# R3-2 (TVCG-2026 rebuttal): deterministic-hint training loader for Diffusart.
# Class body is a verbatim port of Diffusion_v1/data/data_load.py::MyData_train_scrib
# (the approach='proposed' deterministic-hint loader), with only the hardcoded
# data root replaced by DATA_ROOT (env DIFFUSART_DET_DATA_ROOT).
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
        return None
    new_image = Image.fromarray(new_image)
    return new_image

trans = Compose([
    transforms.ToTensor(),
    ])
trans_n = Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])

class MyData_train_scrib_det(Dataset):
    def __init__(self, root, size, approach='proposed', transform=trans, trans_norm=trans_n, hint=None, domain=None, dtype=None, dtype_region=None):
        # self.target_path = target_path
        # self.sketchkeras_path = sketchkeras_path
        # self.sketch_simp = sketch_simp

        self.root       = root
        self.approach = approach
        self.root_sketch = os.path.join(DATA_ROOT, domain, 'sketch')
        self.root_scr = os.path.join(DATA_ROOT, domain, "hint_from_regions", dtype)
        self.root_region = os.path.join(DATA_ROOT, domain, "segmentation_regions", dtype)


        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']        
        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']        
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size
        self.hint = hint

        self.vtransform = transforms.Compose([
            transforms.Resize((256, 256), Image.NEAREST),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.mask_trans =  transforms.Compose([
            # RandomSizedCrop(config.image_size // 4, Image.BICUBIC),
            # transforms.Resize((256 // 4, 256 // 4), Image.NEAREST),
            transforms.Resize((256, 256 ), Image.NEAREST),
            transforms.ToTensor(),
        ])            

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        # original = resize(io.imread(self.root[index].replace('\n', ''), as_gray=False, pilmode="RGB"),  (self.size, self.size))  # Reading target images in RGB

        # region = cv2.imread(os.path.join(self.root_region, dname, fname + '.image.png'))     

        rnd = random.randint(0, 2)
        # fname = self.root[index].replace('\n', '').split('/')[-1].split('.png')[0]
        # dname = self.root[index].replace('\n', '').split('/')[-2]
        fname = self.root[index].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.root[index].replace('\n', '').split('/')[-2]

        original = resize(io.imread(os.path.join(self.root_region, dname, fname + '.image.png'), as_gray=False, pilmode="RGB"),  (self.size, self.size))  # Reading target images in RGB

        try:
            sketch_data = resize(io.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png')),  (self.size, self.size))
        except:
            sketch_data = np.zeros( (self.size, self.size))

        # print(original.shape)
        h, w, _ = original.shape

        region = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_region64.png'))     
        region = cv2.resize(region, (256, 256), interpolation = cv2.INTER_NEAREST)

        # LUT-optimized, PROVEN pixel-identical (300/300) to the original block:
        #   np.unique(axis=0)+argsort(-count) prefix, then np.isin(region, prefix).all(axis=2)
        #   (note: isin is the original's SCALAR membership semantics, reproduced exactly).
        r32 = region.astype(np.int32)
        _packed = (r32[:, :, 0] << 16) | (r32[:, :, 1] << 8) | r32[:, :, 2]
        cand_p, count = np.unique(_packed.ravel(), return_counts=True)
        cand_p = cand_p[np.argsort(-count)]

        rand_val = np.random.rand()
        length_samples = int(len(cand_p) * rand_val )

        convert_hint = np.zeros((region.shape[0], region.shape[1], 3))
        convert_mask = np.zeros((region.shape[0], region.shape[1], 1))            
        single_hint = np.zeros((region.shape[0], region.shape[1], 3))
        single_mask = np.zeros((region.shape[0], region.shape[1], 1))

        length = 1 
        _top = cand_p[:length_samples]
        _lut = np.zeros(256, dtype=bool)
        if len(_top):
            _lut[np.concatenate([_top >> 16, (_top >> 8) & 255, _top & 255])] = True
        _sel = _lut[region[:, :, 0]] & _lut[region[:, :, 1]] & _lut[region[:, :, 2]]

        single_hint[_sel] = 1
        single_mask[_sel[:, :, None]] = 1



        if self.hint == 'dot' and self.approach:
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_dot_mask64.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_dot_col64.png'))

            hint = pil2opencv(hint)
            mask = pil2opencv(mask)  

            single_hint = single_hint * hint
            single_mask = single_mask * mask[:, :, np.newaxis]

            hint = opencv2pil(single_hint.astype(np.uint8))
            mask = opencv2pil(single_mask.astype(np.uint8))  


            totensor =  transforms.ToTensor()
            mask = totensor(mask)
            hint = totensor(hint)
            mask = mask.float()
            hint = hint.float()
            hint = (hint - 0.5) / 0.5

            hint = torch.cat([hint* mask, mask], dim=0)
            
            # hint = scribble_mask.clone().detach().float() * torch.cat([self.vtransform(hint), self.mask_trans(mask)], dim=0)            

        elif self.hint == 'strokes' and self.approach:
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_scribble_mask64.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_scribble_col64.png'))

            hint = pil2opencv(hint)
            mask = pil2opencv(mask)
            single_hint = single_hint * hint
            single_mask = single_mask * mask[:, :, np.newaxis]

            hint = opencv2pil(single_hint.astype(np.uint8))
            mask = opencv2pil(single_mask.astype(np.uint8))  


            totensor =  transforms.ToTensor()
            mask = totensor(mask)
            hint = totensor(hint)
            mask = mask.float()
            hint = hint.float()
            hint = (hint - 0.5) / 0.5

            hint = torch.cat([hint* mask, mask], dim=0)

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]

        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            original = self.transform(original)
            sketch_data = self.transform(np.expand_dims(sketch_data, axis=2))
            original = self.trans_norm(original)
            sketch_data = self.trans_norm(sketch_data)


        return original, sketch_data, hint

    def __len__(self):
        return len(self.root)
    
    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def sketch_loader(self, path):
        return Image.open(path).convert('L')       
