from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import os
import random
import cv2
import numpy as np
import copy
import torch

class ImagePathDataset(Dataset):

    
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name

def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')


def pil2opencv(in_image):
    out_image = np.array(in_image, dtype=np.uint8)
    if out_image.ndim == 2:
        pass
    elif out_image.shape[2] == 3:
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    return out_image

def opencv2pil(in_image):
    new_image = in_image.copy()   #複製
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    else:
        return None
    new_image = Image.fromarray(new_image)
    return new_image



class HintColorizationDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False, domain=None, hint_type=None):
        self.image_size = image_size
        self.image_paths = image_paths

        # NOTE: this class is shadowed by datasets.custom.HintColorizationDataset for
        # the shipped configs; set data_root here only if you use it directly.
        data_root = os.environ.get("DETFILL_DATA_ROOT", "/path/to/dataset")
        self.root_sketch = os.path.join(data_root, domain, "sketch")
        self.root_scr = os.path.join(data_root, domain, "hint_from_regions", "felzenszwalb")
        self.root_region = os.path.join(data_root, domain, "segmentation_regions", "felzenszwalb")
        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']

        self.transform = self.vtransform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.stransform = self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.hint_type = hint_type


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):


        fname = self.image_paths[index].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.image_paths[index].replace('\n', '').split('/')[-2]

        rnd = random.randint(0, 2)


        names = self.image_paths[index].replace('\n', '').split('/')

        
        Cimg = color_loader( os.path.join(self.root_region,  names[-2],  names[-1]))
        try:
            Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png'))
        except:
            Simg = Image.fromarray(np.ones( (512,512)))

        mask = sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_'+str(self.hint_type)+'_mask256.png'))
        hint = color_loader(os.path.join(self.root_scr, dname, fname + '.image_'+str(self.hint_type)+'_col256.png'))



        region = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_region256.png'))     
        region = cv2.resize(region, (256,256), interpolation=cv2.INTER_NEAREST)
        region_uint64 = region.astype(np.uint64)
        id_maps = region_uint64[:, :, 0] * 255 * 255 + region_uint64[:, :, 1] * 255  + region_uint64[:, :, 2]
        cand_vals, count = np.unique(id_maps.reshape(-1), axis=0, return_counts=True)
        np.random.shuffle(cand_vals) # randomly sampled region
        rand_val = np.random.rand()
        length_samples = int(len(cand_vals) * rand_val )
        sample_area = np.isin(id_maps, list(cand_vals[:length_samples]))[:, :, np.newaxis]


        hint = pil2opencv(hint)
        mask = pil2opencv(mask)


        single_hint = copy.deepcopy(sample_area) * hint
        single_mask = copy.deepcopy(sample_area) * mask[:, :, np.newaxis]

        hint = opencv2pil(single_hint.astype(np.uint8))
        mask = opencv2pil(single_mask[:, :, 0].astype(np.uint8))  
        sample_area = opencv2pil(sample_area[:, :, 0].astype(np.uint8))  
      
        Cimg = transforms.Resize((256, 256), interpolation=Image.BICUBIC)(Cimg)
        sample_area = transforms.Resize((256, 256), interpolation=Image.BICUBIC)(sample_area )
        Simg = transforms.Resize((256, 256), interpolation=Image.BICUBIC)(Simg)
        mask = transforms.Resize((256, 256), interpolation=Image.NEAREST)(mask)
        hint = transforms.Resize((256, 256), interpolation=Image.NEAREST)(hint)  

        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)
            sample_area = sample_area.transpose(Image.FLIP_LEFT_RIGHT) 
            mask, hint = mask.transpose(Image.FLIP_LEFT_RIGHT), hint.transpose(Image.FLIP_LEFT_RIGHT)


        Cimg, Simg = self.transform(Cimg), self.stransform(Simg)
        Rimg = self.mask_transform(sample_area)
        null_tensor = torch.zeros(Simg.shape)
        CRimg = torch.cat([Cimg, Rimg, null_tensor], dim=0)

        totensor =  transforms.ToTensor()


        mask = totensor(mask)
        hint = totensor(hint)
        mask = mask.float()
        hint = hint.float()
        hint = (hint - 0.5) / 0.5

        hint = torch.cat([hint* mask, mask], dim=0)



        # return torch.cat([hint, Simg], dim=0), CRimg

        x_cond = torch.cat([hint, Simg], dim=0)
        image_name = Path(os.path.join(self.root_region,  names[-2],  names[-1])).stem

        # return (CRimg.unsqueeze(0), image_name), (x_cond.unsqueeze(0), image_name )
        return (CRimg[:3, :, :], image_name), (x_cond[:3, :, :], image_name )

        # return CRimg, hint, Simg

