import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from Register import Registers
from datasets.base import ImagePathDataset, HintColorizationDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os
import numpy as np
import copy
import torch
import math
@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

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

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

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

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


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


def split_and_random_indices(lst, N):
    # ランダムにインデックスをシャッフルします
    indices = list(range(len(lst)))
    random.shuffle(indices)
    
    # 各サブリストのサイズをランダムに決定します
    sizes = []
    remaining_size = len(lst)
    for i in range(N - 1):
        try:
            size = random.randint(1, remaining_size - (N - 1 - i))
        except:
            size = 0
        sizes.append(size)
        remaining_size -= size
    sizes.append(remaining_size)
    
    # 各サブリストのインデックスを割り当てます
    divided_indices = []
    start = 0
    for size in sizes:
        divided_indices.append(indices[start:start+size])
        start += size
    
    return divided_indices


@Registers.datasets.register_with_name('custom_train_our_model')



class HintColorizationDataset(Dataset):
    # def __init__(self, image_paths, image_size=(64, 64), flip=False, to_normal=False, domain=None, hint_type=None):

    def __init__(self, dataset_config, stage='train', sample_ratio=None, sketch_type=None):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)

        self.stage = stage
        self.sample_ratio = sample_ratio
        self.sketch_type = sketch_type

        self.domain = dataset_config.domain
        self.hint_type = dataset_config.hint_type
        # Training-time hint sampling (Eq. 6): the paper draws p = floor(n*u), u~U[0,1),
        # i.e. p in {0,...,n-1} (the full-hint case p = n never occurs in training).
        # include_full_hint: true samples p uniformly on {0,...,n}. Default false (paper).
        self.include_full_hint = bool(getattr(dataset_config, 'include_full_hint', False))
        self.hint_order = str(getattr(dataset_config, 'hint_order', 'area'))  # 'area' (Table II) | 'label' (Table III)
        if self.hint_order not in ('area', 'label'):
            raise ValueError(f"dataset_config.hint_order must be 'area' or 'label', got {self.hint_order!r}")

        # Two user-configurable layouts (set the roots in the yaml config):
        #  A) dataset_config.scratch_root — flat evaluation layout:
        #     <scratch_root>/segmentations/originals/*.image.png
        #     <scratch_root>/hint_from_regions_64_rev/  and  hint_from_regions_256/
        #     <scratch_root>/sketch/{XDoG,pysimp,sketchkeras}/*.png
        #  B) dataset_config.dataset_path — split-based layout with
        #     {train,valid,test}.txt lists (see detfill/README.md).
        scratch_root = getattr(dataset_config, 'scratch_root', None)
        if scratch_root:
            self.root_sketch = os.path.join(scratch_root, "sketch")
            self.root_scr = os.path.join(scratch_root, "hint_from_regions_64_rev")
            self.root_scr_region = os.path.join(scratch_root, "hint_from_regions_256")
            # hints may live directly under hint_from_regions_64_rev/ or in a
            # '0016' subdirectory (the layout of the original experiments)
            self.hint_subdir = '0016' if os.path.isdir(os.path.join(self.root_scr, '0016')) else ''
            self.flat_sketch = True  # sketch/{XDoG,pysimp,sketchkeras}/<id>.png
            self.root_region = os.path.join(scratch_root, "segmentations")
            self.image_paths = sorted(glob.glob(
                os.path.join(scratch_root, "segmentations/originals/*.image.png")))
        else:
            list_train_imgs, list_valid_imgs, list_test_imgs = init_load_data(dataset_config.domain)
            if stage == 'train':
                self.image_paths = list_train_imgs
            elif stage == 'val':
                self.image_paths = list_valid_imgs
            elif stage == 'test':
                self.image_paths = list_test_imgs

            dataset_path = getattr(dataset_config, 'dataset_path', None)
            if not dataset_path or str(dataset_path).startswith('/path/to'):
                raise ValueError(
                    "Set data.dataset_config.dataset_path (or scratch_root) in the yaml "
                    "config to your dataset root; see detfill/README.md for the layout.")
            self.root_sketch = os.path.join(dataset_path, "sketch")
            self.root_scr = os.path.join(dataset_path, "hint_from_regions_64_rev")
            self.root_scr_region = os.path.join(dataset_path, "hint_from_regions_256")
            self.root_region = os.path.join(dataset_path, "segmentation_regions", "felzenszwalb")



        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']


        # def jitter(x):
        #     ran = random.uniform(0.7, 1)
        #     return x * ran + 1 - ran

        self.transform = self.vtransform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # if stage == 'train':
        self.stransform = self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Lambda(jitter),
            transforms.Normalize((0.5), (0.5))
        ])
        # else:
        #     self.stransform = self.mask_transform = transforms.Compose([
        #         transforms.Resize(self.image_size, Image.BICUBIC),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5), (0.5))
        #     ])            
            


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):


        fname = self.image_paths[index].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.image_paths[index].replace('\n', '').split('/')[-2]


        if self.sketch_type == None:
            rnd = random.randint(0, 2)
        else:
            rnd = self.sketch_type


        names = self.image_paths[index].replace('\n', '').split('/')

        
        Cimg = color_loader( os.path.join(self.root_region,  names[-2],  names[-1])) #.convert('LAB')
        # Cimg = cv2.imread(os.path.join(self.root_region,  names[-2],  names[-1])) #.convert('LAB')
        # try:

        if self.domain == "illust":
            if getattr(self, 'flat_sketch', False):
                Simg = sketch_loader(os.path.join(self.root_sketch, self.sketch_cands[rnd], fname + '.png'))
            else:
                Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png'))
            # Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd],  fname + '.png'))  # for user study (FLAT layout, disabled by evalfix; real data is nested by dname)
        else:
            # Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.image.png'))
            Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd], fname + '.image.png')) # for user study
 
        # except:
        #     Simg = Image.fromarray(np.ones( (512,512)))


        hint_dname = getattr(self, 'hint_subdir', dname)
        region_root = getattr(self, 'root_scr_region', self.root_scr)

        if self.sample_ratio is not None:
            mask = sketch_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_mask64.png'))
            hint = color_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_col64.png'))
            region = cv2.imread(os.path.join(region_root, fname + '.image_region64.png'))
            region = cv2.resize(region, (64,64), interpolation=cv2.INTER_NEAREST)
        else:
            mask = sketch_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_mask64.png'))
            hint = color_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_col64.png'))
            region = cv2.imread(os.path.join(region_root, fname + '.image_region64.png'))
            region = cv2.resize(region, (64,64), interpolation=cv2.INTER_NEAREST)


        mask = sketch_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_mask64.png'))
        hint = color_loader(os.path.join(self.root_scr, hint_dname, fname + '.image_'+str(self.hint_type)+'_col64.png'))

        # mask = sketch_loader(os.path.join(self.root_scr, fname + '.image_'+str(self.hint_type)+'_mask256.png')) # for user study
        # hint = color_loader(os.path.join(self.root_scr, fname + '.image_'+str(self.hint_type)+'_col256.png')) # for user study



        # mask = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_'+str(self.hint_type)+'_mask64.png'))
        # hint = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_'+str(self.hint_type)+'_col64.png')) #.convert('LAB')
        
        region = cv2.imread(os.path.join(region_root, fname + '.image_region64.png'))
        region = cv2.resize(region, self.image_size, interpolation=cv2.INTER_NEAREST)            
        region_uint64 = region.astype(np.uint64)
        id_maps = region_uint64[:, :, 0] * 255 * 255 + region_uint64[:, :, 1] * 255  + region_uint64[:, :, 2]
        cand_vals, count = np.unique(id_maps.reshape(-1), axis=0, return_counts=True)


        if self.sample_ratio is not None:
            # hint_order: 'area' (paper Table II; largest regions first, NumPy default argsort tie order)
            #             'label' (paper Table III; the fixed, area-independent ascending-label order)
            if getattr(self, 'hint_order', 'area') != 'label':
                cand_vals = cand_vals[np.argsort(-count)]
            length_samples = int(len(cand_vals) * self.sample_ratio )
        else:
            np.random.shuffle(cand_vals) # randomly sampled region
            if getattr(self, 'include_full_hint', False):
                length_samples = int(np.random.randint(0, len(cand_vals) + 1))  # p uniform on {0,...,n}
            else:
                rand_val = np.random.rand()
                length_samples = int(len(cand_vals) * rand_val )               # paper: p = floor(n*u), u in [0,1)
        # length_samples = np.random.randint(5, 100)



        # result_indices = split_and_random_indices(cand_vals[:length_samples], 4)
        sample_area = np.isin(id_maps, list(cand_vals[:length_samples]))[:, :, np.newaxis].astype(np.uint8)

        hint = pil2opencv(hint)
        mask = pil2opencv(mask)
        # if self.sample_ratio is not None:
        hint = cv2.resize(hint, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        hint = hint * sample_area
        mask = mask  * sample_area[:, :, 0]


        hint_weight = torch.tensor([1.0 * sample_area.sum() / (sample_area.shape[0] * sample_area.shape[1])])

                  
        # hint = pil2opencv(hint)
        # mask = pil2opencv(mask) 
        Cimg = pil2opencv(Cimg)
        Simg = pil2opencv(Simg)

   

        # if self.stage == 'train':
        #     ul = ( int(self.image_size[0] * (0.25 * random.random())), int((self.image_size[0] * 0.25 * random.random())))
        #     br = ( int((self.image_size[0] * (0.75 + 0.25 * random.random()))), int((self.image_size[0] * (0.75 + 0.25  * random.random()))))

        #     hint = hint[ul[0]:br[0], ul[1]:br[1]]
        #     mask = mask[ul[0]:br[0], ul[1]:br[1]]
        #     Simg = Simg[ul[0]:br[0], ul[1]:br[1]]
        #     Cimg = Cimg[ul[0]:br[0], ul[1]:br[1]]


        # if self.to_lab:
        # hint = cv2.cvtColor(hint, cv2.COLOR_RGB2LAB)
        # Cimg = cv2.cvtColor(Cimg, cv2.COLOR_RGB2LAB)

        hint = opencv2pil(hint.astype(np.uint8))
        mask = opencv2pil(mask.astype(np.uint8)) 
        Cimg = opencv2pil(Cimg.astype(np.uint8))
        Simg = opencv2pil(Simg.astype(np.uint8)) 
                

      
        Cimg = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(Cimg)
        Simg = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(Simg)
        mask = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(mask)
        hint = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(hint)  

        if self.sample_ratio is None:
            if random.random() < 0.5:
                Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)
                mask, hint = mask.transpose(Image.FLIP_LEFT_RIGHT), hint.transpose(Image.FLIP_LEFT_RIGHT)


        Cimg, Simg = self.transform(Cimg), self.stransform(Simg)
        null_tensor = torch.zeros(Simg.shape) 

        CRimg = Cimg

        totensor =  transforms.ToTensor()


        mask = totensor(mask)
        hint = totensor(hint)
        mask = mask.float()
        hint = hint.float()
        hint = (hint - 0.5) / 0.5


        hint_c = torch.cat([mask, hint* mask], dim=0) 
        x_cond = torch.cat([Simg, hint_c], dim=0)
        CRimg = torch.cat([ Simg, mask, Cimg], dim=0)



        image_name = Path(os.path.join(self.root_region,  names[-2],  names[-1])).stem
        # return (CRimg, image_name), (x_cond, image_name )
        return (CRimg, image_name), (x_cond, image_name ), (hint_weight, image_name)

# (a fully commented-out duplicate of the dataset class was removed here during cleanup)

def init_load_data(domain):

    #Training data

    with open(os.path.join('configs', str(domain), 'train.txt'), "r") as a:
        list_train_imgs = a.readlines()

    with open(os.path.join('configs', str(domain), 'valid.txt'), "r") as a:

        list_valid_imgs = a.readlines()

    with open(os.path.join('configs', str(domain), 'test.txt'), "r") as a:
        list_test_imgs = a.readlines()        

    return list_train_imgs, list_valid_imgs, list_test_imgs

