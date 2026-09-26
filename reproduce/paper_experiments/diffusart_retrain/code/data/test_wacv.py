from __future__ import division
import skimage.draw
import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import scipy.stats as stats
import kornia
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
from PIL import ImageFile
import cv2
import scipy.stats as stats
import math
import numbers
import os
import os.path
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Resize, CenterCrop
import kornia
import cv2
import skimage
import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def pil2opencv(in_image):
    out_image = np.array(in_image, dtype=np.uint8)
    if out_image.ndim == 2:
        pass
    # elif out_image.shape[2] == 3:
    #     out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    return out_image

def opencv2pil(in_image):
    new_image = in_image.copy()   #複製
    if new_image.ndim == 2:
        pass
    # elif new_image.shape[2] == 3:
    #     new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    # else:
    #     return None
    new_image = Image.fromarray(new_image)
    return new_image


# def init_load_data():

#     #Training data

#     #Training images list
#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/train_paper.txt', "r") as a:
#         list_train_imgs = a.readlines()

#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/valid_paper.txt', "r") as a:
#         list_valid_imgs = a.readlines()

#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/test_paper.txt', "r") as a:
#         list_test_imgs = a.readlines()        

#     return list_train_imgs, list_valid_imgs


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1, img2
        

        try:
            if w == tw:
                x1 = 0
                y1 = random.randint(0, h - th)
                return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

            elif h == th:
                x1 = random.randint(0, w - tw)
                y1 = 0
                return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))

            elif w - tw < 0 or h - th < 0:
                return img1, img2
            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                return img1.crop((x1, y1, x1 + tw, y1 + th)), img2.crop((x1, y1, x1 + tw, y1 + th))
        except:
            return img1, img2


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.9, 1.) * area
            aspect_ratio = random.uniform(7. / 8, 8. / 7)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root_img, root_sketch):
    images = []
    dnames = os.listdir(root_sketch) # ['pysketchsimplify', 'XDoG', 'sketchkeras']
    for dd in dnames:
        for d in os.listdir(os.path.join(root_sketch, dd)):
            for f in os.listdir(os.path.join(root_sketch, dd, d)):
                images.append(os.path.join(d, f))
    return images


def color_loader(path):
    return Image.open(path).convert('RGB')


def sketch_loader(path):
    return Image.open(path).convert('L')


# class DistributedSampler(Sampler):
#     """Sampler that restricts data loading to a subset of the dataset.
#
#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSampler instance as a DataLoader sampler,
#     and load a subset of the original dataset that is exclusive to it.
#
#     .. note::
#         Dataset is assumed to be of constant size.
#
#     Arguments:
#         dataset: Dataset used for sampling.
#         world_size (optional): Number of processes participating in
#             distributed training.
#         rank (optional): Rank of the current process within world_size.
#     """
#
#     def __init__(self, dataset, round_up=True):
#         self.dataset = dataset
#         self.round_up = round_up
#         self.epoch = 0
#
#         self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
#         if self.round_up:
#             self.total_size = self.num_samples * self.world_size
#         else:
#             self.total_size = len(self.dataset)
#
#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
#         indices = list(torch.randperm(len(self.dataset), generator=g))
#
#         # add extra samples to make it evenly divisible
#         if self.round_up:
#             indices += indices[:(self.total_size - len(indices))]
#         assert len(indices) == self.total_size
#
#         # subsample
#         offset = self.num_samples * self.rank
#         indices = indices[offset:offset + self.num_samples]
#         if self.round_up or (not self.round_up and self.rank < self.world_size - 1):
#             assert len(indices) == self.num_samples
#
#         return iter(indices)
#
#     def __len__(self):
#         return self.num_samples
#
#     def set_epoch(self, epoch):
#         self.epoch = epoch


class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, diter, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.diter = diter
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size * (self.diter + 1)

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[(self.last_iter + 1) * self.batch_size * (self.diter + 1):])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        # each process shuffle all list with same seed
        np.random.seed(0)

        indices = np.arange(len(self.dataset))
        indices = indices[:self.total_size]
        num_repeat = (self.total_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        np.random.shuffle(indices)

        assert len(indices) == self.total_size
        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size


def making_mask_v3(color, size, flatten=True):
    choice = np.random.choice(['width', 'height', 'diag'])


    mask_simp = np.zeros((size, size, 1))

    avg_wht = 0.99
    cnt = 0

    while avg_wht >= 0.6 and cnt < 100:
        cnt = cnt + 1

        if choice == 'width':
            rnd_height = np.random.randint(1, 4)
            rnd_width = np.random.randint(4, 30)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))

            scrib_color = color[:, ii, jj]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 30)
            rnd_width = np.random.randint(1, 4)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[:, ii, jj]
            
        elif choice == 'diag':

            thick = np.random.randint(1, 4)
            rnd_width = np.random.randint(4, 30)

            rnd1 = np.random.randint(size - thick - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.line(rnd1, rnd2, rnd1+rnd_width, rnd2+rnd_width)
            ii = np.hstack([ii + i for i in range(thick)])
            jj = np.tile(jj, thick)
            scrib_color = color[:, ii, jj]

        if choice == 'diag':
            count_white = np.sum(np.all(scrib_color == 1, axis = 1))
            count_black = np.sum(np.all(scrib_color == 0, axis = 1))
            size_pix = scrib_color.shape[1]
        else:
            count_white = np.sum(np.all(scrib_color == 1, axis = 1))
            count_black = np.sum(np.all(scrib_color == 0, axis = 1))
            size_pix = scrib_color.shape[1] * scrib_color.shape[2]

        avg_wht = count_white / size_pix
        avg_black = count_black / size_pix

        if avg_wht < 0.6 and avg_black < 0.6:
            mask_simp[ii, jj, :] = 1

        mean_color = scrib_color.reshape((3,-1))


    mask_simp = np.concatenate([mask_simp * np.mean(mean_color, axis=1), mask_simp], axis=2)

    return mask_simp

# class ImageFolder(data.Dataset):
#     def __init__(self, root_img, root_sketch, type, img_size, root_hint=None, root_hint_mask=None, transform=None, vtransform=None, stransform=None, mtransform=None):
#         imgs = make_dataset(root_img, root_sketch)
#         if len(imgs) == 0:
#             raise (RuntimeError("Found 0 images in folders."))
#         self.root_img = root_img
#         self.root_sketch = root_sketch
        
#         self.imgs = imgs
#         self.transform = transform
#         self.vtransform = vtransform
#         self.stransform = stransform
#         self.mtransform = mtransform

#         self.type = type
#         self.input_size = img_size
#         self.root_hint = root_hint
#         self.root_hint_mask = root_hint_mask
class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, vtransform=None, stransform=None):
        self.root = root
        self.root_sketch = '/home/madorin/gitlab/yuan/hint_colorization/danbooru2021_sketch_png'
        self.sketch_cands =  ['pysketchsimplify', 'XDoG', 'sketchkeras']
        self.transform = transform
        self.vtransform = vtransform
        self.stransform = stransform

        

    def __len__(self):
        return len(self.root)
            
    def __getitem__(self, index):

        # fname = self.imgs[index]
        
        fname = self.root[index].replace('\n', '').split('/')[-1].split('.png')[0]
        dname = self.root[index].replace('\n', '').split('/')[-2]

        rnd = random.randint(0, 2)
        Cimg = color_loader(self.root[index].replace('\n', ''))
        Simg = sketch_loader(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png'))

        Cimg = transforms.Resize((256, 256), interpolation=Image.BICUBIC)(Cimg)
        Simg = transforms.Resize((256, 256), interpolation=Image.BICUBIC)(Simg)



        # mu, sigma = 1, 5e-3
        # X         = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
        # maskS     = 64
        # mask      = torch.rand(1, maskS, maskS).ge(X.rvs(1)[0]).float()
        # vcolored  = self.transform(colored.resize((64, 64)))
        # # print(vcolored.shape, mask.shape)
        # resize = transforms.Resize((64,64), Image.BICUBIC)
        # vcolored = resize(vcolored)
        # hint      = torch.cat((vcolored * mask, mask), 0)
                
        # colored  = self.transform(colored)
        # sketch   = self.stransform(sketch)  
    
            
        if random.random() < 0.5:
            Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)

        Cimg, Vimg, Simg = self.transform(Cimg), self.vtransform(Cimg), self.stransform(Simg)
        return Cimg, Vimg, Simg
    
        # try:
        #     Cimg = color_loader(os.path.join(self.root_img, fname))
        # except:
        #     Cimg = color_loader(os.path.join(self.root_img, fname[:-3] + 'png'))

        # dnames = os.listdir(self.root_sketch) # ['pysketchsimplify', 'XDoG', 'sketchkeras']
        # idx = random.randint(0, 2) 

        # Simg = sketch_loader(os.path.join(self.root_sketch, dnames[idx], fname))
        # # to align the image size and sketch size
        # Simg = Simg.resize(Cimg.size, Image.LANCZOS)


        # flg = random.random() 

        # if flg < 0.5:
        #     Cimg, Simg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT)


        # Cimg = self.transform(Cimg)
        # Simg = self.stransform(Simg)
        # # hint = self.vtransform(Vimg)

        # mask = np.zeros((self.input_size//4, self.input_size//4, 4))

        # repeat = np.random.randint(10, 30)

        # resize = transforms.Resize(self.input_size//4, Image.BICUBIC)


        # for _ in range(repeat):
        #     mask = np.maximum(mask, making_mask_v3( resize((Cimg.detach() + 1.0 ) / 2.0).numpy(), size=self.input_size//4, flatten=True))
        # cv2.imwrite('test_scribbling.png', mask * 255.0)
        # # exit()

        # hint = kornia.utils.image_to_tensor(mask)

        # hint = (hint - 0.5)/0.5
        # hint = hint.float()



        # # print(Cimg.shape, hint.shape, Simg.shape)
        # # colorization image, hint image, sketch image
        # # print(Cimg.shape)
        # # print(hint.shape)
        # # print(Simg.shape)
        # return Cimg, hint, Simg

        # fname = self.imgs[index]

        # try:
        #     Cimg = color_loader(os.path.join(self.root_img, fname))
        # except:
        #     Cimg = color_loader(os.path.join(self.root_img, fname[:-3] + 'png'))

        # Simg = sketch_loader(os.path.join(self.root_sketch, fname))
        # # to align the image size and sketch size
        # Simg = Simg.resize(Cimg.size, Image.LANCZOS)

        # Vimg = color_loader(os.path.join(self.root_hint, fname[:-9] + 'png'))

        # Vimg = Vimg.resize(Cimg.size, Image.LANCZOS)

        # Mimg = sketch_loader(os.path.join(self.root_hint_mask, fname[:-9] + 'png'))

        # Mimg = Mimg.resize(Cimg.size, Image.LANCZOS)


        # flg = random.random() 

        # if flg < 0.5:
        #     Cimg, Simg, Vimg = Cimg.transpose(Image.FLIP_LEFT_RIGHT), Simg.transpose(Image.FLIP_LEFT_RIGHT), Vimg.transpose(Image.FLIP_LEFT_RIGHT)
        #     Mimg = Mimg.transpose(Image.FLIP_LEFT_RIGHT)

        # Cimg = self.transform(Cimg)
        # Simg = self.stransform(Simg)
        # hint = self.vtransform(Vimg)
        # mask = self.mtransform(Mimg)

        # hint = torch.cat([hint, mask], dim=0)

        # rand_erase = transforms.RandomErasing(p=0.5, value=-1)
        # hint = rand_erase(hint)

        # return Cimg, hint, Simg



class ImageFolder_Test(data.Dataset):
        # dataset    = ImageFolder_Test(root, img_size, method, transform, stransform, htransform, ratio)

    def __init__(self, root, img_size, method='strokes', transform=None, stransform=None, htransform=None, vtransform= None, ratio=None, path_length=None, rnd=None, domain=None, dtype=None, dtype_region=None):
        self.root       = root
        # self.root_sketch = '/home/madorin/gitlab/yuan/hint_colorization/danbooru2021_sketch_png'
        # self.root_scr = '/home/madorin/gitlab/yuan/hint_colorization/danbooru2021_scribbling_rev'


        self.root       = root
        # self.root = glob.glob("/scratch/madono/userstudy_imgs/segmentations/originals/*.image.png")
        # self.approach = approach
        # self.root_sketch = os.path.join('/scratch/madono/main_exp', domain, 'sketch')
        # # self.root_sketch = '/scratch/madono/danbooru2021_sketch_png'
        # # self.root_scr = '/scratch/madono/danbooru2021_scribbling_rev'
        # self.root_scr = os.path.join('/scratch/madono/main_exp/', domain, dtype)
        # # self.root_region = '/scratch/madono/region'
        # self.root_region = os.path.join('/scratch/madono/main_exp/', domain, dtype_region)
        # self.root_sketch = os.path.join('/scratch/madono/main_exp', domain, 'sketch')
        # self.root_scr = os.path.join('/scratch/madono/main_exp/', domain, "hint_from_regions", dtype)
        # self.root_region = os.path.join('/scratch/madono/main_exp/', domain, "segmentation_regions", dtype)
        self.domain = domain
        self.root_sketch =  os.path.join("/scratch/madono/main_exp/", self.domain, "sketch")
        self.root_scr = os.path.join("/scratch/madono/main_exp/", self.domain, "hint_from_regions", "felzenszwalb")
        self.root_region =  os.path.join("/scratch/madono/main_exp/", self.domain, "segmentation_regions",  "felzenszwalb")

        # self.root_sketch =  os.path.join("/scratch/madono/userstudy_imgs/", "sketch")
        # self.root_scr = os.path.join("/scratch/madono/userstudy_imgs/",  "hint_from_regions")
        # self.root_region =  os.path.join("/scratch/madono/userstudy_imgs/", "segmentations")

        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']



        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']        
        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']        
        self.transform = Compose([
            transforms.ToTensor(),
        ])
        self.trans_norm = Compose([
            transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        self.size = 256

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

        # self.root_sketch = '/scratch/madono/danbooru2021_sketch_png'
        # self.root_scr = '/scratch/madono/danbooru2021_scribbling_rev'
        # self.root_region = '/scratch/madono/region'


        # self.sketch_cands =  ['pysketchsimplify', 'XDoG', 'sketchkeras']
        # self.transform  = transform
        self.stransform = stransform
        # self.htransform = htransform
        # self.vtransform = vtransform
        # # self.files      = self.init(root)

        # self.mask_trans =  transforms.Compose([
        #     # RandomSizedCrop(config.image_size // 4, Image.BICUBIC),
        #     # transforms.Resize((256 // 4, 256 // 4), Image.NEAREST),
        #     transforms.Resize((256, 256 ), Image.NEAREST),
        #     transforms.ToTensor(),
    # ])    
        self.crop       = RandomCrop(img_size)
        # self.crop_one = RandomCrop_one(img_size)
        self.method     = method
        self.img_size   = img_size
        self.empty      = 0
        self.ratio = ratio
        self.path_length = path_length
        self.rnd = rnd

    def init(self, dir):
        imgs = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dir)
            for file in files
        ]

        # self.image_paths = glob.glob("/scratch/madono/userstudy_imgs/segmentations/originals/*.image.png")
        return [img for img in imgs if self.is_image(img)]

    def is_image(self, filename):
        return any(filename.endswith(extension) for extension in EXT)

    def __len__(self):
        return len(self.root)

    def __getitem__(self, idx):

        rnd = self.rnd


        fname = self.root[idx].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.root[idx].replace('\n', '').split('/')[-2]

        original = resize(io.imread(os.path.join(self.root_region, dname, fname + '.image.png'), as_gray=False, pilmode="RGB"),  (self.size, self.size))  # Reading target images in RGB
        # original = resize(io.imread(os.path.join(self.root_region, "originals", fname + '.image.png'), as_gray=False, pilmode="RGB"),  (self.size, self.size))  # Reading target images in RGB

        # try:
        sketch_data = resize(io.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png')),  (self.size, self.size))
        # sketch_data = resize(io.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], fname + '.png')),  (self.size, self.size))
        # except:
        #     sketch_data = np.zeros( (self.size, self.size))

        # print(original.shape)
        h, w, _ = original.shape

        region = cv2.imread(os.path.join(self.root_scr, dname, fname + '.image_region64.png'))     
        # region = cv2.imread(os.path.join(self.root_scr, fname + '.image_region64.png'))             
        region = cv2.resize(region, (256, 256), interpolation = cv2.INTER_NEAREST)

        # cand_vals = np.unique(region.reshape((-1,3)), axis=0)

        # cand_vals, count = np.unique(region.reshape((-1,3)), axis=0, return_counts=True)
        # cand_vals = cand_vals[np.argsort(-count)]
        region_uint64 = region.astype(np.uint64)
        id_maps = region_uint64[:, :, 0] * 255 * 255 + region_uint64[:, :, 1] * 255  + region_uint64[:, :, 2]
        cand_vals, count = np.unique(id_maps.reshape(-1), axis=0, return_counts=True)

        # remove 
        cand_vals = cand_vals[np.argsort(-count)]

        length_samples = int(len(cand_vals)  * self.ratio )


        # convert_hint = np.zeros((region.shape[0], region.shape[1], 3))
        # convert_mask = np.zeros((region.shape[0], region.shape[1], 1))            
        single_hint = np.zeros((region.shape[0], region.shape[1], 3))
        single_mask = np.zeros((region.shape[0], region.shape[1], 1))

        length = 1 
        # indices = np.argwhere(np.isin(region, cand_vals[:length_samples]).all(axis=2))
        # sample_area = np.isin(id_maps, list(cand_vals[:length_samples]))[:, :, np.newaxis].astype(np.uint8)
        # sample_area = np.isin(id_maps, list(cand_vals[:length_samples]))[:, :, np.newaxis].astype(np.uint8)
        sample_area = np.isin(id_maps, list(cand_vals[:length_samples])).astype(np.uint8)

        # single_hint[indices[:, 0], indices[:, 1], :] = 1
        # single_mask[indices[:, 0], indices[:, 1], :] = 1

        if self.method == "scribble":

            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_scribble_mask64.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_scribble_col64.png'))

            # mask = self.sketch_loader(os.path.join(self.root_scr, fname + '.image_scribble_mask64.png'))
            # hint = self.load_image(os.path.join(self.root_scr,fname + '.image_scribble_col64.png'))            

            hint = pil2opencv(hint)
            mask = pil2opencv(mask)

            hint= cv2.resize(hint, (256, 256), interpolation = cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)


            single_hint = sample_area[:, :, np.newaxis] * hint
            single_mask = sample_area * mask


            # print(sample_area.shape, single_hint.shape, single_mask.shape)
            # iterations = 1
            # length = self.path_length
            # kernel = np.ones((length, length),np.uint8)
            # single_hint = cv2.dilate(single_hint, kernel=kernel, iterations=iterations)        
            # single_mask = cv2.dilate(single_mask, kernel=kernel, iterations=iterations)       

            # single_mask = cv2.GaussianBlur(single_mask, ksize=(3, 3), sigmaX=0)

            hint = opencv2pil(single_hint.astype(np.uint8))
            mask = opencv2pil(single_mask.astype(np.uint8))  


            totensor =  transforms.ToTensor()
            mask = totensor(mask)
            hint = totensor(hint)
            mask = mask.float()
            hint = hint.float()
            # print(hint.shape)
            # print(hint)
            # try:
            # print(torch.max(hint), torch.min(hint))
            # print(torch.max(mask), torch.min(mask))
            # except:
            #     print("ss")
            hint = torch.cat([hint* mask + 0.5 * (1-mask), mask], dim=0)           
            # hint = (hint - 0.5) / 0.5            
            # hint = torch.cat([hint, mask], dim=0)           

        else:

            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_dot_mask64.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_dot_col64.png'))

            hint = pil2opencv(hint)
            mask = pil2opencv(mask)  

            hint= cv2.resize(hint, (256, 256), interpolation = cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)



            single_hint = sample_area[:, :, np.newaxis] * hint
            single_mask = sample_area * mask
            # single_hint = sample_area * hint
            # single_mask = sample_area * mask[:, :, np.newaxis]
            # single_hint = single_hint * hint
            # single_mask = single_mask * mask[:, :, np.newaxis]

            # iterations = 1
            # length = self.path_length
            # kernel = np.ones((length, length),np.uint8)
            # single_hint = cv2.dilate(single_hint, kernel=kernel, iterations=iterations)        
            # single_mask = cv2.dilate(single_mask, kernel=kernel, iterations=iterations)       

            # single_mask = cv2.GaussianBlur(single_mask, ksize=(3, 3), sigmaX=0)

            hint = opencv2pil(single_hint.astype(np.uint8))
            mask = opencv2pil(single_mask.astype(np.uint8))  


            totensor =  transforms.ToTensor()
            mask = totensor(mask)
            hint = totensor(hint)
            mask = mask.float()
            hint = hint.float()

            hint = torch.cat([hint* mask + 0.5 * (1-mask), mask], dim=0)
            # hint = (hint - 0.5) / 0.5

            # hint = torch.cat([hint, mask], dim=0)           


        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]            

        # hint = torch.cat([self.vtransform(hint), self.mask_trans(mask)], dim=0)
        # hint = scribble_mask.clone().detach().float() * torch.cat([self.vtransform(hint),  (self.mask_trans(mask) + 1.0) / 2.0], dim=0)
        # hint = scribble_mask.clone().detach().float() * torch.cat([self.vtransform(hint),  self.mask_trans(mask)], dim=0)

        # print(print(torch.max(hint)), torch.min(hint),hint.shape)
        # hint[:3, :, :] = (hint[:3, :, :] + 1.0) / 2.0
        original  = self.transform(original)
        # sketch   = self.stransform(sketch)
        # sketch   = self.transform(np.expand_dims(sketch_data, axis=2))
        sketch_data   = self.transform(sketch_data)

        # original = self.trans_norm(original)
        # sketch_data = self.trans_norm(sketch_data)        
        # print(print(torch.max(hint)), torch.min(hint),hint.shape)


        # print(original.shape, hint.shape, sketch_data.shape)
        # print(torch.max(hint), torch.max(mask))
        # torchvision.utils.save_image(original.unsqueeze(0), "original.png")
        # torchvision.utils.save_image(hint.unsqueeze(0), "hint.png")
        # torchvision.utils.save_image(sketch_data.unsqueeze(0), "sketch_data.png")
        # exit()

        return self.trans_norm(original), self.trans_norm(sketch_data), self.trans_norm(hint)

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def sketch_loader(self, path):
        return Image.open(path).convert('L')            



def CreateTestLoader(root, batch_size, img_size, method, ratio, path_length, rnd, domain, dtype, dtype_region):


    transform  = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # def jitter(x):
    #     ran = np.random.uniform(0.7, 1)
    #     return x * ran + 1 - ran

    stransform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Lambda(jitter),
        # transforms.Normalize((0.5), (0.5)),
        # lambda x: x.mean(0).unsqueeze(0)
    ])

    VTrans = transforms.Compose([
        transforms.Resize((256, 256), Image.NEAREST),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # htransform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    hTrans = transforms.Compose([
        # RandomSizedCrop(config.image_size // 4, Image.BICUBIC),
        transforms.Resize(256, Image.NEAREST),
        transforms.ToTensor()
    ])

    # dataset    = ImageFolder(root, img_size, method, transform, stransform, htransform, ratio)
    dataset    = ImageFolder_Test(root, img_size, method, transform, stransform, hTrans, VTrans, ratio, path_length, rnd, domain, dtype, dtype_region)

    loader     = data.DataLoader(
        dataset,
        batch_size,
        shuffle     = False,
        pin_memory  = True,
        num_workers = 10,
        drop_last=False
    )

    return loader
