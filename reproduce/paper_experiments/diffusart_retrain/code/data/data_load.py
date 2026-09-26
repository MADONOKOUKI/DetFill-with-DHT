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
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    else:
        return None
    new_image = Image.fromarray(new_image)
    return new_image
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)

        return img

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return img

    # define image transformations (e.g. using torchvision)
trans = Compose([
    transforms.ToTensor(),
    ])
trans_n = Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])

def mask_gen(image_size):
    mu, sigma = 1, 0.005
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
    maskS = image_size
    mask = torch.rand(1, maskS, maskS).ge(X.rvs(1)[0]).float()
    return mask


def making_mask_v3(mask_simp, color, size, hinttype):
    choice = np.random.choice(['width', 'height', 'diag'])

    avg_wht = 0.99

    if hinttype == 'dot':
        mask = np.zeros( (mask_simp.shape[0], mask_simp.shape[1]) )
        
        hh = np.random.randint(256)        
        ww = np.random.randint(256)  

        mask[hh, ww] = 1
        length = random.randint(1, 4)

        kernel = np.ones((length, length),np.uint8)
        mask  = cv2.dilate(mask, kernel=kernel, iterations=1) 

        mask_simp += mask[:, :, np.newaxis]

        return mask_simp              

    while avg_wht >= 0.6:

        if choice == 'width':
            rnd_height = np.random.randint(1, 4)
            rnd_width = np.random.randint(5, 30)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 30)
            rnd_width = np.random.randint(1, 4)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'diag':

            thick = np.random.randint(1, 4)
            rnd_width = np.random.randint(4, 30)

            rnd1 = np.random.randint(size - thick - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.line(rnd1, rnd2, rnd1+rnd_width, rnd2+rnd_width)
            ii = np.hstack([ii + i for i in range(thick)])
            jj = np.tile(jj, thick)
            scrib_color = color[ii, jj]


        if choice == 'diag':
            count_white = np.sum(np.all(scrib_color == 1, axis = 1))
            count_black = np.sum(np.all(scrib_color == 0, axis = 1))
            size_pix = scrib_color.shape[0]
        else:
            count_white = np.sum(np.all(scrib_color == 1, axis = 2))
            count_black = np.sum(np.all(scrib_color == 0, axis = 2))
            size_pix = scrib_color.shape[0] * scrib_color.shape[1]
        avg_wht = count_white / size_pix
        avg_black = count_black / size_pix



        if avg_wht < 0.6 and avg_black < 0.6:
            mask_simp[ii, jj] = 1

    return mask_simp


def making_mask_flat_colors_64(mask_simp, color, color_hint, size, hinttype):
    choice = np.random.choice(['width', 'height', 'diag'])

    avg_wht = 0.99

    if hinttype == 'dot':
        mask = np.zeros( (64, 64) )
        
        hh = np.random.randint(64)        
        ww = np.random.randint(64)  

        mask[hh, ww] = 1
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)
        mask_simp += mask[:, :, np.newaxis]


        color_lists = color[mask==1]
        avg_color = np.mean(color_lists, axis=0)

        color_hint[mask==1] = avg_color 


        return color_hint, mask_simp         

    else:

        if choice == 'width':
            rnd_height = 1
            rnd_width = np.random.randint(5, 30)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'height':
            rnd_height = np.random.randint(4, 30)
            rnd_width = 1

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'diag':

            thick = 1
            rnd_width = np.random.randint(4, 30)

            rnd1 = np.random.randint(size - thick - rnd_width - 1)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.line(rnd1, rnd2, rnd1+rnd_width, rnd2+rnd_width)
            ii = np.hstack([ii + i for i in range(thick)])
            jj = np.tile(jj, thick)
            scrib_color = color[ii, jj]


        if choice == 'diag':
            count_white = np.sum(np.all(scrib_color == 1, axis = 1))
            count_black = np.sum(np.all(scrib_color == 0, axis = 1))
            size_pix = scrib_color.shape[0]
        else:
            count_white = np.sum(np.all(scrib_color == 1, axis = 2))
            count_black = np.sum(np.all(scrib_color == 0, axis = 2))
            size_pix = scrib_color.shape[0] * scrib_color.shape[1]
        avg_wht = count_white / size_pix
        avg_black = count_black / size_pix



        # if avg_wht < 0.6 and avg_black < 0.6:
        mask_single = np.zeros( (64, 64) )
        # mask_simp[ii, jj] = 1
        mask_single[ii,jj] = 1
        # mask_simp = cv2.resize(mask_simp, (256, 256), interpolation = cv2.INTER_NEAREST)
        mask_single = cv2.resize(mask_single, (256, 256), interpolation = cv2.INTER_NEAREST)

        mask_simp = np.maximum(mask_single[:, :, np.newaxis], mask_simp)
        # print(mask_simp.shape, mask_single.shape)
        # exit()
        color_lists = color[mask_single==1]
        avg_color = np.mean(color_lists, axis=0)
        # print(avg_color)
        # print(color_hint.shape, mask_simp.shape)
        color_hint[mask_single==1] = avg_color 

        return color_hint, mask_simp

        
# def making_mask_flat_colors_64(mask_simp, color, color_hint, size, hinttype):
#     choice = np.random.choice(['width', 'height', 'diag'])

#     avg_wht = 0.99

#     if hinttype == 'dot':
#         mask = np.zeros( (64, 64) )
        
#         hh = np.random.randint(64)        
#         ww = np.random.randint(64)  

#         mask[hh, ww] = 1
#         mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_NEAREST)
#         mask_simp += mask[:, :, np.newaxis]


#         color_lists = color[mask==1]
#         avg_color = np.mean(color_lists, axis=0)

#         color_hint[mask==1] = avg_color 


#         return color_hint, mask_simp         

#     else:

#         if choice == 'width':
#             rnd_height = 1
#             rnd_width = np.random.randint(5, 30)

#             rnd1 = np.random.randint(size - rnd_height)
#             rnd2 = np.random.randint(size - rnd_width)

#             ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
#             scrib_color = color[ii, jj]

#         elif choice == 'height':
#             rnd_height = np.random.randint(4, 30)
#             rnd_width = 1

#             rnd1 = np.random.randint(size - rnd_height)
#             rnd2 = np.random.randint(size - rnd_width)
#             ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
#             scrib_color = color[ii, jj]

#         elif choice == 'diag':

#             thick = 1
#             rnd_width = np.random.randint(4, 30)

#             rnd1 = np.random.randint(size - thick - rnd_width - 1)
#             rnd2 = np.random.randint(size - rnd_width)

#             ii, jj = skimage.draw.line(rnd1, rnd2, rnd1+rnd_width, rnd2+rnd_width)
#             ii = np.hstack([ii + i for i in range(thick)])
#             jj = np.tile(jj, thick)
#             scrib_color = color[ii, jj]


#         if choice == 'diag':
#             count_white = np.sum(np.all(scrib_color == 1, axis = 1))
#             count_black = np.sum(np.all(scrib_color == 0, axis = 1))
#             size_pix = scrib_color.shape[0]
#         else:
#             count_white = np.sum(np.all(scrib_color == 1, axis = 2))
#             count_black = np.sum(np.all(scrib_color == 0, axis = 2))
#             size_pix = scrib_color.shape[0] * scrib_color.shape[1]
#         avg_wht = count_white / size_pix
#         avg_black = count_black / size_pix



#         # if avg_wht < 0.6 and avg_black < 0.6:
#         mask_single = np.zeros( (64, 64) )
#         mask_simp[ii, jj] = 1
#         mask_single[ii,jj] = 1
#         mask_simp = cv2.resize(mask_simp, (256, 256), interpolation = cv2.INTER_NEAREST)
#         mask_single = cv2.resize(mask_single, (256, 256), interpolation = cv2.INTER_NEAREST)

#         color_lists = color[mask_single==1]
#         avg_color = np.mean(color_lists, axis=0)

#         color_hint[mask_single==1] = avg_color 

#         return color_hint, mask_simp



def making_mask(mask_simp, size):
    choice = np.random.choice(['width', 'height', 'diag'])

    if choice == 'width':
        rnd_height = np.random.randint(2, 4)
        rnd_width = np.random.randint(6, 24)

        rnd1 = np.random.randint(size - rnd_height)
        rnd2 = np.random.randint(size - rnd_width)
        mask_simp[:, rnd1:rnd1 + rnd_height, rnd2:rnd2 + rnd_width] = 1

    elif choice == 'height':
        rnd_height = np.random.randint(6, 24)
        rnd_width = np.random.randint(2, 4)

        rnd1 = np.random.randint(size - rnd_height)
        rnd2 = np.random.randint(size - rnd_width)
        mask_simp[:, rnd1:rnd1 + rnd_height, rnd2:rnd2 + rnd_width] = 1

    elif choice == 'diag':
        rnd_height = np.random.randint(2, 4)
        rnd_width = np.random.randint(6, 24)

        rnd1 = np.random.randint(size - rnd_height - rnd_width - 1)
        rnd2 = np.random.randint(size - rnd_width)

        for index in range(rnd_width):
            mask_simp[:, rnd1 + index: rnd1 + rnd_height + index, rnd2 + index] = 1

    return mask_simp

class MyData_train(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, transform=trans, trans_norm=trans_n ,size=128):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        original = resize(io.imread(self.target_path[index].replace('\n', '')),  (self.size, self.size))  # Reading target images in RGB

        if rnd <= 0.5:
            sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))

        else:
            sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))

        if rnd_hints <= 0.7:
            mask = mask_gen(image_size = self.size)
        else:
            mask = torch.zeros(1, self.size, self.size).float()

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            original = self.transform(original)
            sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))
            hint = torch.cat((original * mask, mask), 0)

            original = self.trans_norm(original)
            sketch_data = self.trans_norm(sketch_data)
            hint = self.trans_norm(hint)






        return original, sketch_data, hint

    def __len__(self):
        return len(self.target_path)


class MyData_train_luminance(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, transform=trans, trans_norm=trans_n ,size=128, color='yuv'):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size
        self.color = color

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        original = resize(io.imread(self.target_path[index].replace('\n', '')),  (self.size, self.size))  # Reading target images in RGB

        if rnd <= 0.5:
            sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))

        else:
            sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))

        # mask = mask_gen(image_size=self.size)

        if rnd_hints <= 0.7:
            mask = mask_gen(image_size = self.size)
        else:
            mask = torch.zeros(1, self.size, self.size).float()

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            if self.color=='lab':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_lab(original)
                original_ab = original_Lab[1::, :, :] / 127.0
                original_L = torch.unsqueeze((original_Lab[0, :, :] / 100.0), 0)


                sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))



                original_L = self.trans_norm(original_L)
                # original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                # mask_norm = self.trans_norm(mask)
                hint_norm = torch.cat((original_lab * mask, mask), 0)

                original = self.trans_norm(original)
                sketch_data = self.trans_norm(sketch_data)
                # hint = self.trans_norm(hint)

            elif self.color=='yuv':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_yuv(original)
                original_ab = original_Lab[1::, :, :] / 0.5
                original_L = torch.unsqueeze((original_Lab[0, :, :]), 0)

                sketch_data = self.transform(np.expand_dims(sketch_data, axis=2))

                original_L = self.trans_norm(original_L)
                original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                hint_norm = torch.cat((original_L * mask, mask), 0)
                original = self.trans_norm(original)
                # hint = self.trans_norm(hint)
                sketch_data = self.trans_norm(sketch_data)

        return original, sketch_data, original_L, original_ab, hint_norm

    def __len__(self):
        return len(self.target_path)

class MyData_train_color_flat(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, transform=trans, trans_norm=trans_n ,size=128, color='yuv'):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size
        self.color = color

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        img_list = self.target_path[index].replace('\n', '')
        f = img_list[:-4]
        f = ('/home/hec/Phd/dataset/flatten/' + str(f.split('/')[-1]) + '.png')
        original = resize(io.imread(f),  (self.size, self.size))
        if rnd <= 0.5:
            sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))

        else:
            sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))

        mask = mask_gen(image_size=self.size)

        # if rnd_hints <= 0.7:
        #     mask = mask_gen(image_size = self.size)
        # else:
        #     mask = torch.zeros(1, self.size, self.size).float()

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            if self.color=='lab':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_lab(original)
                original_ab = original_Lab[1::, :, :] / 127.0
                original_L = torch.unsqueeze((original_Lab[0, :, :] / 100.0), 0)


                sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))



                original_L = self.trans_norm(original_L)
                # original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                # mask_norm = self.trans_norm(mask)
                hint_norm = torch.cat((original_lab * mask, mask), 0)

                original = self.trans_norm(original)
                sketch_data = self.trans_norm(sketch_data)
                # hint = self.trans_norm(hint)

            elif self.color=='yuv':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_yuv(original)
                original_ab = original_Lab[1::, :, :] / 0.5
                original_L = torch.unsqueeze((original_Lab[0, :, :]), 0)

                sketch_data = self.transform(np.expand_dims(sketch_data, axis=2))

                original_L = self.trans_norm(original_L)
                original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                hint_norm = torch.cat((original_lab_norm * mask, mask), 0)
                original = self.trans_norm(original)
                # hint = self.trans_norm(hint)
                sketch_data = self.trans_norm(sketch_data)

        return original, sketch_data, original_L, original_ab, hint_norm

    def __len__(self):
        return len(self.target_path)

class MyData_train_color(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, transform=trans, trans_norm=trans_n ,size=128, color='lab'):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size
        self.color = color

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        original = resize(io.imread(self.target_path[index].replace('\n', '')),  (self.size, self.size))  # Reading target images in RGB

        if rnd <= 0.5:
            sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))

        else:
            sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))

        mask = mask_gen(image_size=self.size)
        # # mask = mask_gen(image_size=self.size)
        # if rnd_hints <= 0.7:
        #     mask = mask_gen(image_size = self.size)
        # else:
        #     mask = torch.zeros(1, self.size, self.size).float()

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            if self.color=='lab':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_lab(original)
                original_ab = original_Lab[1::, :, :] / 127.0
                original_L = torch.unsqueeze((original_Lab[0, :, :] / 100.0), 0)


                sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))



                original_L = self.trans_norm(original_L)
                original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                # mask_norm = self.trans_norm(mask)
                hint_norm = torch.cat((original_lab_norm * mask, mask), 0)

                original = self.trans_norm(original)
                sketch_data = self.trans_norm(sketch_data)
                # hint = self.trans_norm(hint)
            elif color=='yuv':
                original = self.transform(original)
                original_Lab = kornia.color.rgb_to_yuv(original)
                original_ab = original_Lab[1:, :, :] / 0.5
                sketch_data = self.transform(np.expand_dims(sketch_data, axis=2))
                original_L = torch.unsqueeze((original_Lab[0, :, :]), 0)
                # hint = torch.cat((original * mask, mask), 0)

                original_L = self.trans_norm(original_L)
                original_lab_norm = torch.cat((original_L, original_ab), dim=0)
                hint_norm = torch.cat((original_lab_norm * mask, mask), 0)
                original = self.trans_norm(original)
                sketch_data = self.trans_norm(sketch_data)
                # hint = self.trans_norm(hint)






        return original, sketch_data, original_L, original_ab, hint_norm

    def __len__(self):
        return len(self.target_path)

class MyData_test(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, transform=trans, trans_norm=trans_n ,size=128):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size

    def __getitem__(self, index):

        rnd = np.random.rand()

        original = resize(io.imread(self.target_path[index].replace('\n', '')),  (self.size, self.size))  # Reading target images in RGB

        if rnd <= 0.5:
            sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))

        else:
            sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))

        mask = mask_gen(image_size=self.size)
        # mask = torch.zeros(1, self.size, self.size).float()


        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            original = self.transform(original)
            sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))
            # hint_col = resize(io.imread('/home/hec/Phd/Diffusion_v1/color_t.jpg'), (self.size, self.size))
            # hint_col = self.transform(hint_col)


            hint = torch.cat((original * mask, mask), 0)

            original = self.trans_norm(original)
            sketch_data = self.trans_norm(sketch_data)
            hint = self.trans_norm(hint)






        return original, sketch_data, hint

    def __len__(self):
        return len(self.target_path)


class MyData_train_scrib(Dataset):
    def __init__(self, root, size, approach='proposed', transform=trans, trans_norm=trans_n, hint=None, domain=None, dtype=None, dtype_region=None, fix_thickness=None):
        # self.target_path = target_path
        # self.sketchkeras_path = sketchkeras_path
        # self.sketch_simp = sketch_simp

        self.root       = root
        self.approach = approach
        # self.root_sketch = os.path.join('/scratch/madono/main_exp', domain, 'sketch')
        # # self.root_sketch = '/scratch/madono/danbooru2021_sketch_png'
        # # self.root_scr = '/scratch/madono/danbooru2021_scribbling_rev'
        # self.root_scr = os.path.join('/scratch/madono/main_exp/', domain, dtype)
        # # self.root_region = '/scratch/madono/region'
        # self.root_region = os.path.join('/scratch/madono/main_exp/', domain, dtype_region)
        self.domain = domain

        self.root_sketch =  os.path.join("/scratch/madono/main_exp/", self.domain, "sketch")
        self.root_scr = os.path.join("/scratch/madono/main_exp/", self.domain, "hint_from_regions", "felzenszwalb")
        self.root_region =  os.path.join("/scratch/madono/main_exp/", self.domain, "segmentation_regions",  "felzenszwalb")
        self.sketch_cands =  ['pysimp', 'XDoG', 'sketchkeras']


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



        rnd = random.randint(0, 2)
        rnd_hints = random.randint(0, 2)

        fname = self.root[index].replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = self.root[index].replace('\n', '').split('/')[-2]

        # original = resize(io.imread(os.path.join(self.root_region, dname, fname + '.image.png'), as_gray=False, pilmode="RGB"),  (self.size, self.size))  # Reading target images in RGB

        # try:
        # sketch_data = resize(io.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png')),  (self.size, self.size))
        # except:
        #     sketch_data = np.zeros( (self.size, self.size))

        original = cv2.resize(cv2.imread(os.path.join(self.root_region, dname, fname + '.image.png')),  (256, 256)) 
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # try:
        # sketch_data = resize(io.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.image.png')),  (self.size, self.size))
        sketch_data = cv2.resize(cv2.imread(os.path.join(self.root_sketch,  self.sketch_cands[rnd], dname, fname + '.png')),  (self.size, self.size))

        h, w, _ = original.shape
        mask = np.zeros((256, 256, 1))

        repeat = np.random.randint(5, 25)
        color_hint = np.zeros(original.shape)
        for _ in range(repeat):
           color_hint, mask = making_mask_flat_colors_64(mask, original, color_hint, size=64, hinttype=self.hint)

        # print(np.max(mask), np.max(color_hint))
        # exit()
        color_hint = color_hint * mask + 122.5 * (1 - mask)

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        # if self.transform:
        original = self.trans_norm(self.transform(original))
        sketch_data = self.trans_norm(self.transform(np.expand_dims(sketch_data, axis= 2)))
        # original = self.trans_norm(original)
        # sketch_data = self.trans_norm(sketch_data)

        mask = kornia.utils.image_to_tensor(mask)
        color_hint = kornia.utils.image_to_tensor(color_hint)
        hint = self.trans_norm(torch.cat((color_hint / 255.0, mask), 0))



        return original, sketch_data, hint

 


 

    def __len__(self):
        return len(self.root)
    
    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def sketch_loader(self, path):
        return Image.open(path).convert('L')       

class MyData_test_scrib(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, size, transform=trans, trans_norm=trans_n):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size

    def __getitem__(self, index):

        rnd = np.random.rand()
        rnd_hints = np.random.rand()
        original = resize(io.imread(self.target_path[index].replace('\n', '')),  (self.size, self.size))  # Reading target images in RGB

        # if rnd <= 0.5:
        #     sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')),  (self.size, self.size))
        #
        # else:
        #     sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')),  (self.size, self.size))
        # sketch_data = resize(io.imread(self.sketch_simp[index].replace('\n', '')), (self.size, self.size))
        sketch_data = resize(io.imread(self.sketchkeras_path[index].replace('\n', '')), (self.size, self.size))
        h, w, _ = original.shape
        # mask = torch.zeros((1, h, w))
        mask = np.zeros((h,w, 1))

        repeat = np.random.randint(5, 25)
        for _ in range(repeat):
            mask = making_mask_v3(mask, original, size=self.size)


        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            original = self.transform(original)
            sketch_data = self.transform(np.expand_dims(sketch_data, axis= 2))
            original = self.trans_norm(original)
            sketch_data = self.trans_norm(sketch_data)
            mask = kornia.utils.image_to_tensor(mask)

            hint = torch.cat((original * mask, mask), 0)







        return original, sketch_data, hint

    def __len__(self):
        return len(self.target_path)



class MyData_paper_test(Dataset):
    def __init__(self, target_path, sketchkeras_path, sketch_simp, size, transform=trans, trans_norm=trans_n):
        self.target_path = target_path
        self.sketchkeras_path = sketchkeras_path
        self.sketch_simp = sketch_simp
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size

    def __getitem__(self, index):
        f = self.target_path[index].split('/')
        print(f)
        f[7] = '4489640.png'
        # original = resize(io.imread(self.target_path[index]),  (self.size, self.size))  # Reading target images in RGB
        original = resize(io.imread('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/' + f[7]), (self.size, self.size))  # Reading target images in RGB

        # sketch_data = resize(io.imread('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/sketch/' + f[7]), (self.size, self.size))
        sketch_data = resize(io.imread('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/sketch/' + f[7][:-4] + '.jpg' ), (self.size, self.size))

        mask = resize(io.imread(('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/hints_paper/' + f[7])), (self.size, self.size))

        # alpha = resize(io.imread(('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/hints_paper/test/w_' + f[8])), (self.size, self.size))

        # mask_np = resize(io.imread(('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/hints_paper/test/test.png')),
        #                  (self.size, self.size))
        #
        # mask_np[np.all(mask_np < (100/255), axis=2), :] = 0
        # alpha = np.zeros_like(mask_np)
        # alpha[mask_np>0]=1


        #
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

            # mask_np = self.transform(mask_np)
            # mask_np = self.trans_norm(mask_np)
            # alpha = kornia.utils.image_to_tensor(alpha[:,:, 0:1])
            # torch.zeros_like(mask_np)
            # k = mask_np * alpha
            mask = kornia.utils.image_to_tensor(mask[:, :, 0:1])

            hint = torch.cat((original * mask, mask), 0)

            # hint = torch.cat((k, alpha), 0)

        return original, sketch_data, hint

    def __len__(self):
        return len(self.target_path)