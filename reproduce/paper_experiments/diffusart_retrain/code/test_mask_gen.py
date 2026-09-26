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


def making_mask_v3(mask_simp, color, size, hinttype):
    choice = np.random.choice(['width', 'height', 'diag'])

    avg_wht = 0.99

    if hinttype == 'dot':
        hh = np.random.randint(256)        
        ww = np.random.randint(256)  

        mask_simp[hh, ww] = 1
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


h, w = 256, 256
mask = np.zeros((h,w, 1))
repeat = np.random.randint(5, 25)

original = resize(io.imread("/home/madorin/datasets/tog2024/main_exp/illust/segmentation/0016/3628016.image.png"),  (256, 256))  # Reading target images in RGB

for _ in range(repeat):
    mask = making_mask_v3(mask, original, size=256, hinttype='scribble')

cv2.imwrite('asasasasas.png', mask*255)