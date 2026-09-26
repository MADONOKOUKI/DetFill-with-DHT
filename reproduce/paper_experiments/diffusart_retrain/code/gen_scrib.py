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
import skimage.draw
def making_mask_v3(mask_simp, color, size):
    choice = np.random.choice(['width', 'height', 'diag'])

    avg_wht = 0.99

    while avg_wht >= 0.7:

        if choice == 'width':
            rnd_height = np.random.randint(2, 4)
            rnd_width = np.random.randint(6, 50)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)

            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'height':
            rnd_height = np.random.randint(6, 50)
            rnd_width = np.random.randint(2, 4)

            rnd1 = np.random.randint(size - rnd_height)
            rnd2 = np.random.randint(size - rnd_width)
            ii, jj = skimage.draw.rectangle((rnd1, rnd2), extent=(rnd_height,rnd_width))
            scrib_color = color[ii, jj]

        elif choice == 'diag':

            thick = np.random.randint(2, 4)
            rnd_width = np.random.randint(6, 50)

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

        if avg_wht < 0.7 and avg_black < 0.7:
            mask_simp[ii, jj] = 1

    return mask_simp


def init_load_data_val():

    #Training data

    #Training images list
    with open('/home/hec/Phd/Diffusion_v1/data/img_list_sfw_512.txt', "r") as a:
        list_color = a.readlines()

    with open('/home/hec/Phd/Diffusion_v1/data/sketchSimp_data.txt', "r") as a:
        list_sketchsimp= a.readlines()

    with open('/home/hec/Phd/Diffusion_v1/data/sketchKeras_data.txt', "r") as a:
        list_sketchkeras= a.readlines()

    return list_color[210000:223000], list_sketchsimp[210000:223000], list_sketchkeras[210000:223000]

i = 1
start = i * 3250
fin = (i + 1) * 3250
color_true, t, k = init_load_data_val()
color = color_true[start:fin]
print(start,fin)
size = 256
for index in range(len(color)):

    original = resize(io.imread(color[index].replace('\n', '')),  (size, size))  # Reading target images in RGB
    f = color[index].replace('\n', '').split('/')
    mask = np.zeros((size, size, 1))
    repeat = np.random.randint(25, 50)
    for _ in range(repeat):
        mask = making_mask_v3(mask, original, size=size)
    mask = 255 * mask
    img = mask.astype(np.uint8)
    io.imsave('/home/hec/Phd/dataset/scrib_fix_new/'+str(f[6][:-4])+'.png', img)
    print(index,'/',len(color))
