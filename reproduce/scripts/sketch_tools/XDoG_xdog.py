import cv2
import numpy as np
import random
from scipy import ndimage
import glob
import os

def dog(img, size=(0,0), k=1.6, sigma=0.5, gamma=1):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    return (img1 - gamma * img2)

def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255
    for i in range(0, aux.shape[0]):
        for j in range(0, aux.shape[1]):
            if(aux[i, j] < epsilon):
                aux[i, j] = 1*255
            else:
                aux[i, j] = 255*(1 + np.tanh(phi * (aux[i, j])))
    return aux

def get_xdog_image(img, sigma=0.4, k=2.5, gamma=0.95, epsilon=-0.5, phi=10**9):
    xdog_image = xdog(img, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi).astype(np.uint8)
    return xdog_image

def add_intensity(img, intensity):
    if intensity == 1:
        return img
    inten_const = 255.0 ** (1 - intensity)
    return (inten_const * (img ** intensity)).astype(np.uint8)

def make_xdog(img):
    s = 0.35 + 0.1 * random.random()
    k = 2 + random.random()
    g = 0.95
    return get_xdog_image(img, sigma=s, k=k, gamma=g, epsilon=-0.5, phi=10**9)



# Pout_dir='Osketch_out_XDoG'
# if not os.path.exists(Pout_dir): os.mkdir(Pout_dir)
# for num in ('01', '02', '03', '04', '05', '06', '07', '08', '09', '10a', '10b'):
    # in_dir = "Osketch_out_raw/" + num + "/raw/"
in_dir = '/home/USER/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020/train/'
out_dir = '/home/USER/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020_sketch/XDoG/train'    
# out_dir = os.path.join(Pout_dir, num)
if not os.path.exists(out_dir): os.mkdir(out_dir)
for img in glob.glob(in_dir + '*.image.png'):
    filepath, filename = os.path.split(img)
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    xdog_img = make_xdog(img)
    '''
    width = float(xdog_img.shape[1])
    height = float(xdog_img.shape[0])
    if (width > height):
        xdog_img = cv2.resize(xdog_img, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        xdog_img = cv2.resize(xdog_img, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512
    '''
    # print(filename, os.path.join(out_dir, filename))
    cv2.imwrite(os.path.join(out_dir, filename), xdog_img)



