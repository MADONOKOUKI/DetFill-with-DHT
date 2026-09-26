import cv2
import numpy as np
import random
from scipy import ndimage
import glob
import os
import sys
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
# in_dir = '/home/madorin/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020/train/'
# out_dir = '/home/madorin/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020_sketch/XDoG/train'    

in_dir_root = '/scratch/madono/danbooru2021/original/'
# out_dir = 'sketch_out'
out_dir_root = '/scratch/madono/tog_created/sketch/XDoG/' 

# out_dir = os.path.join(Pout_dir, num)

# dirs = os.listdir(in_dir_root)[int(sys.argv[1]):int(sys.argv[2])]
dirs =  ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

# print(dirs)
# exit()
random.shuffle(dirs)
for d in dirs:
    in_dir = os.path.join(in_dir_root, d)
    out_dir = os.path.join(out_dir_root, d)
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    # print(os.path.join(in_dir + '*.png'))
    # print(glob.glob(os.path.join(in_dir + '/*.png')), in_dir)

    files = glob.glob(in_dir + '/*.png') + glob.glob(in_dir + '/*.jpg')
    random.shuffle(files)
    for img in files:
        filepath, filename = os.path.split(img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        print(os.path.join(out_dir, filename))

        if os.path.exists(os.path.join(out_dir, filename)):
            continue        
        
        img = cv2.resize(img, (512, 512))
        xdog_img = make_xdog(img)

        # print(os.path.join(out_dir, filename[:-4] + '.png'))
        # exit()
        cv2.imwrite(os.path.join(out_dir, filename[:-4] + '.png'), xdog_img)
        # exit()



