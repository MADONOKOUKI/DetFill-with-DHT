# ---------------------------------------------------------------------------
# ARCHIVAL COPY (verbatim) of the original dataset-generation script
# (originally `hint_dot_generation_20240114_illust_64.py`) used to
# build the paper dataset. Kept unmodified for provenance; the paths below
# refer to the original experiment environment and are not meant to be run
# as-is. For new data, use the `hintauc` library or ../D_retrain_gen_hints.py.
# ---------------------------------------------------------------------------
import cv2
import skimage
import numpy as np
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
from numpy.lib.npyio import save
from skimage.morphology import skeletonize
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
# from utils import dilation, dilation_7
import sys
import random
from natsort import natsorted
import gc
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u
import json
import os
import signal
import glob
import time
import copy


def skeleton_to_scribbling(fil):
    # fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    fil.analyze_skeletons(branch_thresh=3* u.pix, skel_thresh=3 * u.pix, prune_criteria='length')

    return fil




def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("FUBAR")


def make_scribbling(img_path, region_path_o, save_path):
    # region_path_o = '1447002.png'


    
    # img= cv2.imread('1447002.png')

    # print(img_path)
    # print(region_path_o)

    img = cv2.imread(img_path)
    print(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # important!!!

    region = cv2.imread(region_path_o)
    # print(img.shape)
    # print(region.shape)

    # img = cv2.resize(img, (region.shape[1], region.shape[0]))
    img = cv2.resize(img, (hint_img_size, hint_img_size))
    print(region.shape[1], region.shape[0])
    # print(img.shape, region.shape)

    bd = np.zeros(region.shape)

    bd[:-1, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 
    bd[1:, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 

    bd[:, :-1, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 
    bd[:, 1:, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 

    bd = np.sum(bd, axis=2)
    bd[bd > 0] = 1
    bd[bd < 0] = 0


    _, labeled_image = cv2.connectedComponents((1-bd.astype(np.uint8))* 255)

    # cv2.imwrite('debug.png', bd * 255)
    # cv2.imwrite('skelton.png', skeletonize(1 - bd) * 255)


    skeleton = skeletonize(1 - bd) 

    # cand_vals = np.unique(region.reshape((-1,3)), axis=0)

    print(region.shape)
    # cand_vals = np.unique(region.reshape((-1)), axis=0)
    # remove background
    # cand_vals = cand_vals[1:]    

    region = cv2.resize(region, (hint_img_size, hint_img_size),  interpolation = cv2.INTER_NEAREST)
    cand_vals = np.unique(region.reshape((-1,3)), axis=0)


    scribble_img = np.zeros(region.shape)
    scribble_mask = np.zeros(region.shape) 
    scribbles_single = np.zeros((hint_img_size, hint_img_size))
    time_sta = time.time()
    skeleton_new = np.zeros((region.shape[0], region.shape[1]))


    dot_mask = np.zeros((hint_img_size, hint_img_size))
    dot_color = np.zeros((hint_img_size, hint_img_size, 3))
    rng = np.random.default_rng()

    for i in tqdm(range(len(cand_vals))):
        mask = np.all(region == cand_vals[i], axis=2) * 255
        _, labeled_image = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        lbs = np.unique(labeled_image)
        # print(lbs)
        if lbs.shape[0] > 2:
            for j in range(2, lbs.shape[0]):
                # print(j)
                flg = True
                while(flg):
                    val = np.array([ (255*rng.random()) // 1 , (255*rng.random()) // 1, (255*rng.random()) // 1]).reshape(1, 3).astype(np.uint8)
                    cflg  = True
                    for v in cand_vals:
                        if np.all(v == val) == True:
                            cflg = False
                    if cflg == True:
                        region[labeled_image == j, :] = val
                        flg = False
                
    # cv2.imwrite(save_path + '_region_adter.png', region)
    # exit()

    cand_vals = np.unique(region.reshape((-1,3)), axis=0)

    nums = []
    for i in tqdm(range(len(cand_vals))):
        
        skeleton_tmp = np.zeros((region.shape[0], region.shape[1]))
        indices = np.argwhere(np.isin(np.all(region == cand_vals[i], axis=2), 1 ))


        skeleton_tmp[indices[:, 0], indices[:, 1]] = 1
        # skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations = 1)    

        skeleton_s =  skeletonize(skeleton_tmp) 
        # skeleton_s  = cv2.dilate(skeleton_s.astype(np.uint8), kernel, iterations = 1)    


        
        skeleton_new += skeleton_s
        mval = copy.deepcopy(img[indices[:, 0], indices[:, 1]].mean(axis=0).astype(np.uint8))

        scribble_img[indices[:, 0], indices[:, 1]] = mval
        # print(mval)

        indices = np.argwhere(np.isin(skeleton_s, 1))

        scribble_mask[indices[:, 0], indices[:, 1]] = 255



        kernel = np.ones((3, 3),np.uint8)

        skeleton_s  = cv2.dilate(skeleton_s.astype(np.uint8), kernel, iterations = 1)  * 255
        fil = FilFinder2D(skeleton_s, distance=250 * u.pc, mask=skeleton_s)
            

        # fil = FilFinder2D(skeleton_new,  distance=250 * u.pc, mask=skeleton_new )
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False,
        use_existing_mask=True)
        fil.medskel(verbose=False)

        flg = 0
        # fil = make_scribbling(fil)
        try:
            print("Before: %s" % time.strftime("%M:%S"))
            fil = skeleton_to_scribbling(fil)
        except Exception as ex:
            print("Gotcha!")
            flg = 1
            continue
        finally:
            signal.alarm(0)
            print("After: %s" % time.strftime("%M:%S"))

        #  if finished, add path
        nums.append(indices.shape[0])            


        # only pick up 
        scribbles_single += (fil.skeleton_longpath * skeleton_tmp)

        indices_scr = np.argwhere(np.isin(fil.skeleton_longpath, 1))

        try:
            mx, my = int(indices_scr[:, 0].mean()), int(indices_scr[:, 1].mean())

            dot_mask[mx,my] = 1
            # dot_color[mx,my, :] = crop_region_col[0, :]
        except:
            continue
        

    cv2.imwrite(save_path + '_region'+str(hint_img_size)+'.png', region )
    cv2.imwrite(save_path + '_scribble_mask'+str(hint_img_size)+'.png', scribbles_single.astype(np.uint8) * 255 )
    cv2.imwrite(save_path + '_scribble_col'+str(hint_img_size)+'.png',  scribble_img * scribbles_single[:, :, np.newaxis] )
    cv2.imwrite(save_path + '_flatten_img'+str(hint_img_size)+'.png',  (scribble_img  ).astype(np.uint8))
    cv2.imwrite(save_path + '_dot_mask'+str(hint_img_size)+'.png', dot_mask.astype(np.uint8) * 255 )
    cv2.imwrite(save_path + '_dot_col'+str(hint_img_size)+'.png',  (scribble_img * dot_mask[:, :, np.newaxis] ).astype(np.uint8))    

    # exit()


if __name__=='__main__':



    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'illust', 'train_paper_64.txt'), "r") as a:
    #     list_train_imgs = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'illust', 'valid_paper_64.txt'), "r") as a:
    #     list_valid_imgs = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'illust', 'test_paper_64.txt'), "r") as a:
    #     list_test_imgs = a.readlines()    

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'train_paper_64.txt'), "r") as a:
    #     list_train_imgs_r = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'valid_paper_64.txt'), "r") as a:
    #     list_valid_imgs_r = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'test_paper_64.txt'), "r") as a:
    #     list_test_imgs_r = a.readlines()    


    with open(os.path.join('/scratch/madono/configs',  'train_paper.txt'), "r") as a:
        list_train_imgs = a.readlines()

    with open(os.path.join('/scratch/madono/configs', 'valid_paper.txt'), "r") as a:
        list_valid_imgs = a.readlines()

    with open(os.path.join('/scratch/madono/configs', 'test_paper.txt'), "r") as a:
        list_test_imgs = a.readlines()    

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'train_paper_64.txt'), "r") as a:
    #     list_train_imgs_r = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'valid_paper_64.txt'), "r") as a:
    #     list_valid_imgs_r = a.readlines()

    # with open(os.path.join('/scratch/madono/configs', 'ablation', 'real', 'test_paper_64.txt'), "r") as a:
    #     list_test_imgs_r = a.readlines()        
                
            



    flists = [list_train_imgs, list_valid_imgs, list_test_imgs] #, list_train_imgs_r, list_valid_imgs_r, list_test_imgs_r]
    # flists = [list_train_imgs_r, list_valid_imgs_r, list_test_imgs_r]


    # dirs = ['sam_hq', 'superpixel/superpixel_250_0.01', 'superpixel/superpixel_250_1', 'superpixel/superpixel_250_10' 
    #                 , 'superpixel/superpixel_500_0.01', 'superpixel/superpixel_500_1', 'superpixel/superpixel_500_10' 
    #                 , 'superpixel/superpixel_750_0.01', 'superpixel/superpixel_750_1', 'superpixel/superpixel_750_10'  ]
    dirs = ['']
    
    # base_path = '/home/madorin/datasets/tog2024/main_exp/illust/segmentation'
    base_path = '/scratch/madono/tog_created/segmentation'

    # /home/madorin/datasets/tog2024/main_exp/illust/deep_illust_pretrained/
    save_path = '/scratch/madono/tog_created/illust/deep_illust_pretrained/'

    

    hint_img_size = 64
    for kk in range(3):
        for jj in range(1):
            random.shuffle(flists[kk])
            # exit()
            for cnt in range(len(flists[kk])):
                print(kk, cnt)
                # print(flists[i])
                file_r = flists[kk][cnt]
                fname = file_r.replace('\n', '').split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
                dname = file_r.replace('\n', '').split('/')[-2]
                # /home/madorin/datasets/tog2024/main_exp/illust/segmentation/0008/1870008.image.png
                # region_path_o = file_r[:-10]+'png'
                image_path = os.path.join(base_path, dirs[jj], dname, fname + '.image.png' )
                region_path_o = os.path.join(base_path, dirs[jj], dname, fname + '.png' )

                save_path_p = os.path.join(save_path, dname)
                if not os.path.exists(save_path_p): os.mkdir(save_path_p)
                save_path_p = os.path.join(save_path, dname, fname)

                if os.path.isfile(save_path_p+'_scribble_mask'+str(hint_img_size)+'.png'):
                    continue
                print(kk, jj, cnt)
                make_scribbling(image_path, region_path_o, save_path_p)
              