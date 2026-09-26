# Archival copy (verbatim, from cayenne1:/scratch/madono/archives_submission/main_hg_illust_64.py, DATE) of the
# 64-px hint generation script whose DOT RULE produced the paper's stored hint maps
# (main_exp_felzenszwalb_fixdot, earlier, and the revision test-split maps released with this repository):
#   dot = the in-region longest-path pixel with the smallest total Manhattan distance to the other in-region
#         path pixels (first index on ties); a single-pixel region whose path was pruned gets the pixel itself.
# Verified DATE on 5,335 regions of six test images: 100% of the stored dots follow this rule.
# canonical/hint_dot_generation.py is the earlier version of the same script, which placed the dot at the
# truncated mean of the whole path; that earlier rule is NOT the one behind the released maps.
# Not meant to be run as-is (hard-coded paths).
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

hint_img_size = 64

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
    # print(region.shape[1], region.shape[0])
    # print(img.shape, region.shape)

    # bd = np.zeros(region.shape)

    # bd[:-1, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 
    # bd[1:, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 

    # bd[:, :-1, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 
    # bd[:, 1:, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 

    # bd = np.sum(bd, axis=2)
    # bd[bd > 0] = 1
    # bd[bd < 0] = 0


    # _, labeled_image = cv2.connectedComponents((1-bd.astype(np.uint8))* 255)

    # # cv2.imwrite('debug.png', bd * 255)
    # # cv2.imwrite('skelton.png', skeletonize(1 - bd) * 255)


    # skeleton = skeletonize(1 - bd) 

    # # cand_vals = np.unique(region.reshape((-1,3)), axis=0)

    # print(region.shape)
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

    cand_vals_check = copy.deepcopy(cand_vals)

    for i in tqdm(range(len(cand_vals))):
        # indices = np.argwhere(np.isin(region, cand_vals[i])) #.all(axis=2))
        # mask = np.zeros((region.shape[0], region.shape[1], 1))
        mask = np.isin(region, cand_vals[i]).all(axis=-1) * 255
        # mask[indices[:, 0], indices[:, 1], :] = 255
        # mask = np.all(region == cand_vals[i], axis=2) * 255
        _, labeled_image = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
        lbs = np.unique(labeled_image)
        if lbs.shape[0] > 2: # background = 0, single bolbs = 1, 
            for j in range(2, lbs.shape[0]):
                flg = True
                while(flg):
                    val = np.array([ (255*rng.random()) // 1 , (255*rng.random()) // 1, (255*rng.random()) // 1]).reshape(1, 3).astype(np.uint8)
                    cflg  = True
                    for v in cand_vals_check:
                        if np.all(v == val) == True:
                            cflg = False
                    if cflg == True:
                        region[labeled_image == j, :] = val
                        cand_vals_check = np.concatenate([cand_vals_check, val], axis=0)
                        flg = False
                


    cand_vals = cand_vals_check


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

        # scribbles_single += fil.skeleton_longpath
        scribbles_single += (fil.skeleton_longpath * skeleton_tmp)

        indices_scr = np.argwhere(np.isin((fil.skeleton_longpath * skeleton_tmp), 1))

        # try:

        best_idx = None
        best_val = 1e12

        print( np.argwhere(np.isin(np.all(region == cand_vals[i], axis=2), 1 )))
        print(np.argwhere(np.isin(skeleton_tmp, 1)))
        print(indices_scr.shape)

        if indices_scr.shape[0] > 0:
            for i in range(indices_scr.shape[0]):
                total_diff = 0
                for j in range(indices_scr.shape[0]):
                    total_diff += abs(indices_scr[i, :] - indices_scr[j,:]).sum()
                if best_val > total_diff:
                    best_idx = i
                    best_val = total_diff 

            # mcoords = np.median(indices_scr, axis=0)
            # mcoords = np.quantile(indices_scr, 0.5, method='lower', axis=0)
            # mx, my = int(indices_scr[:, 0].mean()), int(indices_scr[:, 1].mean())
            # print(mcoords)
            # print(indices_scr)
            # print(indices_scr[ best_idx, 0], indices_scr[ best_idx, 1])
            dot_mask[indices_scr[ best_idx, 0], indices_scr[ best_idx, 1]] = 1
        else:
            indices_scr = np.argwhere(np.isin(skeleton_tmp, 1))
            # print(indices_scr.shape)
            if indices_scr.shape[0] == 1:
                dot_mask[indices_scr[0, 0], indices_scr[0, 1]] = 1      
                scribbles_single += skeleton_tmp
            else:
                continue
            # if indices_scr.shape[0] > 1:
            #     print(indices_scr.shape[0] )
            #     raise ValueError("error!")
        # dot_mask[mx,my] = 1
            # dot_color[mx,my, :] = crop_region_col[0, :]
        # except:
        #     continue
        




    kernel = np.ones((5,5),np.uint8)

    scribbles_single[scribbles_single > 0] = 1
    cv2.imwrite(save_path + '_region'+str(hint_img_size)+'.png', region)
    cv2.imwrite(save_path + '_image'+str(hint_img_size)+'.png', img)    
    cv2.imwrite(save_path + '_scribble_mask'+str(hint_img_size)+'.png', scribbles_single.astype(np.uint8) * 255 )
    cv2.imwrite(save_path + '_scribble_col'+str(hint_img_size)+'.png',  scribble_img * scribbles_single[:, :, np.newaxis] )
    cv2.imwrite(save_path + '_flatten_img'+str(hint_img_size)+'.png',  (scribble_img  ).astype(np.uint8))
    cv2.imwrite(save_path + '_dot_mask'+str(hint_img_size)+'.png', dot_mask.astype(np.uint8) * 255 )
    cv2.imwrite(save_path + '_dot_col'+str(hint_img_size)+'.png',  (scribble_img * dot_mask[:, :, np.newaxis] ).astype(np.uint8))    


    # cv2.imwrite( '_region'+str(hint_img_size)+'.png', region)
    # cv2.imwrite( '_image'+str(hint_img_size)+'.png', img)    
    # cv2.imwrite( '_scribble_mask'+str(hint_img_size)+'.png', scribbles_single.astype(np.uint8) * 255 )
    # cv2.imwrite( '_scribble_col'+str(hint_img_size)+'.png',  scribble_img * scribbles_single[:, :, np.newaxis] )
    # cv2.imwrite( '_flatten_img'+str(hint_img_size)+'.png',  (scribble_img  ).astype(np.uint8))
    # cv2.imwrite( '_dot_mask'+str(hint_img_size)+'.png', dot_mask.astype(np.uint8) * 255 )
    # cv2.imwrite( '_dot_col'+str(hint_img_size)+'.png',  (scribble_img * dot_mask[:, :, np.newaxis] ).astype(np.uint8))    

    # exit()

if __name__=='__main__':

    # import sys
    # idx = int(sys.argv[1])
    idx = 0


    # path = "/data/datasets/ImageNet"

    # region_path = os.path.join("/scratch/madono/tog_created/segmentation_regions/imagenet", str(sys.argv[1]) )
    # out_path = os.path.join("/scratch/madono/tog_created/hint_from_regions/imagenet", str(sys.argv[1]) )
    # region_path = os.path.join("/home/madorin/datasets/labrepo/hint_eval/illust/superpixel", str(sys.argv[1]) )
    # out_path = os.path.join("/home/madorin/datasets/labrepo/hint_eval/illust/superpixel_hint", str(sys.argv[1]) )
    region_path = os.path.join("/scratch/madono/tog_created/segmentation_regions", str(sys.argv[1]) )
    out_path = os.path.join("/scratch/madono/tog_created/hint_from_regions/", str(sys.argv[1]) )       
    path = region_path
    dirs =  ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

    random.shuffle(dirs)    
    for d in reversed(dirs):
        
        if not os.path.exists(out_path): os.mkdir(out_path)
        if not os.path.exists(os.path.join(out_path, d)): os.mkdir(os.path.join(out_path, d))

        # print(os.path.join(path, d, "*.png"))
        files = glob.glob(os.path.join(path, d, "*.image.png")) #+ glob.glob(os.path.join(path, d, "*.jpg"))
        # files = glob.glob(os.path.join(path, d, "*.JPEG"))

        # print(len(files))
        # print(path, d, "*.png")
        # print(files)
        # exit()
        random.shuffle(files)
        for f in reversed(files):
            sf = f.split('/')[-1]

            
            if os.path.isfile(os.path.join(out_path, d,  sf[:-4]+'_scribble_mask'+str(hint_img_size)+'.png')):
                # print("True")
                continue

            img_path = os.path.join(region_path, d, sf[:-4]+'.png')
            region_img_path = os.path.join(region_path, d, sf[:-10]+'.png')


            # print(img_path, region_img_path)
            
            # make_scribbling(img_path, region_img_path, os.path.join(out_path, d,  sf[:-4]))
            # try:
            # # exit()
            make_scribbling(img_path, region_img_path, os.path.join(out_path, d,  sf[:-4]))
            # except:
            #     continue

            # exit()

