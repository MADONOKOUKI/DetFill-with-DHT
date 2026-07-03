# ---------------------------------------------------------------------------
# ARCHIVAL COPY (verbatim) of the original dataset-generation script used to
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
import sys
import random
from natsort import natsorted
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float



def skeleton_to_scribbling(fil):
    fil.analyze_skeletons(branch_thresh=10* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')
    return fil


def timeout_handler(num, stack):
    print("Received SIGALRM")
    raise Exception("FUBAR")


def make_scribbling(img_path, region_path_o, save_path):
    # region_path_o = '1447002.png'

    # img= cv2.imread('1447002.png')
    img= cv2.imread(img_path)
    region = cv2.imread(region_path_o)

    img = cv2.resize(img, (region.shape[0], region.shape[1]))

    bd = np.zeros(region.shape)

    bd[:-1, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 
    bd[1:, :, :] += abs(region[:-1, :, :] - region[1:, :, :]) 

    bd[:, :-1, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 
    bd[:, 1:, :] += abs(region[:, :-1, :] - region[:, 1:, :]) 

    bd = np.sum(bd, axis=2)
    bd[bd > 0] = 1
    # cv2.imwrite('debug.png', bd * 255)

    cv2.imwrite('skelton.png', skeletonize(1 - bd) * 255)


    skeleton = skeletonize(1 - bd) 

    cand_vals = np.unique(region.reshape((-1,3)), axis=0)

    scribble_img = np.zeros(region.shape)
    scribble_mask = np.zeros(region.shape)
    dot_img = np.zeros(region.shape)
    dot_mask_img = np.zeros(region.shape)
    import time
    time_sta = time.time()

    if len(cand_vals) > 2000:
        return 

    for i in tqdm(range(len(cand_vals))):
    # for i in tqdm(range(50)):
        val = cand_vals[i]
        indices = np.argwhere(np.isin(region, cand_vals[i]).all(axis=2))
        center_x, center_y = int(indices[:, 0].mean()), int(indices[:, 1].mean())
        mval = img[indices[:, 1], indices[:, 0]].mean(axis=0).astype(np.int32)

        cv2.circle(dot_img, (center_y, center_x), 4, (int(mval[0]), int(mval[1]), int(mval[2])), thickness=-1)
        cv2.circle(dot_mask_img, (center_y, center_x), 4, (255, 255, 255), thickness=-1)

        scribble_img[indices[:, 0], indices[:, 1]] = mval


        scribble_mask[indices[:, 0], indices[:, 1]] = 255


    dot_img = dot_img  * skeleton[:, :, np.newaxis]

    scribble_mask = scribble_mask * skeleton[:, :, np.newaxis]
    fil = FilFinder2D(scribble_mask[:, :, 0] , distance=250 * u.pc, mask=scribble_mask[:, :, 0]  )
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)




    # signal.signal(signal.SIGALRM, timeout_handler)
    # signal.alarm(5000)

    cv2.imwrite(save_path + '_skeleton.png', scribble_img * skeleton[:, :, np.newaxis])
    cv2.imwrite(save_path + '_skeleton_mask.png', skeleton * 255)
    cv2.imwrite(save_path + '_dot.png', dot_img )
    cv2.imwrite(save_path + '_dot_mask.png', dot_mask_img )

    flg = 0
    # fil = make_scribbling(fil)
    try:
        print("Before: %s" % time.strftime("%M:%S"))
        fil = skeleton_to_scribbling(fil)
    except Exception as ex:
        print("Gotcha!")
        flg = 1
        return

    finally:
        signal.alarm(0)
        print("After: %s" % time.strftime("%M:%S"))

    # fil = skeleton_to_scribbling(fil)
    # if flg == 0:
    scribbles_single =  fil.skeleton_longpath * 255
    # print(save_path)
    # cv2.imwrite(save_path + '_scribble_mask.png', scribbles_single )
    # cv2.imwrite(save_path + '_scribble_col.png', scribble_img * scribbles_single[:, :, np.newaxis] / 255 )


    cv2.imwrite('_scribble_mask.png', scribbles_single )
    cv2.imwrite('_scribble_col.png', scribble_img * scribbles_single[:, :, np.newaxis] / 255 )
    exit()










# cv2.imwrite('scribble.png', scribbles_single)



if __name__=='__main__':

    path = "tog_created"
    out_path = "/scratch/madono/ablation_study_analysis/segmentation_regions/"  


    dirs =  ['0000', '0001', '0002', '0003', '0004']
    # , '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']

    seg_cands =['felzenszwalb', 'slic', 'quickshift', "watershed"]

    if not os.path.exists(out_path): 
        os.mkdir(out_path)
    random.shuffle(dirs)    
    # random.shuffle(dirs)  
    # 
    color_sets = [((i+1), (j+1), (k+1)) for i in range(254) for j in range(254) for k in range(254)]
    random.shuffle(color_sets)

    for pos, s in enumerate(seg_cands):
        print(s)
        if s not in "watershed":
            continue
        for d in tqdm(reversed(dirs)):

            
            if not os.path.exists(os.path.join(out_path, d)): 
                os.mkdir(os.path.join(out_path, d))

            files = glob.glob(os.path.join(path, d, "*.png"))
            random.shuffle(files)
            for f in tqdm(reversed(files)):
                sf = f.split('/')[-1]

                # try:
                # print(f)
                # if os.path.isfile(os.path.join(out_path, d,  sf[:-4]+'.image.png')):
                #     continue
                if os.path.isfile(os.path.join(out_path, d, sf[:-4]+str(s)+'.png')):
                    continue                
                img_path = os.path.join(path, d, sf[:-4]+'.png')

                # https://scikit-image.org/docs/stable/api/skimage.io.html
                img = cv2.imread(img_path)
                if pos == 0:
                    segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=100)
                elif pos == 1:
                    segments = slic(img, n_segments=250, compactness=10, sigma=1, start_label=1)
                elif pos == 2:
                    segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
                elif pos == 3:
                    gradient = sobel(rgb2gray(img))
                    segments = watershed(gradient, markers=250, compactness=0.001)



                img = cv2.resize(img, (2048, 2048))
                segments = cv2.resize(segments, (2048, 2048), interpolation = cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_path, d, sf[:-4]+'.image.png'), img)


                h, w = segments.shape
                
                region = np.zeros((h,w,3))
                pp = np.unique(segments)
                cnt = 0
                pop_idx = 0
                while cnt < pp.shape[0]:
                    val = color_sets[pop_idx]
                    pop_idx = pop_idx + 1                
                # while cnt < pp.shape[0]:
                #     val = (random.randint(1, 256), random.randint(1, 256), random.randint(1, 256) )

                    if np.all(region == val, axis=-1).nonzero()[0].shape[0] == 0:
                        region[segments==cnt, :] = val
                        cnt = cnt + 1
                    else:
                        continue

                cv2.imwrite(os.path.join(out_path, d, sf[:-4]+str(s)+'.png'), region)
                # except:
                #     continue

            files = glob.glob(os.path.join(path, d, "*.jpg"))
            random.shuffle(files)
            for f in tqdm(reversed(files)):
                sf = f.split('/')[-1]

                # try:
                # print(f)                        
                # if os.path.isfile(os.path.join(out_path, d,  sf[:-4]+'.image.png')):
                #     continue
                if os.path.isfile(os.path.join(out_path, d, sf[:-4]+str(s)+'.png')):
                    continue
                img_path = os.path.join(path, d, sf[:-4]+'.jpg')

                # https://scikit-image.org/docs/stable/api/skimage.io.html
                # img = skimage.io.imread(img_path, pilmode='RGB')
                # img = skimage.transform.resize(img, (2048, 2048), anti_aliasing=True)
                # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = img[:,:,::-1]

                img = cv2.imread(img_path)                    

                # RGB to HSV
                # segments = slic(img, n_segments=n_segments, compactness=compactness)
                # segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=100)
                if pos == 0:
                    segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=100)
                elif pos == 1:
                    segments = slic(img, n_segments=250, compactness=10, sigma=1, start_label=1)
                elif pos == 2:
                    segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
                elif pos == 3:
                    gradient = sobel(rgb2gray(img))
                    segments = watershed(gradient, markers=250, compactness=0.001)


                img = cv2.resize(img, (2048, 2048))
                segments = cv2.resize(segments, (2048, 2048), interpolation = cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(out_path, d, sf[:-4]+'.image.png'), img )

                h, w = segments.shape
                
                region = np.zeros((h,w,3))
                pp = np.unique(segments)
                cnt = 0
                pop_idx = 0
                while cnt < pp.shape[0]:
                    val = color_sets[pop_idx]
                    pop_idx = pop_idx + 1
                    # val = (random.randint(1, 256), random.randint(1, 256), random.randint(1, 256) )
                    if np.all(region == val, axis=-1).nonzero()[0].shape[0] == 0:
                        region[segments==cnt, :] = val
                        cnt = cnt + 1
                    else:
                        continue
                    print(cnt, pp.shape[0])

                cv2.imwrite(os.path.join(out_path, d, sf[:-4]+str(s)+'.png'), region)
                # except:
                #     continue
