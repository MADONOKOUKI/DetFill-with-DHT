import numpy as np
import os



def init_load_data(domain):

    #Training data

    #Training images list
    with open(os.path.join('configs', str(domain), 'train_paper.txt'), "r") as a:
        list_train_imgs = a.readlines()

    # with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/valid_paper.txt', "r") as a:
    with open(os.path.join('configs', str(domain), 'valid_paper.txt'), "r") as a:

        list_valid_imgs = a.readlines()

    return list_train_imgs, list_valid_imgs

