# Date 2019/03/26
# Author: Craig Hsin
# Purpose: remove to gray scale, only FD + landmark (warp) and badpose check

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys



from torchvision import transforms
from PIL import Image

import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
import cv2

cv2.setNumThreads(16)

import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
torch.backends.cudnn.deterministic = True
import matplotlib.pyplot as plt
import shutil
#import get_LM_angle_ratio

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#import badpose_detector

import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_dir',type=str)
    #parser.add_argument('--img_dir',type=str)
    parser.add_argument('--division', type=int, default=0, help='process division')
    args = parser.parse_args()


    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_align112x112/Glint360K/celeb_deepglint/glint360k'
    #save_dir = '/mnt/sdd/craig/face_recognition/glint/FR_glint_bad_angle'

    #img_dir = '/mnt/sdd/craig/face_recognition/glint/110k_enhancement'
    #save_dir = '/mnt/sdd/craig/face_recognition/glint/110k_enhancement_bad_angle'

    img_dir = '/mnt/sdd/craig/face_recognition/glint/FR_glint_bad_angle'
    save_dir = '/mnt/sdd/craig/face_recognition/glint/FR_glint_bad_angle_visualization'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("img_dir: ", img_dir)
    for ii, img_dir_i in enumerate(os.listdir(img_dir)):
        #if ii%2==args.division:
        if ii > 20:
            break
        #if img_dir_i=='7172' && 'vgg2_112x112_split'in img_dir:
        #    continue
        #print(img_dir_i)
        if (ii%1000)==0:
            print("processed ", ii, " directories...")
        dst_dir = os.path.join(save_dir,img_dir_i)
        if os.path.isdir(dst_dir):
            continue
        for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
            dst_path = os.path.join(dst_dir, image_file_i)
            if os.path.isfile(dst_path):
                continue
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            #output_image_r = change_color_rgb_r(output_image)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

            shutil.copyfile(image_file, dst_path)
