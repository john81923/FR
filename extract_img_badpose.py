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

import tensorflow as tf
print(tf.__version__)
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

from FD_SSD.fd_ssd_runner import FdSsdRunner
from ONET.onet_runner import OnetRunner



face_detector = FdSsdRunner("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)

#landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")
landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5",False)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import badpose_detector

import cv2

def pre_process(image_path):
    image_rgb = Image.open(image_path).convert("RGB")

    w, h =image_rgb.size

    landmark = landmark_detector.run(image_rgb, [0, 0, w, h])
    #print(image_file_i,landmark)
    #Bad = badpose_detector.checkBadPose_strict(landmark)
    Bad = badpose_detector.checkBadPose_strict_fix(landmark)
    #Bad = badpose_detector.checkBadPose(landmark)
    if Bad:
        return None
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_dir',type=str)
    #parser.add_argument('--img_dir',type=str)
    parser.add_argument('--division', type=int, default=0, help='process division')
    args = parser.parse_args()


    #img_dir = '/mnt/atdata/FacialRecognition/FR_insightface_withNormalglasses/ms1m-retinaface-t1-112x112'
    #save_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict'

    #img_dir = '/mnt/atdata/FacialRecognition/FR_insightface_lighting/'
    #save_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_lighting_andy_rgb_strict'
    #img_dir = '/mnt/sdc/craig/vgg2_112x112_split/train'
    #save_dir = '/mnt/sdc/craig/vgg2_112x112_split_rm_badpose_strict/train'
    #img_dir = '/mnt/sdc/fr_train_data/insightface/ms1m-retinaface-t1-112x112'
    #save_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_rgb_strict'

    #img_dir = '/mnt/sdc/fr_train_data/insightface/ms1m-retinaface-t1-112x112'
    #save_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix'



    #img_dir = '/mnt/sdc/fr_train_data/insightface/ms1m-retinaface-t1-112x112_glass_mixed'
    #save_dir = '/mnt/sdd/craig/insightface/ms1m-retinaface-t1-112x112_glass_mixed_rgb_badpose_fix'


    ##img_dir = '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign'
    #save_dir = '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign_bad_pose'
    img_dir = '/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle'
    save_dir = '/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign_bad_angle_bad_pose'


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("img_dir: ", img_dir)
    for ii, img_dir_i in enumerate(os.listdir(img_dir)):
        #if ii%2==args.division:
        #if ii < 4493:
        #    continue
        #if img_dir_i=='7172' && 'vgg2_112x112_split'in img_dir:
        #    continue
        #print(img_dir_i)
        if (ii%1000)==0:
            print("processed ", ii, " directories...")
        dst_dir = os.path.join(save_dir,img_dir_i)
        for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
            dst_path = os.path.join(dst_dir, image_file_i)
            if os.path.isfile(dst_path):
                continue
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            output_image = pre_process(image_file)
            if output_image is None:
                continue
            #output_image_r = change_color_rgb_r(output_image)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

            shutil.copyfile(image_file, dst_path)
