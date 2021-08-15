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

#import tensorflow as tf
#print(tf.__version__)

'''
def tf_limit_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

tf_limit_gpu_memory()
from torchvision import transforms
'''
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
import get_LM_angle_ratio
'''
from FD_SSD.SSD_infer import ssd_detector
from ONET.onet_infer import onet_detector
face_detector = ssd_detector("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)

landmark_detector = onet_detector("./ONET/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")
'''

#from FD_SSD.fd_ssd_runner import FdSsdRunner
#from ONET.onet_runner import OnetRunner
#face_detector = FdSsdRunner("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
#                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)
#landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5",False)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#import badpose_detector

import cv2

def pre_process(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h =image.size

    landmark = landmark_detector.run(image, [0, 0, w, h])
    #print(image_file_i,landmark)
    #Bad = badpose_detector.checkBadPose_strict(landmark)
    #Bad = badpose_detector.checkBadPose_strict_fix(landmark)
    ratio = get_LM_angle_ratio.lmk_to_angle_ratio(landmark)

    #Bad = badpose_detector.checkBadPose(landmark)
    if ratio>1.5:
        return None
    else:
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_dir',type=str)
    #parser.add_argument('--img_dir',type=str)
    parser.add_argument('--division', type=int, default=0, help='process division')
    args = parser.parse_args()


    #mi-8 - 0608
    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/mi8_liu_112x112_20200608_nir150_new_copy_bad_angle_nir'

    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/mi8_mdong_112x112_20200608_nir150_new_copy_bad_angle_nir'

    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_liu_112x112_20200608_nir150_new_copy_nir'


    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0608/mi8_mdong_112x112_20200608_nir150_new_copy_nir'

    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_nir'

    #img_dir = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0611/20200611_nir150_new_copy_bad_angle_nir'

    #A
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_A_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_A_nir'

    #B
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_B_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_B_nir'

    #C
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_C_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_C_nir'

    #D
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_D_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_D_nir'

    #E
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_E_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_E_nir'

    #F
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_liu_112x112_20200709_F_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_liu_112x112_20200709_F_nir'


    #mdong_A
    #img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_mdong_112x112_20200803_A_nir150'
    #save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_rgb'
    #save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_A_nir'


    #mdong_B
    img_dir = '/mnt/atdata/FR_Asian_Raw_Data_clean/mi8_mdong_112x112_20200803_B_nir150'
    save_dir_rgb = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_rgb'
    save_dir_nir = '/mnt/sdd/craig/face_recognition/mi8/0904/mi8_mdong_112x112_20200803_B_nir'

    if not os.path.isdir(save_dir_rgb):
        os.makedirs(save_dir_rgb)

    if not os.path.isdir(save_dir_nir):
        os.makedirs(save_dir_nir)
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
        dst_dir_rgb = os.path.join(save_dir_rgb,img_dir_i)
        dst_dir_nir = os.path.join(save_dir_nir,img_dir_i)
        #if os.path.isdir(dst_dir_rgb):
        #    continue
        #if os.path.isdir(dst_dir_nir):
        #    continue

        if not os.path.isdir(dst_dir_rgb):
            os.makedirs(dst_dir_rgb)
        if not os.path.isdir(dst_dir_nir):
            os.makedirs(dst_dir_nir)

        for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
            dst_path_rgb = os.path.join(dst_dir_rgb, image_file_i)
            dst_path_nir = os.path.join(dst_dir_nir, image_file_i)
            if os.path.isfile(dst_path_rgb): #folder
                continue
            if os.path.isfile(dst_path_nir): #folder
                continue
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            if 'RGB' in image_file_i:
                shutil.copyfile(image_file, dst_path_rgb)
            elif 'NIR' in image_file_i:
                shutil.copyfile(image_file, dst_path_nir)
            else:
                print("RGN and NIR not in filename:")
                print(image_file_i)
                assert 0


            '''
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            output_image = pre_process(image_file)
            if output_image is None:
                continue
            #output_image_r = change_color_rgb_r(output_image)
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)

            '''
