# Date 2019/06/07
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
from skimage import transform as trans
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

from FD_SSD.fd_ssd_runner import FdSsdRunner
from ONET.onet_runner import OnetRunner
face_detector = FdSsdRunner("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)
landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5",False)



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import badpose_detector

import cv2

def pre_process(image_path):
    #print("image_path: ", image_path)
    #exit(0)
    image = Image.open(image_path).convert("RGB")

    image_bgr = cv2.imread(image_path)
    w, h =image.size

    #new = im.convert(mode='RGB')
    # for gray scale
    img = np.array(image_bgr)
    #print("img.shape: ", img.shape)
    #exit(0)
    landmark = landmark_detector.run(image, [0, 0, w, h])



    image_size=(112,112)
    if np.size(landmark) != 10:
        landmark = landmark[:10]
    assert np.size(landmark) == 10
    landmark = np.reshape(landmark, (5, 2))
    assert len(image_size) == 2
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if image_size[1] == 112:
        src[:, 0] += 8.0

    dst = np.squeeze(np.asarray(landmark, np.float32))
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    assert len(image_size) == 2
    ret = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

    return  ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_dir',type=str)
    #parser.add_argument('--img_dir',type=str)
    parser.add_argument('--division', type=int, default=0, help='process division')
    args = parser.parse_args()



    #img_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix/'
    #save_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle/'


    #img_dir = '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict'
    #save_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict_bad_angle/'
    #img_dir = '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/val/'
    #save_dir = '/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_bad_angle/val/'

    #Guizhou train
    #img_dir = '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train'
    #save_dir = '/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_realign/train'
    #img_dir = '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/val'
    #save_dir = '/mnt/sdd/craig/face_recognition/Guizhou_NIR_dataset_kfr_0304_trainval_realign/val'

    #AsianTestDatasets
    #img_dir = '/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_train'
    #save_dir = '/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_train_realign'

    #img_dir = '/mnt/sdc/craig/Asia_face_dataset/AsianTestDatasets_badpose_val'
    #save_dir = '/mnt/sdd/craig/face_recognition/AsianTestDatasets_badpose_val_realign'

    #img_dir = '/mnt/sdc/craig/Asia_face_dataset/FR_TPE_Street_Video_Labeled_Dataset_badpose_cleaned'
    #save_dir = '/mnt/sdd/craig/face_recognition/FR_TPE_badpose_cleaned_realign'


    #insightface
    #img_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle'
    #save_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_rgb_badpose_fix_bad_angle_realign'

    #img_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict_bad_angle'
    #save_dir = '/mnt/sdd/craig/face_recognition/insightface/ms1m-retinaface-t1-112x112_glass_yuanlin_rgb_strict_bad_angle_realign'

    #img_dir = '/mnt/sdd/craig/face_recognition/glint/celebrity_112'
    #save_dir = '/mnt/sdd/craig/face_recognition/glint/celebrity_112_realign'

    #不是，用的insightface的FR模型，FD和landmarks是mtcnn.
    img_dir = '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data'
    save_dir = '/mnt/sdd/craig/face_recognition/vloggerface/20201204/align_data_realign'



    for ii, img_dir_i in enumerate(os.listdir(img_dir)):
        if (ii%1000)==0:
            print("processed ", ii, " directories...")
        dst_dir = os.path.join(save_dir,img_dir_i)
        #if os.path.isdir(dst_dir):
        #    continue
        for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
            dst_path = os.path.join(dst_dir, image_file_i)
            if os.path.isfile(dst_path):
                continue
            image_file = os.path.join(img_dir,img_dir_i, image_file_i)
            output_image = pre_process(image_file)
            if output_image is None:
                continue
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            cv2.imwrite(dst_path, output_image)
