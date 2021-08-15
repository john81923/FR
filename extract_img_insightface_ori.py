import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
#from pytorchtools import EarlyStopping
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import random
import cv2
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False
from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import onnx
import onnxruntime
import numpy as np
torch.backends.cudnn.deterministic = True
import matplotlib.pyplot as plt
import shutil

sys.path.insert(0, os.path.abspath("FD_SSD"))
from SSD_infer import ssd_detector

import onet_infer 
import badpose_detector
from skimage import transform as trans
import cv2

def change_color_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def change_color_rgb_r(image):
    image_red = np.zeros((image.shape[0],image.shape[1],3))
    ind = 2
    image_red[:,:,0],image_red[:,:,1],image_red[:,:,2]=image[:,:,ind],image[:,:,ind],image[:,:,ind]
    return image_red
    
def change_color_yuv_y(image):    
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y = np.zeros((image.shape[0],image.shape[1],3))
    ind = 0
    image_y[:,:,0],image_y[:,:,1],image_y[:,:,2]=img_yuv[:,:,ind],img_yuv[:,:,ind],img_yuv[:,:,ind]
    return image_y


def pre_process(image_path):
    dets = face_detector.run(image_path)
    if len(dets)==0:
        return None 
    bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])
    image = Image.open(image_file).convert("RGB")
    w, h =image.size
    landmark = landmark_detector.run(image, [0, 0, w, h])
    print(image_file_i,landmark)
    Bad = badpose_detector.checkBadPose(landmark)
    if Bad:
        return None 

    img = cv2.imread(image_file)
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

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--img_dir',type=str)
    args = parser.parse_args()

    img_dir=args.img_dir
    ssd_face_model_path='FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5'
    anchor_face_model_path="FD_SSD/models/anchor_face_ssd7s_cfg2.npy"
    landmark_model_path="model-code/resnet34/evaluate/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5"

    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.isfile(ssd_face_model_path):
        raise ValueError('not find',ssd_face_model_path)
    if not os.path.isfile(anchor_face_model_path): 
        raise ValueError('not find',anchor_face_model_path)   
    face_detector = ssd_detector(ssd_face_model_path, anchor_face_model_path, score_thres=0.5, only_max=False)
    landmark_detector = onet_infer.onet_detector(landmark_model_path)
    
    img_dir = '/mnt/sdc/fr_train_data/insightface/ms1m-retinaface-t1-112x112'
    save_dir = '/mnt/sdc/fr_train_data_R/insightface/ms1m-retinaface-t1-112x112' 
    for ii, img_dir_i in enumerate(os.listdir(img_dir)):
        if ii%5==2:
            dst_dir = os.path.join(save_dir,img_dir_i)
            for image_file_i in os.listdir(os.path.join(img_dir,img_dir_i)):
                dst_path = os.path.join(dst_dir, image_file_i)
                if os.path.isfile(dst_path):
                    continue
                image_file = os.path.join(img_dir,img_dir_i, image_file_i)
                output_image = pre_process(image_file)
                if output_image is None:
                    continue
                output_image_r = change_color_rgb_r(output_image)
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir)
                
                cv2.imwrite(dst_path, output_image_r)

        