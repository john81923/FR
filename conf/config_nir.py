import os
import torch

# Sets device for model and PyTorch tensors
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
NUM_WORKERS = 40  # for data-loading; right now, only 1 works with h5py
GRAD_CLIP = 5.  # clip gradients at an absolute value of
PRINT_FREQ = 100  # print training/validation stats  every __ batches

# Training dataset
# IMAGE_ROOT = None
# IMAGE_ROOT = '/opt/train_data/fr_data/vgg2/vgg2_112x112'
# IMAGE_ROOT = '/opt/train_data/fr_data/faces_glint/all_112'
# IMAGE_ROOT = '/opt/train_data/fr_data/faces_glint/train_kface/msra'
# IMAGE_ROOT = '/opt/train_data/fr_data/insightface/ms1m-retinaface-t1-112x112'
# IMAGE_LIST = None
# IMAGE_LIST = '/opt/train_data/fr_data/vgg2/vgg2.txt'
#IMAGE_ROOT = '/opt/train_data/fr_data/insightface/ms1m-retinaface-t1-112x112'
IMAGE_ROOT = [
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/train',
        '/mnt/sdc/craig/insightface/ms1m-retinaface-t1-112x112',
        #'/mnt/sdc/craig/CASIA_NIR/NIR_kfr_0304',
        ]

MODEL_STRUCTURE = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/'
IMAGE_LIST = None
IMAGE_PER_LABEL = 5

# Validation dataset
PAIR_FOLDER_NAME = "/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pair_gen"
VALID_LIST = [
        #'/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_val',
        '/mnt/sdc/craig/Guizhou_NIR_dataset_kfr_0304_trainval/val',
        ]

# Logging/checkpoint parameters
LOG_DIR = './checkpoint'
TRAIN_LOG_FILE = 'train.log'
EVAL_LOG_FILE = 'train.log'
