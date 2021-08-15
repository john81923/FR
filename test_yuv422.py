import os
#os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
print(tf.__version__)
import cv2
from PIL import Image
import numpy as np
from skimage import transform as trans
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
#face_detector = ssd_detector("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
#                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)

#landmark_detector = onet_detector("./ONET/models/onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5")

from FD_SSD.fd_ssd_runner import FdSsdRunner
from ONET.onet_runner import OnetRunner
face_detector = FdSsdRunner("./FD_SSD/models/ssd7_0.8_epoch-97_loss-0.1407_val_loss-0.0825.h5",
                            "./FD_SSD/models/anchor_face_ssd7s_cfg2.npy", score_thres=0.5, only_max=False)
landmark_detector = OnetRunner("./ONET/models/rgb2gray_Laplacian_onet_base_onet-combined-246_loss-0.7126_val_loss-0.4551.hdf5",False)

import get_LM_angle_ratio


import os
import sys
import numpy as np

import torch
from torch import nn

from PIL import Image
import torchvision

import torch
from torch import nn
from collections import OrderedDict
import argparse

import badpose_detector

def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--inp-size', type=int, default=112, help='input image size')
    parser.add_argument('--block-layout', type=str, default='8 28 6', help='feature block layout')
    #parser.add_argument('--block-size', type=str, default='32 384 1152 2144', help='feature block size')
    parser.add_argument('--block-size', type=str, default='32 32 64 128 256', help='feature block size') # original size
    #parser.add_argument('--block-size', type=str, default='48 48 72 160 288', help='feature block size')
    parser.add_argument('--se-ratio', type=int, default=0, help='SE reduction ratio')
    parser.add_argument('--head', type=str, default='fc', help='head fc or varg')
    #parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--emb-size', type=int, default=256, help='embedding length')
    #parser.add_argument('--do-rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--do-rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
    parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
    parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
    #parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
    parser.add_argument('--focal-loss', type=bool, default=True, help='focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    #parser.add_argument('--checkpoint', type=str, default='None', help='checkpoint')
    parser.add_argument('--checkpoint', type=str, default='/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/checkpoint/kface_fr_gray-r50_nb-E256-av0.9980_0.9730_0.9786_0.9967_0.9260_0.9154.pth', help='checkpoint')
    parser.add_argument('--batch-size', type=int, default=512, help='batch size in each context')
    parser.add_argument('--warmup', type=int, default=0, help='warmup training epochs without validation')
    parser.add_argument('--cooldown', type=int, default=0, help='keep training with repeating the last few epochs')
    parser.add_argument('--max-cool', type=int, default=5, help='Maxium cooling down without improvment')
    parser.add_argument('--end-epoch', type=int, default=100, help='training epoch size.')
    parser.add_argument('--gpus', type=str, default='0', help='running on GPUs ID')
    #parser.add_argument('--log-dir', type=str, default=None, help='Checkpoint/log root directory')
    parser.add_argument('--log-dir', type=str, default='/mnt/storage1/craig/kneron_fr', help='Checkpoint/log root directory')
    #parser.add_argument('-gr', '--grayscale', help='Use grayscale input.', action='store_true')
    parser.add_argument('-nf', '--no-flip', help='No face flip in evaluation.', action='store_false')
    parser.add_argument('-pre', '--pre-norm', help='Preprocessing normalization id.', default='CV-kneron')


    #https://stackoverflow.com/questions/12818146/python-argparse-ignore-unrecognised-arguments
    args_new, unknown = parser.parse_known_args() #<---- change to this
    #args_new = parser.parse_args()
    args_new.block_layout = [int(n) for n in args_new.block_layout.split()]
    args_new.block_size = [int(n) for n in args_new.block_size.split()]
    args_new.gpus = [int(n) for n in args_new.gpus.split()]



    return args_new

    #return args_new

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(channel // reduction),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x).view(-1, self.channel)
        y = self.fc(y).view(-1, self.channel, 1, 1)

        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se_ratio=0):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=False)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU(inplanes)
        self.prelu_out = nn.PReLU(planes)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        if se_ratio > 0:
            self.se = SEBlock(planes, se_ratio)
        else:
            self.se = None
        #self.bnbp = nn.BatchNorm2d(planes, affine=False)


    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        #out = self.bnbp(out)
        out = self.prelu_out(out)

        return out


def input_block(inp_size, inplanes):
    if inp_size == 112:
        return nn.Sequential(
                nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.PReLU(inplanes),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
    elif inp_size == 224:
        return nn.Sequential(
                nn.Conv2d(3, inplanes//2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(inplanes//2),
                nn.PReLU(inplanes//2),
                nn.Conv2d(inplanes//2, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(inplanes),
                nn.PReLU(inplanes),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )


def head_block(inplanes, feat_size, do_rate, emb_size, head='fc', varg_last=1024):
    # use grouped head, otherwise go fc
    if head=='varg':
        return nn.Sequential(
                nn.Conv2d(inplanes, varg_last, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(varg_last),
                nn.PReLU(varg_last),
                nn.Conv2d(varg_last, varg_last, kernel_size=feat_size, stride=1, padding=0, bias=False, groups=varg_last//8),
                nn.BatchNorm2d(varg_last),
                nn.Conv2d(varg_last, varg_last//2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(varg_last//2),
                nn.PReLU(varg_last//2),
                nn.Flatten(),
                nn.Linear(varg_last//2, emb_size),
                nn.BatchNorm1d(emb_size)
                )
    else:
        return nn.Sequential(
                #nn.BatchNorm2d(inplanes),
                nn.Dropout(p=do_rate),
                nn.Flatten(),
                nn.Linear(inplanes*feat_size*feat_size, emb_size),
                nn.BatchNorm1d(emb_size)
                )


class Head(nn.Module):
    def __init__(self, feat_size=7,
            inplanes=256, emb_size=256, do_rate=0.0, head='fc', varg_last=1024):
        super(Head, self).__init__()

        if head=='varg':
            print('varg head')
            self.head = nn.Sequential(
                    nn.Conv2d(inplanes, varg_last, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(varg_last),
                    nn.PReLU(varg_last),
                    nn.Conv2d(varg_last, varg_last, kernel_size=feat_size, stride=1, padding=0, bias=False, groups=varg_last//8),
                    nn.BatchNorm2d(varg_last),
                    nn.Conv2d(varg_last, varg_last//2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(varg_last//2),
                    nn.PReLU(varg_last//2),
                    nn.Flatten(),
                    nn.Linear(varg_last//2, emb_size),
                    nn.BatchNorm1d(emb_size)
                    )

        else:
            self.head = nn.Sequential(
                        nn.BatchNorm2d(inplanes),
                        nn.Dropout(p=do_rate),
                        nn.Flatten(),
                        #nn.Linear(inplanes*feat_size*feat_size, 512),
                        #nn.ReLU(inplace=True),
                        #nn.Linear(512, emb_size),
                        nn.Linear(inplanes*feat_size*feat_size, emb_size),
                        nn.BatchNorm1d(emb_size)
                    )
        self.head.apply(init_weight)

    def forward(self, x):
        x = self.head(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layout, block_size, se_ratio=0,
            inp_size=112, emb_size=512, do_rate=0.4,
            head='fc', varg_last=1024):
        super(ResNet, self).__init__()

        self.input = input_block(inp_size, block_size[0])

        self.layer1 = self._make_layer(block, block_size[0], block_size[1], layout[0], stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, block_size[1], block_size[2], layout[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, block_size[2], block_size[3], layout[2], stride=2, se_ratio=se_ratio)
        self.layer4 = self._make_layer(block, block_size[3], block_size[4], layout[3], stride=2, se_ratio=se_ratio)

        self.head = head_block(block_size[4], 7, do_rate, emb_size, head)
    '''
    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
    '''
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, se_ratio=0):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes,  planes, stride, downsample, se_ratio=se_ratio))
        for i in range(1, blocks):
            layers.append(block(planes, planes, se_ratio=se_ratio))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.input(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)

        return x


DEFAULT_LAYER = 50
BLOCK_LAYOUT = {
    18:  [2, 2,  2, 2],
    34:  [3, 4,  6, 3],
    50:  [3, 4, 14, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
    }

DEFAULT_SIZE = 'm'
BLOCK_SIZE = {
    's' : [16, 16, 32, 64, 128],
    'se': [32, 32, 64, 64, 128],
    'sf': [32, 32, 64, 128, 128],
    'm' : [32, 32, 64, 128, 256],
    'l' : [64, 64, 128, 256, 512],
    'l1': [64, 64, 128, 512, 512],
    'l2': [64, 64, 256, 256, 512],
    'l3': [64, 64, 256, 512, 512],
    'l7': [64, 128, 256, 512, 512],
    'x_8': [64, 128, 256, 512, 1024],
    'x'  : [128, 128, 256, 512, 1024],
    }


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.affine:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def build_resnet(block=IRBlock, se_ratio=0, inp_size=112,
            block_layout=BLOCK_LAYOUT[DEFAULT_LAYER], block_size=BLOCK_SIZE[DEFAULT_SIZE],
            emb_size=512, do_rate=0.4, head='fc'):

    model = ResNet(block, block_layout, block_size, se_ratio=se_ratio,
            inp_size=inp_size, emb_size=emb_size, do_rate=do_rate, head=head)

    model.apply(init_weight)

    return model



def resnet50(args=None, **kwargs):
    if args:
        model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[50], block_size=args.block_size,
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)
    else:
        model = build_resnet(block_layout=BLOCK_LAYOUT[50], se_ratio=16)

    return model


class Merged(nn.Module):
    def __init__(self, original_model, nir_head):
        super(Merged, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:])
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.rgb_head = nn.Sequential(*list(original_model.children())[-1])
        self.nir_head = nn.Sequential(*list(nir_head.children()))
        #print(self.features)

    def forward(self, x):
        features = self.backbone(x)
        rgb_head_output = self.rgb_head(features)
        nir_head_output = self.nir_head(features)
        return rgb_head_output, nir_head_output

args_new = parse_args()
res50_rgb_model = resnet50(args_new) #Res50_RGB(emb_size = 256)
target_model = res50_rgb_model #nn.Sequential(merged_nir_model.backbone, merged_nir_model.nir_head) #concatenate
#target_model.load_state_dict(torch.load('/mnt/models/FR_models/FR_RGB_Craig/0620_versions/resnet_mi8_v12_4_state_dict.tar'))
target_model.load_state_dict(torch.load('/mnt/models/FR_models/FR_RGB_Craig/0930_versions/resnet_mi8_v12_16_3_state_dict.tar'))

target_model.eval()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
target_model.to(device)

def create_folder (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def display(img, dets=None, landmarks_list=None, save_path='0911_display.png'):
    """
    show detection result.
    """
    if isinstance(img, str):
        img = Image.open(img)

    img = np.array(img)
    img = np.squeeze(img)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

    if dets:
        for i, box in enumerate(dets):
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', fill=False)
            ax.add_patch(rect)
            ax.text(box[0], box[1], "%.2f"%box[4], bbox=dict(facecolor='red', alpha=0.5))

    if landmarks_list:
        for i in range(len(landmarks_list)):
            for j in range(0, 10, 2):
                circle = patches.Circle((int(landmarks_list[i][j + 0]), int(landmarks_list[i][j + 1])), max(1,img.shape[0]/200), color='g')
                ax.add_patch(circle)
            ax.text(landmarks_list[i][8], landmarks_list[i][9], "%.2f"%landmarks_list[i][-1], bbox=dict(facecolor='green', alpha=0.5))




    if save_path:
        fig.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    return


def pre_process(image_path):
    dets = face_detector.run(image_path)
    ret = None
    #test_path = "/home/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/00001_0001.png"
    test_path = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1208_issue/Register_ID/NIR_2020-12-08-04-10-50-783.png"

    if len(dets)>0:
        #for i in range(len(dets)): # only 1 face
        bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])

        image = Image.open(image_path).convert("RGB")
        img = np.array(image)
        #landmark = landmark_detector.run(img, bbox)
        landmark = landmark_detector.run(image_path, dets[0])
        #print("landmark: ", landmark)

        ''' ========== for testing start ===============================
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')

        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', fill=False)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], "1.0", bbox=dict(facecolor='red', alpha=0.5))


        for j in range(0, 10, 2):
            circle = patches.Circle((int(landmark[j + 0]), int(landmark[j + 1])), max(1,img.shape[0]/200), color='g')
            ax.add_patch(circle)
        ax.text(landmark[8], landmark[9], "%.2f"%landmark[-2], bbox=dict(facecolor='green', alpha=0.5))

        display(test_path, dets, [landmark])
        #plt.plot(img)
        #plt.show()
        fig.savefig('1208_NIR_2020-12-08-04-10-50-783.png')
        plt.close()
        exit(0)

        ========== for testing end  =============================== '''
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
        #print("M: ", M)
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        assert len(image_size) == 2
        ret = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        ret_size = ret.shape
        #print("ret_size: ", ret_size)
        #exit(0)
        #ret = np.expand_dims(ret, axis=-1)
        return len(dets), ret
    else:
        return len(dets), ret



def embedding(image_path):
    dets = face_detector.run(image_path)
    normed_emb_ori = None
    test_path = "/home/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/00001_0001.png"
    #print("dets: ")
    #print(dets)
    if len(dets)>0:
        #for i in range(len(dets)): # only 1 face
        bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])

        image = Image.open(image_path).convert("RGB")
        img = np.array(image)
        #landmark = landmark_detector.run(img, bbox)
        landmark = landmark_detector.run(image_path, dets[0])

        bad_angle = False
        ratio = get_LM_angle_ratio.lmk_to_angle_ratio(landmark)
        if ratio>1.5:
            bad_angle = True

        bad_pose= False
        Bad = badpose_detector.checkBadPose_strict_fix(landmark)
        #Bad = badpose_detector.checkBadPose(landmark)
        if Bad:
            bad_pose = True


        image_size=(112,112)
        if np.size(landmark) != 10:
            landmark = landmark[:10]
        assert np.size(landmark) == 10
        landmark = np.reshape(landmark, (5, 2))
        #print("landmark: ")
        #print(landmark)
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
        #print("M: ", M)
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        assert len(image_size) == 2
        resized_img = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        '''
        resized_img = cv2.cvtColor(resized_img,cv2.COLOR_RGB2BGR)
        #cv2.imwrite("testing/1208_NIR_2020-12-08-04-10-50-783.jpg", resized_img)
        cv2.imwrite("testing/NIR_2020-12-08-04-12-45-717.jpg", resized_img)
        #image_path2 = 'testing/face_occulusion_out.jpeg'
        #dets = face_detector.run(image_path2)
        #print(dets)
        exit(0)
        '''
        #ret_size = ret.shape
        #print("ret_size: ", ret_size) #112,112,3
        #exit(0)
        #ret = np.expand_dims(ret, axis=-1)

        resized_img_np = np.asarray(resized_img, dtype='f')

        ori_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),  # 4, 112, 112
            torchvision.transforms.Normalize(std=[256., 256., 256.], mean=[128, 128, 128]),
        ])

        face_ori = ori_transform(resized_img_np)
        # (4) run model (single model)
        input_imgs_tensor = torch.zeros([1, 3, 112, 112], dtype=torch.float)
        input_imgs_tensor[0] = face_ori
        input_imgs_tensor = input_imgs_tensor.to(device)  ###

        with torch.no_grad():
            ori_output = target_model(input_imgs_tensor)
            #print("ori_output: ")
            #print(ori_output)
            emb_ori = ori_output[0].cpu().numpy()
            normed_emb_ori = emb_ori / np.linalg.norm(emb_ori)
            normed_emb_ori_size = normed_emb_ori.shape
        return normed_emb_ori, bad_angle, bad_pose
        #print("normed_emb_ori_size: ", normed_emb_ori_size)
        #exit(0)






def test_yuv422(rgb_image_path, nir_image_path, index = ""):
    dets = face_detector.run(rgb_image_path)
    dets_nir = face_detector.run(nir_image_path)
    ret = None

    if len(dets)>0:
        #for i in range(len(dets)): # only 1 face
        bbox =(dets[0][0], dets[0][1], dets[0][2], dets[0][3])

        image = Image.open(rgb_image_path).convert("RGB")
        img = np.array(image)
        image_nir = Image.open(nir_image_path).convert("RGB")
        #landmark = landmark_detector.run(img, bbox)
        landmark = landmark_detector.run(rgb_image_path, dets[0])
        landmark_nir = landmark_detector.run(nir_image_path, dets_nir[0])
        #print("landmark: ", landmark)

        ''' ========== for testing start ===============================
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')

        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', fill=False)
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], "1.0", bbox=dict(facecolor='red', alpha=0.5))


        for j in range(0, 10, 2):
            circle = patches.Circle((int(landmark[j + 0]), int(landmark[j + 1])), max(1,img.shape[0]/200), color='g')
            ax.add_patch(circle)
        ax.text(landmark[8], landmark[9], "%.2f"%landmark[-2], bbox=dict(facecolor='green', alpha=0.5))

        display(test_path, dets, [landmark])
        #plt.plot(img)
        #plt.show()
        fig.savefig('1208_NIR_2020-12-08-04-10-50-783.png')
        plt.close()
        exit(0)

        ========== for testing end  =============================== '''
        image_size=(112,112)
        #image_size=(140,112)
        if np.size(landmark) != 10:
            landmark = landmark[:10]
            landmark_nir = landmark_nir[:10]
        assert np.size(landmark) == 10
        landmark = np.reshape(landmark, (5, 2))
        landmark_nir = np.reshape(landmark_nir, (5, 2))
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112 or image_size[1] == 140:
            src[:, 0] += 8.0

        dst = np.squeeze(np.asarray(landmark, np.float32))
        dst_nir = np.squeeze(np.asarray(landmark_nir, np.float32))
        tform = trans.SimilarityTransform()
        tform_nir = trans.SimilarityTransform()
        tform.estimate(dst, src)
        tform_nir.estimate(dst_nir, src)
        M = tform.params[0:2, :]
        M_nir = tform_nir.params[0:2, :]
        #print("M: ", M)
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

        assert len(image_size) == 2

        #img = np.array(image)
        img_bgr = cv2.imread(rgb_image_path)
        print("img_bgr.shape:", img_bgr.shape)
        #https://github.com/opencv/opencv/issues/14163
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        # Create YUYV from YUV
        y0 = np.expand_dims(img_yuv[...,0][::,::2], axis=2)
        u = np.expand_dims(img_yuv[...,1][::,::2], axis=2)
        y1 = np.expand_dims(img_yuv[...,0][::,1::2], axis=2)
        v = np.expand_dims(img_yuv[...,2][::,::2], axis=2)
        #img_yuyv = np.concatenate((y0, u, y1, v), axis=2)
        print("y0.shape: ", y0.shape)
        y =  np.expand_dims(img_yuv[...,0], axis=2)
        #aaa = (y0, u, v, y1, u, v)
        #print("aaa.shape: ", aaa.shape)
        img_yuyv = np.concatenate((y0, u, v, y1, u, v), axis=2)
        img_yyy = np.concatenate((y, y, y), axis=2)


        img_nir = cv2.imread(nir_image_path)
        print("img_nir.shape:", img_nir.shape)

        img_yuv_nir = cv2.cvtColor(img_nir, cv2.COLOR_BGR2YUV)

        #y0_nir = np.expand_dims(img_yuv_nir[...,0][::,::2], axis=2)
        #y1_nir = np.expand_dims(img_yuv_nir[...,0][::,1::2], axis=2)
        #exit(0)
        img_yuyv_cvt = img_yuyv.reshape(img_yuyv.shape[0], img_yuyv.shape[1] * 2, 3)

        print("img_yuyv_cvt.shape:", img_yuyv_cvt.shape)
        # Convert back to BGR results in more saturated image.

        #exit(0)
        #img_bgr_restored = cv2.cvtColor(resized_img, cv2.COLOR_YUV2BGR_YUYV)

        resized_img = cv2.warpAffine(img_yuyv_cvt, M, (image_size[1], image_size[0]), borderValue=0.0)
        resized_img_nir = cv2.warpAffine(img_yuv_nir, M_nir, (image_size[1], image_size[0]), borderValue=0.0)
        resized_img_yyy = cv2.warpAffine(img_yyy, M, (image_size[1], image_size[0]), borderValue=0.0)
        #resized_img = cv2.warpAffine(img2, M, (image_size[1], image_size[0]), borderValue=0.0)
        #ret_size = ret.shape
        img_bgr_restored = cv2.cvtColor(resized_img, cv2.COLOR_YUV2BGR)

        #print(resized_img_nir[:,:,0].shape)
        #print(resized_img[:,:,1:].shape)
        y_nir = np.expand_dims(resized_img_nir[:,:,0], axis=2)
        uv_rgb =  resized_img[:,:,1:]
        img_bgr_restored_nir_add = np.concatenate((y_nir, uv_rgb), axis=2)
        print("img_bgr_restored_nir_add.shape:", img_bgr_restored_nir_add.shape)
        img_bgr_restored_nir = cv2.cvtColor(resized_img_nir, cv2.COLOR_YUV2BGR)
        img_bgr_restored_combined = cv2.cvtColor(img_bgr_restored_nir_add, cv2.COLOR_YUV2BGR)
        #img_bgr_restored_yyy= cv2.cvtColor(resized_img_yyy, cv2.COLOR_YUV2BGR)
        #resized_img = cv2.cvtColor(resized_img,cv2.COLOR_RGB2BGR)
        #cv2.imwrite("testing/1208_NIR_2020-12-08-04-10-50-783.jpg", resized_img)
        #cv2.imwrite("testing/0120_yuv/result/output_0121_v3.jpg", img_bgr_restored)
        #cv2.imwrite("testing/0120_yuv/result/output_0123_size_140_112_"+str(index)+".png", img_bgr_restored_nir)
        cv2.imwrite("testing/0120_yuv/result/output_0123_size_112_112_"+str(index)+".png", img_bgr_restored_nir)
        #cv2.imwrite("testing/0120_yuv/result/output_0121_v5.jpg", img_bgr_restored_combined)
        #cv2.imwrite("testing/0120_yuv/result/mix_occ_size_140_112_with_uv_605_"+str(index)+".png", img_bgr_restored_combined)
        #cv2.imwrite("testing/0120_yuv/result/yyy_425.png", resized_img_yyy)
        #image_path2 = 'testing/face_occulusion_out.jpeg'
        #dets = face_detector.run(image_path2)
        #print(dets)
        #exit(0)

        #print("ret_size: ", ret_size)
        #exit(0)
        #ret = np.expand_dims(ret, axis=-1)
        return len(dets), ret
    else:
        return len(dets), ret

if __name__ == "__main__":


    rgb_106 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/106.png'
    rgb_test = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/test.png'

    rgb_605 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/RGB_2020-11-28-03-17-27-605.png'
    nir_605 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/NIR_2020-11-28-03-17-27-605.png'
    mix_605 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_605.png'
    yyy_605 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/yyy_605.png'
    mix_605_uv_from_425 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_605_uv_from_425.png'
    mix_605_uv_from_106 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_605_uv_from_106.png'

    rgb_425 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/RGB_2020-11-29-11-20-59-425.png'
    nir_425 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/NIR_2020-11-29-11-20-59-425.png'
    mix_425 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_425.png'
    yyy_425 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/yyy_425.png'
    mix_425_uv_from_605 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_425_uv_from_605.png'
    mix_425_uv_from_106 = '/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/mix_425_uv_from_106.png'
    #test_yuv422(rgb_test,rgb_test)
    #test_yuv422(rgb_605,nir_425)

    image_list = []
    for ext in ('*.bmp', '*.png', '*.jpg'):
        #image_list.extend(sorted(glob.glob(os.path.join('/mnt/sdd/craig/home_craig/hw_runner/kneron_hw_models/test/test_seg_0121', ext))))
        image_list.extend(sorted(glob.glob(os.path.join('/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/0120_yuv/occ2', ext))))
        print("image_list: ", image_list)

    for image_idx, image_path in enumerate(image_list):
        test_yuv422(rgb_605,image_path, image_idx)

    exit(0)
    print ("img1: ", rgb_605)
    embedding1, _, _= embedding(rgb_605)

    print ("img2: ", mix_425_uv_from_106)
    embedding2, _, _= embedding(mix_425_uv_from_106)
    diff = np.subtract(embedding1, embedding2)

    dist = np.sum(np.square(diff))
    print ("dist: ", dist)


    print ("img1: ", rgb_425)
    embedding1, _, _= embedding(rgb_425)

    print ("img2: ", mix_425_uv_from_106)
    embedding2, _, _= embedding(mix_425_uv_from_106)
    diff = np.subtract(embedding1, embedding2)

    dist = np.sum(np.square(diff))
    print ("dist: ", dist)
    exit(0)
    '''
    w=h=112
    image_path1 = 'testing/mouth_occ_align.jpg'
    dets = face_detector.run(image_path1)
    print("occ: ",dets)
    landmark = landmark_detector.run(image_path1, [0, 0, w, h])
    print("occ: ", landmark)

    image_path2 = 'testing/no_occ_align.jpg'
    dets2 = face_detector.run(image_path2)
    print("no_occ: ",dets2)
    landmark = landmark_detector.run(image_path2, [0, 0, w, h])
    print("no_occ: ", landmark)
    #print(dets2)

    a = cv2.imread(image_path1)
    b = cv2.imread(image_path2)
    c = cv2.hconcat([a[:,0:60],b[:,60:112]])
    cv2.imwrite("testing/hconcat_c.jpg", c)
    image_path = 'testing/hconcat_c.jpg'
    dets2 = face_detector.run(image_path)
    print("c: ", dets2)
    landmark = landmark_detector.run(image_path, [0, 0, w, h])
    print("c: ", landmark)
    #for j in range(0, 10, 2):
    #    circle = patches.Circle((int(landmark[j + 0]), int(landmark[j + 1])), max(1,img.shape[0]/200), color='g')
    #    ax.add_patch(circle)
    #ax.text(landmark[8], landmark[9], "%.2f"%landmark[10], bbox=dict(facecolor='green', alpha=0.5))
    #cv2.imwrite("testing/hconcat_c.jpg", c)

    d= cv2.hconcat([b[:,0:60],a[:,60:112]])
    cv2.imwrite("testing/hconcat_d.jpg", d)
    image_path = 'testing/hconcat_d.jpg'
    dets2 = face_detector.run(image_path)
    print("d: ",dets2)
    landmark = landmark_detector.run(image_path, [0, 0, w, h])
    print("d: ", landmark)

    e = cv2.vconcat([a[0:60,:],b[60:112,:]])
    cv2.imwrite("testing/vconcat_e.jpg", e)
    image_path = 'testing/vconcat_e.jpg'
    dets2 = face_detector.run(image_path)
    print("e: ",dets2)
    landmark = landmark_detector.run(image_path, [0, 0, w, h])
    print("e: ", landmark)

    f= cv2.vconcat([b[0:60,:],a[60:112,:]])
    cv2.imwrite("testing/vconcat_f.jpg", f)
    image_path = 'testing/vconcat_f.jpg'
    dets2 = face_detector.run(image_path)
    print("f: ",dets2)
    landmark = landmark_detector.run(image_path, [0, 0, w, h])
    print("f: ", landmark)

    g= b
    g[0:56,0:56] = a [0:56,0:56]
    cv2.imwrite("testing/left_top.jpg", g)
    image_path = 'testing/left_top.jpg'
    dets2 = face_detector.run(image_path)
    print("left_top: ",dets2)
    landmark = landmark_detector.run(image_path, [0, 0, w, h])
    print("left_top: ", landmark)

    exit(0)
    '''
    #'''
    #embedding1= embedding(img1)
    '''
    print ("img1: ", img1)
    embedding1= embedding(img1)

    print ("img2: ", img2)
    embedding2= embedding(img2)
    diff = np.subtract(embedding1, embedding2)

    dist = np.sum(np.square(diff))
    print ("dist: ", dist)
    '''

    #test_path = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1208_issue/Register_ID/NIR_2020-12-08-04-10-50-783.png"
    #test_path = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1208_issue/Unlock_images/NIR_2020-12-08-04-12-45-717.png"
    #embedding(test_path)
    #exit(0)
    min_distance = 10000.0
    min_dist_list = []
    min_image_name = "None"
    version_name = 'v12_16_3'
    f = open('1209_result_'+version_name+'_distance_detailed.txt', 'w')
    counter = 0
    #register_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1110_issue/Register_ID/"
    #unlock_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1110_issue/Unlock_images/"
    #register_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1208_issue/Register_ID/"
    #unlock_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1208_issue/Unlock_images/"
    register_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1209_issue/Register_ID/"
    unlock_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/1209_issue/Unlock_images/"
    for register_filename in os.listdir(register_folder):
        print("counter: ", counter)
        #print("register_filename: ", register_filename)
        counter +=1
        register_path = os.path.join(register_folder,register_filename)

        #if counter == 2:
        #    break
        min_distance = 10000.0
        min_image_name = "None"
        min_bad_angle =  False
        min_bad_pose = False
        embedding_register, bad_angle_register, bad_pose_register = embedding(register_path)
        #print("embedding_register: ", embedding_register)
        if embedding_register is None:
            continue
        #f.write("---------------------------------\n")
        #f.write("register_filename: " + str(register_filename) + "\n")
        #f.write("bad_angle: " + str(bad_angle_register) + "\n")
        #f.write("bad_pose: " + str(bad_pose_register) + "\n")
        for unlock_filename in os.listdir(unlock_folder):
            if 'RGB' in register_filename and 'NIR' in unlock_filename:
                continue
            if 'NIR' in register_filename and 'RGB' in unlock_filename:
                continue
            unlock_path = os.path.join(unlock_folder,unlock_filename)
            embedding_unlock, bad_angle_unlock, bad_pose_unlock= embedding(unlock_path)
            if embedding_unlock is None:
                continue
            diff = np.subtract(embedding_register, embedding_unlock)
            dist = np.sum(np.square(diff))

            f.write("---------------------------------\n")
            f.write("register_filename: " + str(register_filename) + "\n")
            f.write("corresponding_unlock_image: " + str(unlock_filename) + "\n")
            f.write("distance: " + str(dist) + "\n")
            if dist < min_distance:
                min_distance = dist
                min_image_name = unlock_filename
                min_bad_angle = bad_angle_unlock
                min_bad_pose = bad_pose_unlock

        #f.write("min_distance: " + str(min_distance) + "\n")
        #f.write("corresponding_unlock_image: " + str(min_image_name) + "\n")
        #f.write("corresponding_unlock_image_bad_angle: " + str(min_bad_angle) + "\n")
        #f.write("corresponding_unlock_image_bad_pose: " + str(min_bad_pose) + "\n")
        min_dist_list.append(min_distance)
    f.close()
    #pickle.dump(min_dist_list, open( "min_distance"+version_name+".pkl", "wb" ) )
    #favorite_color = pickle.load( open( "save.p", "rb" ) )

    #data = [a, b]
    #plt.boxplot(data)
    '''
    test_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/RGB_test/"
    output_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/RGB_output/"
    #test_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/NIR_test/"
    #output_folder = "/mnt/sdd/craig/home_craig/kneron_tw_models/framework/face_recognition/kfr/testing/NIR_output/"

    for filename in os.listdir(test_folder):
        test_path = os.path.join(test_folder,filename)
        """
        outname = 'out_' +filename
        output_path = os.path.join(output_folder,outname)

        image = Image.open(test_path).convert("RGB")
        w, h =image.size
        dets = face_detector.run(image)
        x1,y1,w,h =int(dets[0][0]),int(dets[0][1]),int(dets[0][2]),int(dets[0][3])
        landmarks = landmark_detector.run(image, [x1, y1, w, h])#rectangle: list, [x, y, w, h]
        """
        #pre_process(test_path)
        embedding(test_path)
        #landmarks = landmark_detector.run(image, [0, 0, w, h])
        #landmark = landmark_detector.run(image_path, dets[0])
        #landmarks = detector.run(image, [0, 0, w, h])
        #print(landmarks)
        #display(test_path, dets = dets, landmarks_list = [landmarks], save_path=output_path)
    '''
