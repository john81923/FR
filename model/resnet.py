import torch
from torch import nn
from collections import OrderedDict

import sys
from .timm.models import efficientnet as effnet
from .timm.models import resnest as resnest

import torch.nn.functional as F

import torchvision.models as torch_models

#sys.path.append('/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pycls')
sys.path.append('/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/pycls/')

'''
import pycls.pycls.core.model_builder as model_builder
import pycls.pycls.utils.checkpoint as cu
import pycls.pycls.utils.logging as lu
import pycls.pycls.utils.metrics as mu
from pycls.pycls.core.config import assert_and_infer_cfg, cfg
'''
'''
import pycls.core.model_builder as model_builder
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
#import pycls.utils.metrics as mu
from pycls.core.config import assert_and_infer_cfg, cfg


logger = lu.get_logger(__name__)
#cfg_file = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/pycls_0513/configs/dds_baselines/regnetx/RegNetX-600MF_dds_8gpu.yaml'
#cfg.merge_from_file(cfg_file)
#cfg.TEST.WEIGHTS = '/mnt/storage1/craig/regnet/model/RegNetX-600MF_dds_8gpu.pyth'
#RGB_fix_weight = '/mnt/storage1/craig/regnet/model/regnetx_006-85ec1baa.pth'

cfg_file = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/pycls_0513/configs/dds_baselines/regnetx/RegNetX-1.6GF_dds_8gpu.yaml'
cfg.merge_from_file(cfg_file)
cfg.TEST.WEIGHTS = '/mnt/storage1/craig/regnet/model/RegNetX-1.6GF_dds_8gpu.pyth'
RGB_fix_weight = '/mnt/storage1/craig/regnet/model/regnetx_016-65ca972a.pth'
cfg.OUT_DIR = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/'
#cfg.merge_from_list(None)
assert_and_infer_cfg()
cfg.freeze()
'''
#logger.info("Config:\n{}".format(cfg))

#Dropblock:
#https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py
class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob=0.1, block_size=5):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)


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

        #print("x.size(): ", x.size())
        #print("residual_ori.size(): ", residual.size())
        if self.downsample:
            residual = self.downsample(x)
        #print("out.size(): ", out.size())
        #print("residual.size(): ", residual.size())
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, se_ratio=0):
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

class hard_swish_layer(nn.Module):
    def __init__(self):
        super(hard_swish_layer, self).__init__()
    def forward(self, x):
        return x.mul_(F.relu6(x + 3.) / 6.)
#self.act_fn_outer = hard_swish_layer()

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
            inplanes=528, emb_size=256, do_rate=0.0, head='fc', varg_last=1024):
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
                        #nn.BatchNorm2d(inplanes),
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


class ResNet_bottleneck(nn.Module):

    def __init__(self, block, layout, block_size, se_ratio=0,
            inp_size=112, emb_size=256, do_rate=0.4,
            head='fc', varg_last=1024):
        super(ResNet_bottleneck, self).__init__()
        self.expend_ratio = 4
        self.input = input_block(224, block_size[0])
        self.inplanes = block_size[0]
        self.layer1 = self._make_layer(block, block_size[0], block_size[1], layout[0], stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, block_size[1], block_size[2], layout[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, block_size[2], block_size[3], layout[2], stride=2, se_ratio=se_ratio)

        #self.layer4 = self._make_layer(block, block_size[3], block_size[4], layout[3], stride=2, se_ratio=se_ratio)

        # 512x7x7             128*4 = 512, 256*4= 1024
        self.head = head_block(block_size[3]*self.expend_ratio, 7, do_rate, emb_size, head)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, se_ratio=0):
        downsample = None

        if stride != 1 or self.inplanes != (self.expend_ratio*planes):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*self.expend_ratio, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expend_ratio),
            )

        layers = []
        layers.append(block(self.inplanes,  planes, stride, downsample, se_ratio=se_ratio))
        self.inplanes = planes*self.expend_ratio
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se_ratio=se_ratio))


        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.input(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)

        x = self.head(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layout, block_size, se_ratio=0,
            inp_size=112, emb_size=512, do_rate=0.4,
            head='fc', varg_last=1024):
        super(ResNet, self).__init__()
        self.expend_ratio = 4
        self.input = input_block(inp_size, block_size[0])
        self.inplanes = block_size[0]
        self.layer1 = self._make_layer(block, block_size[0], block_size[1], layout[0], stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, block_size[1], block_size[2], layout[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, block_size[2], block_size[3], layout[2], stride=2, se_ratio=se_ratio)

        self.layer4 = self._make_layer(block, block_size[3], block_size[4], layout[3], stride=2, se_ratio=se_ratio)

        if block == Bottleneck:
            self.head = head_block(block_size[4]*self.expend_ratio, 7, do_rate, emb_size, head)
        else:
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

        if block == Bottleneck:
            if stride != 1 or self.inplanes != (self.expend_ratio*planes):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes*self.expend_ratio, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes*self.expend_ratio),
                )
        else:
            if stride != 1 or inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        if block == Bottleneck:
            layers = []
            layers.append(block(self.inplanes,  planes, stride, downsample, se_ratio=se_ratio))
            self.inplanes = planes*self.expend_ratio
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, se_ratio=se_ratio))
        else:
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





class ResNet_dropblock(nn.Module):

    def __init__(self, block, layout, block_size, se_ratio=0,
            inp_size=112, emb_size=512, do_rate=0.4,
            head='fc', varg_last=1024):
        super(ResNet_dropblock, self).__init__()
        self.expend_ratio = 4
        self.input = input_block(inp_size, block_size[0])
        self.inplanes = block_size[0]
        self.dropblock = DropBlock2D(drop_prob=0.1, block_size=5)
        self.layer1 = self._make_layer(block, block_size[0], block_size[1], layout[0], stride=1, se_ratio=se_ratio)
        self.layer2 = self._make_layer(block, block_size[1], block_size[2], layout[1], stride=2, se_ratio=se_ratio)
        self.layer3 = self._make_layer(block, block_size[2], block_size[3], layout[2], stride=2, se_ratio=se_ratio)

        self.layer4 = self._make_layer(block, block_size[3], block_size[4], layout[3], stride=2, se_ratio=se_ratio)

        if block == Bottleneck:
            self.head = head_block(block_size[4]*self.expend_ratio, 7, do_rate, emb_size, head)
        else:
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

        if block == Bottleneck:
            if stride != 1 or self.inplanes != (self.expend_ratio*planes):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes*self.expend_ratio, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes*self.expend_ratio),
                )
        else:
            if stride != 1 or inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        if block == Bottleneck:
            layers = []
            layers.append(block(self.inplanes,  planes, stride, downsample, se_ratio=se_ratio))
            self.inplanes = planes*self.expend_ratio
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, se_ratio=se_ratio))
        else:
            layers = []
            layers.append(block(inplanes,  planes, stride, downsample, se_ratio=se_ratio))
            for i in range(1, blocks):
                layers.append(block(planes, planes, se_ratio=se_ratio))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.input(x)

        x = self.dropblock(self.layer1(x))
        x = self.dropblock(self.layer2(x))
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)

        return x


DEFAULT_LAYER = 50
BLOCK_LAYOUT = {
    18:  [2, 2,  2, 2],
    34:  [3, 4,  6, 3],
    34_2:  [3, 4,  10, 3],
    34_3:  [3, 4,  12, 3],
    #34_3:  [3, 4,  10, 3],
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
    'ml' : [64, 64, 96, 128, 256],
    'l' : [64, 64, 128, 256, 512],
    'l1': [64, 64, 128, 512, 512],
    'l2': [64, 64, 256, 256, 512],
    'l3': [64, 64, 256, 512, 512],
    'l7': [64, 128, 256, 512, 512],
    'x_8': [64, 128, 256, 512, 1024],
    'x'  : [128, 128, 256, 512, 1024],
    'large' : [48, 48, 72, 160, 288],
    }





def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.affine:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def build_resnet(block=IRBlock, se_ratio=0, inp_size=112,
            block_layout=BLOCK_LAYOUT[DEFAULT_LAYER], block_size=BLOCK_SIZE[DEFAULT_SIZE],
            emb_size=512, do_rate=0.4, head='fc', dropblock = False):

    if block ==IRBlock:
        if dropblock:
            model = ResNet_dropblock(block, block_layout, block_size, se_ratio=se_ratio,
                    inp_size=inp_size, emb_size=emb_size, do_rate=do_rate, head=head)
        else:
            model = ResNet(block, block_layout, block_size, se_ratio=se_ratio,
                    inp_size=inp_size, emb_size=emb_size, do_rate=do_rate, head=head)

    else:
        model = ResNet_bottleneck(block, block_layout, block_size, se_ratio=se_ratio,
                inp_size=inp_size, emb_size=emb_size, do_rate=do_rate, head=head)

    model.apply(init_weight)

    return model



def resnet18(args=None, **kwargs):
    if args:
        model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[18], block_size=args.block_size,
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)
    else:
        model = build_resnet(block_layout=BLOCK_LAYOUT[18], se_ratio=16)

    return model


def resnet34(args=None, **kwargs):
    if args:
        model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[34], block_size=args.block_size,
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)
    else:
        model = build_resnet(block_layout=BLOCK_LAYOUT[34], se_ratio=16)

    return model

'''
def resnet50_large(args=None, **kwargs):
    if args:
        model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[50], block_size=BLOCK_SIZE['large'],
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)
    else:
        model = build_resnet(block_layout=BLOCK_LAYOUT[50], se_ratio=16)

    return model
'''
def resnet50_large(args=None, **kwargs):

    model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
        block_layout=BLOCK_LAYOUT[34_2], block_size=BLOCK_SIZE['m'],
        emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)

    return model

#this is actually res34 with changing 3,4,6,3 to 3,4,14,3
def resnet50(args=None, **kwargs):
    if args:
        print("=== drop-block setting: ", args.dropblock)
        model = build_resnet(se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[50], block_size=args.block_size,
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head, dropblock = args.dropblock)
    else:
        model = build_resnet(block_layout=BLOCK_LAYOUT[50], se_ratio=16)

    return model

def resnet50_bottleneck(args=None, **kwargs):
    model = build_resnet(block = Bottleneck, se_ratio=args.se_ratio, inp_size=args.inp_size,
            block_layout=BLOCK_LAYOUT[34_2], block_size=BLOCK_SIZE['l'], #block_size=args.block_size,
            emb_size=args.emb_size, do_rate=args.do_rate, head=args.head)
    #print(model)
    #exit(0)
    return model




def resnet_face18(se_ratio=0, **kwargs):
    model = build_resnet(block_layout=BLOCK_LAYOUT[18],  block_size=BLOCK_SIZE[DEFAULT_SIZE],
            se_ratio=args.se_ratio)
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

#https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
class ResNet50Backbone(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Backbone, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:])
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        #print(self.features)

    def forward(self, x):
        x = self.features(x)
        return x

class RGBHead(nn.Module):
    def __init__(self, original_model):
        super(RGBHead, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:])
        self.features = nn.Sequential(*list(original_model.children())[-1])
        #print(self.features)

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = resnet152().to('cpu')
    summary(model, (3, 112, 112), batch_size=1, device='cpu')

'''
def log_model_info(model):
    """Logs model info"""
    print("Params: ", mu.params_count(model))
    print("Flops: ", mu.flops_count(model))
    print("Acts: ", mu.acts_count(model))
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(mu.params_count(model)))
    logger.info("Flops: {:,}".format(mu.flops_count(model)))
    logger.info("Acts: {:,}".format(mu.acts_count(model)))
'''

'''
class Merged_Resnet(nn.Module):
    def __init__(self, nir_emb_size):
        super(Merged_Resnet, self).__init__()
        """
        block_size = [32, 32, 64, 128, 256]
        backbone_model = build_resnet(se_ratio=0, inp_size=112,
            block_layout=BLOCK_LAYOUT[50], block_size=block_size,
            emb_size=256, do_rate=0.0, head='fc')
        """
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.rgb_head = nn.Sequential(*list(backbone_model.children())[-1])
        self.nir_head =  Head(emb_size=nir_emb_size, head='fc')
        #self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        #self.nir_head = nn.Sequential(*list(nir_head.children()))
        #print(self.features)
        #print(self.backbone)
        #exit(0)

    def forward(self, x):
        features = self.backbone(x)
        rgb_head_output = self.rgb_head(features)
        nir_head_output = self.nir_head(features)
        return rgb_head_output, nir_head_output
'''
"""
class Merged_RGB_TIMM(nn.Module):
    def __init__(self, emb_size, load_pretrain = False):
        super(Merged_RGB, self).__init__()
        # Build the model (before the loaders to speed up debugging)
        backbone_model = model_builder.build_model()
        #log_model_info(backbone_model)
        print("load from: ", cfg.TEST.WEIGHTS)

        if load_pretrain:
            #cu.load_checkpoint(cfg.TEST.WEIGHTS, backbone_model)
            cu.load_checkpoint_rgb(cfg.TEST.WEIGHTS, backbone_model)
        logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
        print("loading ok")
        exit(0)
        #self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        #print(self.backbone)
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet32_3stage_state_dict_m.tar')
        #exit(0)
        self.rgb_head = Head(emb_size=emb_size, head='fc')
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_backbone_state_dict_m.tar')

        #torch.save(self.rgb_head.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_fc_state_dict_m.tar')
        #exit(0)

#RegNet16
class Merged_RGB(nn.Module):
    def __init__(self, emb_size, load_pretrain = False):
        super(Merged_RGB, self).__init__()
        # Build the model (before the loaders to speed up debugging)
        backbone_model = model_builder.build_model()
        #log_model_info(backbone_model)
        print("load from: ", cfg.TEST.WEIGHTS)
        if load_pretrain:
            #cu.load_checkpoint(cfg.TEST.WEIGHTS, backbone_model)
            cu.load_checkpoint_rgb(cfg.TEST.WEIGHTS, backbone_model, RGB_fix_weight)
        logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

        #self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        #for regnet 600MF
        #self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        #for regnet 1.6GF
        ##a = list(list(backbone_model.children())[-3].children())[:-2]

        # 408,912
        #self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
        #912
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        #print(self.backbone)
        #print(a)
        #exit(0)
        #print(self.backbone)
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet32_3stage_state_dict_m.tar')
        #exit(0)
        #for regnet 600MF
        #self.rgb_head = Head(emb_size=emb_size, head='fc')
        #for regnet 1.6GF
        #self.rgb_head = Head(inplanes=408, emb_size=emb_size, head='fc')
        #self.rgb_head = Head(inplanes=408, emb_size=emb_size, head='varg')
        self.rgb_head = Head(inplanes=912, feat_size=4, emb_size=emb_size, head='varg')

        for m in self.rgb_head.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_backbone_state_dict_m.tar')

        #torch.save(self.rgb_head.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_fc_state_dict_m.tar')
        #exit(0)

    def forward(self, x):
        features = self.backbone(x)
        rgb_head_output = self.rgb_head(features)

        return rgb_head_output
"""

class ResNet50Backbone(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Backbone, self).__init__()
        #self.features = nn.Sequential(*list(original_model.children())[:])
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        #print(self.features)

    def forward(self, x):
        x = self.features(x)
        return x

class Merged_Resnet(nn.Module):
    def __init__(self, original_model):
        super(Merged_Resnet, self).__init__()
        """
        block_size = [32, 32, 64, 128, 256]
        backbone_model = build_resnet(se_ratio=0, inp_size=112,
            block_layout=BLOCK_LAYOUT[50], block_size=block_size,
            emb_size=256, do_rate=0.0, head='fc')
        """
        self.model = nn.Sequential(*list(backbone_model.children())[:-1])
        self.rgb_head = nn.Sequential(*list(backbone_model.children())[-1])
        #self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        #self.nir_head = nn.Sequential(*list(nir_head.children()))
        #print(self.features)
        #print(self.backbone)
        #exit(0)

    def forward(self, x):
        features = self.backbone(x)
        rgb_head_output = self.rgb_head(features)
        return rgb_head_output



class mobilenetv2_100(nn.Module):
    def __init__(self, load_pretrain = False):
        super(mobilenetv2_100, self).__init__()


        self.keep = [1, 2, 4]
        #self.uplayer_shape = [24, 32, 96]
        self.uplayer_shape = [32, 40, 112]
        #self.output_channels = 320
        self.output_channels = 384

        #backbone = effnet._gen_mobilenet_v2('mobilenetv2_100', 1.0, pretrained=load_pretrain)
        backbone = effnet._gen_mobilenet_v2(
            'mobilenetv2_120d', 1.2, depth_multiplier=1.4, fix_stem_head=True, pretrained=load_pretrain)
        self.mbv2_bb = nn.Sequential(*list(backbone.children())[:-5])
        #print(self.mbv2_bb)
        #exit(0)

    def forward(self, x):
        outs = []
        for index, item in enumerate(self.mbv2_bb):
            if index<3:
                x = item(x)
            elif index ==3:
                for index_sub, item_sub in enumerate(item):
                    x = item_sub(x)
                    if index_sub in self.keep:
                        outs.append(x)

                outs.append(x)
            else:
                raise RuntimeError(f"Unsupport mode: index in mbv2 module")

        return outs


#'''
class resnest_50(nn.Module):
    def __init__(self, load_pretrain = False):
        super(resnest_50, self).__init__()

        pool1 = nn.AvgPool2d(3, stride=2)

        #backbone = resnest.resnest26d(pretrained=load_pretrain)
        backbone = resnest.resnest50d(pretrained=load_pretrain)
        #backbone = torch_models.resnet50(pretrained=True)
        #remove max-pooling, last layer is 2048
        #self.resnest_26d_backbone = nn.Sequential(*list(backbone.children())[:3] , *list(backbone.children())[4:-2])
        #remove final layer
        #self.resnest_backbone = nn.Sequential(*list(backbone.children())[:-3])
        #remove final two layers
        self.resnest_backbone = nn.Sequential(pool1, *list(backbone.children())[:-4])
        #self.resnest_50_backbone = nn.Sequential(*list(backbone.children())[:])
        #print(self.resnest_backbone)
        #self.rgb_head = Head(inplanes=1024, emb_size=256, head='fc')
        self.rgb_head = Head(inplanes=512, feat_size=7, emb_size=256, head='fc')
        #exit(0)

    def forward(self, x):
        features = self.resnest_backbone(x)
        rgb_head_output = self.rgb_head(features)
        return rgb_head_output
    '''
    def forward(self, x):
        outs = []
        for index, item in enumerate(self.mbv2_bb):
            if index<3:
                x = item(x)
            elif index ==3:
                for index_sub, item_sub in enumerate(item):
                    x = item_sub(x)
                    if index_sub in self.keep:
                        outs.append(x)

                outs.append(x)
            else:
                raise RuntimeError(f"Unsupport mode: index in mbv2 module")

        return outs
    '''
#'''

# Conv BatchNorm Activation
class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Up Sample Module
class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,  bias=False, mode="UCBA", alignment = False):
        super(UpModule, self).__init__()
        self.mode = mode

        if self.mode == "UCBA":
            #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            if alignment:
                self.up = nn.Upsample(size=(7,7), scale_factor=None, mode='bilinear', align_corners=True)
            else:
                self.up = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias)
        elif self.mode == "DeconvBN":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.mode == "DeCBA":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError(f"Unsupport mode: {mode}")

    def forward(self, x):
        if self.mode == "UCBA":
            return self.conv(self.up(x))
        elif self.mode == "DeconvBN":
            return F.relu(self.bn(self.dconv(x)))
        elif self.mode == "DeCBA":
            return self.conv(self.dconv(x))


# SSH Context Module
class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()

        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1)

    def forward(self, x):

        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


# SSH Detect Module
class DetectModule(nn.Module):
    def __init__(self, in_channels):
        super(DetectModule, self).__init__()

        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1)
        self.context = ContextModule(in_channels)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


# Job Head Module
class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, has_ext=False):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.has_ext = has_ext

        if has_ext:
            self.ext = CBAModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):

        if self.has_ext:
            x = self.ext(x)
        return self.head(x)



# DBFace Model
class DBFace_mbv2(nn.Module):
    def __init__(self, has_landmark=False, wide=128, has_ext=True, upmode="UCBA", load_pretrain=False, do_rate = 0.0, emb_size = 256, head = 'fc' ):
        super(DBFace_mbv2, self).__init__()
        self.has_landmark = has_landmark

        # define backbone
        self.bb = mobilenetv2_100(load_pretrain=load_pretrain)

        # Get the number of branch node channels
        # stride4, stride8, stride16
        c0, c1, c2 = self.bb.uplayer_shape

        self.conv3 = CBAModule(self.bb.output_channels, wide, kernel_size=1, stride=1, padding=0, bias=False) # s32
        #self.connect0 = CBAModule(c0, wide, kernel_size=1)  # s4
        self.connect1 = CBAModule(c1, wide, kernel_size=1)  # s8
        self.connect2 = CBAModule(c2, wide, kernel_size=1)  # s16

        self.up0 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode, alignment= True) # s16
        self.up1 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode) # s8
        #self.up2 = UpModule(wide, wide, kernel_size=2, stride=2, mode=upmode) # s4
        self.detect = DetectModule(wide)

        # origin: 256x7x7 = 12544, 64x28x28 = 50176, 64x14x14 = 12544
        #self.head = Head(feat_size=28, inplanes=wide, emb_size=emb_size, head='fc') #head_block(wide, 28, do_rate, emb_size, head)
        self.head = Head(feat_size=14, inplanes=wide, emb_size=emb_size, head='fc') #head_block(wide, 28, do_rate, emb_size, head)
        #self.head = Head(feat_size=28, inplanes=wide, emb_size=emb_size, head='fc') #head_block(wide, 28, do_rate, emb_size, head)

        #self.center = HeadModule(wide, 1, has_ext=has_ext)
        #self.box = HeadModule(wide, 4, has_ext=has_ext)

        #if self.has_landmark:
        #    self.landmark = HeadModule(wide, 10, has_ext=has_ext)


    def init_weights(self):

        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19

        # Load backbone weights from ImageNet
        #self.bb.load_pretrain()
        self.center.init_normal(0.001, d)
        self.box.init_normal(0.001, 0)

        #if self.has_landmark:
        #    self.landmark.init_normal(0.001, 0)


    def load(self, file):
        checkpoint = torch.load(file, map_location="cpu")
        self.load_state_dict(checkpoint)


    def forward(self, x):
        s4, s8, s16, s32 = self.bb(x)
        s32 = self.conv3(s32)

        s16 = self.up0(s32) + self.connect2(s16)
        s8 = self.up1(s16) + self.connect1(s8)
        '''
        s4 = self.up2(s8) + self.connect0(s4)
        x = self.detect(s4)
        '''
        x = self.detect(s8)

        #print("x_ori.size(): ", x.size())

        x = self.head(x)
        #x = self.head(x)

        #print("x.size(): ", x.size())
        #exit(0)
        return x
