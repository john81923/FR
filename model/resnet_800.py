import torch
from torch import nn
from collections import OrderedDict

import sys

#sys.path.append('/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pycls')
#sys.path.append('/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/pycls')

import pycls.core.model_builder as model_builder
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
#import pycls.utils.metrics as mu
from pycls.core.config import assert_and_infer_cfg, cfg

logger = lu.get_logger(__name__)
cfg_file = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/model/pycls_0513/configs/dds_baselines/regnetx/RegNetX-800MF_dds_8gpu.yaml'
cfg.merge_from_file(cfg_file)
cfg.TEST.WEIGHTS = '/mnt/storage1/craig/regnet/model/RegNetX-800MF_dds_8gpu.pyth'
cfg.OUT_DIR = '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/'
#cfg.merge_from_list(None)
assert_and_infer_cfg()
cfg.freeze()

#logger.info("Config:\n{}".format(cfg))

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

class Head(nn.Module):
    def __init__(self, feat_size=7,
            inplanes=672, emb_size=256, do_rate=0.0, head='fc', varg_last=1024):
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

class Merged_RGB_800(nn.Module):
    def __init__(self, emb_size, load_pretrain = False):
        super(Merged_RGB_800, self).__init__()
        # Build the model (before the loaders to speed up debugging)
        backbone_model = model_builder.build_model()
        #log_model_info(backbone_model)
        if load_pretrain:
            cu.load_checkpoint(cfg.TEST.WEIGHTS, backbone_model)
        logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

        #self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
        self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])

        #print(self.backbone)
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet32_3stage_state_dict_m.tar')
        #exit(0)
        self.rgb_head = Head(emb_size=emb_size, head='fc')
        #torch.save(self.backbone.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_backbone_state_dict_m.tar')

        #print(self.rgb_head)
        #exit(0)
        #torch.save(self.rgb_head.state_dict(), '/home/craig/kneron_development/kneron_tw_models/framework/face_recognition/kfr/log_regnet/regnet06_fc_state_dict_m.tar')
        #exit(0)

    def forward(self, x):
        features = self.backbone(x)
        rgb_head_output = self.rgb_head(features)
        return rgb_head_output
