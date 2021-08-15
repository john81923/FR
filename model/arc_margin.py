import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from ..conf.config import DEVICE

class ArcMarginModel(nn.Module):
    def __init__(self, num_classes, args):
        super(ArcMarginModel, self).__init__()

        self.num_classes = num_classes
        self.weight = Parameter(torch.FloatTensor(self.num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m


    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        #print("cosine.shape: ", cosine.shape) #cosine.shape:  torch.Size([8, 280])  (Batcg x class)
        #exit(0)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=DEVICE)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

'''
def update_arc_margin(model, margin_m=0.5):
    model.module.m = margin_m

    model.module.cos_m = math.cos(model.module.m)
    model.module.sin_m = math.sin(model.module.m)
    model.module.th = math.cos(math.pi - model.module.m)
    model.module.mm = math.sin(math.pi - model.module.m) * model.module.m
'''

def update_arc_margin(model, margin_m=0.5):
    model.m = margin_m

    model.cos_m = math.cos(model.m)
    model.sin_m = math.sin(model.m)
    model.th = math.cos(math.pi - model.m)
    model.mm = math.sin(math.pi - model.m) * model.m


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

#https://github.com/HuangYG123/CurricularFace
class CurricularFace(nn.Module):
    #def __init__(self, in_features, out_features, m = 0.5, s = 64.):
    def __init__(self, num_classes, args, m = 0.5, s = 64., emb_size = 256):
        super(CurricularFace, self).__init__()
        self.in_features = emb_size
        self.out_features = num_classes
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(emb_size, num_classes ))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis = 1)
        kernel_norm = l2_norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output #, origin_cos * self.s
