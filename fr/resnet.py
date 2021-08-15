import torch
from torch import nn

def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding)


class flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


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
        self.bn0 = nn.BatchNorm2d(inplanes)
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
        out = self.prelu_out(out)

        return out


def input_block(inp_size, inplanes):
    return nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.PReLU(inplanes),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )


def head_block(inplanes, feat_size, do_rate, emb_size, head='fc', varg_last=1024):
    return nn.Sequential(
            #nn.BatchNorm2d(inplanes),
            nn.Dropout(p=do_rate),
            nn.Flatten(),
            nn.Linear(inplanes*feat_size*feat_size, emb_size),
            nn.BatchNorm1d(emb_size)
            )


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

