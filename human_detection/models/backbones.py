from copy import deepcopy
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


def ConvBnLeaky(in_, out_, k, s):
    '''
    in_: input channel, e.g. 32
    out_: output channel, e.g. 64
    k: kernel size, e.g. 3 or (3,3)
    s: stride, e.g. 1 or (1,1)
    '''
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_, out_, k, s, padding=pad, bias=False),
        nn.BatchNorm2d(out_, eps=1e-5, momentum=0.1),
        nn.LeakyReLU(0.1)
    )


class DarkBlock(nn.Module):
    '''
    basic residual block in Darknet53
    in_out: input and output channels
    hidden: channels in the block
    '''
    def __init__(self, in_out, hidden):
        super().__init__()
        self.cbl_0 = ConvBnLeaky(in_out, hidden, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(hidden, in_out, k=3, s=1)

    def forward(self, x):
        residual = x
        x = self.cbl_0(x)
        x = self.cbl_1(x)

        return x + residual


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.netlist = nn.ModuleList()

        # first conv layer
        self.netlist.append(ConvBnLeaky(3, 32, k=3, s=1))

        # Downsampled by 2 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(32, 64, k=3, s=2))
        for _ in range(1):
            self.netlist.append(DarkBlock(in_out=64, hidden=32))

        # Downsampled by 4 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(64, 128, k=3, s=2))
        for _ in range(2):
            self.netlist.append(DarkBlock(in_out=128, hidden=64))

        # Downsampled by 8 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(128, 256, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=256, hidden=128))
        assert len(self.netlist) == 15

        # Downsampled by 16 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(256, 512, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=512, hidden=256))
        assert len(self.netlist) == 24

        # Downsampled by 32 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(512, 1024, k=3, s=2))
        for _ in range(4):
            self.netlist.append(DarkBlock(in_out=1024, hidden=512))
        assert len(self.netlist) == 29
        # end creating Darknet-53 back bone layers

    def forward(self, x):
        for i in range(0,15):
            x = self.netlist[i](x)
        small = x
        for i in range(15,24):
            x = self.netlist[i](x)
        medium = x
        for i in range(24,29):
            x = self.netlist[i](x)
        large = x

        return small, medium, large


class ResNetBackbone(nn.Module):
    '''
    Args:
        tv_model: torch vision model
    '''

    def __init__(self, tv_model):
        super().__init__()
        self.conv1 = tv_model.conv1
        self.bn1 = tv_model.bn1
        self.relu = tv_model.relu
        self.maxpool = tv_model.maxpool

        self.layer1 = tv_model.layer1
        self.layer2 = tv_model.layer2
        self.layer3 = tv_model.layer3
        self.layer4 = tv_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        small = self.layer2(x)
        medium = self.layer3(small)
        large = self.layer4(medium)
        return small, medium, large


def resnet34():
    print('Using backbone ResNet-34. Loading ImageNet weights...')
    model = torchvision.models.resnet34(pretrained=True)
    return ResNetBackbone(model)


def resnet50():
    print('Using backbone ResNet-50. Loading ImageNet weights...')
    model = torchvision.models.resnet50(pretrained=True)
    return ResNetBackbone(model)


def resnet101():
    print('Using backbone ResNet-101. Loading ImageNet weights...')
    model = torchvision.models.resnet101(pretrained=True)
    return ResNetBackbone(model)


class YOLOBranch(nn.Module):
    '''
    Args:
        in_: int, input channel number
        out_: int, output channel number, typically = 3 * 6 [x,y,w,h,a,conf]
        has_previous: bool, True if this is not the first detection layer
        prev_ch: (int,int), the Conv2d channel for the previous feature,
                 default: None
    '''
    # def __init__(self, in_, out_=18, has_previous=False, prev_ch=None):
    def __init__(self, in_, out_=18, prev_ch=None):
        super(YOLOBranch, self).__init__()
        assert in_ % 2 == 0, 'input channel must be divisible by 2'

        # tmp_ch = prev_ch if prev_ch is not None else (in_, in_//2)
        if prev_ch:
            self.process = ConvBnLeaky(prev_ch[0], prev_ch[1], k=1, s=1)
            in_after_cat = in_ + prev_ch[1]
        else:
            self.process = None
            in_after_cat = in_

        self.cbl_0 = ConvBnLeaky(in_after_cat, in_//2, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_2 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_3 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_4 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_5 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.to_box = nn.Conv2d(in_, out_, kernel_size=1, stride=1)

    def forward(self, x, previous=None):
        '''
        Args:
            x: feature from backbone, for large/medium/small size
            previous: feature from the lower spatial resolution
        '''
        if previous is not None:
            pre = self.process(previous)
            pre = F.interpolate(pre, scale_factor=2, mode='nearest')
            x = torch.cat((pre, x), dim=1)

        x = self.cbl_0(x)
        x = self.cbl_1(x)
        x = self.cbl_2(x)
        x = self.cbl_3(x)
        feature = self.cbl_4(x)
        x = self.cbl_5(feature)
        detection = self.to_box(x)

        return detection, feature

##################################################################
##################################################################
##################################################################


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class yolov5(nn.Module):
    '''
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    '''

    def __init__(self, cfg, ch=3):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        nc = self.yaml['nc'] # number of classes
        anchors = self.yaml['anchors'] # anchors

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        # m = self.model[-1]  # Detect()
        # if isinstance(m, Detect):
        #     s = 256  # 2x min stride
        #     m.inplace = self.inplace
        #     m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        #     m.anchors /= m.stride.view(-1, 1, 1)
        #     check_anchor_order(m)
        #     self.stride = m.stride
        #     self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        # self.info()
        print('')

    def forward(self, x):
        i_small = 17  # P3
        i_medium = 20  # P4
        i_large = 23  # P5
        
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            
            if m.i == i_small:
                small = x
            elif m.i == i_medium:
                medium = x
            elif m.i == i_large:
                large = x

        return small, medium, large


def parse_model(d, ch):  # model_dict, input_channels(3)
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head'][:-1]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPPF, C3]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = math.ceil(c2 * gw / 8) * 8
            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # elif m is Detect:
        #     args.append([ch[x] for x in f])
        #     if isinstance(args[1], int):  # number of anchors
        #         args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
