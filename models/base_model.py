# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch.nn as nn

from .alexnet import alexnet
from .googlenet import GoogLeNet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class BaseModel(nn.Module):
    def __init__(self, name, num_classes):
        super(BaseModel, self).__init__()
        if name == 'alexnet':
            self.base = alexnet(num_classes=num_classes)
        elif name == 'googlenet':
            self.base = GoogLeNet(num_classes=num_classes)
        elif name == 'resnet':
            self.base = resnet34(num_classes=num_classes)
        else:
            raise ValueError('Input model name is not supported!!!')

    def forward(self, x):
        return self.base(x)

