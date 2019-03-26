'''
Layers and operations for masker
including FPN(ResNet50 backbone), RPN head, ROIAlign, mask generator, losses calculator, etc.
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torchvision.models as models

import utils.opencv_utils as cv
import utils.tools as tl

class FeaturePyramidNet(nn.Module):
    def __init__(self):
        super(FeaturePyramidNet, self).__init__()
        resnet50=models.resnet50(pretrained=True)
        self.conv1=resnet50.conv1
        self.bn1=resnet50.bn1
        self.relu=resnet50.relu
        self.maxpool=resnet50.maxpool
        self.layer1=resnet50.layer1
        self.layer2=resnet50.layer2
        self.layer3=resnet50.layer3
        self.layer4=resnet50.layer4
        self.avgpool=resnet50.avgpool
        self.fc=resnet50.fc

    def forward(self, x):
        '''
        x is the batch of input square imgs
        '''

