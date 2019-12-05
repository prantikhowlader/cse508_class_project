import torch
import torch.nn as nn
import os
import random
import sys
import torchvision.models as models
import torch.optim as optim
import logging
import time
import warnings
#import fine_grained_dataset_linear
#from fine_grained_dataset_linear import fine_grained_linear_dataset

from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform
import copy
import skimage
from skimage import io,transform
import kornia
#from utils import getImagesInFolder
#from tqdm import tqdm
import argparse



class SiameseLinearNetwork(nn.Module):
    def __init__(self):
        super(SiameseLinearNetwork, self).__init__()
        cnn = models.resnet50(pretrained=True)

        self.sim_model = cnn

    def forward(self, input1 = None):

        output1 = self.sim_model(input1)
        #output2 = self.fc1(output1)

        return output1

class LinearNetwork(nn.Module):
    def __init__(self, in_dim=1000, out_dim=200):
        super(LinearNetwork, self).__init__()


        self.fc1 = nn.Sequential(
            #nn.Linear(1000, 1000),
            #nn.ReLU(),
            nn.Linear(in_dim, out_features=out_dim),
            #nn.Softmax(dim = 1)
        )


    def forward(self, input1 = None):

        #output1 = self.sim_model(input1)
        output2 = self.fc1(input1)

        return output2


class ResNet50(nn.Module):
    def __init__(self,
                 pretrained=True,
                 lin_dim_in=1000,
                 lin_dim_out=200):
        super(ResNet50, self).__init__()

        # define the resnet152
        resnet = models.resnet50(pretrained=pretrained)

        # isolate the feature blocks
        self.features = nn.Sequential(resnet.conv1,
                                      resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      resnet.layer1,
                                      resnet.layer2,
                                      resnet.layer3,
                                      resnet.layer4)

        # average pooling layer
        self.avgpool = resnet.avgpool

        # classifier
        self.classifier1 = resnet.fc
        self.classifier2 = LinearNetwork(in_dim=lin_dim_in,
                                         out_dim=lin_dim_out)

        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x, track_grads=False):

        # extract the features
        out_x = self.features(x)

        # register the hook
        if track_grads:
            h = out_x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(out_x)
        x = x.view((x.shape[0], -1))
        x = self.classifier1(x)
        x = self.classifier2(x)
        if track_grads:
            return x, out_x
        else:
            return x, None