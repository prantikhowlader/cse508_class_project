##the siamese resnet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import os
import random
import sys
import torchvision.models as models
import torch.optim as optim
import logging
import time
import warnings
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import os

class CNN(nn.Module):

    COSINE_SIMILARITY = 0
    L1_SIMILARITY = 1
    L2_SIMILARITY = 2
    PARAM_DISTANCE_DECAY_RATE = 0.5
    
    @staticmethod
    def exp_manhattan_distance(v1,v2) :

        d = torch.pow(nn.functional.pairwise_distance(v1,v2,1),CNN.PARAM_DISTANCE_DECAY_RATE)
        return 1.0 - torch.exp(-d)
#'../cub200/raw/CUB_200_2011/CUB_200_2011/classes.txt'
    @staticmethod
    def exp_euclidian_distance(v1,v2) :
        d = torch.pow(nn.functional.pairwise_distance(v1,v2,2), CNN.PARAM_DISTANCE_DECAY_RATE)
        return 1.0 - torch.exp(-d)

    @staticmethod
    def inv_cosine_similarity(v1,v2) :
        assert len(v1.shape) == 2
        assert len(v2.shape) == 2
        assert v1.shape[0] == v2.shape[0]
        v1_norm = F.normalize(v1, p=2, dim=1)
        v2_norm = F.normalize(v2, p=2, dim=1)
        return 1.0 - F.cosine_similarity(v1_norm,v2_norm)

    sim_fns = [inv_cosine_similarity,  exp_manhattan_distance, exp_euclidian_distance]
    sim_fn = sim_fns[COSINE_SIMILARITY]
    
    def build_model(self, v):
        
        if v == 18:
            cnn = models.resnet18(pretrained=True)
        elif v == 34:
            cnn = models.resnet34(pretrained=True)
        elif v == 50:
            cnn = models.resnet50(pretrained=True)
        elif v == 101:
            cnn = models.resnet101(pretrained=True)
        elif v == 152:
            cnn = models.resnet152(pretrained=True)
        else:
            #print("I am in resnet 152")
            cnn = models.resnet152(pretrained=True)
            
        #lastlayer_in = cnn.fc.in_features
        #cnn.fc = nn.Linear(lastlayer_in, 1000)
        
        #active_layers = {"fc.weight":1,"fc.bias":1}
        #for name, param in cnn.named_parameters():
            #if name not in active_layers:
                #param.requires_grad = False
         
        self.sim_model = cnn
        #self.fn1 = nn.Sequential(
        #nn.Linear(41,41) 
            #nn.Softmax(dim = 1)
        #)
       
    def __init__(self, similarity_dims, version=152):
        
        super(CNN, self).__init__()
        self.build_model( version)

                                    
    def parameters(self):
        return self.sim_model.fc.parameters()
    
    def forward(self, img1 = None, img2 = None):
        if(img2 is None):
            cnn_out1 = self.sim_model(img1)
            #output = self.fn1(cnn_out1)
            return cnn_out1
            
            
        cnn_out1 = self.sim_model(img1)
        cnn_out2 = self.sim_model(img2)
        #print("direct the cnn output check the dimensions")
        #print(cnn_out1.size())
        #print(cnn_out2.size())
        assert cnn_out1.size() == cnn_out2.size()
       
        return CNN.sim_fn(cnn_out1, cnn_out2)