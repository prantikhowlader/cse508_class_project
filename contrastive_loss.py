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
def contrastive_loss(distance, labels):
    margin = 3.0
    
    is_diff = (labels).float()
    #loss = torch.mean(((1-is_diff) * torch.pow(distance, 2)) +
    #                    ((is_diff) * torch.pow(torch.abs(labels - distance), 2)))
    #loss = torch.mean((1-is_diff) * torch.pow(distance, 2) +
    #                                  (is_diff) * torch.pow(torch.clamp(margin - distance, min=0.0), 2))
    #assert distance.shape[1] == 1
    assert distance.shape[0] == is_diff.shape[0]
    loss = F.hinge_embedding_loss(distance, target=is_diff, margin=1.0)
    #print("check the loss")
    #print(loss)
    #print("check the loss vector shape")
    #print(loss.size())
    #assert loss.shape[0] == 1
    #assert loss.shape[1] == 1 
        
    return loss