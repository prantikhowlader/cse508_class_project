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
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import os
import skimage
from skimage import io

class adv_dataset(Dataset):
    def __init__(self,txt_path, train,img_transform1):
        #/workspace/data_fine_grain/cub200/raw/CUB_200_2011/(base) ironman@bigbox:~$ CUB_200_2011
        self.dir_path = './imagenet_fgsm/imagenet/fgsm/'
        self.image_list = []
        self.label_list = []
        self.train = train
        with(open(txt_path)) as f:
            lines = f.readlines()
            for i in lines:
                self.image_list.append(i.rstrip())
                self.label_list.append(i.split('_')[-2])
            
        self.img_transform1 = img_transform1

        
            
    def __getitem__(self,index):
        if(self.train):
            img1_path = self.image_list[index]
            img1_path = img1_path.rstrip()
            
            same_first = random.randint(0,1) 
            while True:
                #keep looping till the same class image is found
                img2_path = random.randint(0,len(self.image_list)-1)
                img2_path = self.image_list[img2_path].rstrip()
                if img1_path.split('_')[-2]==img2_path.split('_')[-2]:
                    break
            while True:
                #keep looping till the diff class image is found
                img3_path = random.randint(0,len(self.image_list)-1)
                img3_path = self.image_list[img3_path].rstrip()
                if img1_path.split('_')[-2]!=img3_path.split('_')[-2]:
                    break
            img1 = io.imread(img1_path)
            if(img1.shape.__len__() == 2):
                img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2) 
         
            img2 = io.imread(img2_path)
            if(img2.shape.__len__() == 2):
                img2 = np.repeat(img2[:, :, np.newaxis], 3, axis=2) 
            
            img3 = io.imread(img3_path)
            if(img3.shape.__len__() == 2):
                img3 = np.repeat(img3[:, :, np.newaxis], 3, axis=2) 
            #cv2.imread(img3_path)
            #img1 = img1.astype(np.float)/255.0
            #img2 = img2.astype(np.float)/255.0
            #img3 = img3.astype(np.float)/255.0
            img1 = cv2.resize(img1,(256,256),interpolation = cv2.INTER_AREA)
            img2 = cv2.resize(img2,(256,256), interpolation = cv2.INTER_AREA)
            img3 = cv2.resize(img3,(256,256), interpolation = cv2.INTER_AREA)
            #print("check the shape in the dataset")
            #print(img1.shape)
            #print(np.unique(img1))
            
            img1 = self.img_transform1(img1)
            img2 = self.img_transform1(img2)
            img3 = self.img_transform1(img3)
            #print("perform a check")
            #print("check the values of image")
            #print(np.unique(img1))
            
            #print(img1_path)
            #print(img2_path)
            #print(img3_path)
            #same_first=0
            if(same_first):
                return {"img1": img1, "img2": img2, "img3": img3, "label2": 1, "label3":-1 }
            else:
                #print("here")
                return {"img1": img1, "img2": img3, "img3": img2, "label2": -1, "label3":1 }
        else:
            img1_path = self.image_list[index]
            img1_path = img1_path.rstrip()
            img1 = io.imread(img1_path)
            if(img1.shape.__len__() == 2):
                img1 = np.repeat(img1[:, :, np.newaxis], 3, axis=2)
            #img1 = img1.astype(np.float)/255.0
            img1 = cv2.resize(img1,(256,256),interpolation = cv2.INTER_AREA)
            img1 = self.img_transform1(img1)
            label = self.label_list[index]
            label = int(label)
            return {"img1": img1, "label": label}

        #return img1,img2,label
    def __len__(self):
        return len(self.label_list)