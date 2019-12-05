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
import os
import adv_dataset
from adv_dataset import adv_dataset

import contrastive_loss
from contrastive_loss import contrastive_loss
import network
from network import CNN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str,default="0", help="gpu_id")
parser.add_argument("--cnn_size", type=str,default="50", help="gpu_id")
parser.add_argument("--epochs", type=str,default="0", help="epoch")
#parser.add_argument("--gamma", type=str,default="1", help="gamma")
args = parser.parse_args()

######### Log files for changing the 
log_file = "./logs_{}".format(args.cnn_size)
if not os.path.exists(log_file):
    os.mkdir(log_file)
    #print("Directory " , dirName ,  " Created ")
writer = SummaryWriter(log_file)





model_save_path = "./model_output_{}/".format(args.cnn_size)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
    #print("Directory " , dirName ,  " Created ")



#print(os.getcwd())
writer = SummaryWriter(log_file)
print(os.getcwd())

CHECKPOINT_PATH = os.path.join(model_save_path, 'resnet_simaese.pth') #save the final model
CHECKPOINT_STATE_PATH = os.path.join(model_save_path, 'model_info.pth')
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(int(args.gpu_id))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform1 = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop([224,224]),transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.RandomRotation((-15,15)),transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)) ])

#for train the file: train_images.txt, sample_train_images.txt
#for test the file: test_images.txtself.image_list1_res_imagen
train_set = adv_dataset('./fgsm_train.txt',True,transform1)

trainloader = DataLoader(train_set,batch_size = 5,shuffle = True, num_workers =0)

test_set =  adv_dataset('./fgsm_test.txt',True,transform1)
testloader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

val_set =  adv_dataset('./fgsm_val.txt',True,transform1)
valloader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)
#learning_rate
lr = 1e-4
num_epochs = int(args.epochs)


similarity_dims = 1000
optimizer = 'SGD'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
OPTIMIZER = {'Adam': optim.Adam, 'SGD': optim.SGD}

logger.info('Build the model')
#################################################use resnet 50
model = CNN(int(args.cnn_size))
#testloader = DataLoader(test_set, batch_size=10, shuffle=False, num_workers=2)
optimizer = OPTIMIZER[optimizer](model.parameters(), lr= lr)

# Save arguments used to create model for restoring the model later


similarity_margin = 0.03
def test_model(model, test_dl):
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    iteration = 0
        
    for data in test_dl:
      
        img1 = data['img1'].to(DEVICE)
        #img1 = img1.float()
        img1 = img1.view(-1,img1.shape[-3],img1.shape[-2],img1.shape[-1])
        img1 = img1.float()
        img2 = data['img2'].to(DEVICE)
        #img2 = img2.float()(base) ironman@bigbox:~$ 
        img2 = img2.view(-1,img2.shape[-3],img2.shape[-2],img2.shape[-1])
        img2 = img2.float()
        img3 = data['img3'].to(DEVICE)
        #img2 = img2.float()
        img3 = img3.view(-1,img2.shape[-3],img2.shape[-2],img2.shape[-1])
        img3 = img3.float()
        label2 = data['label2'].to(DEVICE).float()
        label2 = label2.view(-1)
        label3 = data['label3'].to(DEVICE).float()
        label3 = label3.view(-1)
    
        distance = model.forward(img1,img2)
        distance1 =  model.forward(img1,img3)
        loss = contrastive_loss(distance, label2)
        loss2 = contrastive_loss(distance1, label3)
        iteration +=1
        #predictions = (torch.abs(distance - labels) < similarity_margin).int()
        
        running_loss += loss.item()
        running_loss += loss2.item()
        #running_corrects += torch.sum(predictions)
    
    test_loss = running_loss / (2*iteration)
    #test_acc = running_corrects.double() / len(test_dl.dataset)
    
    logger.info('Test set: Average loss: {:.8f}\n'.format(test_loss))
    
    return test_loss

BEST_MODEL_METRIC = {
    'train-loss': 1000.0,
    'train-acc': 0.0,
    'test-loss': 1000.0,
    'test-acc': 0.0
}

def train_sim_model(model, train_dl, test_dl, optimizer, num_epochs= num_epochs):
    
    
        since = time.time()
        best_loss = 1000.0
        model = model.to(DEVICE)
        global_iter = 0.0
        for epoch in range(num_epochs):

            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logger.info('-' * 10)

            # Each epoch has a training and validation phase
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0
            iteration = 0
            i = 0
            tb_iter = 0
            # Iterate over data.
            for data in train_dl:
                #print("i am inside")
                #return {"img1": img1, "img2": img2, "img3": img3, "label2": 0, "label3":1 }

                img1 = data['img1'].to(DEVICE)
                #img1 = img1.float()
                img1 = img1.view(-1,img1.shape[-3],img1.shape[-2],img1.shape[-1])
                img1 = img1.float()
                
                img2 = data['img2'].to(DEVICE)
                #img2 = img2.float()
                img2 = img2.view(-1,img2.shape[-3],img2.shape[-2],img2.shape[-1])
                img2 = img2.float()
                
                img3 = data['img3'].to(DEVICE)
                #img2 = img2.float()
                img3 = img3.view(-1,img2.shape[-3],img2.shape[-2],img2.shape[-1])
                img3 = img3.float()
                
                label2 = data['label2'].to(DEVICE).float()
                label2 = label2.view(-1)
                
                label3 = data['label3'].to(DEVICE).float()
                label3 = label3.view(-1)

                # zero the parameter gradients
                optimizer.zero_grad()

                distance = model.forward(img1,img2)
                distance2 = model.forward(img1,img3)
                #print("distance")
                #print(distance)
                #print("check the distance")
                #print(distance)
                #print("labels")
                #print(label1)

                loss1 = contrastive_loss(distance, label2)
                loss2 = contrastive_loss(distance2, label3)
                (loss1 + loss2).backward()
                #loss2.backward()
                optimizer.step()
                print("Curr LossL {}".format((loss1 + loss2).item()))
                # statistics(base) ironman@bigbox:~$ 
                #predictions = (torch.abs(distance - labels) < similarity_margin).int()
                running_loss += loss1.item()
                running_loss += loss2.item()
                #running_corrects += torch.sum(predictions)
                iteration += 1
                #if global_iter%10:
                    #print("Logging to tb")
                    #writer.add_scalar('Train/Loss', (loss1 + loss2).item(), global_iter)
                global_iter += 1    
                #if i %10 == 0 :
                    
                    #iteration +=10
                    #writer.add_scalar('Train/Loss', (loss1.item()+loss2.item())/2, iteration)
                    #counter.append(iteration_number)
                    #loss_history.append(loss_contrastive.item())
            #print("done") 
            #if tb_iter%2:
            #         print("Epoch number {}\n Current loss {}\n".format(epoch,(running_loss)/(2*len(train_dl.dataset)) ))
            #         writer.add_scalar('Train/Loss', (running_loss)/(2*len(train_dl.dataset)), iteration)
            #tb_iter = tb_iter+1
            BEST_MODEL_METRIC['train-loss'] = running_loss/(2*iteration) 
            writer.add_scalar('Train/Loss', running_loss/(2*iteration), epoch)
            #BEST_MODEL_METRIC['train-acc'] = running_corrects.double() / len(train_dl.dataset)

            logger.info('Training set: Average loss: {:.8f} \n'
                        .format(BEST_MODEL_METRIC['train-loss']))
            #logger.info('Training set: Average loss: {:.8f}\n'
                        #.format(BEST_MODEL_METRIC['train-loss']))
            #with torch.no_grad():
            #for i, (image1, image2, labels) in enumerate(testloader):
                #for data in testloader:
        
            #img1 = data['img1']
            #plt.imshow(image1)
            #img1 = image1.to(DEVICE)
            #img1 = img1.float()
            #img1 = img1.view(-1,img1.shape[-3],img1.shape[-2],img1.shape[-1])
            #img1 = img1.float()
            #output = model(img1)tonybrook-cse508-fall19.hotcrp.com/
            #print(output)

            BEST_MODEL_METRIC['test-loss']= test_model(model,test_dl)
            writer.add_scalar('Val/Loss', BEST_MODEL_METRIC['test-loss'], epoch)

            # checkpoint the best model
            if  BEST_MODEL_METRIC['test-loss'] < best_loss:
                best_loss = BEST_MODEL_METRIC['test-loss']
                print("going to save the model")

                logger.info('Saving the best model: {}'.format(best_loss))
                with open(CHECKPOINT_PATH, 'wb') as f:
                    print("the path where it is saved")
                    print(CHECKPOINT_PATH)
                    torch.save(model.state_dict(), f)
                

        time_elapsed = time.time() - since
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logger.info('Best Loss: {:8f}'.format(best_loss))

        # Load the best saved model.
        #with open(CHECKPOINT_PATH, 'rb') as f:tonybrook-cse508-fall19.hotcrp.com/
            #model.load_state_dict(torch.load(f))

     
        print("exception")
        

                   
        return model

model = train_sim_model(model, trainloader, valloader, optimizer, num_epochs = num_epochs)



