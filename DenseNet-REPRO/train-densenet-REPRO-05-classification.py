################################################################################################
## IMPORT LIBRARIES ############################################################################
################################################################################################
## python3 -m visdom.server -port 8097

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from visdom import Visdom
# import pywt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from utilities import adjust_learning_rate, AverageMeter, VisdomLinePlotter, DatasetMC

import h5py as h5
import warnings
warnings.filterwarnings("ignore")
import nibabel as nib
#To detect anomaly
torch.autograd.set_detect_anomaly(True)

#To make the model deterministic
torch.manual_seed(1701)
np.random.seed(1701)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#########################################################################
#########################################################################
global plotter
plotter = VisdomLinePlotter(env_name='Loss Plots')
#########################################################################
##### LOAD DATA #########################################################
#########################################################################
filename = "./../source-images/Repro-Structural-Data.h5"
f = h5.File(filename, "r")
##########################################################################
##########################################################################
# device = torch.device("cuda:1")

#########################################################################
#########################################################################

batch_size_ = 30
##########################################################################
##########################################################################
#To detect anomaly
torch.autograd.set_detect_anomaly(True)
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
##########################################################################
##########################################################################
# classes = {0:'Good', 1:'Sufficient', 2:'Bad'}
classes = {0:'T1-axial-1', 1:'T1-axial-2', 2:'T1-axial-3', # 
           3:'T1-coronal-1', 4:'T1-coronal-2', 5:'T1-coronal-3', #
           6:'T1-sagittal-1', 7:'T1-sagittal-2', 8:'T1-sagittal-3', # 

           9:'T2-axial-1', 10:'T2-axial-2', 11:'T2-axial-3', # 
           12:'T2-coronal-1', 13:'T2-coronal-2', 14:'T2-coronal-3', # 
           15:'T2-sagittal-1', 16:'T2-sagittal-2', 17:'T2-sagittal-3', # 

           18:'PD-axial-1', 19:'PD-axial-2', 20:'PD-axial-3', # 
           21:'PD-coronal-1', 22:'PD-coronal-2', 23:'PD-coronal-3', #

           24:'Flair-axial-1', 25:'Flair-axial-2', 26:'Flair-axial-3', # 
           27:'Flair-coronal-1', 28:'Flair-coronal-2', 29:'Flair-coronal-3', # 
           30:'Flair-sagittal-1', 31:'Flair-sagittal-2', 32:'Flair-sagittal-3', # 
           }

"""
classes = {0:'T1-axial-1', 1:'T1-axial-2', 2:'T1-axial-3', 3:'T1-axial-4', 4:'T1-axial-5',
           5:'T1-coronal-1', 6:'T1-coronal-2', 7:'T1-coronal-3', 8:'T1-coronal-4', 9:'T1-coronal-5',
           10:'T1-sagittal-1', 11:'T1-sagittal-2', 12:'T1-sagittal-3', 13:'T1-sagittal-4', 14:'T1-sagittal-5',

           15:'T2-axial-1', 16:'T2-axial-2', 17:'T2-axial-3', 18:'T2-axial-4', 19:'T2-axial-5',
           20:'T2-coronal-1', 21:'T2-coronal-2', 22:'T2-coronal-3', 23:'T2-coronal-4', 24:'T2-coronal-5',
           25:'T2-sagittal-1', 26:'T2-sagittal-2', 27:'T2-sagittal-3', 28:'T2-sagittal-4', 29:'T2-sagittal-5',

           30:'PD-axial-1', 31:'PD-axial-2', 32:'PD-axial-3', 33:'PD-axial-4', 34:'PD-axial-5',
           35:'PD-coronal-1', 36:'PD-coronal-2', 37:'PD-coronal-3', 38:'PD-coronal-4', 39:'PD-coronal-5',

           40:'Flair-axial-1', 41:'Flair-axial-2', 42:'Flair-axial-3', 43:'Flair-axial-4', 44:'Flair-axial-5',
           45:'Flair-coronal-1', 46:'Flair-coronal-2', 47:'Flair-coronal-3', 48:'Flair-coronal-4', 49:'Flair-coronal-5',
           50:'Flair-sagittal-1', 51:'Flair-sagittal-2', 52:'Flair-sagittal-3', 53:'Flair-sagittal-4', 54:'Flair-sagittal-5'
           }
"""
################################################################################################
################################################################################################
################################################################################################

print('\n ... loading DenseNet-161 ... \n')

import torchvision.models as models
net = models.densenet161()
net.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.classifier = nn.Linear(in_features=2208, out_features=33, bias=True)
net.cuda() #to(device)
################################################################################################
################################################################################################
################################################################################################

initialLearningRate = 0.001
lrDecayNEpoch = 10
lrDecayRate = 0.1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initialLearningRate, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=initialLearningRate)

numb_epochs = 1000

# valset = DatasetMC(valdata, transform=test_transform)
valset = DatasetMC(f, train_=False, val_=True, test_=False,transform=None)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_, shuffle=True)
nval = len(valset) 

#print('\n')
print('   ... starting training ... \n')
print('   Batch size: ', batch_size_,'\n')
print('   Total number epochs: ', numb_epochs, '\n')

#print('\n')

best_val_loss = 999999999.
for epoch in range(numb_epochs):
    # adjust_learning_rate(optimizer, epoch, initialLearningRate, lrDecayNEpoch, lrDecayRate)
    # trainset = DatasetMC(traindata, transform=train_transform)
    trainset = DatasetMC(f, train_=True, val_=False, test_=False, transform=None)
    ntrain = len(trainset) #print(len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, shuffle=True)

    #############################################################################
    # Training stage: ###########################################################
    #############################################################################
    net.train()
    running_loss = 0.0
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.cuda(), labels.cuda()# imgs.to(device), labels.to(device)
        imgs_train.requires_grad = True
        optimizer.zero_grad()  # 
        output_train = net(imgs_train.float())
        # print(output_train, labels_train.long())
        loss = criterion(output_train, labels_train.long().squeeze()) # test for CrossEntropyLoss
        loss.backward() #
        optimizer.step() #
        # print statistics
        running_loss += loss.item()
        if idx % 5 == 4:    
            print('[%d, %14d] Training loss: %.9f' % (epoch + 1, idx + 1, running_loss / 5))
            x_trainp = (idx+(epoch*ntrain/batch_size_))/(ntrain/batch_size_)
            y_trainp = loss.item()
            plotter.plot('loss', 'train', 'Loss Plots DenseNet-161 3 classes of motion',x_trainp, y_trainp)
            running_loss = 0.0


    #############################################################################
    # calculate validation loss #################################################
    #############################################################################
    net.eval()
    mean_val_loss = 0.0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_val, labels_val = imgs.cuda(), labels.cuda() #to(device), labels.to(device)
            val_outputs = net(imgs_val.float())
            val_loss = criterion(val_outputs, labels_val.long().squeeze())
            mean_val_loss += val_loss.item()
            if idx % 5 == 4:    
                print('[%d, %14d] Validation loss: %.9f' % (epoch + 1, idx + 1, mean_val_loss / 5))
                x_valp = (idx+(epoch*nval/batch_size_))/(nval/batch_size_)
                y_valp = val_loss.item()
                plotter.plot('loss', 'validation', 'Loss Plots DenseNet-161 3 classes of motion',x_valp, y_valp)
                mean_val_loss /= len(labels)

    #############################################################################
    #############################################################################
    #############################################################################
    print('Mean-val-loss:  '+str(mean_val_loss)+', Best-val-loss: '+str(best_val_loss))    
    # save model if it has a better validation loss than all before
    if mean_val_loss < best_val_loss:

        print("Saving best checkpoint:  "+str(epoch))
        path_ = './checkpoints-test/DenseNet-161-3-classes-best-latest.pth'
        count_ = format(epoch+1, '04d')
        path_tmp_ = './checkpoints-test/DenseNet-161-3-classes-best-'+str(count_)+'.pth'
        torch.save(net.state_dict(), path_tmp_)
        torch.save(net.state_dict(), path_)
        best_val_loss = mean_val_loss

print('Finished Training')

PATH = './checkpoints-test/DenseNet-161-3-classes-'+str(numb_epochs)+'-latest.pth'
torch.save(net.state_dict(), PATH)


# print(torch.nn.functional.softmax(output_train[0], dim=0))
# print( labels_train.long().size()  , torch.max(output_train[0], dim=1).indices.size() )
# output_train = torch.unsqueeze( torch.max(output_train[0], dim=1).indices, dim=1 )
# print(output_train.double().size(), labels_train.long().size())
# output_train =  torch.max(output_train[0], dim=1).indices #torch.unsqueeze( torch.max(output_train[0], dim=1).indices, dim=1 )
# print(output_train, labels_train.long().view(batch_size_))
