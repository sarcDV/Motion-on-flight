################################################################################################
## IMPORT LIBRARIES ############################################################################
################################################################################################
## python3 -m visdom.server -port 8097

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

PATH = './checkpoints-test/DenseNet-161-3-classes-best-latest.pth'

import torchvision.models as models
net = models.densenet161()
net.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.classifier = nn.Linear(in_features=2208, out_features=33, bias=True)
net.load_state_dict(torch.load(PATH))
net.cuda() #to(device)
################################################################################################
################################################################################################
################################################################################################
testset = DatasetMC(f, train_=False, val_=False, test_=True,transform=None)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_, shuffle=True)
ntest = len(testset) 
print(ntest)
################################################################################################
################################################################################################
################################################################################################
correct = 0
total = 0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
#------
y_true = []
y_pred = []
#-----
counter=0
with torch.no_grad():
    for data in test_loader:
        #
        counter=counter+1
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print(labels, predicted)
        #-------------
        y_true.append(np.array(labels.cpu())) 
        y_pred.append(np.array(predicted.cpu()))  
        #------------- 
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
        c = (predicted == labels.long()).squeeze()
        cc_ = int(np.array(labels.size()))
        for i in range(cc_):
            label = labels[i]
            class_correct[label.int()] += c[i].item()
            class_total[label.int()] += 1

        print('...processing group:  ',counter, images.size(), labels.size(),'  temporary accuracy: ', 100*correct/total)
 
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

np.save('Confusion-matrix-densenet-3-classes.npy',  (y_true, y_pred))


print('Accuracy of the network on the '+str(ntest)+' test images: %d %%' % (100 * correct / total))


for i in range(0,len(classes)):
    # print(classes[i], class_correct[i], class_total[i])
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# print(ntest)

## for confusion matrix 
## from sklearn.metrics import confusion_matrix 
# y_true = []
# y_pred = []

##
"""
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        # print(images.dtype, labels.dtype)
        outputs = net(images)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        #print(np.array(predicted.cpu()), np.array(labels.cpu()))
        #y_true.append(np.array(labels.cpu())) 
        #y_pred.append(np.array(predicted.cpu()))   

        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
"""

#y_true = np.asarray(y_true)
#y_pred = np.asarray(y_pred)

## np.save('Confusion-matrix-densenet161-05-nii.npy',  (y_true, y_pred))
