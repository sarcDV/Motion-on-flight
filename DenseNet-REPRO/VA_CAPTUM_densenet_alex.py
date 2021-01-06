import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
from numpy import newaxis
from utilities import adjust_learning_rate, AverageMeter, VisdomLinePlotter, DatasetMC

import h5py as h5
import warnings
warnings.filterwarnings("ignore")
import nibabel as nib
from PIL import Image

# --- torch ------ # 
import torch
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset

# --- captum ------ # 
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

# --- utilities and warnings ------ #
# from utilities import adjust_learning_rate, AverageMeter, VisdomLinePlotter, MyDataset
import warnings
warnings.filterwarnings("ignore")

# --- -.-.-.- ------ #
#To detect anomaly
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
batch_size_ = 30
##########################################################################
##########################################################################
#To detect anomaly
torch.autograd.set_detect_anomaly(True)
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
###########################################
# --- classes ------ ######################
###########################################

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

occlusion = Occlusion(net)
integrated_gradients = IntegratedGradients(net)
gradient_shap = GradientShap(net)
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
n_ = 0

test_img_  = np.zeros((448,448,60))
attr2_img_ = np.zeros((448,448,60))
attr3_img_ = np.zeros((448,448,60))
attr4_img_ = np.zeros((448,448,60))

with torch.no_grad():
    for data in test_loader:
        #
        if n_ > 1:
            break
        
        images, labels = data        
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for ii in range(0, batch_size_):
            print(classes[int(labels[ii].cpu())], classes[int(predicted[ii].cpu())])
            # print((counter+ii)+(counter*(batch_size_-1)))
            aa_ = ((counter+ii)+(counter*(batch_size_-1)))
            tmp_img = images[ii,0,:,:].cpu().detach().numpy()
            tmp_img = tmp_img/tmp_img.max()
            test_img_[:,:,aa_] = tmp_img

            img_1_= nib.Nifti1Image(test_img_, np.eye(4))
            nib.save(img_1_,'original_test_.nii.gz')

            # torch.Size([1, 1, 256, 256]) torch.Size([1])
            """
            tmp_img = images[ii,0,:,:].unsqueeze(0).unsqueeze(0)
            
            attributions_occ2 = occlusion.attribute(tmp_img.cuda(),
                                        strides = (1, 8, 8),
                                        target=labels[ii].unsqueeze(0).long(),
                                        sliding_window_shapes=(1, 15, 15),
                                        baselines=0)
            
            attributions_ig = integrated_gradients.attribute(tmp_img.cuda(), 
                                                target=labels[ii].long(), 
                                                n_steps=20)

            # Defining baseline distribution of images
            rand_img_dist = torch.cat([tmp_img.cuda()*0.5, tmp_img.cuda() * 1])
                
            attributions_gs = gradient_shap.attribute(tmp_img.cuda(),
                                                n_samples=8,
                                                stdevs=0.0001,
                                                baselines=rand_img_dist,
                                                target=labels[ii].long())
                
            
            
            attr2_img_[:,:,aa_] = attributions_occ2.squeeze().cpu().detach().numpy()
            # attr3_img_[:,:,aa_] = attributions_ig.squeeze().cpu().detach().numpy()
            attr4_img_[:,:,aa_] = attributions_gs.squeeze().cpu().detach().numpy()

            ## renormalize maps:
            # attr2_img_[:,:,aa_] = np.abs(attr2_img_[:,:,aa_])/np.abs(attr2_img_[:,:,aa_]).max()
            # attr3_img_[:,:,aa_] = np.abs(attr3_img_[:,:,aa_])/np.abs(attr3_img_[:,:,aa_]).max()
            # attr4_img_[:,:,aa_] = np.abs(attr4_img_[:,:,aa_])/np.abs(attr4_img_[:,:,aa_]).max()

            img_2_= nib.Nifti1Image(attr2_img_, np.eye(4))
            nib.save(img_2_,'attributes2.nii.gz')

            # img_3_= nib.Nifti1Image(attr3_img_, np.eye(4))
            # nib.save(img_3_,'attributes3.nii.gz')

            img_4_= nib.Nifti1Image(attr4_img_, np.eye(4))
            nib.save(img_4_,'attributes4.nii.gz')
            """
            ####################################################
            # ------------------------------------------------ #
            ####################################################

        counter = counter+1    
        n_ = (n_+1)
        

###########################################
### Occlusion - Large sliding window  #####
### Gradient-based attribution ############
###########################################
"""
occlusion = Occlusion(net)
integrated_gradients = IntegratedGradients(net)
gradient_shap = GradientShap(net)

attr1_img_ = np.zeros((testdata.shape))
attr2_img_ = np.zeros((testdata.shape))
attr3_img_ = np.zeros((testdata.shape))
attr4_img_ = np.zeros((testdata.shape))
##
correct = 0
total = 0
counter = 0
for data in test_loader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    ####################################################
    # ------------------------------------------------ #
    ####################################################
        
    attributions_occ = occlusion.attribute(images,
        							strides = (1, 50, 50),
                                    target=labels.long(),
                                    sliding_window_shapes=(1, 60, 60),
                                    baselines=0)
    attributions_occ2 = occlusion.attribute(images,
        							strides = (1, 8, 8),
                                    target=labels.long(),
                                    sliding_window_shapes=(1, 15, 15),
                                    baselines=0)

    attributions_ig = integrated_gradients.attribute(images, 
        							target=labels.long(), 
        							n_steps=20)
        
    # Defining baseline distribution of images
    rand_img_dist = torch.cat([images*0.5, images * 1])
    
    attributions_gs = gradient_shap.attribute(images,
                                    n_samples=8,
                                    stdevs=0.0001,
                                    baselines=rand_img_dist,
                                    target=labels.long())
		
    attr1_img_[:,:,counter] = attributions_occ.squeeze().cpu().detach().numpy()
    attr2_img_[:,:,counter] = attributions_occ2.squeeze().cpu().detach().numpy()
    attr3_img_[:,:,counter] = attributions_ig.squeeze().cpu().detach().numpy()
    attr4_img_[:,:,counter] = attributions_gs.squeeze().cpu().detach().numpy()

    ## renormalize maps:
    
    attr1_img_[:,:,counter] = np.abs(attr1_img_[:,:,counter])/np.abs(attr1_img_[:,:,counter]).max()
    attr2_img_[:,:,counter] = np.abs(attr2_img_[:,:,counter])/np.abs(attr2_img_[:,:,counter]).max()
    attr3_img_[:,:,counter] = np.abs(attr3_img_[:,:,counter])/np.abs(attr3_img_[:,:,counter]).max()
    attr4_img_[:,:,counter] = np.abs(attr4_img_[:,:,counter])/np.abs(attr4_img_[:,:,counter]).max()

    img_1_= nib.Nifti1Image(attr1_img_, np.eye(4))
    nib.save(img_1_,'attributes1.nii.gz')

    img_2_= nib.Nifti1Image(attr2_img_, np.eye(4))
    nib.save(img_2_,'attributes2.nii.gz')

    img_3_= nib.Nifti1Image(attr3_img_, np.eye(4))
    nib.save(img_3_,'attributes3.nii.gz')

    img_4_= nib.Nifti1Image(attr4_img_, np.eye(4))
    nib.save(img_4_,'attributes4.nii.gz')

    ####################################################
    # ------------------------------------------------ #
    ####################################################
    
    counter = counter + 1
    print(np.round_(counter/testdata.shape[2]*100,2)," percent complete         \r", end=" ")
"""