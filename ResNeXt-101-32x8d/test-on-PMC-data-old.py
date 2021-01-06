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
from utilities import adjust_learning_rate, AverageMeter, VisdomLinePlotter, \
            generate_motion_rnd, MyDataset, MyDatasetNolabel, index_subset, create_subset
from skimage import io 
import warnings
warnings.filterwarnings("ignore")
import nibabel as nib
from skimage.filters import gabor_kernel, gaussian
from skimage import io
from skimage import exposure
import cv2
#################################################################

def segment_nlevels(img, levels):
    b = np.zeros(img.shape)
    c = np.zeros(img.shape)
    d = np.zeros(img.shape)

    dist_ = np.linspace(0.,1., levels)

    for ii in range(0, len(dist_)-1):
        b[img>=dist_[ii]]=ii+1
        c[img<dist_[ii]]=ii+1
        d = (b*c)/(ii+1)

    return d


def remove_upper_outliers(img, levels):
    b = img
    b[img>=levels]=levels
    b = b/b.max()

    return b

def remove_lower_outliers(img, levels):
    b = img
    b[img<=levels]=levels
    b = b/b.max()

    return b

def crop_square(img):
    dims_ = np.array([img.shape[0], img.shape[1]])
    min_dim_ = dims_.min()

    tmp_ = np.zeros((min_dim_, min_dim_, img.shape[2]))

    for kk in range(0, img.shape[2]):
        tmp_[:,:,kk]=img[int(img.shape[0]/2)-int(min_dim_/2):int(img.shape[0]/2)+int(min_dim_/2) ,
                         int(img.shape[1]/2)-int(min_dim_/2):int(img.shape[1]/2)+int(min_dim_/2) ,kk]    


    return tmp_

def resize_to_256x256(img): 
    n = 256
    tmp_ = np.zeros((n,n,img.shape[2]))
    dim = (n,n)
    for kk in range(0, img.shape[2]):
        # tmp_[:,:,kk] = cv2.resize((img[:,:,kk]*255).astype(np.uint8), dim, interpolation = cv2.INTER_AREA)
        tmp_[:,:,kk] = cv2.resize(img[:,:,kk], dim, interpolation = cv2.INTER_AREA)
    
    return tmp_   

#####################################################
# import trained model ##############################
#####################################################
device = torch.device("cuda:0")
# print(device)

import torchvision.models as models        

PATH = './checkpoints-test/resnext101_32x8d-05-classification-best.pth'
net = models.resnext101_32x8d()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
net.load_state_dict(torch.load(PATH))
net.to(device)


##############################################################
test_transform = transforms.Compose([transforms.ToTensor()])
batch_size_ = 1

def classify_iqa(testset):
    testset = MyDatasetNolabel(testdata, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_, shuffle=False)

    y_pred = []

    ##
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images = data
            images = images.to(device)
            # print(images.dtype, labels.dtype)
            outputs = net(images)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred.append(np.array(predicted.cpu()))   

    y_pred = np.asarray(y_pred)
    # y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[1],1))    
    y_pred = 10-y_pred

    #print(np.mean(y_pred), np.std(y_pred))
    return np.mean(y_pred), np.std(y_pred)
#########################################################################################
#########################################################################################
#########################################################################################

## T1 OFF example: xx##_T1_OFF.nii
subjects=['au70','dl43','dn20','dp20','iy25','kc73','me21','mf79',
          'nv85','pa30','qg37','sc17','um68','ut70','vk04','vq83',
          'ws31','ww25','xi27','xx54','yv98']

modality = 'ON'  # 'OFF' or 'ON'
#sbj_ = 10
valori_medi_ = []
deviazioni_  = []
for kk in range(0, len(subjects)):
    #########################################################
    ########## for T1-w #####################################
    #########################################################
    """
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/T1/'+str(subjects[kk])+'_T1_'+str(modality)+'.nii')).get_fdata()
    # if T1 get subset:
    testdata = testdata[:,:,80:336]
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    ## remove upper outliers
    testdata = remove_upper_outliers(testdata, 0.35)
    testdata = testdata/testdata.max()
    testdata = resize_to_256x256(testdata)
    """
    #########################################################
    ########## for T2-w #####################################
    #########################################################
    """
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/T2/'+str(subjects[kk])+'_T2_'+str(modality)+'.nii')).get_fdata()
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    testdata = crop_square(testdata)
    testdata = resize_to_256x256(testdata)
    """
    #########################################################
    ########## for PD-w #####################################
    #########################################################
    """
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/PD/'+str(subjects[kk])+'_PD_'+str(modality)+'.nii')).get_fdata()
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    testdata = crop_square(testdata)
    testdata = resize_to_256x256(testdata)
    """
    #########################################################
    ########## for T2*-w 05 #################################
    #########################################################
    """
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/T2star-05/'+str(subjects[kk])+'_05_'+str(modality)+'.nii')).get_fdata()
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    testdata = crop_square(testdata)
    testdata = resize_to_256x256(testdata)
    """
    #########################################################
    ########## for T2*-w 035 #################################
    #########################################################
    """
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/T2star-035/'+str(subjects[kk])+'_035_'+str(modality)+'.nii')).get_fdata()
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    testdata = crop_square(testdata)
    testdata = resize_to_256x256(testdata)
    """
    #########################################################
    ########## for T2*-w 025 #################################
    #########################################################
    
    testdata =  (nib.load('./../../PMC-data-7T/'+str(modality)+'/T2star-025/'+str(subjects[kk])+'_025_'+str(modality)+'.nii')).get_fdata()
    ## normalize 0-1:
    testdata = testdata/testdata.max()
    testdata = crop_square(testdata)
    testdata = resize_to_256x256(testdata)
    
    #########################################################
    ## classification #######################################
    #########################################################
    media_, deviazione_ = classify_iqa(testdata)
    # print(str(subjects[kk]),' T1 ' ,modality, media_, deviazione_)
    # print(str(subjects[kk]),' T2 ' ,modality, media_, deviazione_)
    # print(str(subjects[kk]),' PD ' ,modality, media_, deviazione_)
    # print(str(subjects[kk]),' T2*-05 ' ,modality, media_, deviazione_)
    # print(str(subjects[kk]),' T2*-035 ' ,modality, media_, deviazione_)
    print(str(subjects[kk]),' T2*-025 ' ,modality, media_, deviazione_)
    
    valori_medi_.append(media_)
    deviazioni_.append(deviazione_)


valori_medi_ = np.asarray(valori_medi_)
deviazioni_ = np.asarray(deviazioni_)

print(np.mean(valori_medi_), np.std(valori_medi_))
