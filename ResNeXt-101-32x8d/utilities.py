################################################################################################
## IMPORT LIBRARIES ############################################################################
################################################################################################
## python3 -m visdom.server -port 8097
import torch
import torchvision
import torchvision.transforms as transforms
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from visdom import Visdom
import numpy as np
# import pywt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random 
import kornia
import cv2
import time
import matplotlib.pyplot as plt
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h
from skimage.feature import daisy
from skimage.feature import hog
from skimage import data, exposure
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")
from skimage import io  

##########################################################################
# aux functions  #########################################################
##########################################################################
def generate_motion_rnd(input_tensor_3d, gpu_device): 
    print('\n')
    print('......................................')
    print('.... generating motion artifacts .....')
    print('......................................')
    print('\n')

    labels = []   
    tran_x, tran_y = 0, 0 # set translations to zero 
    ## convert input to tensor and transfer to the gpu
    # input_tensor_3d = torch.from_numpy(input_data)
    # input_tensor_3d = input_tensor_3d.to(gpu_device)
    ####
    dims_ = (input_tensor_3d.size())
    output_ = torch.empty(dims_, dtype=torch.float)
    # start_time = time.time()
    
    for ii in range(0, dims_[2]):
        ############################################
        # translations    
        ############################################
        data = input_tensor_3d[:,:,ii]
        img_a = torch.roll(data, tran_x, dims=0)
        data = torch.roll(img_a, tran_y, dims=1)
        data = data.unsqueeze(0)
        data = data.unsqueeze(0)
        # print(data.size())
        aux_t_ = torch.empty( 1, 1, dims_[0], int(dims_[1]/2)+1, 2).to(gpu_device)#.cuda()
        
        
        rotation = np.random.uniform(0,5.0)# 10)
        labels.append(rotation)
        for kk in range(0,dims_[0]):

            alpha: float = rotation*random.randint(-1,1)#45.0  # in degrees
            angle: torch.tensor = torch.ones(1) * alpha
            
            data_warped = kornia.rotate(data.float(), angle.to(gpu_device))
            img_cax = torch.rfft(data_warped, 2)#.cuda()
            
            aux_t_[:,:,kk,:,:]=img_cax[:,:,kk,:,:]
            
        img_dax = torch.irfft(aux_t_, 2)#.cuda()
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print i/len(some_list)*100," percent complete         \r",
        print(np.round_(ii/dims_[2]*100,2)," percent complete         \r", end=" ")
        # print ii/dims_[2]*100," percent complete         \r",
        # output_[:,:,ii] = torch.clamp(img_dax[0,0,:,:dims_[1]], min=0.000001, max=1.0)    
        output_[:,:,ii] = torch.abs(img_dax[0,0,:,:dims_[1]]) 

        # convert back to numpy array

    output_ = np.array(output_)
    labels = np.asarray(labels)
    labels = np.reshape(labels, (dims_[2]))

    return output_, labels
###########################################


# def generate_motion_2d_torch(input_tensor_3d, rotation, tran_x, tran_y, device = torch.device("cuda:0")):
# def generate_motion_2d_torch(input_tensor_3d, rotation, tran_x, tran_y, device = torch.device("cuda:0")):
def generate_motion(input_tensor_3d, vect_mot, gpu_device): 
    print('\n')
    print('......................................')
    print('.... generating motion artifacts .....')
    print('......................................')
    print('\n')

    labels = []   
    tran_x, tran_y = 0, 0 # set translations to zero 
    ## convert input to tensor and transfer to the gpu
    # input_tensor_3d = torch.from_numpy(input_data)
    # input_tensor_3d = input_tensor_3d.to(gpu_device)
    ####
    dims_ = (input_tensor_3d.size())
    output_ = torch.empty(dims_, dtype=torch.float)
    # start_time = time.time()
    
    for ii in range(0, dims_[2]):
        ############################################
        # translations    
        ############################################
        data = input_tensor_3d[:,:,ii]
        img_a = torch.roll(data, tran_x, dims=0)
        data = torch.roll(img_a, tran_y, dims=1)
        data = data.unsqueeze(0)
        data = data.unsqueeze(0)
        # print(data.size())
        aux_t_ = torch.empty( 1, 1, dims_[0], int(dims_[1]/2)+1, 2).to(gpu_device)#.cuda()
        
        ind_ = np.random.randint(len(vect_mot), size=1)
        rotation = vect_mot[int(ind_)]
        labels.append(ind_)
        for kk in range(0,dims_[0]):

            alpha: float = rotation*random.randint(-1,1)#45.0  # in degrees
            angle: torch.tensor = torch.ones(1) * alpha
            
            data_warped = kornia.rotate(data.float(), angle.to(gpu_device))
            img_cax = torch.rfft(data_warped, 2)#.cuda()
            
            aux_t_[:,:,kk,:,:]=img_cax[:,:,kk,:,:]
            
        img_dax = torch.irfft(aux_t_, 2)#.cuda()
        # print("--- %s seconds ---" % (time.time() - start_time))
        # print i/len(some_list)*100," percent complete         \r",
        print(np.round_(ii/dims_[2]*100,2)," percent complete         \r", end=" ")
        # print ii/dims_[2]*100," percent complete         \r",
        # output_[:,:,ii] = torch.clamp(img_dax[0,0,:,:dims_[1]], min=0.000001, max=1.0)    
        output_[:,:,ii] = torch.abs(img_dax[0,0,:,:dims_[1]]) 

        # convert back to numpy array

    output_ = np.array(output_)
    labels = np.asarray(labels)
    labels = np.reshape(labels, (dims_[2]))

    return output_, labels

#########################################################################

def edge_generation(input_):
    empty_= np.zeros((input_.shape))
    for jj in range(0, input_.shape[2]):
        tmp_ = sobel(input_[:,:, jj])
        tmp_ = tmp_/tmp_.max()
        empty_[:,:,jj]=tmp_

    return empty_

#########################################################################


def isMultipleof5(n):       
    while ( n > 0 ): 
        n = n - 5
    if ( n == 0 ): 
        return 1
  
    return 0

##########################################################################

def adjust_learning_rate(optimizer, epoch, lrInitial, lrDecayNEpoch, lrDecayRate):
    """Sets the learning rate to the initial LR decayed by lrDecayRate every lrDecayNEpoch epochs"""

    lr = lrInitial * (lrDecayRate ** (epoch // lrDecayNEpoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##########################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

##########################################################################

class MyDataset(Dataset):
    
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        #print(len(self.labels))   
        return len(self.labels)   

    def __getitem__(self, idx):
        
        # CrossEntropyLoss
        label = (self.labels[idx])
        img = Image.fromarray(self.images[:,:,idx])
        
        if self.transform:
            img = self.transform(img)

        return img, label

class MyDatasetNolabel(Dataset):
    
    def __init__(self, images, transform=None):
        self.numbimg= images.shape[2]
        self.images = images
        self.transform = transform

    def __len__(self):  
        return self.numbimg#len(self.images)

    def __getitem__(self, idx):
        
        img = Image.fromarray(self.images[:,:,idx])
        
        if self.transform:
            img = self.transform(img)

        return img


#################################################################################


def index_subset(leng, num_samples):
    index_list_ = np.random.randint(leng, size=(num_samples))
    return index_list_

def create_subset(dataset, index_subset_):
    subset_ = np.zeros((dataset.shape[0], dataset.shape[1],len(index_subset_)))
    
    for kk in range(0, len(index_subset_)):
        subset_[:,:,kk]=dataset[:,:,index_subset_[kk]]
    
    return subset_ 

def create_subset_mot(dataset, datalabel,  index_subset_):
    subset_ = np.zeros((dataset.shape[0], dataset.shape[1],len(index_subset_)))
    subsetlabel_ = np.zeros((len(index_subset_)))
    for kk in range(0, len(index_subset_)):
        subset_[:,:,kk]=dataset[:,:,index_subset_[kk]]
        subsetlabel_[kk] = datalabel[index_subset_[kk]]
    
    return subset_ , subsetlabel_

############################################################

def decompose_image(input_img):
    # central mask:
    img_ = input_img
    central_mask_ = img_[int(img_.shape[0]/2)-int(img_.shape[0]/4):int(img_.shape[0]/2)+int(img_.shape[0]/4),
                     int(img_.shape[1]/2)-int(img_.shape[0]/4):int(img_.shape[1]/2)+int(img_.shape[0]/4)]

    outer_mask_ = np.vstack((img_[0:int(img_.shape[0]/4),:],img_[img_.shape[0]-int(img_.shape[0]/4):, :]))

    outer_maskc_ = np.hstack((img_[int(img_.shape[0]/4):int(img_.shape[0]/2)+int(img_.shape[0]/4), 0:int(img_.shape[0]/4)],
                          img_[int(img_.shape[0]/4):int(img_.shape[0]/2)+int(img_.shape[0]/4),img_.shape[1]-int(img_.shape[0]/4):]))


    outer_maska_, outer_maskb_ = outer_mask_[:,0:int(outer_mask_.shape[1]/2)], outer_mask_[:,int(outer_mask_.shape[1]/2):]

    return central_mask_, outer_maska_, outer_maskb_, outer_maskc_

############################################################

def to_daisy_img(A,B,C,D): ## size 128x128 for A,B,...
    X,  = daisy(A, step=90, radius=63, rings=3, histograms=6,orientations=8, visualize=False)
    Y,  = daisy(B, step=90, radius=63, rings=3, histograms=6,orientations=8, visualize=False)
    Z,  = daisy(C, step=90, radius=63, rings=3, histograms=6,orientations=8, visualize=False)
    T,  = daisy(D, step=90, radius=63, rings=3, histograms=6,orientations=8, visualize=False)
    tmp_ = np.concatenate((np.squeeze(X),np.squeeze(Y),np.squeeze(Z),np.squeeze(T)))
    tmp_ = tmp_[16:len(tmp_)-16]
    img_ = np.reshape(tmp_,(24,24))

    return img_

###########################################################

def to_hog_img(img_):   ## size 256x256 for img_
    fd = hog(img_, orientations=8, pixels_per_cell=(16, 16),
            cells_per_block=(1, 1), visualize=False, multichannel=False)

    fd = fd[12:len(fd)-11]
    fd = np.reshape(fd,(45,45))

    return fd

###########################################################

def data_to_daisy(img_):
    print('\n')
    print('......................................')
    print('.... daisy conversion ................')
    print('......................................')
    print('\n')
    test_daisy = np.zeros((24,24,img_.shape[2]))
    for ii in range(0, img_.shape[2]):
        A_,B_,C_,D_ = decompose_image(img_[:,:,ii])
        test_daisy[:,:,ii] = to_daisy_img(A_,B_,C_,D_)
        print(np.round_(ii/img_.shape[2]*100,2)," percent complete         \r", end=" ")

    return test_daisy

###########################################################

def data_to_hog(img_):
    print('\n')
    print('......................................')
    print('.... hog conversion ..................')
    print('......................................')
    print('\n')
    test_hog = np.zeros((45,45,img_.shape[2]))
    for ii in range(0, img_.shape[2]):
        test_hog[:,:,ii] = to_hog_img(img_[:,:,ii])
        print(np.round_(ii/img_.shape[2]*100,2)," percent complete         \r", end=" ")

    return test_hog


###########################################################

def features_extraction(A,B,C,D):
    # print('\n')
    # print('......................................')
    # print('.... features extraction .............')
    # print('......................................')
    # print('\n')
    AA, BB, CC, DD = np.abs(1-A), np.abs(1-B),np.abs(1-C),np.abs(1-D)
    features_ = [np.mean(A), np.std(A), skew(A.reshape((A.shape[0]*A.shape[1]))), kurtosis(A.reshape((A.shape[0]*A.shape[1]))),
                 np.mean(B), np.std(B), skew(B.reshape((B.shape[0]*B.shape[1]))), kurtosis(B.reshape((B.shape[0]*B.shape[1]))),
                 np.mean(C), np.std(C), skew(C.reshape((C.shape[0]*C.shape[1]))), kurtosis(C.reshape((C.shape[0]*C.shape[1]))),
                 np.mean(D), np.std(D), skew(D.reshape((D.shape[0]*D.shape[1]))), kurtosis(D.reshape((D.shape[0]*D.shape[1]))),

                 np.mean(AA), np.std(AA), skew(AA.reshape((AA.shape[0]*AA.shape[1]))), kurtosis(AA.reshape((AA.shape[0]*AA.shape[1]))),
                 np.mean(BB), np.std(BB), skew(BB.reshape((BB.shape[0]*BB.shape[1]))), kurtosis(BB.reshape((BB.shape[0]*BB.shape[1]))),
                 np.mean(CC), np.std(CC), skew(CC.reshape((CC.shape[0]*CC.shape[1]))), kurtosis(CC.reshape((CC.shape[0]*CC.shape[1]))),
                 np.mean(DD), np.std(DD), skew(DD.reshape((DD.shape[0]*DD.shape[1]))), kurtosis(DD.reshape((DD.shape[0]*DD.shape[1])))]
    

    return features_
