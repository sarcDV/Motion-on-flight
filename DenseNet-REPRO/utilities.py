################################################################################################
## IMPORT LIBRARIES ############################################################################
################################################################################################
## python3 -m visdom.server -port 8097
import torch
import torchvision
import torchvision.transforms as transforms
import nibabel as nib
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
from PIL import Image
from visdom import Visdom
import numpy as np
# import pywt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random 
import cv2
import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from skimage import io  

##########################################################################
# aux functions  #########################################################
##########################################################################

def adjust_learning_rate(optimizer, epoch, lrInitial, lrDecayNEpoch, lrDecayRate):
    """Sets the learning rate to the initial LR decayed by lrDecayRate every lrDecayNEpoch epochs"""

    lr = lrInitial * (lrDecayRate ** (epoch // lrDecayNEpoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

##########################################################################
##########################################################################
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

class VisdomImageShow(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def showimages(self, var_name, split_name, title_name, img):
        self.plots[var_name]= self.viz.image(img,
            opts={'title': var_name, 'caption': 'Image'})
        #self.viz.image(predictedimg, title=title_name)
        #self.viz.image(groundtruthimg, title=title_name)
##########################################################################
##########################################################################
##########################################################################
def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype="float64")

def generate_2d_motion_rnd(input_2d_img_): #, gpu_device):
    # device=gpu_device
    input_tensor_2d = torch.from_numpy(input_2d_img_).cuda()
    labels = []   
    tran_x, tran_y = 0, 0 # set translations to zero 
    dims_ = (input_tensor_2d.size())
    output_ = torch.empty(dims_, dtype=torch.float)
    ############################################
    # translations #############################
    ############################################
    data = input_tensor_2d
    # tran_x = int(np.round(np.random.uniform(-20.0,20.0)))
    # tran_y = int(np.round(np.random.uniform(-20.0,20.0)))
    # img_a = torch.roll(data, tran_x, dims=0)
    # data = torch.roll(img_a, tran_y, dims=1)
    # shifted_ = data.cpu().detach().numpy()
    data = data.unsqueeze(0)
    data = data.unsqueeze(0)
    ############################################
    # rotations ################################   
    ############################################
    aux_t_ = torch.empty( 1, 1, dims_[0], int(dims_[1]/2)+1, 2).cuda()#to(gpu_device)#.cuda()
    ############################################
    class_ = np.random.randint(5, size=1)
    
    if class_ == 0:
        rotation = np.random.uniform(0,0.25)
    elif class_ == 1:
        rotation = np.random.uniform(1,1.25)
    elif class_ == 2:
        rotation = np.random.uniform(2,2.25)
    elif class_ == 3:
        rotation = np.random.uniform(3,3.25)
    else:
        rotation = np.random.uniform(4,5)

    # print(class_, rotation)
    #rotation = np.random.uniform(0,5.0) #10)
    labels.append(rotation)
    #######################################################
    for kk in range(0,dims_[0]):

        angle = np.deg2rad(rotation*random.randint(-1,1))
        rotm = rotz(angle)[:2, :].reshape(2, 3, 1)
        rotm = torch.FloatTensor(rotm).permute(2, 0, 1).repeat(data.shape[0], 1, 1).cuda()
        affine_grid = F.affine_grid(rotm, data.size()).cuda() #to(device)

        with torch.no_grad():
            x_r = F.grid_sample(data.float().cuda(), affine_grid.cuda()) #to(device), affine_grid.to(device))
            img_cax = torch.rfft(x_r, 2)
            aux_t_[:,:,kk,:,:]=img_cax[:,:,kk,:,:]

    # print(aux_t_.size())
    img_dax = torch.irfft(aux_t_, 2)#.cuda()
    slice_ = torch.abs(img_dax[0,0,:,:dims_[1]])/torch.max(torch.abs(img_dax[0,0,:,:dims_[1]]))
    # convert back to numpy array
    # output_ =slice_.cpu().detach().numpy()
    # leave as tensor and normalize!
    output_ = slice_/slice_.max()
    ######################################
    # output_ = output_/output_.max()
    # shifted_ = shifted_/shifted_.max()
    ######################################
    labels = np.asarray(class_)

    return output_, labels

def generate_motion_3classes(input_2d_img_, rotation, tran_x, tran_y, label): 
    """ generate motion artifcats for 2d mr image: """
    input_tensor_2d = torch.from_numpy(input_2d_img_).cuda()
    labels = []   
    dims_ = (input_tensor_2d.size())
    output_ = torch.empty(dims_, dtype=torch.float)
    data = input_tensor_2d
    aux_t_ = torch.empty( 1, 1, dims_[0], int(dims_[1]/2)+1, 2).cuda()
    data = data.unsqueeze(0)
    data = data.unsqueeze(0)
    #######################################
    #if (label == 3) and ((tran_x+tran_y)==0):
    #    tran_x = 1
    #######################################
    for kk in range(0,dims_[0]):
        img_a = torch.roll(data, tran_x*random.randint(-1, 1), dims=2)
        data = torch.roll(img_a, tran_y*random.randint(-1, 1), dims=3)
        angle = np.deg2rad(rotation*random.randint(-1,1))
        rotm = rotz(angle)[:2, :].reshape(2, 3, 1)
        rotm = torch.FloatTensor(rotm).permute(2, 0, 1).repeat(data.shape[0], 1, 1).cuda()
        affine_grid = F.affine_grid(rotm, data.size()).cuda()
        with torch.no_grad():
            x_r = F.grid_sample(data.float().cuda(), affine_grid.cuda())
            img_cax = torch.rfft(x_r, 2)
            aux_t_[:,:,kk,:,:]=img_cax[:,:,kk,:,:]
    # ifft2 of the corrupted image: 
    img_dax = torch.irfft(aux_t_, 2)
    # normalization [0,1]:
    slice_ = torch.abs(img_dax[0,0,:,:dims_[1]])/torch.max(torch.abs(img_dax[0,0,:,:dims_[1]]))
    output_ = slice_/slice_.max()
    ## class:
    # print(tran_x, tran_y, rotation, label)
    return output_, label-1 

#############################

def rnd_slice(nslices_, percent_=0.2):
    nmin_ = (np.ceil(nslices_*percent_))
    nmax_ = (nslices_ - np.ceil(nslices_*percent_))
    slice_ = int(np.random.randint(low=nmin_, high=nmax_, size=1))

    return slice_, int(nmin_), int(nmax_)
#############################

def patch_to_center(input_img, sqsize=448): # modify this function to work directly withe tensor!!!
    empty_ = np.zeros((sqsize, sqsize))
    imgs_ = input_img.shape

    empty_[int(sqsize/2)-int(imgs_[0]/2): int(sqsize/2)+int(imgs_[0]/2), 
           int(sqsize/2)-int(imgs_[1]/2): int(sqsize/2)+int(imgs_[1]/2)] = input_img

    return empty_

def patch_to_center_tensor(input_, sqsize=448):
    empty_ = torch.zeros(sqsize, sqsize)
    dim_ = input_.size()

    empty_[int(sqsize/2)-int(dim_[0]/2): int(sqsize/2)+int(dim_[0]/2), 
           int(sqsize/2)-int(dim_[1]/2): int(sqsize/2)+int(dim_[1]/2)] = input_
    
    b=torch.unsqueeze(empty_,0)
    
    return b 


##########################################################################
##########################################################################
##########################################################################


class DatasetMC(Dataset):
    def __init__(self, h5file, train_=False, val_=False, test_=False, transform=None, 
    contrasts_=None, per_slice=0.2, datadict_csv=None):
    # def __init__(self, images, transform=None):
        
        self.images = h5file
        self.train_ = train_
        self.val_ = val_
        self.test_ = test_
        self.transform = transform

        if datadict_csv is None:
            if contrasts_ is None:
                contrasts_ =  ['T1-axial','T1-coronal','T1-sagittal',
                            'T2-axial','T2-coronal','T2-sagittal',
                            'PD-axial','PD-coronal',
                            'Flair-axial','Flair-coronal','Flair-sagittal']

            s = {}
            s['contrast'] = []
            s['file'] = []
            s['slice'] = []
            for (cin, contrast) in enumerate(contrasts_):
                data_= self.images.get(contrast)
                nfiles_ = len(list(data_)) 
                if train_:
                    low=0
                    high=np.ceil(nfiles_*0.7)
                    csvtype = 'train'
                elif val_:
                    low=np.ceil(nfiles_*0.7)
                    high=np.ceil(nfiles_*0.85)
                    csvtype = 'val'
                elif test_:
                    low=np.ceil(nfiles_*0.85)
                    high=nfiles_
                    csvtype = 'test'
                datalist = list(data_)
                for datumID in range(int(low), int(high)):
                    datum = datalist[datumID]
                    # print(datum)
                    d = data_.get(datum)
                    nslice = d.shape[-1]
                    _, minslc, maxslc = rnd_slice(nslice, percent_=per_slice)
                    for i in range(minslc, maxslc):
                        s['contrast'].append(cin)
                        s['file'].append(datum)
                        s['slice'].append(i)
            df = pd.DataFrame.from_dict(s)
            df.to_csv("datadict_"+csvtype+"_.csv")
        
        else:
            df = pd.read_csv(datadict_csv)

        self.df = df
        self.contrasts_ = contrasts_



    def __len__(self):  
        return len(self.df)  #len(self.images)

    def __getitem__(self, idx):
        ###########################################
        ### input #################################
        ###########################################
        
        datum = self.df.iloc[idx]
        img_ = self.images.get(self.contrasts_[datum['contrast']]).get(datum['file'])
        a = img_[:,:,datum['slice']]
        rnd_class = random.randint(0, 2)
        if rnd_class == 0:
            b, label = generate_motion_3classes(a, rotation = np.random.uniform(0,0.25), tran_x = 0, tran_y =0, label=1)
        elif rnd_class == 1:
            b, label = generate_motion_3classes(a, rotation = np.random.uniform(0.25,1.0), tran_x = 0, tran_y =0, label=2)
        else:
            tran_x = random.randint(0,1)
            tran_y = random.randint(0,1)
            if ((tran_x+tran_y)==0):
                tran_x = 1
            b, label = generate_motion_3classes(a, rotation = np.random.uniform(0, 5.0), tran_x =tran_x, tran_y =tran_y, label=3)

        # b, label =  generate_2d_motion_rnd(a) #, gpu_device="cuda:0")
        # img = patch_to_center(b)
        img = patch_to_center_tensor(b)
        label= label + (datum['contrast']*3)  #5) only 3 classes of motion corruption
        """
        b, label =  generate_2d_motion_rnd(a) #, gpu_device="cuda:1")
        # img = patch_to_center(b)
        img = patch_to_center_tensor(b)
        label= label + (datum['contrast']*5)
        """

        if self.transform:
            img = self.transform(img)
        
        return img, label

        """
        sel_cont_ = np.random.randint(len(contrasts_), size=1)
        data_= (self.images).get(contrasts_[int(sel_cont_)]) # f.get(contrasts_[int(sel_cont_)])
        files_, nfiles_ = list(data_), len(list(data_)) 

        if self.train_ == True:
            rndvol_ = np.random.randint(np.ceil(nfiles_*0.7), size=1)
        elif self.val_ == True:
            rndvol_ = np.random.randint(low=np.ceil(nfiles_*0.7), high=np.ceil(nfiles_*0.85), size=1)
        elif self.test_ == True:
            rndvol_ = np.random.randint(low=np.ceil(nfiles_*0.85), high=nfiles_, size=1)
        
        img_ = data_.get(files_[int(rndvol_)])
        slice_ = rnd_slice(img_.shape[2], 0.2)
        a = img_[:,:,slice_]
        b, label =  generate_2d_motion_rnd(a, gpu_device="cuda:0")
        img = patch_to_center(b)
        label= label + (sel_cont_*5)
        """


        
