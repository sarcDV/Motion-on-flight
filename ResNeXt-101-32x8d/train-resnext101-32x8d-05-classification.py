################################################################################################
## IMPORT LIBRARIES ############################################################################
################################################################################################
## python3 -m visdom.server -port 8097
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
			generate_motion_rnd, MyDataset, index_subset, create_subset_mot 
from skimage import io 
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
###############################
##### LOAD DATA ###############
###############################

# traindata = np.loadtxt('./data-features-extracted/train/train_features_0001.txt')
traindata = np.zeros((256,256,1))
trainlabel = np.zeros((1))
for kk in range(0,50):
    tmp_str_ = format(kk+1, '04d')
    print('....reading file ....  '+str(tmp_str_))
    tmp_data_ = (nib.load('./../mot-data-nii-txt/train/train_data_05_'+str(tmp_str_)+'.nii.gz')).get_fdata()
    traindata = np.concatenate((traindata, tmp_data_), axis=2)
    tmp_label_ = np.loadtxt('./../mot-data-nii-txt/train/train_label_05_'+str(tmp_str_)+'.txt')
    trainlabel = np.concatenate((trainlabel, tmp_label_), axis=0)

traindata=traindata[:,:,1:]
trainlabel = trainlabel[1:]
print(traindata.shape, trainlabel.shape)


valdata =  (nib.load('./../mot-data-nii-txt/val/val_data_05_.nii.gz')).get_fdata()
vallabel = np.loadtxt('./../mot-data-nii-txt/val/val_label_05_.txt')

print(traindata.shape, trainlabel.shape, valdata.shape, vallabel.shape)

##########################################################################
##########################################################################
device = torch.device("cuda:0")
#########################################################################
#########################################################################
# num_samples_train=traindata.shape[2] 
num_samples_train=15000
num_samples_val = valdata.shape[2] 
#########################################################################
#########################################################################
global plotter
plotter = VisdomLinePlotter(env_name='Loss Plots')
##########################################################################
##########################################################################
#To detect anomaly
torch.autograd.set_detect_anomaly(True)

train_transform = transforms.Compose([
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])

test_transform = transforms.Compose([transforms.ToTensor()])

batch_size_ = 4

aux_train_size_ =int(num_samples_train/batch_size_)
aux_val_size_ =int(num_samples_val/batch_size_) 

classes = ('1.0', '2.0', '3.0', '4.0','5.0', '6.0', '7.0', '8.0', '9.0', '10.0')

################################################################################################
################################################################################################
################################################################################################
# from ResNetClassifier2D_DO import ResNetClassifier as Net
# net = Net(do_percent=0.0)
# from lenet_2D_512i import Net
import torchvision.models as models

net = models.resnext101_32x8d()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
#net.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#net.classifier = nn.Linear(in_features=2208, out_features=10, bias=True)

#from torchsummary import summary
#summary(net, (1, 256,256), device='cpu')

#Total params: 86,756,554
#Trainable params: 86,756,554
#Non-trainable params: 0
#----------------------------------------------------------------
#Input size (MB): 0.25
#Forward/backward pass size (MB): 1009.02
#Params size (MB): 330.95
#Estimated Total Size (MB): 1340.22
#----------------------------------------------------------------

print(net)
net.to(device)
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

vallabel = np.ceil(vallabel*2)-1
valset = MyDataset(valdata, vallabel, transform=test_transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size_, shuffle=True)
# trainlabel = np.ceil(trainlabel*2)-1
# trainset = MyDataset(traindata, trainlabel, transform=train_transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, shuffle=True) 


best_val_loss = 999999999.
for epoch in range(numb_epochs):#(25):  # loop over the dataset multiple times
    # adjust_learning_rate(optimizer, epoch, initialLearningRate, lrDecayNEpoch, lrDecayRate)

    subset_train = index_subset(traindata.shape[2], num_samples=num_samples_train)
    traindatasub, trainlabelsub = create_subset_mot(traindata, trainlabel,  subset_train)
    trainlabelsub = np.ceil(trainlabelsub*2)-1

    trainset = MyDataset(traindatasub, trainlabelsub, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_, shuffle=True) 

    net.train()
    running_loss = 0.0
    for idx, (imgs, labels) in enumerate(train_loader):
        imgs_train, labels_train = imgs.to(device), labels.to(device)
        imgs_train.requires_grad = True
        labels_train.requires_grad = True
        optimizer.zero_grad()  # 
        output_train = net(imgs_train)
        # print(output_train, labels_train.long())
        loss = criterion(output_train, labels_train.long()) # test for CrossEntropyLoss
        loss.backward() #
        optimizer.step() #
        # print statistics
        running_loss += loss.item()
        if idx % 100 == 99:    # print every 200 mini-batches
            print('[%d, %14d] Training loss: %.9f' %
                  (epoch + 1, idx + 1, running_loss / 100))
            # print(idx+(epoch)*aux_train_size_, loss.item())
            # x_trainp = (idx+(epoch)*aux_train_size_)/(len(trainlabel)/batch_size_)
            x_trainp = (idx+(epoch)*aux_train_size_)/(len(trainlabelsub)/batch_size_)
            y_trainp = loss.item()
            plotter.plot('loss', 'train', 'Loss Plots resnext101_32x8d',x_trainp, y_trainp)
            running_loss = 0.0


    #############################################################################
    # calculate validation loss
    net.eval()
    mean_val_loss = 0.0
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            imgs_val, labels_val = imgs.to(device), labels.to(device)
            val_outputs = net(imgs_val)
            val_loss = criterion(val_outputs, labels_val.long())
            mean_val_loss += val_loss.item()
            if idx % 5 == 4:    # print every 200 mini-batches
                print('[%d, %14d] Validation loss: %.9f' %
                      (epoch + 1, idx + 1, mean_val_loss / 5))
                # print(idx+(epoch)*aux_train_size_, loss.item())
                x_valp = (idx+(epoch)*aux_val_size_)/(aux_val_size_)
                y_valp = val_loss.item()
                plotter.plot('loss', 'validation', 'Loss Plots resnext101_32x8d',x_valp, y_valp)
                mean_val_loss /= len(labels)

    print('Mean-val-loss:  '+str(mean_val_loss)+', Best-val-loss: '+str(best_val_loss))    
    # save model if it has a better validation loss than all before
    if mean_val_loss < best_val_loss:
        # print("Saving best checkpoint:  "+str(epoch))
        # path_ = './checkpoints-test/resnext101_32x8d-05-best.pth'
        # torch.save(net.state_dict(), path_)

        print("Saving best checkpoint:  "+str(epoch))
        path_ = './checkpoints-test/ResNeXt-class-best-latest.pth'
        count_ = format(epoch+1, '04d')
        path_tmp_ = './checkpoints-test/ResNeXt-class-best-'+str(count_)+'.pth'
        torch.save(net.state_dict(), path_tmp_)
        torch.save(net.state_dict(), path_)
        best_val_loss = mean_val_loss

print('Finished Training')

PATH = './checkpoints-test/resnext101_32x8d-05-'+str(numb_epochs)+'-latest.pth'
torch.save(net.state_dict(), PATH)
