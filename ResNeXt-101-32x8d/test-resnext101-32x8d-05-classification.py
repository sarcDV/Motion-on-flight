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
            generate_motion_rnd, MyDataset, index_subset, create_subset
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

################################################################################################
classes = ('1.0', '2.0', '3.0', '4.0','5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
################################################################################################
################################################################################################
################################################################################################
################################################################################################
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")
print(device)
import torchvision.models as models


################################################################################################
################################################################################################
################################################################################################
## 
## Test set
## 
testdata =  (nib.load('./../mot-data-nii-txt/test/test_data_05_.nii.gz')).get_fdata()
testlabel = np.loadtxt('./../mot-data-nii-txt/test/test_label_05_.txt')
testlabel = np.ceil(testlabel*2)-1


print("Total number of samples for testing: "+str(testdata.shape[2]))
#print(testdata.shape, testlabel.shape)


test_transform = transforms.Compose([transforms.ToTensor()])

batch_size_ = 4
testset = MyDataset(testdata, testlabel, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_, shuffle=False)
#######################################################
#######################################################
# PATH = './checkpoints-test/resnext101_32x8d-05-best-first-attempt.pth'
# PATH = './checkpoints-test/resnext101_32x8d-05-classification-best.pth'
PATH = './checkpoints-test/ResNeXt-class-best-latest.pth'
# PATH = './checkpoints-test/ResNeXt-class-best-0189.pth'

net = models.resnext101_32x8d()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
#print(net)
net.load_state_dict(torch.load(PATH))
net.to(device)

## for confusion matrix 
## from sklearn.metrics import confusion_matrix 
y_true = []
y_pred = []

##
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # print(images.dtype, labels.dtype)
        outputs = net(images)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        #print(np.array(predicted.cpu()), np.array(labels.cpu()))
        y_true.append(np.array(labels.cpu())) 
        y_pred.append(np.array(predicted.cpu()))   

        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()


y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

# np.save('Confusion-matrix-resnext101-05-nii.npy',  (y_true, y_pred))

print('Accuracy of the network on the '+str(len(testlabel))+' test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels.long()).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label.int()] += c[i].item()
            class_total[label.int()] += 1
            

for i in range(0,10):
    # print(classes[i], class_correct[i], class_total[i])
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


