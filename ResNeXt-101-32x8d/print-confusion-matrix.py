import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

classes = ('1.0', '2.0', '3.0', '4.0','5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
y_true, y_pred =np.load('Confusion-matrix-resnext101-05-nii.npy')

# print(y_true.dtype, y_true.shape, y_pred.dtype, y_pred.shape)

y_true = np.reshape(y_true,(y_true.shape[0]*y_true.shape[1], 1))
y_pred = np.reshape(y_pred,(y_pred.shape[0]*y_pred.shape[1], 1))

# print(y_true.dtype, y_true.shape, y_pred.dtype, y_pred.shape)

cfmtx_ = confusion_matrix(y_true, y_pred)


#cfmtx_ = confusion_matrix(str(y_true), str(y_pred), labels=classes)
# print(cfmtx_.shape, cfmtx_.max())
# print((cfmtx_/cfmtx_.max())*100)


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

print(confusion_matrix(y_true, y_pred))