# Classification of Motion Corrupted Brain MR Images using Deep Learning Techniques


## Introduction

Despite continuing efforts in solving the problem of motion artifacts in MRI, 
subject motion remains one of themajor sources of image degradation [1], 
in research applications and more importantly, in clinical routine acquisitions. 
It is fundamental to differentiate between images with an acceptable level of motion corruption and those that cannot be used for diagnosis and require rescanning [2-3]. The quality assessment is done visually by the radiographer or the technician. It requires additional time and is biased by the subjective quality perception. We comparatively evaluate two deep learning techniques for automatic classification of the level of motion artifacts. We aim at providing a supplementary tool that assists the clinicians in image quality assessment.

## Subjects/Methods

For this work, images of 29 healthy subjects acquired at 3T were used. 
Our dataset comprises T1, T2, PD and T2-FLAIR images with different orientation and resolution, Fig. 1.
The data (2D slices) were extracted from hundreds of 3D volumes and split as follows: 
34936, 7362, 6768 images for training, validation and testing, respectively. 
Two neural network models were chosen for the classification task: ResNeXt-101_8d [4] and DenseNet-161 [5]. 
In particular, the task was set to identify not only the level of corruption due to motion artifacts, 
but also the type of image contrast and the orientation. 
Motion artifacts have been generated artificially using an ad-hoc algorithm in order to emulate realistic motion artifacts as close as possible. 
We defined a total of 33 classes in the format < contrast-orientation-motion corruption >, 
e.g.: T1-sagittal-class-1, where, class 1 indicates no corruption, class 2, 
a mild level of corruption and class 3, the image is heavily degraded, Fig. 1. 

![Figure 1.](https://user-images.githubusercontent.com/33011208/74101378-2ef4e880-4b5f-11ea-8e9d-5ae1d811a35a.png)

Get my latest tutorials
Email
Related tutorials
How to create a tag in GitHub Repository
How to clone all branches from a remote git repository
How to change a remote URL in Git
How to modify the commit messages in Git
How to push all tags to remote in Git

## Results/Discussion

The average classification accuracy levels reached were 98.39% and 98.92%, for ResNeXt and DenseNet, respectively. 
The errors of the classification are shown in Fig. 2 using the confusion matrix. From Fig. 2,
right side, it is possible to observe a pair of outliers for the DenseNet model. 
Considering the specific outliers,the model has erroneously classified T2-axial-3 and PD-axial-3,
both classes with high level of degradation, clearly missing the contrast information. 
However, these results are promising, yet, the robustness of the networks performance wrt. different datasets (with varying acquisition properties such as sequence parameters and resolution) remains to be investigated. 
It is possible that these models, given their complex architecture, 
have learned features that go beyond the simple detection of motion artifacts, and probably learned details related to the structures. 
We will further analyse these aspects and, if necessary, design a harmonization pre-processing step, to ensure robustness and flexibility.
