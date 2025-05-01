This page is maintained by Chuoyue Wang, graduate student major in computer science at
CSU Channel Islands. Thank you to be my advisor @Michael Soltys for
providing suggustion and support. 


### Introduction

This repository contains the files about my final project, two core codes of image classification,
one presentation slide of the project, and two image files for testing trained models. 

The codes are:
```
masterproject_savefile.py
test_image.py
```
The code (masterproject_savefile.py) is the whole model training scripts.
The code (test_image.py) is to use the trained model to classify new images.

The slide is:
```
final_project_landscape.pptx
```
I put all the information, thoughts, and findings about my projects in this slide. 

The images for testing are:
```
new_image_land.zip
new_image_city.zip
```
The images screen shoot from Google Map and prepared for testing the trained model. New images can be added in this file. 
Testing code is test_image.py. 




### Summary about my project
Satellite imagery plays a critical role in diverse applications such as land use planning, environmental monitoring, disaster management, urban water budgeting, and eco- logical conservation. Traditionally, the identification of objects and facilities within satellite images has relied on manual interpretation, a process that is both time- intensive and prone to error. Automation of this task is essential; however, basic classification algorithms, such as logistic regression, were tested and found to lack the accuracy and robustness required for practical use. This study proposes a deep learning-based framework for automated satellite image classification. Implemented in Python using the PyTorch deep learning library, the framework evaluates five mod- els: SqueezeNet1 0, MobileNetV2, ResNet18, a pretrained Vision Transformer, and a pretrained ReXNet150. Experiments were conducted on two labeled satellite imagery datasets (a cityscape dataset comprising 10 object classes and a landscape dataset comprising 15 classes), each supporting different analytical applications. Among the models evaluated, the pretrained ReXNet150 achieved the highest classification accu- racy, exceeding 95%. These findings demonstrate the effectiveness of deep learning, particularly ReXNet150, in improving the efficiency and reliability of satellite imagery analysis for real-world applications.


### Further resources
The two datasets for this poject are downloaded from Kaggle. 

Dataset1. https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset/data

Dataset2 https://www.kaggle.com/datasets/yessicatuteja/skycity-the-city-landscape-dataset/data 

Regards
