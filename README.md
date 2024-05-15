# Chest-Xray-multilabel-classifier

This project aimed to detect low ejection fraction (EF) and valvular heart disease from chest Xrays using a multi-head classifier.

## Task
Heart failure and different valvular diseases may cause the heart anatomy to change which is visible in the chest radiography. These abnormalities may occur simultaneously. For instance, a person with low EF may simultaneously have different valve diseases. This requires a deep learning model that can identify several outcomes at the same time (a multi-label and not a multi-class classifier). As a result, I tried to develop a model that can detect low EF, Mitral Stenosis, Aortic Stenosis, Mitral insufficiency, Aortic Insufficiency, and Tricuspid Insufficiency. Valvular heart diseases were categorized as moderate to severe (labeled as 1) and normal to mild (labeled as 0), while EF was categorized as EF<40 (label 1) and EF> 40 (label 0). Valvular disease and EF detection require echocardiography which is more costly and time-consuming compared to radiographs. Developing a model that can screen radiographs for heart diseases can be a valuable asset in healthcare to reduce costs. 

## Data
I used 2007 chest X-rays extracted from the EHR of Tehran Heart Center. After removing poor-quality images and radiographs without associated echocardiography 1705 samples remained. Outcome labels were based on echocardiographies performed by cardiologists. 

## Methods
All radiographs were extracted as DICOM files. Numpy arrays were extracted from images. I used several models like Efficientnet B0 and effnetV2S. In the EfficientnetB0 experiment. Imagenet weights were used for all models. Baseline layers were frozen while the final model layers were trainable. As Imagenet is based on 3-channel photos, 2 additional dimensions were added to all numpy arrays. All data was normalized according to Imagenet values. Additional augmentations were performed like flipping, cropping, and changing the intensity. I also tried an experimental Vision transformer model changing the final MLP layer to suit a multi-label task. I used BCE with logit loss for every possible outcome and prediction and then used the average for all losses as the overall model loss. The initial experimental code was developed in pytorch and torchvision, while final model training was performed using MONAI.

## Some examples of the possible augmentations used



![download (3)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/374dba54-ea83-4b0a-aaaa-b8b1035cee90)![download (4)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/97271c45-cb6b-42a2-a215-04b5de555697)
![download (4)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/97271c45-cb6b-42a2-a215-04b5de555697)

![download (3)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/374dba54-ea83-4b0a-aaaa-b8b1035cee90)
![download (5)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/c5ae36c9-4f71-470a-b71c-b318f88fd039)
![download (5)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/c5ae36c9-4f71-470a-b71c-b318f88fd039)

![download (6)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/b5c5e317-a60b-4142-b06a-448331bf9bb3)
![download (6)](https://github.com/Sepehr-76/Chest-Xray-multilabel-classifier/assets/136221815/b5c5e317-a60b-4142-b06a-448331bf9bb3)

