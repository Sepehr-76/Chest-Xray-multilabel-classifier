## Chest-Xray-multilabel-classifier

This project aimed to detect low ejection fraction (EF) and valvular heart disease from chest Xrays using a multi-head classifier.

# Task
Heart failure and different valvular diseases may cause the heart anatomy to change which is visible in the chest radiography. These abnormalities may occur simultaneously. For instance, a person with low EF may simultaneously have different valve diseases. This requires a deep learning model that can identify several outcomes at the same time (a multi-label and not a multi-class classifier). As a result, I tried to develop a model that can detect low EF, Mitral Stenosis, Aortic Stenosis, Mitral insufficiency, Aortic Insufficiency, and Tricuspid Insufficiency. Valvular heart diseases were categorized as moderate to severe (labeled as 1) and normal to mild (labeled as 0), while EF was categorized as EF<40 (label 1) and EF> 40 (label 0).

# Data
Data


