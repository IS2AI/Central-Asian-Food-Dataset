# Central-Asian-Food-Dataset
Central Asian Food Dataset (CAFD) for food classification contains 21,288 images across 42 food classes of Central Asian cuisine. The dataset is web-scraped and manually annotated. Images have various resolution.
Figure below illustrates the samples and name for each class.

<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/samples.png" width="750" height="700">

The dataset is unbalaced. The statistics across all 42 classes is shown on Figure below.

<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/stats_plot.png" width="500" height="600">

# Download the dataset

The dataset can be downloaded using the link below. If there are some issues with the link, please, email us on issai@nu.edu.kz

https://drive.google.com/drive/folders/1mnfShcKkADjESW9_TuOT9_m5IhhZg1h6?usp=sharing

# Pre-trained models

To illustrate the performance of different classification models on CAFD we have trained different models. We used the largest publicly available fine-grained dataset Food1K [[1]](#1) that contains 1,000 food classes to evaluate the performance of classifier with the 1,042 food categories.

|Model| CAFD (Top-1 Acc.)| CAFD (Top-5 Acc.)| Food1K+CAFD (Top-1 Acc.)| Food1K+CAFD (Top-5 Acc.)|
|-----|-----------------|-----------------|------------------------|------------------------|
|VGG-16|72.12|95.06|66.49|89.65|
|Squeezenet1_0|82.73|97.65|71.66|91.54|
|ResNet50|89.41|98.84|82.97|97.09|
|ResNet101|90.10|98.99|84.21|97.43|
|ResNet152|90.66|99.25|85.30|97.74|
|ResNext50_32|89.90|99.10|83.37|96.80|
|Wide ResNet-50|90.10|99.15|85.27|97.81|
|Inception-v3|88.90|98.03|85.78|97.82|
|DenseNet-121|89.60|99.09|83.24|97.28|
|EfficientNet-b4|84.64|98.46|87.75|98.01|

Pre-trained model weights of the best performing models: ResNet152 on KFD and EfficientNet-b4 on Food1K+KFD can be downloaded using these links:

## ResNet152 trained on CAFD: 

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/KFD_pre_trained_models/kfd_resnet152.pt


## EfficientNet-b4 trained on Food1K+CAFD:

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/KFD_pre_trained_models/food1k_kfd_efficientnet.pt

## Model training and testing

To train and test using pre-trained models use train.py and test.py files. 

## References
<a id="1">[1]</a> 
Min, Weiqing and Wang,  Zhiling (2021). 
Large Scale Visual Food Recognition. 
arXiv.

