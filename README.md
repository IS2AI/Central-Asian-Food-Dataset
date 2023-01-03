# Kazakh-Food-Dataset
Kazakh Food Dataset (KFD) for food classification contains 21,288 images across 42 food classes of Kazakh and Central Asian cuisine. The dataset is web-scraped and manually annotated. Images have various resolution.
Figure below illustrates the samples and name for each class.

<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/samples.png" width="750" height="700">

The dataset is unbalaced. The statistics across all 42 classes is shown on Figure below.

<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/stats_plot.png" width="500" height="600">

# Pre-trained models

To illustrate the performance of different classification models on KFD we have trained different models. 

|Model| KFD (Top-1 Acc.)| KFD (Top-5 Acc.)| Food1K+KFD (Top-1 Acc.)| Food1K+KFD (Top-5 Acc.)|
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


https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/KFD_pre_trained_models/kfd_resnet152.pt

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/KFD_pre_trained_models/1k_resnet152.pt

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/KFD_pre_trained_models/food1k_kfd_efficientnet.pt

