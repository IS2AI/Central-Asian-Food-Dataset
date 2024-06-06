# Central-Asian-Food-Dataset
In this work, we propose the first Central Asia Food Scenes Dataset that contains 21,306 images with 69,856 instances across 239 food classes. To make sure that the dataset contains various food items, we took as a benchmark the ontology of Global Individual Food Tool developed by Food and Agriculture Organization (FAO) together with the World Health Organization (WHO)~\cite{FAO2022}. The dataset contains food items across 18 coarse classes: vegetables, baked flower-based products, cooked dishes, fruits, herbs, meat dishes, desserts, salads, sauces, drinks, dairy, fast-food, soups, sides, nuts, pickled and fermented food, egg product, and cereals. Fig.~\ref{fig: stats} illustrates the overall distribution of the class instances based on these categories. 


The dataset contains open source web-scraped images from the search engines (15,939 images) (i.e., Google, YouTube, and Yandex) and our own collected food images from everyday life (2,324 images). To additionally extend the number of instances of the underrepresented classes, we have scraped open-source videos and extracted frames at a rate one frame per second (3,043 images). The dataset has been checked and cleaned for duplicates using the Python Hash Image library. Furthermore, we have also filtered out images less than 30 kB in size and replaced them by performing additional iterative data scraping and duplicate check to make sure the high quality of the dataset.



<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/samples.png" width="750" height="700">

The dataset is unbalaced. The statistics across all 42 classes is shown on Figure below.

<img src="https://github.com/IS2AI/Kazakh-Food-Dataset/blob/main/figures/stats_plot.png" width="500" height="600">

# Download the dataset

The dataset can be downloaded using the link below. If there are some issues with the link, please, email us on issai@nu.edu.kz

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/CAFD/CAFD.zip

# Pre-trained models

To illustrate the performance of different classification models on CAFD we have trained different models. We used the largest publicly available fine-grained dataset Food1K [[1]](#1) that contains 1,000 food classes to evaluate the performance of classifier with the 1,042 food categories.

|Model| CAFD (Top-1 Acc.)| CAFD (Top-5 Acc.)| Food1K+CAFD (Top-1 Acc.)| Food1K+CAFD (Top-5 Acc.)|
|-----|-----------------|-----------------|------------------------|------------------------|
|VGG-16|86.03|98.33|80.87|96.19|
|Squeezenet1_0|79.58|97.29|69.16|90.15|
|ResNet50|88.03|98.44|83.22|97.25|
|ResNet101|88.51|98.44|84.20|97.45|
|ResNet152|88.70|98.59|84.75|97.58|
|ResNext50_32|87.95|98.44|84.81|97.65|
|Wide ResNet-50|88.21|98.59|85.27|97.81|
|DenseNet-121|86.95|98.26|82.45|96.93|
|EfficientNet-b4|81.28|97.37|87.75|98.01|

Pre-trained model weights of the best performing models: ResNet152 on KFD and EfficientNet-b4 on Food1K+KFD can be downloaded using these links:

## ResNet152 trained on CAFD: 

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/CAFD/cafd_resnet152.pt

## EfficientNet-b4 trained on Food1K+CAFD:

https://issai.nu.edu.kz/wp-content/themes/issai-new/data/models/CAFD/food1k_kfd_efficientnet.pt

## Model training and testing

To train and test using pre-trained models use train.py and test.py files. 

## References
<a id="1">[1]</a> 
Min, Weiqing and Wang,  Zhiling (2021). 
Large Scale Visual Food Recognition. 
arXiv.

# In case of using our dataset and/or pre-trained models, please cite our work:
```
@Article{nu15071728,
AUTHOR = {Karabay, Aknur and Bolatov, Arman and Varol, Huseyin Atakan and Chan, Mei-Yen},
TITLE = {A Central Asian Food Dataset for Personalized Dietary Interventions},
JOURNAL = {Nutrients},
VOLUME = {15},
YEAR = {2023},
NUMBER = {7},
ARTICLE-NUMBER = {1728},
URL = {https://www.mdpi.com/2072-6643/15/7/1728},
ISSN = {2072-6643}
}
```
