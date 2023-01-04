from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import pandas as pd
import pickle
import csv
from efficientnet_pytorch import EfficientNet

# set feature extraction mode to False to unfreeze all weights and perform finetuning
feature_extract = False

# models list
model_list = ['squeezenet1-0','resnet-152', 'densenet-121', 'efficientnet-b4']

# path to the dataset
data_dir = '../KFD'

# select the model
model_name = model_list[3]

# set hyperparameters of the model
learning_rate = 0.001
epoch_number = 40
batch = 68
learning_rate_scheduler = True
cudnn.benchmark = True
plt.ion()   # interactive mode

# LOAD THE DATA 
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(380),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
                
# initiate data loaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# extract class names
class_names = image_datasets['train'].classes

# set the device to train the model
device = torch.device("cuda:2") #if torch.cuda.is_available() else "cpu")

# dictionary to record the training progress
results = {'train_loss':[], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[]}

# TRAINING THE MODEL
train_loss = []
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):

    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase=='train':
                results['train_loss'].append(epoch_loss)
                results['train_accuracy'].append(epoch_acc)
            if phase=='val':
                results['val_loss'].append(epoch_loss)
                results['val_accuracy'].append(epoch_acc)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# finetune the convnet
if model_name == "squeezenet1-0":
    model_ft = models.squeezenet1_0(pretrained=True)
    model_ft.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = len(class_names)
elif model_name == "resnet-152":
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
elif model_name == "densenet-121":
    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(class_names)) 
elif model_name == "efficientnet-b4":
    model_ft = EfficientNet.from_pretrained('efficientnet-b4')
    
    # To use the custom weights of the previouls trained model
    # net_weight = 'pretrained_custom.pt'
    # state_dict = torch.load(net_weight)
    # model_ft.load_state_dict(state_dict)
    
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, len(class_names))
else:
    print('invalid model, exiting...')
    exit()


if feature_extract == True:
    for param in model_ft.parameters():
        param.requires_grad = False

# transfer model to selected device
model_ft = model_ft.to(device)

# set the training loss type
criterion = nn.CrossEntropyLoss()

# observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

# decay LR by a factor of 0.1 every 7 epochs
if learning_rate_scheduler==True:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=epoch_number)
else:
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=epoch_number)
PATH = "results/kfd_{}.pt".format(model_name)

# save the model
torch.save(model_ft, PATH)

# plotting the train and validation loss curves
plt.style.use("ggplot")
plt.figure()
plt.plot(results['train_loss'], label="train_loss")
plt.plot(results['val_loss'], label="val_loss")
plt.title("Training and validation lossess")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig('results/loss-curve_food1k_{}.png'.format(model_name)) 



