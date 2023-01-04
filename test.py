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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# data pre-processing and normalization 
data_transforms = {
		test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# path to the dataset folder
data_dir = '../KFD/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
                  
# loaad data to Data Loaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=False, num_workers=4)
              for x in ['test']}
              
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

# extract ground truth class names 
class_names = image_datasets['test'].classes

# select the device to train the model
device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu")

# define a function to run test cases
def test(model):
    # set the model to evaluation mode
    model.eval()
    images_so_far = 0
	
	# dataframe to store the predictions for accuracy metrics
    df = pd.DataFrame(columns=['ground_truth', 'top1', 'top5'])

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
        	# transfer the inputs and labels to selected device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
			# model prediction 
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
			
			# record in the dataframe the ground truth of the test sample	
            gt = labels.cpu().detach().numpy()[0]
            df.at[i, 'ground_truth'] = gt
            
            # extract the predicted class
            pred = preds.cpu().detach().numpy()[0]
            
			# extract probabilities of all prediction 
            predByClass = outputs.cpu().detach().numpy()
            predByClass = predByClass[0].tolist()
            top5 = set()
            
            # extract top 5 predictions with maximum probabilities
            for j in range(5):
                max_val = max(predByClass)
                max_ind = predByClass.index(max_val)
                top5.add(max_ind)
                predByClass[max_ind] = -10000
            df.at[i, 'pred']=pred
            
            # Top-5 Accuracy - check if ground truth is within the top 5 maximum probability predictions
            if gt in top5:
                df.at[i, 'top5']=1 
            else:
                df.at[i, 'top5']=0
			
			# Top-1 Accuracy - check if ground truth matches exactly the prediction of the model
            if gt == pred:
                df.at[i, 'top1']=1
            else:
                df.at[i, 'top1']=0
   
   	# print Top-1 and Top-5 accuracy
    top1_acc = 100*df['top1'].sum()/len(df)
    top5_acc = 100*df['top15'].sum()/len(df)
	print("Top-1 Accuracy = {} and Top-5 Accuracy = {}".format(top1_acc, top5_acc)
	
	# extract classification report
	y_test = df.ground_truth.values.tolist()
	y_pred = df.pred.values.tolist()
	report = classification_report(y_test, y_pred, output_dict=True)
	
	# convert the report to dataframe
	report = pd.DataFrame(report).transpose()
	
	# save classification report to csv file
	report.to_csv('classification_report_kfd.csv')

# load saved model
model = torch.load('results/kfd_resnet152.pt')

# run testing
test(model)






