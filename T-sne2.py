
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from bioinfokit.visuz import cluster

from sklearn.manifold import TSNE

import Tsnenet as NeuralNet
from nibabel.testing import data_path
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from ReadData import mydataLoader
# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

from torch.utils.data import Dataset, DataLoader
import glob
import torchio as tio
import pandas as pd

torch.cuda.empty_cache()
torch.set_printoptions(sci_mode=False)
# put the csv fle in your home folder with this file
df = pd.read_csv("dataset_with_filenames_allADNI_val4.csv", engine='python', sep=',')
df2 = pd.read_csv("dataset_with_filenames_allADNI_test4.csv", engine='python', sep=',')
FN = 0
TP  = 0
TN = 0
FP = 0
allx = []
alldata = []
diagnosis = []
diagnosis_int = []  
num_classes = 2
confidences = []
datasets = []
Gender = []
saved = []
true = []
for epoch in range(1):  # loop over the dataset multiple times
    x = []
    saved = []
    df = pd.read_csv("testdatafromCV_mixed2"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
    df1 = pd.read_csv("dataset_with_fieldstrenghts_AIBL"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
    df1 = df1.drop(['Dataset'], axis='columns')
    df1 = df1.rename(columns = {'Imaging Protocol': 'Dataset'})
    df2 = pd.read_csv("dataset_with_filenames_ItalianADNI"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])[1:]
    df2 = df2.rename(columns = {'GENDER': 'Sex'})
    df = pd.concat([df,df1,df2])
    df = df.loc[(df['Group'] == 'AD') | (df['Group'] == 'CN')]
    diagnosis = df['Group'].astype('str').tolist() 
    Gender = df['Sex'].astype('str').tolist() 
    print(set(Gender))
    datasets = df['Dataset'].astype('str').tolist() 

	#convert list of str to list of int
	#print(diagnosis_int[0], diagnosis[0])


    paths = df['Path'].astype('str').tolist()

    dataset_val = mydataLoader(paths, Gender, 1)

    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    print("TESTING MODEL:", str(epoch))
    my_nn = NeuralNet.Net()

#    my_nn.load_state_dict(torch.load("new_best-model_learningrate_0"), strict=False)# map_location=torch.device('cpu')), strict=False)
    my_nn.load_state_dict(torch.load("new_best-model_epoch_" + str(120) + "l=" + str(0) + "6layersCV_mixed3"), strict=False)# map_location=torch.device('cpu')), strict=False)

    my_nn.eval()
    my_nn.to(device)

    train_correct = 0
    train_all = 0
    losses = []
    count = 0
    loss_list = []
    iteration_list = []
    lossepoch = []
    CN = 0
    AD = 0
    accuracy_list = []
    accavg = []
    
    running_loss = 0.0
    print(len(dataloader_val))
    for i, (val, label) in enumerate(dataloader_val):
        count += 1
        correct = 0
        total = 0
        # Iterate through test dataset
        val = val.to(device)
#        labels  = label.to(device)
        # Forward propagation
        outputs = my_nn(val)[0]
        # Get predictions from the maximum value
        predicted = (outputs.data)
        predicted2 = torch.max(predicted)
        predicted1 = torch.max(outputs.data, 1)[1]
        saved.append(my_nn(val)[1].flatten().tolist())
        print(np.array(saved).shape)
        x = predicted2.flatten().tolist()
        y = 10
        x = x[0]
        if (x < 0.6):
            y = 4
        elif (x < 0.7):
            y = 3
        elif (x < 0.8):
            y = 2
        elif (x < 0.9):
            y = 1
        elif (x < 1):
            y = 0
        else:
            y = 10
        map = {'CN': 0, 'AD': 1}
        diagnosis_int = [map[word] for word in diagnosis]
        z = [int(((predicted1.tolist())[0]) != diagnosis_int[i])]
        print((predicted1.tolist())[0])
        true = true + z
        confidences = confidences + [y]
       
X = saved
X_embedded = TSNE(n_components=2).fit_transform(X)

print(set(datasets))
for epoch in range(6):     
    alldata = alldata + saved
    if (epoch == 0):
    
        map = {'ADNI1': 0, 'ADNI1_3T': 0, 'ADNI2' : 0, 'Field Strength=1.5': 1, 'Field Strength=3.0': 1, 'ItalianADNI_15T': 2, 'ItalianADNI_3T': 2}
        diagnosis_int = [map[word] for word in datasets]
        x= diagnosis_int
        cluster.tsneplot(score=X_embedded)
        #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
        cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#a03050','#072969','#147E27'), figname='sitewise', legendpos='upper right', legendanchor=(1.15, 1))

    elif (epoch ==1):
       map = {'ADNI1': 0, 'ADNI1_3T': 1, 'ADNI2' : 1, 'Field Strength=1.5': 0, 'Field Strength=3.0': 1, 'ItalianADNI_15T': 0, 'ItalianADNI_3T': 1}
       diagnosis_int = [map[word] for word in datasets]
       x= diagnosis_int
       cluster.tsneplot(score=X_embedded)
       #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
       cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#d690f8','#460664'), figname='fieldstrengt.png', legendpos='upper right', legendanchor=(1.15, 1))

    elif (epoch ==2):
    
       map = {'CN': 0, 'AD': 1}
       diagnosis_int = [map[word] for word in diagnosis]
       x = diagnosis_int
       cluster.tsneplot(score=X_embedded)
       #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
       cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#072969','#ed2121',), figname='diagnosis.png', legendpos='upper right', legendanchor=(1.15, 1))

    elif (epoch ==3):
       unique = set(Gender)
       map = {'M': 0, 'F': 1, 'X': 0}
       diagnosis_int = [map[word] for word in Gender]
       x= diagnosis_int
       cluster.tsneplot(score=X_embedded)
       #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
       cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#072969','#ed2121','#c61010'), figname='Gender.png', legendpos='upper right', legendanchor=(1.15, 1))

    elif (epoch == 4):
       x= confidences
       cluster.tsneplot(score=X_embedded)
       #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
       cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#f69292','#ee2f2f','#c61010', '#840a0a', '#420505'), figname='Confidences.png', legendpos='upper right', legendanchor=(1.15, 1))

    else:
       x= true
       cluster.tsneplot(score=X_embedded)
       #c = '#dc5c34', '#de653f', '#072969', '#4e87f3', '#147E27', '#4e8713', '#7aea8e', '#b57df4', '#210541', '#C2C297', '#434327',  '#f05050'
       cluster.tsneplot(score=X_embedded, colorlist=np.array(x), colordot=('#ed2121', '#49DC49'), figname='Wrongs.png', legendpos='upper right', legendanchor=(1.15, 1))

    
