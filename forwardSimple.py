
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import NeuralNetsimple1 as NeuralNet
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
  
x = []
num_classes = 2
for epoch in range(124,125):  # loop over the dataset multiple times
#    torch.cuda.empty_cache()

#    df1 = pd.read_csv("ValCVRound"+ str(0) + "women.csv", engine='python', sep=',')#pd.concat([df,df2])
#    df = pd.read_csv("dataset_with_filenames_allADNI_test4"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
    df = pd.read_csv("dataset_with_fieldstrenghts_AIBL"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
#    df = pd.read_csv("dataset_with_filenames_ItalianADNI"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
  # df = df.sample(frac=1)
    df = df.loc[(df['Group'] == 'AD') | (df['Group'] == 'CN')]
    df = df.drop_duplicates(subset='Subject', keep="first")

#    df = pd.concat([df,df1])
#    df = df1
    df = df.loc[(df['Imaging Protocol'] == 'Field Strength=3.0')]# | (df['Dataset'] == 'ADNI1_3T') ]

#    df = (df.groupby('Group').head(31))[:(31+14)]
	#df = df.sort_values(by=['Subject', 'Acq Date'])
	#print(df['Group'])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    diagnosis = df['Group'].astype('str').tolist() 

	#convert list of str to list of int
    unique = set(diagnosis)
    map = {'1': 1, '0': 0, 'AD' : 1, 'CN': 0}
    diagnosis_int = [map[word] for word in diagnosis]
    print(map)
    print("a", diagnosis_int.count(0))
    print("b", diagnosis_int.count(1))

	#print(diagnosis_int[0], diagnosis[0])


    paths = df['Path'].astype('str').tolist()

    dataset_val = mydataLoader(paths, diagnosis_int, 1)

    dataloader_val = DataLoader(dataset_val, batch_size=3, shuffle=True, num_workers=1)
    print("TESTING MODEL:", str(epoch))
    my_nn = NeuralNet.Net()

    my_nn.load_state_dict(torch.load("new_best-model_epoch_120l=06layersCV_3T"), strict=False)# map_location=torch.device('cpu')), strict=False)
#    my_nn.load_state_dict(torch.load("basemodel/CV_6layers_model/new_best-model_epoch_" + str(130) + "l=" + str(0) + "6layersCV_TRAINEDONALLADNI"), strict=False)# map_location=torch.device('cpu')), strict=False)

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
    for i, (val, label) in enumerate(dataloader_val):
        count += 1
        correct = 0
        total = 0
        # Iterate through test dataset
        val = val.to(device)
        labels  = label.to(device)
        # Forward propagation
        outputs = my_nn(val)
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                
        total += len(label)
        confusion_vec = predicted / labels
        print("confusion_vec:", confusion_vec)
        CN += torch.sum(labels == 0).item()
        AD += torch.sum(labels == 1).item()
        print(CN, AD)
        TP += torch.sum(confusion_vec == 1).item()
        TN += torch.sum(torch.isnan(confusion_vec)).item()
        FP += torch.sum(confusion_vec == float('inf')).item()
        FN += torch.sum(confusion_vec == 0).item()
        correct += (predicted.squeeze() == labels.squeeze()).sum()


        accuracy = 100 * correct / float(total)
        print("true:", labels)
        print("predicted:", predicted)
        # store loss and iteration
        iteration_list.append(count)
        accuracy_list.append(accuracy)
        accavg.append(sum(accuracy_list)/len(accuracy_list))

        if count % 1 == 0:
    #        # Print Loss
    #        print('Iteration: {}  Accuracy: {}  avgAcc: {} %'.format(count, accuracy, sum(accuracy_list)/len(accuracy_list)))
            print('TP: {} TN: {} FP: {} FN {} %'.format(TP, TN, FP, FN))
    #       
    #        plt.figure()
    #        plt.plot(accavg)
#   #         plt.savefig(str(count) + '_accuraciesavg.png')
    x.append(torch.round(sum(accuracy_list)/len(accuracy_list)).cpu().item())
    print(x)
    plt.hist(np.round(x,1), bins = range(70,90))
    plt.show()
    plt.savefig('hist2.png')
    print('Iteration: {}  Accuracy: {}  avgAcc: {} %'.format(count, accuracy, sum(accuracy_list)/len(accuracy_list)))
    print('TP: {} TN: {} FP: {} FN {} %'.format(TP, TN, FP, FN))
print('Finished Testing')
