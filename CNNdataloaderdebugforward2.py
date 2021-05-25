
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import NeuralNetBasaia as NeuralNet
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

x = []
num_classes = 2
for epoch in range(1):  # loop over the dataset multiple times
#    torch.cuda.empty_cache()

#    df = pd.read_csv("ValCVRoundBasaia"+ str(epoch) + "", engine='python', sep=',')#pd.concat([df,df2])
#    df = pd.read_csv("testdatafromCV_mixed15Tand3T"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
    df = pd.read_csv("dataset_with_fieldstrenghts_AIBL"+ ".csv", engine='python', sep=',')#pd.concat([df,df2])
   # df = df.sample(frac=1)
    df = df.loc[(df['Group'] == 'AD') | (df['Group'] == 'CN')]
    df = df.loc[(df['Imaging Protocol'] == 'Field Strength=3.0')]

#    df = (df.groupby('Group').head(74))[:(74+23)]
	#df = df.sort_values(by=['Subject', 'Acq Date'])
	#print(df['Group'])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    diagnosis = df['Group'].astype('str').tolist() 

	#convert list of str to list of int
    unique = set(diagnosis)
    map = {'AD': 1, 'CN': 0}
    diagnosis_int = [map[word] for word in diagnosis]
    print(map)
    print("a", diagnosis_int.count(0))
    print("b", diagnosis_int.count(1))

	#print(diagnosis_int[0], diagnosis[0])


    paths = df['Path'].astype('str').tolist()

    dataset_val = mydataLoader(paths, diagnosis_int, 1)

    dataloader_val = DataLoader(dataset_val, batch_size=3, shuffle=True, num_workers=3)
    print("TESTING MODEL:", str(epoch))
    my_nn = NeuralNet.Net()

#    my_nn.load_state_dict(torch.load("new_best-model_epoch_" + str(epoch) + "l=0.0001allADNI_Basaia2-08052021NopriorTrain"), strict=False)# map_location=torch.device('cpu')), strict=False)
    my_nn.load_state_dict(torch.load("basemodel/BasaiaCV/new_best-model_epoch_"+ str(199) +"l=" + str(5) + "allADNI_BasaiaCV"), strict=False)# map_location=torch.device('cpu')), strict=False)
#    my_nn.load_state_dict(torch.load("basemodel/new_best-model_epoch_"+ str(195) +"l=" + str(1) + "allADNI_BasaiaCV"), strict=False)# map_location=torch.device('cpu')), strict=False)

    my_nn.eval()
    my_nn.to(device)

    train_correct = 0
    train_all = 0
    losses = []
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    accavg = []
    lossepoch = []
    FN = 0
    TP  = 0
    TN = 0
    FP = 0
    CN = 0
    AD = 0

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
        print(outputs)
        # Get predictions from the maximum value
        predicted = (outputs > 0.5).float().squeeze()
                
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
    plt.hist(np.round(x,1), bins = range(70,95))
    plt.show()
    plt.savefig('hist2Basaia.png')

    print('Iteration: {}  Accuracy: {}  avgAcc: {} %'.format(count, accuracy, sum(accuracy_list)/len(accuracy_list)))
    print('TP: {} TN: {} FP: {} FN {} %'.format(TP, TN, FP, FN))
print('Finished Testing')
