import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from nibabel.testing import data_path
from matplotlib import pyplot as plt
import matplotlib
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

from ReadData import mydataLoader
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
from numpy.random import randint as ri
import NeuralNetBasaia as NeuralNet
import torchio as tio
import pandas as pd
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()

# This file is for training the model from Basaia


# put the csv fle in your home folder with this file. The following csv files contain relevant trainingdata
df3 = pd.read_csv("dataset_with_filenames_allADNI_train4.csv", engine='python', sep=',')
df2 = pd.read_csv("dataset_with_filenames_allADNI_val4.csv", engine='python', sep=',')
df1 = pd.read_csv("dataset_with_filenames_allADNI_test4.csv", engine='python', sep=',')
df = pd.concat([df3, df1,df2], ignore_index=True)

# balance data
df_new = df.groupby('Group').head(119)
df.drop(df_new[:2*119].index, axis=0,inplace=True)
df.to_csv( "testdatafromCVBasaia.csv")
df = df_new
Data = df['Path'].astype('str').tolist()
Target = df['Group'].astype('str').tolist()

# print options - not really needed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#convert list of str to list of int

unique = set(Target)
map = {word: i for i, word in enumerate(unique)}

Target = [map[word] for word in Target]


my_nn = NeuralNet.Net()
#my_nn.load_state_dict(torch.load("new_best-model_epoch_199l=0.0001allADNI_Basaia"), strict=False)
#my_nn.train()
my_nn.to(device)
learning_rates = [0.0001]   
criterion = nn.BCELoss()
n_datapoints = len(Target)
#print(Data)
testset_length = int(n_datapoints/6)
splits = 6

for l in range(splits):
    X_train, X_test, y_train, y_test = Data[:testset_length*5], Data[-testset_length:], Target[:testset_length*5],  Target[-testset_length:]
#    save_df_val = pd.DataFrame()
#    save_df_val['Path'] = X_test
#    save_df_val['Group'] = y_test
#    save_df_val.to_csv("ValCVRoundBasaia" + str(l))
#    save_df_train = pd.DataFrame()
#    save_df_train['Path'] = X_train
#    save_df_train['Group'] = y_train
#    save_df_train.to_csv("TrainCVRoundBasaia" + str(l))

# different test and validationset for each loop

    Target = np.roll(Target, testset_length)
    Data = np.roll(Data, testset_length)

    Target = (list(Target))
    Data = list(Data)
    print("Number of 0 patients in traindataset:", y_train.count(0))
    print("Number of 1 patients in traindataset:", y_train.count(1))

    print("Number of 1 patients in valdataset:", y_test.count(1))
    print("Number of 0 patients in valdataset:", y_test.count(0))
    
    dataset_train = mydataLoader(X_train, y_train, 0) 
    dataset_val = mydataLoader(X_test, y_test, 1)

    dataloader_train = DataLoader(dataset_train, batch_size=3,
                        shuffle=True, num_workers=3)
    dataloader_val = DataLoader(dataset_val, batch_size=3,
                        shuffle=True, num_workers=3)

# Not really needed forloop. Used for training the model several times for few epochs at development
    for m in range(1):
        my_nn = NeuralNet.Net()
        my_nn.to(device)

        optimizer = torch.optim.Adam(my_nn.parameters(), learning_rates[0])
        losses = []
        count = 0
        accepoch = []
        lossepoch = []
        v_lossepoch = []

        for epoch in range(400):  # loop over the dataset multiple times
            train_correct = 0
            val_correct = 0
            total_val = 0
            total_train = 0
               
            running_loss = 0.0
            v_running_loss = 0.0
#train
            for i, (data, target) in enumerate(dataloader_train): 
                #if i > 100:
                #    break
                #batch_size = (10,4,4,2)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data.to(device), ((torch.tensor(target))).float().to(device)
                #print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = my_nn(inputs)
                
                loss = criterion(outputs, labels.unsqueeze(1))
                #losses.append(loss)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                count += 1
                total_train += len(target)
                print(total_train)
                
                predicted = (outputs>0.5).float()
                print(predicted)
                train_correct += (labels.squeeze() == predicted.squeeze()).sum().item()
                    # Iterate through test dataset
                print(train_correct)
#validate:
            for j, (v_images, v_labels) in enumerate(dataloader_val):
                val = v_images.to(device)
                v_labels  = (torch.tensor(v_labels)).float().to(device)
                # Forward propagation
                v_outputs = my_nn(val)
                total_val += len(v_labels)
                v_loss = criterion(v_outputs, v_labels.unsqueeze(1))
                v_running_loss += v_loss.item()
                        # Get predictions from the maximum value
                v_predicted = (v_outputs>0.5).float()

                        # Total number of labels

                val_correct += (v_predicted.squeeze() == v_labels.squeeze()).sum().cpu().item()

                    # store loss and iteration

#save and evaluate            
            torch.save(my_nn.state_dict(), "modelname")
            print('new augmentation Learningrate: {} Testrun nr: {} Epoch: {}  Train_Loss: {}  Val_Loss: {} Train Accuracy: {} % Val Accuracy: {} %'.format(l, m, epoch, running_loss/len(dataloader_train), v_running_loss/len(dataloader_val), (100*train_correct)/total_train, (100*val_correct)/total_val))
            lossepoch.append(running_loss/len(dataloader_train))
            v_lossepoch.append(v_running_loss/len(dataloader_val))      
            plt.figure()
            tloss, = plt.plot(lossepoch)
            vloss, = plt.plot(v_lossepoch)
            tloss.set_label('Training Loss')
            vloss.set_label('Validation loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("something")
            plt.savefig('round' + str(m) + 'name.png')

 

        torch.save(my_nn.state_dict(), "name" + str(l)  )
print('Finished Training')
