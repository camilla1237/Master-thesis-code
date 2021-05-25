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
import NeuralNetsimple1 as NeuralNet
import torchio as tio
import pandas as pd
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()

# put the csv fle in your home folder with this file
df = pd.read_csv("dataset_with_filenames_allADNI_train4.csv", engine='python', sep=',')
#df = df.loc[(df['Group'] == 'AD') | (df['Group'] == 'CN')]

df1 = pd.read_csv("dataset_with_filenames_allADNI_val4.csv", engine='python', sep=',')
df2 = pd.read_csv("dataset_with_filenames_allADNI_test4.csv", engine='python', sep=',')
df = pd.concat([df, df1], ignore_index=True)
#df = df.loc[(df['Sex'] == 'F')]
ADNI1 = pd.read_csv("dataset_with_filenames_allADNI.csv", engine='python', sep=',')
ADNI1 = ADNI1.drop_duplicates(subset='Subject', keep="first")
ADNI1['Dataset']= ADNI1['Dataset'].replace('ADNI1_3T', 'ADNI2')
ADNI1 = ADNI1.loc[(ADNI1['Group'] == 'AD') | (ADNI1['Group'] == 'CN')]
#df_new = ADNI1.groupby(['Dataset', 'Group']).head(100)
#ADNI1.drop(ADNI1[:2*250].index, axis=0,inplace=True)
ADNI1.to_csv( "testdatafromCV_ALLADNI.csv")
#df = df_new
#print(len(df))
#df_new = df.groupby('Group').head(70)
#df.drop(df_new[:2*100].index, axis=0,inplace=True)

#df = pd.read_csv( "traindatafromCV_mixed2.csv")

#df = df.sample(frac=1, random_state=1)


Data = df['Path'].astype('str').tolist()
Datasets = df['Dataset'].astype('str').tolist()

Target = df['Group'].astype('str').tolist()


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#convert list of str to list of int

print(Target.count(1))

#my_nn = NeuralNet.Net()
#my_nn.load_state_dict(torch.load("new_best-model_epoch_199l=0.0001allADNI_Basaia"), strict=False)
#my_nn.train()
#my_nn.to(device)
learning_rates = [0.0001]   
criterion = nn.CrossEntropyLoss()
n_datapoints = len(Target)
print(n_datapoints)
#print(Data)
testset_length = int(n_datapoints/15)
splits = 15

for l in range(splits):
    X_train, X_test, y_train, y_test = Data[:testset_length*splits], Data[-testset_length+1:], Target[:testset_length*splits],  Target[-testset_length+1:]
    save_df_val = pd.DataFrame()
    save_df_val['Path'] = X_test
    save_df_val['Group'] = y_test
    save_df_val['Dataset'] = Datasets[-testset_length+1:]
    save_df_val.to_csv("ValCVRound" + str(l) + "_3T2.csv")


    save_df_train = pd.DataFrame()
    save_df_train['Path'] = X_train
    save_df_train['Group'] = y_train
    save_df_train.to_csv("TrainCVRound" + str(l) + "_3T2.csv")
    Target = np.roll(Target, testset_length)
    Data = np.roll(Data, testset_length)
    Target = (list(Target))
    Data = list(Data)
    print("Number of 0 patients in traindataset:", y_train.count('CN'))
    print("Number of 1 patients in traindataset:", y_train.count('AD'))

    print("Number of 1 patients in valdataset:", y_test.count('CN'))
    print("Number of 0 patients in valdataset:", y_test.count('AD'))
    map = {'AD': 1, 'CN': 0}
   
    y_train = [map[word] for word in y_train]
    y_test = [map[word] for word in y_test]

    dataset_train = mydataLoader(X_train, y_train, 0) 
    dataset_val = mydataLoader(X_test, y_test, 1)

    dataloader_train = DataLoader(dataset_train, batch_size=3,
                        shuffle=True, num_workers=3)
    dataloader_val = DataLoader(dataset_val, batch_size=3,
                        shuffle=True, num_workers=3)
    for m in range(1):
        my_nn = NeuralNet.Net()
        my_nn.to(device)

        optimizer = torch.optim.Adam(my_nn.parameters(), learning_rates[0])
        losses = []
        count = 0
        accepoch = []
        accuracyVal = []
        accuracyTrain = []
        lossepoch = []
        v_lossepoch = []

        for epoch in range(140):  # loop over the dataset multiple times
            train_correct = 0
            val_correct = 0
            total_val = 0
            total_train = 0
               
            running_loss = 0.0
            v_running_loss = 0.0
            for i, (data, target) in enumerate(dataloader_train):
                #if i > 100:
                #    break
                #batch_size = (10,4,4,2)
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data.to(device), ((torch.tensor(target))).to(device)
                #print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = my_nn(inputs)
                
                loss = criterion(outputs, labels)
                #losses.append(loss)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                count += 1
                total_train += len(target)
                #print(total_train)
                
                predicted =  torch.max(outputs.data, 1)[1]
                #print(predicted)
                train_correct += (labels.squeeze() == predicted.squeeze()).sum().item()
                    # Iterate through test dataset
               # print(train_correct)
            for j, (v_images, v_labels) in enumerate(dataloader_val):
                val = v_images.to(device)
                v_labels  = (torch.tensor(v_labels)).to(device)
                # Forward propagation
                v_outputs = my_nn(val)
                total_val += len(v_labels)
                v_loss = criterion(v_outputs, v_labels)
                v_running_loss += v_loss.item()
                        # Get predictions from the maximum value
                v_predicted =  torch.max(v_outputs.data, 1)[1]

                        # Total number of labels
                print("v1", v_predicted, "v2", v_labels)

                val_correct += (v_predicted.squeeze() == v_labels.squeeze()).sum().cpu().item()

                    # store loss and iteration

            if ((epoch % 10 == 0) and (epoch > 100)):
            	torch.save(my_nn.state_dict(), "new_best-model_epoch_" + str(epoch) + "l=" + str(l) + "6layersCV_ALL")
            print('new augmentation Learningrate: {} Testrun nr: {} Epoch: {}  Train_Loss: {}  Val_Loss: {} Train Accuracy: {} % Val Accuracy: {} %'.format(l, m, epoch, running_loss/len(dataloader_train), v_running_loss/len(dataloader_val), (100*train_correct)/total_train, (100*val_correct)/total_val))
            lossepoch.append(running_loss/len(dataloader_train))
            v_lossepoch.append(v_running_loss/len(dataloader_val))
            accuracyVal.append((100*val_correct)/total_val)      
            accuracyTrain.append((100*train_correct)/total_train)      

            plt.figure()
            tloss, = plt.plot(lossepoch)
            vloss, = plt.plot(v_lossepoch)
            tloss.set_label('Training Loss')
            vloss.set_label('Validation loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Learningrate: " + str(l) + "6layersCV_ALL")
            plt.savefig('round' + str(l+1) + 'allADNI_6layersCV_ALL.png')


        torch.save(my_nn.state_dict(), "new_best-model_learningrate_" + str(l)  )
print('Finished Training')
