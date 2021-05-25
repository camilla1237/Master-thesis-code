
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from nibabel.testing import data_path
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob
import torchio as tio
import pandas as pd

num_classes = 1

# Create CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 50,5, 1)
        self.conv_layer2 = self._conv_layer_set(50, 50, 5, 2)
        self.conv_layer3 = self._conv_layer_set(50, 100, 3, 1)
        self.conv_layer4 = self._conv_layer_set(100, 250, 3, 2)
        self.conv_layer5 = self._conv_layer_set(250, 400, 3, 1)
        self.conv_layer6 = self._conv_layer_set(400, 650, 3, 2)
        self.conv_layer7 = self._conv_layer_set(650, 800, 3, 1)
        self.conv_layer8 = self._conv_layer_set(800, 950, 3, 2)
        self.conv_layer9 = self._conv_layer_set(950, 1100, 3, 1)
        self.conv_layer10 = self._conv_layer_set(1100, 1250, 3, 2)
        self.conv_layer11 = self._conv_layer_set(1250, 1400, 3, 1)
        self.conv_layer12 = self._conv_layer_set(1400, 1600, 3, 2)
 
#        self.fc1 = nn.Linear(92345408,128) #8 layers, new imsize       
#        self.fc1 = nn.Linear(6912000, 128) #4 layers, new imsize
#        self.fc1 = nn.Linear(3456000, 128) #2 layers 
#        self.fc1 = nn.Linear(1769472, 128) #4 layers
#        self.fc1 = nn.Linear(262144, 128) #8 layers 
        self.fc1 = nn.Linear(19200, 128) #Basaia layers

#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.out = nn.Sigmoid()
        self.batch=nn.BatchNorm1d(128)
        self.batch1=nn.BatchNorm1d(512)
  
        self.drop=nn.Dropout(p=0.15)        
        
    def _conv_layer_set(self, in_c, out_c, ks, strides, batch=False):
        if batch:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(ks, ks, ks), padding=0, stride=strides),
                nn.ReLU(), 
                nn.BatchNorm3d(out_c)       
            )
        else:
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(ks, ks, ks), padding=1, stride=strides),
                nn.ReLU(),
#        nn.MaxPool3d((2, 2, 2))
#        nn.Dropout(p=0.15)
            )
        return conv_layer
    

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        print(out.size())
        out = self.conv_layer2(out)
        #out = self.drop(out)
        out = self.conv_layer3(out)
        print(out.size())
        out = self.conv_layer4(out)
        print(out.size())
#        out = self.drop(out)
        out = self.conv_layer5(out)
        print(out.size())
#        out = self.drop(out)
        out = self.conv_layer6(out)

        print(out.size())
        out = self.conv_layer7(out)
        print(out.size())

        out = self.conv_layer8(out)

        print(out.size())
        out = self.conv_layer9(out)

        print(out.size())
        out = self.conv_layer10(out)
        print(out.size())
        out = self.conv_layer11(out)
        print(out.size())
        out = self.conv_layer12(out)
     

        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
     #   out = self.batch1(out)
    
      #  out = self.fc2(out)
      #  out = self.relu(out)
      #  out = self.fc3(out)
      #  out = self.relu(out)
        out = self.batch(out)

       # out = self.fc4(out)
       # out = self.relu(out)

 #       out = self.drop(out)
        out = self.fc2(out)
        out = self.out(out)
        
        return out

