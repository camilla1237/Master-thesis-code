
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

num_classes = 2

# Create CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 32,3, 1)
        self.conv_layer2 = self._conv_layer_set(32, 64, 3, 2)
        self.conv_layer3 = self._conv_layer_set(64, 128, 3, 1)
        self.conv_layer4 = self._conv_layer_set(128, 256, 3, 2)
        self.conv_layer5 = self._conv_layer_set(256, 512, 5, 1)
        self.conv_layer6 = self._conv_layer_set(512, 1024, 3, 2)
        self.conv_layer7 = self._conv_layer_set(1024, 2048, 3, 1)
        self.conv_layer8 = self._conv_layer_set(2048, 4096, 3, 2)
 
#        self.fc1 = nn.Linear(92345408,128) #8 layers, new imsize       
#        self.fc1 = nn.Linear(6912000, 128) #4 layers, new imsize
#        self.fc1 = nn.Linear(3456000, 128) #2 layers 
        self.fc1 = nn.Linear(1734656, 128) #4 layers
#        self.fc1 = nn.Linear(262144, 128) #8 layers 

        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.out = nn.Sigmoid()
        self.batch=nn.BatchNorm1d(128)
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
                nn.Conv3d(in_c, out_c, kernel_size=(ks, ks, ks), padding=0, stride=strides),
                nn.ReLU(),
#        nn.MaxPool3d((2, 2, 2))
#        nn.Dropout(p=0.15)
            )
        return conv_layer



    def forward(self, x):
        # Set 1
        out3 = x
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
     #   #out = self.drop(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
#        out = self.drop(out)
        out = self.conv_layer5(out)
#        out = self.drop(out)
        out = self.conv_layer6(out)
#        out = self.conv_layer7(out)
#        out = self.conv_layer8(out)
#        out = self.conv_layer9(out)
     #   out = self.conv_layer10(out)
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out2 = out
        out = self.batch(out)

        out = self.drop(out)
        out = self.fc2(out)
        out = self.out(out)
        
        return out, out2

