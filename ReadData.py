import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from nibabel.testing import data_path
from matplotlib import pyplot as plt
import matplotlib

import torchvision as tv
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
from numpy.random import randint as ri
import NeuralNetBasaia as NeuralNet
import torchio as tio
import pandas as pd
from sklearn.model_selection import train_test_split


class mydataLoader(Dataset):
    def __init__(self, paths, label, testvalortrain):
        self.testvalortrain = testvalortrain
        self.image_paths = paths
        self.labels = label
           
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        transformR = tv.transforms.Compose([tv.transforms.RandomRotation(ri(0,15))])
        transformC = tv.transforms.Compose([tv.transforms.RandomResizedCrop((256, 160)), tv.transforms.Resize((145,121))])
        transformD = tv.transforms.Compose([tio.transforms.RandomElasticDeformation(ri(7,11), ri(11,16))])
        transformF = tv.transforms.Compose([tio.transforms.RandomFlip(('P'), 1.0)])

        path = self.image_paths[index]
        label = self.labels[index]
        image = tio.ScalarImage(path)
        image = np.array(image)
        image = torch.from_numpy(image).float()
#        print("orig", image.size())
#        matplotlib.image.imsave('name.png', image[0,:,100,:])
        Augment = ri(0,2)
        if ((Augment == 1) and (self.testvalortrain==0)):
            cAugment = ri(0,3)
            if (cAugment == 0):
                image = transformR(image)
            if (cAugment == 1):
                image = transformF(image)
            if (cAugment == 2):
                image = transformD(image)
        return image, label
    
