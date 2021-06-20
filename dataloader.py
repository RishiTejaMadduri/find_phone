
from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import random


def make_dataset(data_dir, names):
    items = []
    labels = []
    for i in names:
        items.append(os.path.join(data_dir, i))
        labels.append(names[i])
    return items, labels



#Custom PyTorch Dataset
class MyDataLoader(Dataset):
    def __init__(self, data_dir, joint_transform=None, sliding_crop=None, transform=None, target_transform=None):
        
        self.data_dir = data_dir
        self.file = self.data_dir + "labels.txt"
        self.df = pd.read_csv(self.file, sep=" ", names=["Name", "X", "Y"])
        self.dict_ids = {}
        
        #Creating a dict such that {Name.jpg} = [x, y] label
        for i in range(0, len(self.df.Name)):
            name = self.df.Name[i]
            self.dict_ids[name] = [self.df.X[i], self.df.Y[i]]
            
        #For each image instance, load the image and label
        self.imgs, self.labels = make_dataset(self.data_dir, self.dict_ids)
        
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        
        
    def __getitem__(self, index):
        
        img_path = self.imgs[index]
        label = self.labels[index]
        img = np.asarray(Image.open(img_path), dtype=np.float64)
        img = np.transpose(img, (2,0,1))
        return img, label
    
    def __len__(self):
        return len(self.imgs)

