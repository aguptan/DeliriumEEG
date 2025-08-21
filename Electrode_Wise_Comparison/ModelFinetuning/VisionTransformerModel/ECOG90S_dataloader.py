from torch.utils.data import Dataset
import os
import PIL
import pandas as pd

from PIL import Image

from torchvision.datasets import ImageFolder
import pandas as pd

import os
import numpy as np
import glob
import torch

import random

from numpy.random import randint
from torchvision import transforms
from natsort import natsorted

class ECOG90S_train(Dataset):
    """ECOG90s train channel wise dataset."""

    def __init__(self, data_location,in_chans=3,transform=None):
        self.data_location = data_location     
        self.channel = in_chans
        self.dataframe=pd.read_csv(os.path.join(data_location,'train.csv'),names=['name','label'])
        self.transform = transform
        

    def __len__(self):
        return len(self.dataframe)
    
        print("Total number of train images:")
        print(len(self.dataframe))

            
        print("First 5 train image names:")
        print(self.dataframe.head())

    def __getitem__(self, idx):
        img_path=os.path.join(self.data_location, str(self.dataframe.iloc[idx]['name']) + '.png')        
        label = self.dataframe.iloc[idx]['label']
        
        if self.channel == 1: 
            image = PIL.Image.open(img_path).convert('L')
        elif self.channel == 3: 
            image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(image) , label
        else: 
            return image,label
       
class ECOG90S_test(Dataset):
    """ECOG90s train channel wise dataset."""

    def __init__(self, data_location,in_chans=3,transform=None):
        self.data_location = data_location     
        self.channel = in_chans
        self.dataframe=pd.read_csv(os.path.join(data_location,'test.csv'),names=['name','label'])
        self.transform = transform
        

    def __len__(self):
        return len(self.dataframe)
    
    
        print("Total number of test images:")
        print(len(self.dataframe))

            
        print("First 5 test image names:")
        print(self.dataframe.head())

    def __getitem__(self, idx):
        img_path=os.path.join(self.data_location, str(self.dataframe.iloc[idx]['name']) + '.png')        
        label = self.dataframe.iloc[idx]['label']
        
        if self.channel == 1: 
            image = PIL.Image.open(img_path).convert('L')
        elif self.channel == 3: 
            image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(image) , label
        else: 
            return image,label



class DataAugmentation_finetune(object):
    def __init__(self, istrain, input_size, in_chans=1):
        
        if in_chans == 3:
            mean_, std_ = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean_, std_ = (0.485), (0.229)

        normalize = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean_, std_)])
        
        
        if istrain == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(
                    0.8, 1), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)], p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(
                    kernel_size=(5, 9), sigma=(0.1, 5))], p=0.2),
                transforms.RandomApply(
                    [transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.2),
                transforms.RandomApply(
                    [transforms.RandomAutocontrast()], p=0.2),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([input_size, input_size], interpolation=Image.BICUBIC),
                normalize
            ])

    def __call__(self, image):
        return self.transform(image)

