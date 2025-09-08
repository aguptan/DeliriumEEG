# -*- coding: utf-8 -*-
"""
@author: aguptan (minor changes from lyonm's original code)

PyTorch Dataset and DataLoader classes for ECoG image data.

This script uses a unified Dataset class for training and testing downstream
for loading ECoG data from corresponding CSV files. It also includes the 
`DataAugmentation_finetune` class to handle image transformations for training
and testing.

Changelog
---------
Last Modified: June 10, 2025

Summary of Major Changes:
1.  **Enabled Patient ID Tracking in Test Set:**
    - The `ECOG90S_test` class was modified to support patient-wise analysis.
    - The `__getitem__` method now reads the patient ID from the CSV file for 
      each test sample.
    - It was changed to return three items: `(image, label, patient_id)`,
      instead of the original two.

2.  **No Change to Training Set:**
    - The `ECOG90S_train` class was intentionally left unchanged, as the 
      patient ID is not required during the model's training phase.

"""
import os
import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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

# For LeaveOnePairOut
from itertools import product
from sklearn.utils import check_random_state


class LOO_ECOG_Dataset(Dataset):    
    """A unified ECOG dataset that works directly with pandas DataFrames."""

    def __init__(self, data_location, dataframe, in_chans=3, transform=None):
    
        self.data_location = data_location
        self.channel = in_chans
        self.transform = transform
        self.dataframe = dataframe.reset_index(drop=True)
        self.patient_ids = self.dataframe['patient_id'].unique()
        
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.data_location, str(self.dataframe.iloc[idx]['filename'])) + '.png'
        label = self.dataframe.iloc[idx]['label']
        patient_id = self.dataframe.iloc[idx]['patient_id']

        if self.channel == 1:
            image = PIL.Image.open(img_path).convert('L')
        elif self.channel == 3:
            image = PIL.Image.open(img_path).convert('RGB')
        else:
            raise ValueError(f"Unsupported number of input channels: {self.channel}")
        
        if self.transform:
            image = self.transform(image)

        
        return image, label, patient_id



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




class LeaveOnePairOut:
    """
    Cross-validator for Leave-One-Pair-Out CV.
    Each fold holds out one class 0 patient and one class 1 patient.
    Remaining patients are used for training.
    """
    def __init__(self, labels, patient_ids, random_state=None):
        self.labels = np.array(labels)
        self.patient_ids = np.array(patient_ids)
        self.random_state = check_random_state(random_state)

        # Unique patients and their class
        self.patient_df = self._build_patient_df()

        # Split patients by class
        self.class0_patients = self.patient_df[self.patient_df['label'] == 0]['patient_id'].tolist()
        self.class1_patients = self.patient_df[self.patient_df['label'] == 1]['patient_id'].tolist()

    def _build_patient_df(self):
        # Drop duplicate patient entries
        unique_patients = {}
        for pid, label in zip(self.patient_ids, self.labels):
            if pid not in unique_patients:
                unique_patients[pid] = label
            else:
                if unique_patients[pid] != label:
                    raise ValueError(f"Patient {pid} has inconsistent labels.")
        return pd.DataFrame({'patient_id': list(unique_patients.keys()),
                             'label': list(unique_patients.values())})

    def split(self, X, y=None, groups=None):
        for p0, p1 in product(self.class0_patients, self.class1_patients):
            test_ids = [p0, p1]
            test_idx = np.where(np.isin(self.patient_ids, test_ids))[0]
            train_idx = np.where(~np.isin(self.patient_ids, test_ids))[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.class0_patients) * len(self.class1_patients)
