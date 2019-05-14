from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils import data
import math
import glob

data_transforms = {
    'TRAIN' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'TEST' : transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}

def load_train_val_test(train_path, test_path, valid_split=0.2):
    """
    Inputs:
        train_path: path to training data
        test_path: path to test data
        valid_split: proportion of training data to be set aside as validation set
    Returns:
        np_train_set: numpy array of training data with dims (n_train x 3 x 224 x 224)
        np_train_labels: numpy array of training label with dims (n_train, )
        np_valid_set_normalized: numpy array of normalized training data with dims (n_valid x 3 x 224 x 224)
        np_valid_labels: numpy array of training label with dims (n_valid, )
        np_test_set: numpy array of test data with dims (n_test x 3 x 224 x 224)
        np_test_labels: numpy array of training label with dims (n_test, )
    """
    train_set = datasets.ImageFolder(train_path, data_transforms['TRAIN'])
    test_set = datasets.ImageFolder(test_path, data_transforms['TEST'])
    train_size = len(train_set)
    indices = list(range(train_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(valid_split * train_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        sampler=train_sampler,
        num_workers=0,
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        sampler=valid_sampler,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )
    train_images, train_labels = iter(train_loader).next()
    valid_images, valid_labels = iter(valid_loader).next()
    test_images, test_labels = iter(test_loader).next()
    np_train_set = train_images.numpy()
    np_train_labels = train_labels.numpy()
    np_valid_set = valid_images.numpy()
    np_valid_labels = valid_labels.numpy()
    np_test_set = test_images.numpy()
    np_test_labels = test_labels.numpy()
    return np_train_set, np_train_labels, np_valid_set, np_valid_labels, np_test_set, np_test_labels

def normalize_train(np_train_set):
    """
    Inputs:
        np_train_set: numpy array of training data with dims (n_train x 3 x 224 x 224)
    Returns:
        np_train_set: numpy array of normalized training data (subtracted mean of training batch, divided by std of training batch) 
        mean: mean of training batch
        std: std of training batch
    """
    mean = np.mean(np_train_set)
    std = np.std(np_train_set)
    np_train_set -= mean
    np_train_set /= std
    return np_train_set, mean, std

def normalize_test(np_test_set, train_mean, train_std):
    """
    Inputs:
        np_test_set: numpy array of training data with dims (n_test x 3 x 224 x 224)
        train_mean: mean of training batch
        test_std: std of training batch
    Returns:
        np_test_set: numpy array of normalized test data (subtracted mean of training batch, divided by std of training batch)
    """
    np_test_set -= train_mean
    np_test_set /= train_std
    return np_test_set
     
def get_train_val_test(train_path, test_path, valid_size):
    """
    Inputs:
        train_path: path to training data
        test_path: path to test data
        valid_size: proportion of training data to be set aside as validation set
    Returns:
        np_train_set_normalized: numpy array of normalized training data with dims (n_train x 3 x 224 x 224)
        np_train_labels: numpy array of training label with dims (n_train, )
        np_valid_set_normalized: numpy array of normalized training data with dims (n_valid x 3 x 224 x 224)
        np_valid_labels: numpy array of training label with dims (n_valid, )
        np_test_set_normalized: numpy array of normalized test data with dims (n_test x 3 x 224 x 224)
        np_test_labels: numpy array of training label with dims (n_test, )
    """
    np_train_set, np_train_labels, np_valid_set, np_valid_labels, np_test_set, np_test_labels = load_train_val_test(train_path, test_path, valid_size)
    np_train_set_normalized, train_mean, train_std = normalize_train(np_train_set)
    np_valid_set_normalized = normalize_test(np_valid_set, train_mean, train_std)
    np_test_set_normalized = normalize_test(np_test_set, train_mean, train_std)
    return np_train_set_normalized, np_train_labels, np_valid_set_normalized, np_valid_labels, np_test_set_normalized, np_test_labels

get_train_val_test('train_data', 'test_data', valid_size=0.2)
