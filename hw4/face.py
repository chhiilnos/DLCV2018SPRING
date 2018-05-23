from __future__ import print_function
from PIL import Image
from scipy import misc
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pandas as pd
import torch.utils.data as data

class FACE(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            # train list
            print("in train_list")
            train_csv = pd.read_csv(os.path.join(root,"train.csv"))
            #train_csv = pd.read_csv(os.path.join(root,"test.csv"))
            train_list = train_csv["image_name"] 
            
            # train data
            print("in train_data")
            self.train_data = []
            for png_file in train_list :
                png_frame = misc.imread(os.path.join(root,"train",png_file))
                #png_frame = misc.imread(os.path.join(root,"test",png_file))
                self.train_data.append(png_frame)
            
            # train labels
            print("in train_labels")
            self.train_labels = []
            for idx,png_file in enumerate(train_list) :
                label = train_csv.loc[idx,"Bangs":"Wearing_Lipstick"]
                self.train_labels.append(label.tolist())
        else:
            # test_list
            print("in test_list")
            test_csv = pd.read_csv(os.path.join(root,"test.csv"))
            test_list = test_csv["image_name"] 
            
            # test_data
            print("in test_data")
            self.test_data = []
            for png_file in test_list :
                png_frame = misc.imread(os.path.join(root,"test",png_file))
                self.test_data.append(png_frame)
            
            # test_labels
            print("in test_labels")
            self.test_labels = []
            for idx,png_file in enumerate(test_list) :
                #label = test_csv.loc[idx,"Bangs":"Wearing_Lipstick"]
                label = test_csv.loc[idx,"Bangs":]
                self.test_labels.append(label.tolist())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
