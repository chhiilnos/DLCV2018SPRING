import os
import torch
import torchvision.transforms as transforms
import pickle

class cnn_dataset():
    def __init__(self, path):
        self.path = path
        self.usage = os.path.basename(path)
        if self.usage == 'train':
            self.length = 3236
        elif self.usage == 'valid':
            self.length = 517
    
    def __getitem__(self, index): 
        xy_file = open(os.path.join(self.path,str(index+1)+'.pkl'),'rb')
        xy = pickle.load(xy_file)
        #print(type(xy))
        return xy

    def __len__(self):
        return self.length





