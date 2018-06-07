import os
import torch
import torchvision.transforms as transforms
import pickle

class rnn_dataset():
    def __init__(self, path):
        self.path = path
        self.usage = os.path.basename(path)
        self.length = len([name for name in os.listdir(path) if name.endswith('.pkl')])
    
    def __getitem__(self, index): 
        xy_file = open(os.path.join(self.path,str(index+1)+'.pkl'),'rb')
        xy = pickle.load(xy_file)
        #print(type(xy))
        return xy

    def __len__(self):
        return self.length





