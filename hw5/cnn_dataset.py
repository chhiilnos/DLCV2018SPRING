import os
import torch
import torchvision.transforms as transforms
import numpy as np

class cnn_dataset():
    def __init__(self, path):
        self.path = path
        self.usage = os.path.basename(path)
        if self.usage == 'train':
            self.length = 3236
        elif self.usage == 'valid':
            self.length = 517
    
    def __getitem__(self, index): 
        xy = np.load(os.path.join(self.path,str(index+1)+'.npy'))
        x = torch.FloatTensor(xy[0]) / 255
        y = xy[1]
        return x, y

    def __len__(self):
        return self.length





