import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import scipy.misc
import os, cv2, random
import shutil
import scipy.misc
import csv
import time
import skvideo.io
from skimage.io import imsave
from resnet import resnet50

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        # Everything except the last linear layer
        print('arch.starswith(\'resnet\')')
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax()
        )
        self.modelName = 'resnet'
        for p in self.features.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        ## get features
        f = self.features(x)
        f = f.view(f.size(0), -1)
        return f


'''
remove original hw5_rnn 
and creates hw5_rnn with :
hw5_rnn/train/0/
hw5_rnn/train/1/
hw5_rnn/train/2/
...
hw5_rnn/train/10/
hw5_rnn/valid/0/
hw5_rnn/valid/1/
hw5_rnn/valid/2/
...
hw5_rnn/valid/10/
'''
def recreate_dir(dest_dir):
   
  # remove if dest_dir already exists 
  if os.path.exists(dest_dir):
     shutil.rmtree(dest_dir)
     os.mkdir(dest_dir)
  
  # create_dir
  data_usage  = ['train','valid']
  if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)
  for i in data_usage:
    usage_dir = os.path.join(dest_dir,i)
    if not os.path.exists(usage_dir):
      os.mkdir(usage_dir)


def process_clip(model,video_path, nframe):
    frame_seq = skvideo.io.vread(video_path) 
    frame_seq = np.asarray(frame_seq)
    #print('frame_seq.shape={}'.format(frame_seq.shape))
    frame_seq = np.moveaxis(frame_seq,3,1)
    len_frames = frame_seq.shape[0] 
    step = int(len_frames/nframe)
    index = np.array([step*i for i in range(nframe)]) 
    frame_seq = frame_seq[index,:,:,:]
    frame_seq = torch.FloatTensor(frame_seq) 
    frame_seq = torch.autograd.Variable(frame_seq,requires_grad=False)
    #print('frame_seq.shape={}'.format(frame_seq.shape))
    feature_seq = model(frame_seq)
    return feature_seq

'''
processes raw videos from 
data_dir = HW5_data
to pkls (by assigned dsr)
dest_dir = hw5_rnn

dest_dir contains :
hw5_rnn/train/0/OP01-R01-PastaSalad-66680-6813.pkl
train can be train/valid
0 can be 0,1,..,10
'''


def data_to_framelist(model,data_dir,dest_dir,image_size,nframe):
  # start time
  start_time = time.time()
  
  # recreate dest_dir
  recreate_dir(dest_dir)

  # process video to frame
  data_usage  = ['train','valid']
  for i in data_usage:
    csv_file = os.path.join(data_dir,'TrimmedVideos','label','gt_'+i+'.csv')
    usage_path = os.path.join(data_dir,'TrimmedVideos','video',i)
    reader = csv.DictReader(open(csv_file,'r'))
    count = 0
    for row in reader:
      video_index = row['Video_index']
      if (int(video_index)%20==0):
        print(video_index)
      video_name = row['Video_name']
      video_category = row['Video_category']
      label = row['Action_labels']
      video_cat_path = os.path.join(usage_path,video_category)
      video_names = [file for file in os.listdir(video_cat_path) if file.startswith(video_name)]
      for video in video_names :
        # video_path should be something like HW5_data/TrimmedVideos/video/train/OP01-R01-PastaSalad/OP01-R01-PastaSalad-66680-68130.......mp4
        video_path = os.path.join(usage_path,video_category,video)
        x = process_clip(model,video_path, nframe)
        x = x.cpu().numpy()
        #print('type(x)={}'.format(type(x)))
        #print('x.shape={}'.format(x.shape))
        #x = int(label)
        #y = np.eye(11)[int(label)]
        y = int(label)
        #print(y)
        save_path = os.path.join(dest_dir,i,video_index+'.pkl')
        with open(save_path,'wb') as f:
            pkl.dump((x,y),f)

  # end time
  elapsed_time = time.time() - start_time 
  print('Processing data to frame takes:', int(elapsed_time / 60), 'minutes')
 
if __name__ == '__main__':   
  model = FineTuneModel(resnet50(),'resnet50',11)
  model = torch.nn.DataParallel(model).cuda()
  checkpoint = torch.load('model_best.pth.tar')
  model.load_state_dict(checkpoint['state_dict'])
  data_to_framelist(model,'HW5_data', 'hw5_rnn', (240,320), 16) 
