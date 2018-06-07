import scipy.misc
import os, cv2, random
import shutil
import numpy as np
import scipy.misc
import csv
import time
import skvideo.io
import pickle as pkl
import torch
from skimage.io import imsave
'''
remove original hw5_cnn 
and creates hw5_cnn with :
hw5_cnn/train/0/
hw5_cnn/train/1/
hw5_cnn/train/2/
...
hw5_cnn/train/10/
hw5_cnn/valid/0/
hw5_cnn/valid/1/
hw5_cnn/valid/2/
...
hw5_cnn/valid/10/
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

def process_clip(video_path, nframe):
    frames = skvideo.io.vread(video_path)
    len_frames = frames.shape[0]
    if(len_frames<16):
      print('{}<16'.format(video_path))
    step = int((len_frames-1)/(nframe-1))
    index = np.array([step*i for i in range(nframe-1)]+[len_frames-1])
    frames = frames[index,:,:] / 255
    frames = torch.FloatTensor(frames)
    #print(frames.shape)
    return frames
  

'''
processes raw videos from 
data_dir = HW5_data
to pngs (by assigned dsr)
dest_dir = hw5_cnn

dest_dir contains :
hw5_cnn/train/0/OP01-R01-PastaSalad-66680-6813_1.png
train can be train/valid
0 can be 0,1,..,10
'''
def data_to_frames(data_dir,dest_dir,nframe):
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
      if(int(video_index)%20==0):
        print(video_index)
      video_name = row['Video_name']
      video_category = row['Video_category']
      label = row['Action_labels']
      video_cat_path = os.path.join(usage_path,video_category)
      video_names = [file for file in os.listdir(video_cat_path) if file.startswith(video_name)]
      assert len(video_names) == 1
      for video in video_names :
        # video_path should be something like HW5_data/TrimmedVideos/video/train/OP01-R01-PastaSalad/OP01-R01-PastaSalad-66680-68130.......mp4
        video_path = os.path.join(usage_path,video_category,video)
        x = process_clip(video_path, nframe)
        y = int(label)
        save_path = os.path.join(dest_dir,i,video_index+'.pkl')
        with open(save_path,'wb') as f:
          pkl.dump((x,y),f)

  # end time
  elapsed_time = time.time() - start_time 
  print('Processing data to frame takes:', int(elapsed_time / 60), 'minutes')
 
if __name__ == '__main__':
  data_to_frames('HW5_data', 'hw5_cnn', 16) 
