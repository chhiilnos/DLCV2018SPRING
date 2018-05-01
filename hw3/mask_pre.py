import pickle
import os
import numpy as np
import cv2

######## train, val file path and data, label dir ########
current_dir = os.path.dirname(os.path.realpath(__file__))
train_file_path = os.path.join(current_dir,'data/train_files.txt')
val_file_path		= os.path.join(current_dir,'data/val_files.txt')
train_dir = os.path.join(current_dir,'data/train')
val_dir = os.path.join(current_dir,'data/validation')
label_suffix='_mask.png'
class_suffix = '_label.pkl'


with open(train_file_path,"r") as tfp :
	lines = tfp.readlines()
	for line in lines:
		print(line)
		line = line[:-1]
		path = ("{}/{}{}".format(train_dir, line, label_suffix))
		img = cv2.imread(path)
		label_img = np.zeros((512,512))
		for x,y in [(x,y) for x in range(512) for y in range(512)]:
			a = img[x][y]/255
			label_img[x][y] = 4*a[0]+2*a[1]+a[2]
		label_img.dump("{}/{}{}".format(train_dir,line,class_suffix))

with open(val_file_path,"r") as vfp :
	lines = vfp.readlines()
	for line in lines:
		print(line)
		line = line[:-1]
		path = ("{}/{}{}".format(val_dir, line, label_suffix))
		img = cv2.imread(path)
		label_img = np.zeros((512,512))
		for x,y in [(x,y) for x in range(512) for y in range(512)]:
			a = img[x][y]/255
			label_img[x][y] = 4*a[0]+2*a[1]+a[2]
		label_img.dump("{}/{}{}".format(val_dir,line,class_suffix))

