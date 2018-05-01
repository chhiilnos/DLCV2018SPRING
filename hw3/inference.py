import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

from models import *

def class2rgb(a):
    a = bin(a)[2:]
    zeros = 3 - len(a);
    for i in range (zeros):
      a = "0"+a
    b = 255*np.asarray([int(a[2]),int(a[1]),int(a[0])])
    return b

def inference(model_name, weight_file, image_size, data_dir, save_dir, label_suffix='_mask.png', data_suffix='_sat.jpg'):

    ############## load model and checkpoint #################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # mean_value = np.array([104.00699, 116.66877, 122.67892])
    batch_shape = (1, ) + image_size + (3, )
    save_path = os.path.join(current_dir, 'Models/'+model_name)
    #model_path = os.path.join(save_path, "model_"+sys.argv[4]+".json")
    checkpoint_path = os.path.join(save_path, weight_file)
    # model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)
    # model.summary()

    #################### start inference #####################
    total = 0
    image_list = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]
    for image_name in image_list:
        # load image and label
        total += 1
        print('#%d: %s' % (total,image_name))
        image = Image.open('%s/%s' % (data_dir,image_name))
        image = img_to_array(image)  # , data_format='default')

        # padding and preprocessing
        img_h, img_w = image.shape[0:2]
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        image = np.lib.pad(image, ((int(pad_h/2), pad_h - int(pad_h/2)), (int(pad_w/2), pad_w -int(pad_w/2)), (0, 0)), 'constant', constant_values=0.)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # predict label
        result = model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)
        #print("result.shape = {}".format(result.shape)) 
        result_img = np.asarray([[class2rgb(result[i][j]) for j in range(512)] for i in range(512)]).astype(np.uint8)
        #print("result_img.shape = {}".format(result_img.shape)) 
        #print("result_img[0] = {}".format(result_img[0]))
        #print(result_img.dtype)
        result_img = Image.fromarray(result_img, mode='RGB')
        #result_img.palette = label.palette
        #result_img = result_img.crop((int(pad_w/2), int(pad_h/2), int(pad_w/2)+img_w, int(pad_h/2)+img_h))
        # result_img.show(title='result')
        
        # save image  
        result_img.save(os.path.join(save_dir, image_name[:-8]+label_suffix))

if __name__ == '__main__':    
    ########### model name and weight file ###################
    model_name = sys.argv[3] 
    #model_name = 'AtrousFCN_Vgg16_16s' 
    #model_name = 'FCN_Vgg16_32s' 
    # model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    # model_name = 'DenseNet_FCN'
    weight_file = 'checkpoint_weights_'+sys.argv[4]+'.hdf5'
    
    #################### start inference #####################
    # function parameters
    image_size = (512, 512)
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]
    
    results = inference(model_name, weight_file, image_size, data_dir, save_dir)
