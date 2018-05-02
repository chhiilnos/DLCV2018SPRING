import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
#from keras.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *


def top(x, input_shape, classes, activation, weight_decay):

    x = Conv2D(classes, (1, 1), activation='linear',
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)

    if K.image_data_format() == 'channels_first':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    if activation is 'sigmoid':
        x = Reshape((row * col * classes,))(x)

    return x

def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=8):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print(x.shape) 

    # Load vgg weight first
    model = Model(img_input, x)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(current_dir,'weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights(weights_path, by_name=True)
    
    # New sequential model for the convenience of concatenating
    new_model = Sequential()
    for layer in model.layers:
      new_model.add(layer)
    
    # Convolutional layers transfered from fully-connected layers
    top_model = Sequential()
    top_model.add(Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay), input_shape = (10,10,512)))

    # top_model.add(Dropout(0.5))
    top_model.add(Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay)))

    # top_model.add(Dropout(0.5))
    
    #classifying layer
    top_model.add(Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay)))
    top_model.add(Conv2DTranspose(filters = classes, kernel_size = (64, 64), strides = (32, 32), padding='same'))
    new_model.add(top_model) 
    top_model.summary()
    return new_model

def AtrousFCN_Vgg16_32s_aad(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=8):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay), trainable = False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    print(x.shape) 
    
    # Load vgg weight first
    model = Model(img_input, x)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(current_dir,'weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights(weights_path, by_name=True)
    
    # New sequential model for the convenience of concatenating
    new_model = Sequential()
    for layer in model.layers:
      new_model.add(layer)
    
    # Convolutional layers transfered from fully-connected layers
    top_model = Sequential()
    top_model.add(Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2,2), name='ast1', kernel_regularizer=l2(weight_decay), input_shape = (10,10,512)))
    top_model.add(Conv2D(4096, (1, 1), activation='relu', padding='same', dilation_rate=(2,2), name='ast2', kernel_regularizer=l2(weight_decay)))
    top_model.add(Dropout(0.5))
    
    #classifying layer
    top_model.add(Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay)))
    top_model.add(Conv2DTranspose(filters = classes, kernel_size = (64, 64), strides = (32, 32), padding='same'))
    new_model.add(top_model) 
    top_model.summary()
    return new_model

