#!/bin/bash
# 1
echo $1
python3 train.py $1 50 False
cp Models/$1/checkpoint_weights.hdf5 Models/$1/checkpoint_weights_50.hdf5
cp Models/$1/model.json Models/$1/model_50.json


