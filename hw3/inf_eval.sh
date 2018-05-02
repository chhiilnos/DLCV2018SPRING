#!/bin/bash
echo $1
mkdir report/$1
mkdir report/$1/mask
python3 inference.py data/validation report/$1/mask $1 50
python3 mean_iou_evaluate.py -g data/validation -p report/$1/mask > report/$1/meaniou.txt

