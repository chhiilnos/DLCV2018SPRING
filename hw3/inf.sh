#!/bin/bash
echo $1
mkdir $3
mkdir report/$1
mkdir report/$1/mask
python3 inference.py $2 report/$1/mask $1 50
cp report/$1/mask/* $3/

