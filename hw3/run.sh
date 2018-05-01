mkdir report/$1
#!/bin/bash
# 1
echo $1
echo "epoch 1 start"
python3 train.py $1 1 False
cp Models/$1/checkpoint_weights.hdf5 Models/$1/checkpoint_weights_1.hdf5
cp Models/$1/model.json Models/$1/model_1.json
python3 inference.py data/validation results/validation $1 1
mkdir report/$1/1
cp results/validation/0008_mask.png report/$1/1/
cp results/validation/0097_mask.png report/$1/1/
cp results/validation/0107_mask.png report/$1/1/
python3 mean_iou_evaluate.py -g data/validation -p results/validation > report/$1/1.txt
echo "epoch 1 start"

# 5
echo $1
echo "epoch 2 start"
python3 train.py $1 4 True
cp Models/$1/checkpoint_weights.hdf5 Models/$1/checkpoint_weights_5.hdf5
cp Models/$1/model.json Models/$1/model_5.json
python3 inference.py data/validation results/validation $1 5
mkdir report/$1/5
cp results/validation/0008_mask.png report/$1/5/
cp results/validation/0097_mask.png report/$1/5/
cp results/validation/0107_mask.png report/$1/5/
python3 mean_iou_evaluate.py -g data/validation -p results/validation > report/$1/5.txt
echo "epoch 5 done"

# 10 and more
for i in {10..50..5}
  do
    echo "epoch $i start"
    python3 train.py $1 5 True
    cp Models/$1/checkpoint_weights.hdf5 Models/$1/checkpoint_weights_$i.hdf5
    cp Models/$1/model.json Models/$1/model_$i.json
    python3 inference.py data/validation results/validation $1 $i
    mkdir report/$1/$i
    cp results/validation/0008_mask.png report/$1/$i/
    cp results/validation/0097_mask.png report/$1/$i/
    cp results/validation/0107_mask.png report/$1/$i/
    python3 mean_iou_evaluate.py -g data/validation -p results/validation > report/$1/$i.txt
    echo "epoch $i done"
  done

