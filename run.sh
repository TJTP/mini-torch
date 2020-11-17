#!/bin/bash

####train#####
#python fit_func.py \
#--train --save \
#--train_epoch_num 300 \
#--train_batch_size 64 --lr 3e-4 \
#--predict --draw \

#####predict####
python fit_func.py \
--predict --draw \
--model_dir ./models/model_642_300 \

