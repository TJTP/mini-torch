#!/bin/bash

####train#####
python fit_func.py \
--train --save \
--data_dir ./data/ \
--train_epoch_num 300 \
--train_batch_size 64 --lr 6e-4 \
--predict --draw 

#####predict####
# python fit_func.py \
# --predict --draw \
# --model_dir ./models/[model_name]\

