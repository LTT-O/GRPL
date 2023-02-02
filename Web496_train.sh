#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6,7
export DATA='/data1/Web496/web-bird/'
export N_CLASSES=200
export DATA_TYPE='bird-gcn-0813'

python main.py \
--data ${DATA} \
--lr 1e-3 \
--batch-size 16 \
--weight-decay 1e-4 \
--num_class ${N_CLASSES} \
--data_type ${DATA_TYPE} 
# --resume "/data0/LTT/GPL/exp/checkpoint/checkpoint_bir-gcn-0812-pro-cls-S.pth"



