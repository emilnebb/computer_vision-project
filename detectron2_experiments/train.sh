#!/bin/bash
export CUDA_VISIBLE_DEVICES=4 # just set GPU:10 visible; torch can't handle more than 16 GPUs by default

python trainer.py \
--config-file config.yaml \
--num-gpus 1 
#SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025 \
#--opts MODEL.DEVICE cpu \
#--webcam \
#--input "../datasets/tdt4265/dataset/images/val/*" \
#--output output/
