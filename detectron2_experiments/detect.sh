#!/bin/bash
export CUDA_VISIBLE_DEVICES=11 # just set GPU:10 visible; torch can't handle more than 16 GPUs by default

python demo.py \
--config-file config.yaml \
--input "../datasets/tdt4265/images/val/*" \
--output detections/
#--webcam \
#--opts MODEL.DEVICE cpu \