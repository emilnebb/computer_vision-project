#!/bin/bash

python demo.py \
--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--input "../datasets/tdt4265/dataset/images/val/*" \
--output output/
#--webcam \
#--opts MODEL.DEVICE cpu \