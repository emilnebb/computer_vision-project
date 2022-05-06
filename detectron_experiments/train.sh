#!/bin/bash

python tools/plain_train_net.py \
--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#--opts MODEL.DEVICE cpu \
#--webcam \
#--input "../datasets/tdt4265/dataset/images/val/*" \
#--output output/