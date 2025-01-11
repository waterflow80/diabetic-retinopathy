#!/bin/bash

drdetector --mode train --data_path ./data/ --images_dir train_images_part && \
drdetector --mode test --data_path ./data/ --images_dir test_images_part --model_path \
./models/cnn_alexnet_freeze_backbone_False.pth


