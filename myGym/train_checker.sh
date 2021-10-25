#!/bin/sh
rm -r ./trained_models/check
python train.py --config ./configs/train_check.json --steps 20000
python test.py -cfg ./trained_models/check/reach_table_kuka_joints_gt_ppo2/train.json
rm -r ./trained_models/check