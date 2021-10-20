#!/bin/sh
rm -r ./trained_models/tester
python train.py --config ./configs/tester.json
python test.py -cfg ./trained_models/tester/reach_table_kuka_joints_gt_ppo2/train.json
rm -r ./trained_models/tester