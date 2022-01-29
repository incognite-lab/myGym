#!/bin/sh
#This is minimal unit test to check, whether the current version of myGym is able to train. 
# It will do simle train and evaluate in 2 minutes. If success rate is above 30 percdents, it works.
rm -r ./trained_models/check
python train.py --config ./configs/speed_reach.json --max_velocity .5 --max_force 30 --action_repeat 1
python test.py -cfg ./trained_models/check/reach_table_kuka_magnetic_joints_ppo2/train.json
rm -r ./trained_models/check