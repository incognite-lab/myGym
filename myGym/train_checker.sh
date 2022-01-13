#!/bin/sh
#This is minimal unit test to check, whether the current version of myGym is able to train. 
# It will do simle train and evaluate in 2 minutes. If success rate is above 30 percdents, it works.
rm -r ./trained_models/check
python train.py --config ./configs/train_reach.json --algo ppo2 --max_episode_steps 256 --algo_steps 256 --steps 20000 --eval_episodes 10 --logdir ./trained_models/check
python test.py -cfg ./trained_models/check/reach_table_kuka_joints_ppo2/train.json
rm -r ./trained_models/check