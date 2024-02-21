import os

mode = "test"
testconfig = "./configs/train_reach_nico_finger_r.json"
trainfolder = "./trained_models/nico_dualhead_reach/"
eval_episodes = 5
max_velocity = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gui = 1
if mode == "test":
  for velocity in max_velocity:
    os.system("python test.py --config {} --speed -ct IKsolver -ba absolute --eval_episodes {} --gui {} --max_velocity {}".format(testconfig, eval_episodes, gui, velocity))
elif mode == "eval":
  root, dirs, files = next(os.walk(str(trainfolder)))
  for folder in dirs:
    os.system("python test.py --config {}{}/train.json --speed --eval_episodes {} --gui {}".format(trainfolder,folder, eval_episodes, gui))