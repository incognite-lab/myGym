
![alt text](myGym/images/mygymlogo.png "myGym")



We introduce myGym, a toolkit suitable for fast prototyping of neural networks in the area of robotic manipulation and navigation. Our toolbox is fully modular, so that you can train your network with different robots, in several environments and on various tasks. You can also create a curriculum of tasks  with increasing complexity and test your network on them. We also included an automatic evaluation and benchmark tool for your developed model. We have pretained the Yolact network for visual recognition of all objects in the simulator, so that you can reward your networks based on visual sensors only. 

We keep training the current state-of-the-art algorithms to provide baselines for the tasks in the toolbox. There is also a leaderboard showing algorithms with the best generalization capability, tested on the tasks in our basic curriculum. From version 2.0 it is possible to train multiple networks within one task and switch between them based on reward or adaptively. The number of neteworks is specified in config file.


[![Generic badge](https://img.shields.io/badge/OS-Linux-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Computation-CPU,GPU-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Language-Python:3.7-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Physics-Bullet-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Env-Gym-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Learning-TF,Torch-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Docs-Yes-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Maintained-Yes-green.svg)](https://shields.io/)


## Advantages

* [Separate modules](https://mygym.readthedocs.io/en/latest/index.html) for fast prototyping (task.py, reward.py, env.py)

![alt text](myGym/images/schemas/mygym_scheme.png "scheme")

* [Pretrained vision](https://mygym.readthedocs.io/en/latest/user_guide/train_camera.html) for instance wise semantic segmentation

![alt text](myGym/images/results/kuka_reach_yolact_fast.gif "yolact")

* [Customizable distractors](https://mygym.readthedocs.io/en/latest/user_guide/train_with_distractors.html) for real enviroment simulation.

![alt text](myGym/images/workspaces/small_chaotic_distractors.gif "training")

* Human-robot collaboration environments (WIP)

![alt text](myGym/images/workspaces/panda_push.gif "training")



From version 2.1:

* Multi-step tasks defined inside [config file](myGym/configs/train_pnp_2n_multitask4.json#L20)  with customizable [observations](myGym/configs/train_pnp_2n_multitask4.json#L30) 

![alt text](myGym/images/workspaces/multireach_jaco.gif "Multistep")

* Multi-goal rewards for training long horizon [tasks](myGym/envs/rewards.py#L1365)

![alt text](myGym/images/workspaces/pnp/pnp3n3x_kuka.gif "Multireward")

* REAL robotic gripping based on [friction](myGym/envs/robots/franka_emika/panda/urdf/panda1.urdf) or [containment](myGym/envs/robots/franka_emika/panda/urdf/panda_cgripper.urdf)

![alt text](myGym/images/workspaces/pnp/pnp3n3x_panda_boxgripper.gif "Realgrip")

* Multi-network training - three networks switching in Pick and rotate task

![alt text](myGym/images/workspaces/train_multi_5000000.gif "Pnr")


## Installation

Clone the repository:

`git clone https://github.com/incognite-lab/mygym.git`

`cd mygym`

Create Python 3.7 conda env (later Python versions does not support TF 0.15.5 neccesary for Stable baselines ):

`conda create -n  mygym Python=3.7`

`conda activate mygym`

Install myGym:

`python setup.py develop`

If you face troubles with mpi4py dependency install the lib:

`sudo apt install libopenmpi-dev`

If you want to use the pretrained visual modules, please download them first:

`cd myGym`
`sh download_vision.sh`

If you want to use the pretrained baseline models, download them here:

`cd myGym`
`sh download_baselines.sh`

Check, whether the toolbox works:

`sh ./speed_checker.sh`

If everything is correct, the toolbox will train for two minutes without GUI and then shows the test results (at least 30% success rate)

## Overview


| Environment  | Gym-v0 is suitable both single-step and multi-step manipulation and navigation|
|---|---|
| Workspaces | Table, Collaborative table, Maze, Vertical maze, Drawer, Darts, Football, Fridge, Stairs, Baskets |
| Vision  | Cartesians, RGB, Depth, Class, Centroid, Bounding Box, Semantic Mask, Latent Vector |
| Robots  | 9 robotic arms, 2 dualarms, humanoid |
| Robot actions  | Absolute, Relative, Joints |
| Objects  | 54 objects in 5 categories |
| Tasks  | Reach, Press, Switch, Turn, Push, Pick, Place, PicknPlace, Poke,MultiReach, MultiPNP|
| Randomizers  | Light, Texture, Size, Camera position |
| Baselines  | Tensorflow, Pytorch |
| Physics  | Bullet, Mujoco deprecated from version 2.0 |




Learn more about the toolbox in our [documentation](https://mygym.readthedocs.io/en/latest/)


## Test the environments prior training

You can visualize the virtual gym env prior to the training. 

`python test.py`

There will be the default workspace activated.  

EXPERIMENTAL - You can control the robot and gripper from keyboard (arrows and A and Z for third axis in caartesian), spawn object to test the task (WIP)

![alt text](myGym/images/workspaces/gym_table_test2.png "test_work")

There are also visual outputs from the active cameras (both RGB and Depth):

![alt text](myGym/images/workspaces/gym_table_test_cameras.png "test_work")

Find more details about this function in the [documentation](https://mygym.readthedocs.io/en/latest/user_guide/visualization.html)

## Training

Run the default training without specifying the parameters:

`python train.py`

The training will start with the GUI window and a standstill visualization. Wait until the first evaluation to check the progress: 

There are more training tutorials in the [documentation](https://mygym.readthedocs.io/en/latest/user_guide/basic_training.html)


### Single-step tasks

### [Reach](myGym/images/workspaces/kuka100000.gif "training")

``python train.py --config ./configs/train_reach.json``

### [Press](myGym/images/workspaces/press/kuka500000.gif "training")

``python train.py --config ./configs/train_press.json``

For details see [documentation](https://mygym.readthedocs.io/en/latest/user_guide/train_press.html)


### [Switch](myGym/images/workspaces/switch/kuka250000.gif)

``python train.py --config ./configs/train_switch.json``

For details see [documentation](https://mygym.readthedocs.io/en/latest/user_guide/train_switch.html)

### [Turn](myGym/images/workspaces/turn/kuka500000.gif)

``python train.py --config ./configs/train_turn.json``

For details see [documentation](https://mygym.readthedocs.io/en/latest/user_guide/train_turn.html)

### Push (WIP)

``python train.py --config ./configs/train_push.json``

### Poke (WIP)

``python train.py --config ./configs/train_poke.json``

### Pick and Place

``python train.py --config ./configs/train_pnp_1n.json``

## Multi-step tasks

### [Multi reach](myGym/images/workspaces/multireach_jaco.gif)

``python train.py --config ./configs/train_reach_multitask.json``

### [Multi PNP](myGym/images/workspaces/pnp/pnp3n2m_panda1.gif)

``python train.py --config ./configs/train_pnp_3n_multitask2.json``

### [Build a tower](myGym/images/workspaces/pnp/pnp3n3x_kuka.gif)

``python train.py --config ./configs/train_pnp_3n_multitask4.json``



##  Parametric Training

As myGym is modular, you can easily train with different robots:

`python train.py --robot jaco`

You can also change the workspace within the gym, the task or the goal object. If you want to store an ouput video, just add the record parameter:

`python train.py  --workspace collabtable --robot panda --task push --task_objects wrench --record 1`

Learn more about the simulation parameters in the [documentation](https://mygym.readthedocs.io/en/latest/user_guide/tutorial_parametric.html)


## Parallel Training

We have developed scripts for parallel training to speed up this process. You can edit the desired parameter in train_parallel.py and run it:

`python train_parallel.py`


## Pretrained models

You can use the test script for the visualization of pretrained models:

`python test.py --config ./trained_models/yourmodel/train.json`

It will load the pretrained model and test it in the task and workspace defined in the config file.


## Evaluation

There is automatic evaluation and logging included in the train script. It is controlled by parameters --eval_freq and --eval_episodes. The log files are stored in the folder with the trained model and you can easily visualize the learning progress after the training. There are also gifs for each eval period stored to compare the robot performance during training. We have also implemented evaluation in tensorboard:

`tensorboard --logdir ./trained_models/yourmodel`

![alt text](myGym/images/results/tensorboard_eval.png "training")

If you want to interactively compare different parameters, just run tensorboard without model dir specification:

![alt text](myGym/images/results/tensorboard_compare.png "training")


There are also other visualization scripts (Documerntation in preparation)

## Environment

As myGym allows curriculum learning, the workspaces and tasks are concentrated in single gym, so that you can easily transfer the robot. The basic environment is called Gym-v0. There are more gyms for navigation and multi-agent collaboration in preparation. 

## Robots

| Robot  |Type| Gripper  | DOF  | Parameter value |
|---|---|---|---|---|
| UR-3  | arm  | no gripper  |  6 | ur3  |
| UR-5  | arm  | no gripper  |  6 | ur5  |
| UR-10  | arm  | no gripper  |  6 | ur10  |
| Kuka IIWA | arm  | magnetic, gripper  |  6 |  kuka |
| Reachy  | arm  | passive palm  |  7 | reachy  |
| Leachy  | arm  | passive palm  |  7 | leachy  |
|  Franka-Emica | arm  | gripper  | 7  |  panda |
| Jaco arm  | arm  |  two finger |  13 | jaco  |
| Gummiarm  | arm  | passive palm  |  13 | gummi  |
| Human Support Robot (HSR) | arm | gripper | 7 | hsr |
| ABB Yumi  | dualarm  | two finger  |  12 | yummi  |
| ReachyLeachy  | dualarm  | passive palms  |  14 | reachy_and_leachy |
| Pepper  | humanoid | --  |  20 | WIP  |
| Thiago | humanoid  | --  |  19 | WIP  |
| Atlas  | humanoid  | --  |  28 | WIP  |

## Workspaces

| Name  |Type| Suitable tasks | Parameter value |
|---|---|---|---|
| Tabledesk | manipulation  | Reach,Press, Switch, Turn, PicknPlace | table |
| Drawer | manipulation  | Pick, Place, PicknPlace  | drawer |
| Fridge | manipulation  | Push, Open, Close, Pick | fridge |
| Baskets | manipulation  | Throw, Hit | baskets |
| Darts | manipulation  | Throw, Hit | darts |
| Football | manipulation  | Throw, Hit | football |
| Collaborative table | collaboration  | Give, Hold, Move together | collabtable |
| Vertical maze | planning  | -- | veticalmaze |
| Maze | navigation  | -- | maze |
| Stairs | navigation  | -- | stairs |

## Baselines

| Workspace  |Reach| Pick | Place | PicknPlace | Push | Press | Switch | Turn | Poke | Multistep PNP | Multistep Reach |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tabledesk | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Drawer | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Collaborative table | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

## Learnability 

The new global evaluation metric, which we call \textit{learnability}, allows the user to evaluate and compare algorithms in a more systematic fashion. Learnability is defined as a general ability to learn irrespective of environmental conditions. The goal is to test an algorithm with respect to the complexity of environment. We have decomposed the environment complexity into independent scales. The first scale is dedicated to the complexity of the task. Second scale exploits the complexity of the robotic body that is controlled by the neural network. The third scale stands for the temporal complexity of the environment. 

![alt text](myGym/images/schemas/multieval_visualization.png "learnability")

## Leaderboard

Learnability is represented as a single value metric that evaluates algorithms under various conditions, allowing us to compare different RL algorithms. The number of conditions is limited for practical reasons, as the number of training configurations grows exponentially with each new condition, and each configuration requires standalone training and evaluation. Therefore, we limited the total number of combinations to $3^3$ = 27, which can be evaluated in few hours with a standard computer infrastructure.

| Pos. | Algorhitm | Score |
|---|---|---|
|1.| PPO2 | 30.11  |
|2.| TRPO | 28.75  |
|3.| ACKTR | 27.5  |
|4.| SAC | 27.43 |
|5.| PPO | 27.21 |
|5.| myAlgo | 15.00  |

## Authors


![alt text](myGym/images/incognitelogo.png "test_work")


[Incognite lab - CIIRC CTU](https://incognite-lab.github.io) 

Core team:

[Michal Vavrecka](https://kognice.wixsite.com/vavrecka)

[Gabriela Sejnova](https://www.linkedin.com/in/gabriela-sejnova/)

[Megi Mejdrechova](https://www.linkedin.com/in/megi-mejdrechova)

[Nikita Sokovnin](https://www.linkedin.com/in/nikita-sokovnin-250939198/)

Contributors:

Radoslav Skoviera, Peter Basar, Michael Tesar, Vojtech Pospisil, Jiri Kulisek, Anastasia Ostapenko, Sara Thu Nguyen

## Citation

'@INPROCEEDINGS{9643210,
  author={Vavrecka, Michal and Sokovnin, Nikita and Mejdrechova, Megi and Sejnova, Gabriela},
  
  
  booktitle={2021 IEEE 33rd International Conference on Tools with Artificial Intelligence (ICTAI)}, 
  
  
  title={MyGym: Modular Toolkit for Visuomotor Robotic Tasks}, 
  
  
  year={2021},
  volume={},
  number={},
  pages={279-283},
  
  
  doi={10.1109/ICTAI52525.2021.00046}}'

## Paper

[myGym: Modular Toolkit for Visuomotor Robotic Tasks](https://arxiv.org/abs/2012.11643)
