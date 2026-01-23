
![alt text](myGym/images/mygymlogo310.png "myGym")


We introduce myGym, a toolkit suitable for fast prototyping of neural networks in the area of robotic manipulation and navigation. Our toolbox is fully modular, so that you can train your network with different robots, in several environments and on various tasks. You can also create a curriculum of tasks with increasing complexity and test your network on them. 

From version 3.10 there is SB3 and Gymnasium implemented and there is a basic set of protorewards to create any manipulation task from their combination. Their composition is semi automated and will be fully automated in next release. It is possible to train multiple networks within one task and switch between them based on reward or adaptively. The number of networks is specified in config file.


[![Generic badge](https://img.shields.io/badge/OS-Linux-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Computation-CPU,GPU-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Language-Python:3.7-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Physics-Bullet-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Env-Gym-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Learning-TF,Torch-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Docs-Yes-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Maintained-Yes-green.svg)](https://shields.io/)

## Install myGym 3.10

From myGym 3.10 there is Stable Baseline 3 and Gymnasium. 
If you want to use old myGym 3.7. with Stable Baselines1 and Gym, switch to branch mygym-3.7


Clone the repository:

`git clone https://github.com/incognite-lab/mygym.git`

`cd mygym`

Create Python 3.10 conda env:

`conda create -n mygym Python=3.10`

`conda activate mygym`

Install myGym:

`pip install -e .`

If you face troubles with mpi4py dependency install the lib:

`sudo apt install libopenmpi-dev`


## Utility Functions

### visualize_robot

The `visualize_robot` function provides an interactive visualization tool for robots. It allows you to load and manipulate robot models in a 3D environment with joint sliders to control each joint independently.

**Purpose:**
- Visualize robot URDF models in PyBullet
- Interactively control robot joints using sliders
- Test and save default joint configurations
- Extract end-effector poses for robot setup

**Usage:**

```bash
# Interactive robot selection
python myGym/visualize_robot.py

# Specify robot directly
python myGym/visualize_robot.py -b kuka

# Enable interactive selection explicitly
python myGym/visualize_robot.py -i
```

**Arguments:**
- `-b, --robot`: Robot key from r_dict (e.g., 'kuka', 'panda', 'ur5')
- `-i, --interactive`: Enable interactive robot selection

**Interactive Controls:**
- Use sliders to control each joint
- Press 't' to print current joint values and save them to helpers.py

**Example:**
```bash
# Visualize the Kuka robot
python myGym/visualize_robot.py -b kuka

# Interactive selection from available robots
python myGym/visualize_robot.py -i
```

### visualize_robot_ik

The `visualize_robot_ik` function provides inverse kinematics (IK) testing and visualization for robots. It allows you to test IK solutions by specifying target positions and orientations.

**Purpose:**
- Test inverse kinematics for robot end-effectors
- Visualize reachable workspace
- Control gripper joints
- Verify IK solutions in real-time

**Usage:**

```bash
# Interactive robot and workspace selection
python myGym/visualize_robot_ik.py

# Specify robot directly
python myGym/visualize_robot_ik.py --robot-key panda

# Enable workspace mode
python myGym/visualize_robot_ik.py --workspace
```

**Arguments:**
- `--urdf`: Path to the robot URDF file
- `--interactive`: Interactive selection of robot from registry
- `--robot-key`: Robot key from r_dict to use
- `--workspace`: Workspace mode - select workspace then robot

**Interactive Controls:**
- Use position sliders (X, Y, Z) to move the IK target
- Control gripper joints with dedicated sliders
- Press 'i' to solve IK and print joint values
- Press 't' to save current configuration

**Example:**
```bash
# Test IK for Panda robot
python myGym/visualize_robot_ik.py --robot-key panda

# Interactive workspace and robot selection
python myGym/visualize_robot_ik.py --workspace
```

### taskchecker

The `taskchecker` is a lightweight interactive selector for task/config JSON files. It scans configuration files and allows you to quickly test tasks with different control modes.

**Purpose:**
- Browse and select task configurations
- Quick testing with different control modes (oraculum, keyboard, slider, random)
- Edit and view configuration files
- Test tasks with specific robots

**Usage:**

```bash
# Default - scans ./configs directory
python myGym/taskchecker.py

# Custom directory
python myGym/taskchecker.py --root ./my_configs

# Plain CLI mode (no curses UI)
python myGym/taskchecker.py --no-curses

# Dry run - show command without executing
python myGym/taskchecker.py --dry-run

# Run without GUI
python myGym/taskchecker.py --no-gui
```

**Arguments:**
- `--root`: Directory to scan for JSON files (default: ./configs)
- `--no-curses`: Force plain CLI mode
- `--dry-run`: Show command but do not execute
- `--order`: Order listing by 'alpha' or 'date'
- `--no-gui`: Use `-g 0` instead of `-g 1`
- `--print-only`: Just list found JSON files

**Interactive Commands:**
- `<index>`: Run test with selected config
- `o <index>`: Run with oraculum control mode
- `k <index>`: Run with keyboard control mode
- `s <index>`: Run with slider control mode
- `r <index>`: Run with random control mode
- `v <index>`: View file content
- `e <index>`: Edit file
- `c <index>`: Choose robot
- `q`: Quit

**Example:**
```bash
# Interactive selection from configs directory
python myGym/taskchecker.py

# List all configs and test the first one with oraculum
python myGym/taskchecker.py --order date
# Then press 'o 0' to run with oraculum

# Dry run to see what command would be executed
python myGym/taskchecker.py --dry-run
```

### test

The `test` function allows you to test and visualize tasks without training. You can control the robot using different methods like keyboard, oraculum (automatic solver), or random actions.

**Purpose:**
- Test task configurations before training
- Visualize robot behavior in different scenarios
- Debug task setups and configurations
- Record episodes for analysis

**Usage:**

```bash
# Default test with default config
python myGym/test.py

# Test with specific config
python myGym/test.py --config ./configs/train_A.json -g 1

# Test with oraculum control
python myGym/test.py --config ./configs/train_A.json -ct oraculum -g 1

# Test with keyboard control
python myGym/test.py --config ./configs/train_A.json -ct keyboard -g 1

# Test with specific robot
python myGym/test.py --config ./configs/train_A.json -b kuka -g 1
```

**Key Arguments:**
- `-cfg, --config`: Path to configuration JSON file
- `-ct, --control`: Control mode (keyboard, observation, random, oraculum, slider)
- `-g, --gui`: Use GUI (1 for on, 0 for off)
- `-b, --robot`: Robot to use (e.g., kuka, panda, jaco)
- `-vs, --vsampling`: Visualize sampling area
- `-vt, --vtrajectory`: Visualize gripper trajectory
- `-vn, --vinfo`: Visualize info

**Control Modes:**
- `keyboard`: Manual control using keyboard
- `oraculum`: Automatic task solver
- `slider`: Control with GUI sliders
- `random`: Random actions
- `observation`: Use trained model

**Example:**
```bash
# Test reach task with oraculum
python myGym/test.py --config ./configs/reach_config.json -ct oraculum -g 1 -ba absolute_gripper

# Test with keyboard control and trajectory visualization
python myGym/test.py --config ./configs/pick_place_config.json -ct keyboard -g 1 -vt

# Test with Panda robot using slider control
python myGym/test.py -b panda -ct slider -g 1
```

### train

The `train` function is used to train reinforcement learning agents on robotic manipulation tasks.

**Purpose:**
- Train RL agents using Stable Baselines3
- Support multiple algorithms (PPO, SAC, TD3, A2C)
- Multiprocessing for faster training
- Save models and training logs

**Usage:**

```bash
# Default training
python myGym/train.py

# Train with specific config
python myGym/train.py --config ./configs/train_A.json

# Train with GUI enabled
python myGym/train.py --config ./configs/train_A.json -g 1

# Train with multiprocessing
python myGym/train.py --config ./configs/train_A.json -i 4

# Train for specific number of steps
python myGym/train.py --config ./configs/train_A.json -s 100000
```

**Key Arguments:**
- `-cfg, --config`: Path to configuration JSON file
- `-s, --steps`: Number of training steps
- `-g, --gui`: Use GUI (1 for on, 0 for off)
- `-a, --algo`: Learning algorithm (ppo2, her, sac, td3)
- `-w, --train_framework`: Training framework (pytorch, tensorflow)
- `-i, --multiprocessing`: Enable multiprocessing (specify number of processes)
- `-b, --robot`: Robot to train (kuka, panda, jaco, etc.)
- `-ws, --workspace`: Workspace name (table, drawer, fridge, etc.)
- `-l, --logdir`: Directory for logs and saved models
- `-e, --eval_episodes`: Number of evaluation episodes
- `-ef, --eval_freq`: Evaluation frequency in steps

**Training Frameworks:**
- `pytorch`: PyTorch-based training (Stable Baselines3)
- `tensorflow`: TensorFlow-based training

**Example:**
```bash
# Train PPO agent on pick and place task
python myGym/train.py --config ./configs/pick_place.json -a ppo2 -s 500000

# Train with 4 parallel environments
python myGym/train.py --config ./configs/reach.json -i 4 -s 1000000

# Continue training from pretrained model
python myGym/train.py --config ./configs/train_A.json -ptm ./trained_models/model.zip -s 100000

# Train Kuka robot on table workspace
python myGym/train.py -b kuka -ws table -tt reach -s 200000 -l ./logs/kuka_reach
```

### show

The `show` function is an interactive launcher for viewing training results and testing trained models. It discovers trained model directories and allows you to quickly test them.

**Purpose:**
- Browse trained models
- Visualize training results
- Test trained models interactively
- View evaluation metrics

**Usage:**

```bash
# Default - scans ./trained_models directory
python myGym/show.py

# Custom directory
python myGym/show.py --root ./my_models

# Scan non-recursively
python myGym/show.py --no-recursive

# Plain CLI mode
python myGym/show.py --no-curses

# Print all found models and exit
python myGym/show.py --print-only
```

**Arguments:**
- `--root`: Root directory to scan (default: ./trained_models)
- `--name`: Exact filename to match (default: train.json)
- `--no-recursive`: Disable recursive search
- `--no-curses`: Force CLI mode (disable curses UI)
- `--dry-run`: Do not execute, just show command
- `--no-loop`: Run once and exit
- `--once`: Select & run first entry automatically
- `--print-only`: Print discovered paths and exit
- `--order`: Order listing by 'alpha' or 'date'
- `--no-gui`: Run with GUI disabled

**Interactive Features:**
- Browse trained models with arrow keys
- Select model to test
- View evaluation results inline
- Visualize training progress

**Example:**
```bash
# Interactive selection from trained models
python myGym/show.py

# Show models ordered by date (newest first)
python myGym/show.py --order date

# Test models in specific directory
python myGym/show.py --root ./experiments/pick_place

# List all trained models
python myGym/show.py --print-only
```

## Unit Tests

The repository includes comprehensive unit tests in the `myGym/unittest/` directory to ensure code quality and functionality.

### Available Tests

#### test_robots.py
Tests robot URDF joint limit reachability for all robots in the robot dictionary.

```bash
# Run without GUI
python3 myGym/unittest/test_robots.py

# Run with GUI visualization
python3 myGym/unittest/test_robots.py --gui
```

#### test_robot_reachability.py
Tests robot IK reachability across a 3D volume. Generates a 3D plot of reachable points and bounding box.

```bash
# Interactive robot selection
python3 myGym/unittest/test_robot_reachability.py

# Test specific robot
python3 myGym/unittest/test_robot_reachability.py --robot kuka

# Test with GUI and custom volume
python3 myGym/unittest/test_robot_reachability.py --robot panda --gui --min 0.2 0.2 0.2 --max 0.8 0.8 0.8
```

#### test_train_configs.py
Tests train.py with all configuration files in the `./configs` folder. Reports which configs train successfully.

```bash
# Test all configs (default 10000 steps)
python3 myGym/unittest/test_train_configs.py

# Test with custom step count
python3 myGym/unittest/test_train_configs.py --steps 5000

# Test specific config
python3 myGym/unittest/test_train_configs.py --config train_A_nico.json
```

#### test_oraculum_configs.py
Tests the oraculum (automatic solver) with all configuration files. Verifies task feasibility.

```bash
# Test all configs (5 trials per config)
python3 myGym/unittest/test_oraculum_configs.py

# Test with custom trials
python3 myGym/unittest/test_oraculum_configs.py --trials 10

# Test specific config
python3 myGym/unittest/test_oraculum_configs.py --config train_A.json
```

### Test Output

All tests provide:
- ✔ OK marks for successful tests
- Detailed error messages for failures
- Summary tables with statistics
- Exit codes (0 for success, 1 for failures)

### Running All Tests

```bash
cd myGym/unittest
python3 test_robots.py
python3 test_robot_reachability.py --robot kuka
python3 test_train_configs.py
python3 test_oraculum_configs.py
```

For more details, see `myGym/unittest/README.md`.

## Authors


![alt text](myGym/images/incognitelogo.png "test_work")


[Incognite lab - CIIRC CTU](https://incognite-lab.github.io) 

Core team:

[Michal Vavrecka](https://kognice.wixsite.com/vavrecka)

[Gabriela Sejnova](https://www.linkedin.com/in/gabriela-sejnova/)

[Megi Mejdrechova](https://www.linkedin.com/in/megi-mejdrechova)

[Nikita Sokovnin](https://www.linkedin.com/in/nikita-sokovnin-250939198/)

[Frederik Albl](https://incognite-lab.github.io)

[Sofia Ostapenko](https://incognite-lab.github.io)

[Radoslav Skoviera](https://incognite-lab.github.io)

Contributors:

Peter Basar, Michael Tesar, Vojtech Pospisil, Jiri Kulisek, Anastasia Ostapenko, Sara Thu Nguyen

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
