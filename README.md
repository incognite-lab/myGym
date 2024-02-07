
![alt text](myGym/images/mygymlogo.png "myGym")


We introduce myGym, a toolkit suitable for fast prototyping of neural networks in the area of robotic manipulation and navigation. Our toolbox is fully modular, so that you can train your network with different robots, in several environments and on various tasks. You can also create a curriculum of tasks  with increasing complexity and test your network on them. We also included an automatic evaluation and benchmark tool for your developed model. We have pretained the Yolact network for visual recognition of all objects in the simulator, so that you can reward your networks based on visual sensors only. 


[![Generic badge](https://img.shields.io/badge/OS-Linux-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Computation-CPU,GPU-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Language-Python:3.7-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Physics-Bullet-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Env-Gym-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Learning-TF,Torch-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Docs-Yes-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Maintained-Yes-green.svg)](https://shields.io/)

## Learn more about the toolbox in our [documentation](https://mygym.readthedocs.io/en/latest/)


# Nico - sim2reach branch 


This branch allows to control real Nico robot from myGym simulator both manually and pretrained RL networks. This is description how to run the code on real Nico robot. Without real robot you can train RL algorithms in simulator and deploy them on real Nico robot. This is the last version dependent on Python 3.7 and Tensorflow v1. The new version is WIP.


## Installation

### myGym simulator

Clone the repository:

`git clone https://github.com/incognite-lab/mygym.git`

`cd mygym`

Create Python 3.7 conda env as follows (later Python versions does not support TF 0.15.5 neccesary for Stable baselines ):

`conda env create -f environment.yml `

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

`sh ./train_checker.sh`

If everything is correct, the toolbox will train for two minutes without GUI and then shows the test results (at least 30% success rate)

### Real Nico robot software

Setup communication ports:

`sudo adduser $USER dialout`

`sudo chmod 777 /dev/ttyACM*`

`sudo apt-get install setserial`


Install modified Pypot:

`git clone https://github.com/knowledgetechnologyuhh/pypot.git` 

`cd pypot`

`python setup.py install`

Install Nico software

`git clone https://github.com/knowledgetechnologyuhh/NICO-software.git` 

`cd Nico-software\api\src\nicomotion`

`python setup.py install`

`cd Nico-software\api\src\nicovision`

`python setup.py install`


Patch the pypot and nicomotion (copu files from the link to the directories at conda/envs/mygym/lib/python3.7/site_packages)

[Patches](https://github.com/andylucny/nico/tree/main/nicogui-pypot/patches)

# Usage 


## Test the standalone simulator

You can visualize the virtual gym env. 

`python test.py`

There will be the Nico robot at the standard touchscreen setup and you can control eah joint with the slider (sim2real is off)


## Training RL algorithms

You can train the basic reaching task.

`python train.py`

After the training you can see the trained behavior.

`python test.py --config ./trained_models/reach/[modelpath]/train.json --gui 1`


There are more training tutorials in the [documentation](https://mygym.readthedocs.io/en/latest/user_guide/basic_training.html)


## Sim2real

If the robot is connected to the computer you can control it.

`python sim2real.py`


There will be visualization of the robot and you can control it by moving the joint sliders

If you want to control the robot by draging the finger with mouse or IK:

`python sim2real.py -ba absolute`

If you want to control end effector from keyboard (arrows + AZ keys for z axis):

`python sim2real.py -ba absolute -ct keyboard`

If you want to run pretrained RL behavior on real robot:

WIP


# Howto

Change the robot body (different gripper or hand, fixing or enabling body joints)

Edit this file





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
