.. _train_switch.rst:

Train a robot - switch
=============
Train robot to switch a lever


Setup
-----------

To train robot with a different behaviour, change values of following coefficients in config file.
Values have to be between 0 - 1.

::

   #Coefficients
   "coefficient_kw"        : 0.1,
   "coefficient_kd"        : 0.1,
   "coefficient_ka"        : 1,

Reward is splitted into 3 parts, each part is multiplied by specific coefficient and at the end of episode summed up.

Coefficient_kw - is multiplied by distance between position of robot's gripper and line - (initial position of robot, final position of robot)

Coefficient_kd - is multiplied by distance between task_object and robot's gripper

Coefficient_ka - is multiplied by angle of switch


Training
-----------

* To train model with default coefficients settings run following command
``python train.py --config ./configs/train_switch.json``

The training will start with gui window and standstill visualization. New directory 
is created in the logdir, where tranining checkpoints, final model and other relevant 
data are stored.

Wait until the first evaluation after 50000 steps to check the progress:

.. figure:: ../../../myGym/images/workspaces/switch/kuka50000.gif
   :alt: training

After 250000 steps the arm is able to switch the lever with 80% accuracy:

.. figure:: ../../../myGym/images/workspaces/switch/kuka250000.gif
   :alt: training