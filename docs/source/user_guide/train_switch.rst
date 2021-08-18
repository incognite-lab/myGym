.. _train_switch.rst:

Train a robot - switch
=============

Setup
-----------

To train robot with a different behaviour, change values of following coefficients in config file.
Values have to be between 0 - 1.

::

   #Coefficients
   "coefficient_kw"        : 0.6,
   "coefficient_kd"        : 0.5,
   "coefficient_ka"        : 0.5

Reward is splitted into 3 parts, each part is multiplied by specific coefficient and at the end of episode summed up.

Coefficient_kw - multiplies distance between position of robot's gripper and line - (initial position of robot, final position of robot)

Coefficient_kd - multiplies distance between task_object and robot's gripper

Coefficient_ka - multiplies angle of switch


Training
-----------

Run training using following command

``python train.py --robot kuka --reward switch --task_objects switch --task_type switch --gui 1``

The training will start with gui window and standstill visualization. New directory 
is created in the logdir, where tranining checkpoints, final model and other relevant 
data are stored.

Wait until the first evaluation after 50000 steps to check the progress:

.. figure:: ../../../myGym/images/workspaces/switch/kuka50000.gif
   :alt: training

After 250000 steps the arm is able to switch the lever with 80% accuracy:

.. figure:: ../../../myGym/images/workspaces/switch/kuka250000.gif
   :alt: training