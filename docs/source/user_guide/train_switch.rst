.. _basic_training:

Train a robot - switch
=============

Run training using following command

``python train.py --robot kuka --reward switch --task_objects switch --task_type switch --gui 1``

The training will start with gui window and standstill visualization. New directory 
is created in the logdir, where tranining checkpoints, final model and other relevant 
data are stored. 

If you want to train robot with the same goal, but with different behaviour, change values of coefficients
in reward.py under class SwitchReward: k_w, k_d, k_a, values have to be <0; 1>.


Wait until the first evaluation after 50000 steps to check the progress:

.. figure:: ../../../myGym/images/workspaces/switch/kuka50000.gif
   :alt: training

After 250000 steps the arm is able to switch the lever with 80% accuracy:

.. figure:: ../../../myGym/images/workspaces/switch/kuka250000.gif
   :alt: training