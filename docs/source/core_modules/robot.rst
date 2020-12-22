.. _robot:

Robot
=====

The Robot Module in myGym takes care of interaction between robot and environment. In this module, *actions* received from trained model are executed on simulated robot. You can specify name of a robot you would like to train and Robot module will add it to your environment. This module automatically loads robot's configuration and does the robot control for you. The robots are equiped with various types of grippers as well. 

Currently available robots are: **kuka**, **panda**, **jaco**, **reachy**, **leachy**, **reachy_and_leachy**, **gummi**, **ur3**, **ur5**, **ur10**, **yumi**. Note that for jaco and gummi with complex grippers there are also versions with fixed gripper joints **jaco_fixed** and **gummi_fixed**. See more details in :ref:`mygym_robots`.

.. automodule:: myGym.envs.robot
  :members:
