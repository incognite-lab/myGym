.. _train_multimodular:

Train with more networks on one workspace
================

It is possible to delegate your task into more subtasks each handled by different network in one training.

Setup
-----------

To setup parameters of training with multimodular algorithm you will need to set these parameters in your training config file.

::

   #Train
   "algo"              :"multi",
   "num_networks"      :2,

The number of networks can be larger than 2, but you need to write your own reward for that to implement network switching.

Training
-----------

* To succesfully train poke with two networks, one for aiming and second for poking, with static goal positions, we recommend at least 500000 steps of training.

 *Results after a four million steps:*

 .. figure:: ../../../myGym/images/workspaces/static_poke.gif
   :width: 400
   :alt: trained_model

* To succesfully train poke with two networks, one for aiming and second for poking, with dynamic goal positions, we recommend at least 20000000 steps of training.

 *Result after twenty million steps:*

 .. figure:: ../../../myGym/images/workspaces/dinamic_poke.gif
   :width: 400
   :alt: trained_model

* To train a successful pick and place, we recommend 100000 steps with 2 networks and 1000000 steps with 3 networks when gripper is not part of the action space.

 *Result of one hundred thousand steps long training:*

 .. figure:: ../../../myGym/images/workspaces/pick_and_place.gif
   :width: 400
   :alt: trained_model
