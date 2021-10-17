.. _train_multimodular:

Train with more networks on one workspace
================

It is possible, to delegate your task into more subtasks each handled by different network in one training.

Setup
-----------

To setup parameters of training with multimodular algorithm you will need to set these parameters in your training config file.

::

   #Train
   "train_framework"   :"tensorflow",
   "algo"              :"multi",
   "diagram"           :[0, 0],
   ...
   "multiprocessing"   :false,


1. .. rubric:: train_framework:
      :name: train_framework

   This feature is implemented only in tensorflow framework yet.

2. .. rubric:: algo:
      :name: algo

   THere is stated, that it is derired to delegate the task into multiple subtasks.

3. .. rubric:: diagram:
      :name: diagram

   This list states how many networks will be used and what is their observation space (0 stands for deafult which is set by environment).

4. .. rubric:: multiprocessing:
      :name: multiprocessing

   this is archaic leftover which you should not use if you want everything to work.

Training
-----------

* To succesfully train poke with two networks, one for aiming and second for poking, with static goal positions, we recommend at least 500000 steps of training.

 *Result of two four milion steps long training:*

 .. figure:: ../../../myGym/images/workspaces/static_poke.gif
   :width: 400
   :alt: trained_model

* To succesfully train poke with two networks, one for aiming and second for poking, with dinamic goal positions, we recommend at least 20000000 steps of training.

 *Result of twenty milion steps long training:*

 .. figure:: ../../../myGym/images/workspaces/dinamic_poke.gif
   :width: 400
   :alt: trained_model

* To train 100% successfull pick and place we recommend 100000 steps with 2 networks and 1000000 steps with 3 networks when gripper is not part of NN astions space.

 *Result of one hundred thousand steps long training:*

 .. figure:: ../../../myGym/images/workspaces/pick_and_place.gif
   :width: 400
   :alt: trained_model

* To train XXX% successfull pick and place we recommend 3000000 steps with 3 networks and 1000000 steps with 3 networks when gripper IS part of NN astions space.

 *Result of one hundred thousand steps long training:*

 .. figure:: ../../../myGym/images/workspaces/pick_and_place.gif
   :width: 400
   :alt: trained_model
