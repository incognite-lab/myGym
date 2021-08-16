.. _train_with_distractors:

Train with distractors
================

It is possible to add distractors to your training environment and make your task harder for the robot to succesfully complete.

Setup
-----------

To setup parameters of training with distractors you can change theese parameters in config file:

::

   #Distractor
   "distractors"                   : null,
   "distractor_moveable"           : 1,
   "distractor_constant_speed"     : 0,
   "distractor_movement_dimensions": 3,
   "distractor_movement_endpoints" : [-0.3, 0.3, 0.4, 0.7, 0.1, 0.3],
   "observed_links_num"            : 5,


1. .. rubric:: distractors:
      :name: distractors

   Takes in array of distractors to be placed in the training environment, or null when no distractors are desired.

2. .. rubric:: distractor_moveable:
      :name: distractor_moveable

   To let distractor move during the training episodes pass in 1, otherwise 0.

3. .. rubric:: distractor_constant_speed:
      :name: distractor_constant_speed

   Pass in 1 for constant speed of the distractor, 0 for randomly changing speed of distractors during the training episodes.

4. .. rubric:: distractor_movement_dimensions:
      :name: distractor_movement_dimensions

   Number of axis of movement (up to 3)

5. .. rubric:: distractor_movement_endpoints:
      :name: distractor_movement_endpoints

   Borders of distractor's movement 
   ``[Xmin, Xman, Ymin, Ymax, Zmin, Zmax]``

6. .. rubric:: observed_links_num:
      :name: observed_links_num

   Number of links of the robotic arm included into observation. Counted from end effector of the arm. Used for excluding links which can't interfere 
   with the distractor (or any other task related object) from the observation.

Training
-----------

* To succesfully train reach with one immobile distractor we recommend at least 2000000 steps of training.

 *Result of two milion steps long training:*

 .. figure:: ../../../myGym/images/workspaces/static_distractor_left.gif
   :width: 400
   :alt: trained_model

* To train 60% succesfull reach with one moving distractor we recommend at least 15000000 steps of training

 .. figure:: ../../../myGym/images/workspaces/dinamic_distractor.gif
   :width: 400
   :alt: trained_model

* To train 90% successfull reach with one small chaotically moving distractor we recommend at least 500000 steps of training
 
 .. figure:: ../../../myGym/images/workspaces/small_chaotic_distractors.gif
   :width: 400
   :alt: trained_model
