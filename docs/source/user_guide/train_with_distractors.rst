.. _train_with_distractors:

Train with distractors
================

You can add distractors to your training environment and make your task harder for the robot to succesfully complete.

Setup
-----------

To setup parameters of training with distractors you can change theese parameters in config file:

#Distractor
"distractors"                   : null,
"distractor_moveable"           : 1,
"distractor_constant_speed"     : 0,
"distractor_movement_dimensions": 3,
"distractor_movement_endpoints" : [-0.3, 0.3, 0.4, 0.7, 0.1, 0.3],
"observed_links_num"            : 5,


1. distractors:

   Takes in array of distractors you want to place in the training environment, or null when no distractors are desired.

2. distractor_moveable:

   If you want the distractor to move during the training episodes pass in 1, otherwise 0.

3. distractor_constant_speed:

   Pass in 1 for constant speed of the distractor, 0 for randomly changing speed of distractors during the training episodes.

4. distractor_movement_dimensions:

   Number of axis of movement (up to 3)

5. distractor_movement_endpoints:

   Borders of distractors movement (Xmin, Xman, Ymin, Ymax, Zmin, Zmax)

6. observed_links_num:

   Number of links of the robotic arm included into observation. Counted from end effector of the arm. Used for excluding links which can't interfere 
   with the distractor (or any other task related object) from the observation.

Training
-----------

To succesfully train reach with one immobile distractor you need 2000000 steps of training.
To train 60% succesfull reach with one moving distractor you need 15000000 steps of training
to train 90% successfull reach with one small chaotically moving distractor you need 500000 steps of training