How to train iCub robot

You need to begin knowing that you have the URDF file and the meshes correctly set up, with an end effector and all the joints working properly. And a configuration file in this case for the icub robot is AGMicub.json a file that defines the task, the simulation environment, the observations, the training parameters and the reward.
The task definition tells whats is the goals the inicial object and objetive object. 
The observations tells the object, the objetive and the robot state.
The reward defines how progress is measured. This case uses Euclidean distance, meaning that the robot is rewarded for bringing the object closer to the objetive.
Also you need the train.py that loads the config file, create the environment, inicialices the lerning process and run the training process.
