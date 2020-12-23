.. _create_workspace:

Create custom workspace
=======================

MyGym toolkit is designed to be modular and so is the environment 
:ref:`gym_env`. It includes several :ref:`workspace` among which you
can easily alternate to add variability to your training. This tutorial
shows how to create and add your custom workspace into Gym and train
your task there.

The definition of :ref:`gym_env` is `here <https://github.com/incognite-lab/myGym/blob/master/myGym/envs/gym_env.py>`_. There you can find the definitions of existing
workspaces in the *self.workspace_dict* dictionary. 

Choose a key for
your workspace and add new value with its definition into this 
*self.workspace_dict*:

.. code-block:: python

    self.workspace_dict =  {'myworkspace':  {'urdf': 'myworkspace.urdf', 'texture': 'mytexture.jpg',
                                            'transform': {'position':[1.5, -1.0, -1.05], 'orientation':[0.0, 0.0, -0.5*np.pi]},
                                            'robot': {'position': [0.0, 0.0, 0.0], 'orientation': [0.0, 0.0, 0.5*np.pi]}, 
                                            'camera': {'position': [[0.56, -1.71, 0.6], [-1.3, 3.99, 0.6], [-3.43, 0.67, 1.0], [2.76, 2.68, 1.0], [-0.54, 1.19, 3.4]], 
                                                        'target': [[0.53, -1.62, 0.59], [-1.24, 3.8, 0.55], [-2.95, 0.83, 0.8], [2.28, 2.53, 0.8], [-0.53, 1.2, 3.2]]},
                                            'boarders':[-0.7, 0.7, 0.3, 1.3, -0.9, -0.9]},
                                            ...}

You need to provide name of the urdf definition file of the model of 
your workspace and optionally a path to the texture you want to apply 
on the model. The urdf should contain correct position, orientation and 
scale of the model, to be loaded into the Gym properly. You will need 
two urdf definition files - one with visual and collision definitions 
and one with visual definition only. Store these files to the proper 
locations: `collision <https://github.com/incognite-lab/myGym/tree/master/myGym/envs/rooms/collision>`_ 
and `visual <https://github.com/incognite-lab/myGym/tree/master/myGym/envs/rooms/visual>`_
respectively. This way you ensure that the collision detection will 
work during training in your custom workspace and that a visual copy
of your workspace will be visible while training in another.

To make things easier, we recommend to place the origin of the
Cartesian coordinate frame to the position and orientation known with
respect to the workspace. Use the transform value for this purpose.

Define position and orientation within your workspace, where the robot
should be initialized. If you applied transform, use the new coordinates
now. 

Define position and target position of cameras accordingly. The cameras
are used for visualization and image rendering for vision modules.

Lastly, define volume within your workspace, where task objects are 
allowed to occur. You can also adjust this parameter later using 
*object_sampling_area* parameter.

To start your custom training, set your new workspace 
keyword as the *workspace* parameter in the config file:

``"workspace":"myworkspace",``

or in the command line:

``python train.py --workspace=myworkspace``