.. _reward_class:

Reward
======

Reward is the key element in Reinforcement Learning. Each training, an instance of reward class is created and serves for reward signal generation during training steps. In myGym Reward Module there are three basic types of reward signal implemented: **Distance**, **Complex Distance** and **Sparse**. You can choose on of them to be used in training by specifying **reward=** eiher **distance**, **complex_distance** or **sparse** in *config file* or pass it as a command line argument.

.. automodule:: myGym.envs.rewards
  :members: