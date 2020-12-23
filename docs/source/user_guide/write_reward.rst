.. _write_reward:

Create custom reward
====================

Reward function is the essential element of Reinforcement learning mechanism. 
The way how the reward signal is composed may depend on the type of learned task,
task complexity, available agent's observation, learning algorithm etc. There are 
three basic reward functions prepared in myGym, see :ref:`reward_class`, and this 
tutorial shows how to create a custom one.

The definition of :ref:`reward_class` class is `here <https://github.com/incognite-lab/myGym/blob/master/myGym/envs/rewards.py>`_, it 
reads two input arguments - *env* object and *task* object, which are initialized
at the beginning of each training and contain useful characteristics of the 
particular learning session. :ref:`reward_class` class has implemented visualization 
methods, *compute* and *reset* method.

Create your custom reward as a child of :ref:`reward_class` class:

.. code-block:: python

    class MyReward(Reward):
    def __init__(self, env, task):
        super(MyReward, self).__init__(env, task)

and define custom *compute* and *reset* methods:

.. code-block:: python

    def compute(self, observation):
        reward = *your function*
        return reward

    def reset(self):
        *reset any variable if needed*

Choose a keyword for your custom reward and crete an instance of **MyReward** 
during the initialization of training environment :ref:`gym_env` 
here TODO:ref to mygym/envs/gym_env.py:

.. code-block:: python

    if reward == 'myreward':
        self.reward = MyReward(env=self, task=self.task)

The method *compute* is called by :ref:`gym_env` during each execution of *step*,
the *reset* method is called at the end of an episode by the *reset* method 
of :ref:`gym_env`.

To launch a training with your custom reward, pass your keyword as the reward
argument either in the *config file*:

``"reward": "myreward",``

or in the command line:

``python train.py --reward=myreward``

.. Note:
    In case you would need to customize agent's observation for the computation of
    a more complex reward signal, have a look at the *get_observation* method in
    :ref:`gym_env`.
