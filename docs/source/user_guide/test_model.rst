.. _test_model:

Test trained model
==================

After a model is trained you can test how it performs using *test.py*. 

A good practice is to use the copy of the config file used for training, that
was saved to the lodgir together with your trained model: 

``python test.py --config path_to_the_copy_of_the_config_file/config_name.json``

This way, the correct workspace, robot etc. is set and the path to the trained 
model is loaded automatically. The robot starts performing learned task for a given
set of evaluation episodes and some statistics are shown and stored.

You can vary the length of evaluation:

``--eval_episodes=`` number of evaluation episodes to perform

It is possible to record a gif from selected camera during evaluation:

``--record=1 --camera=0``

.. figure:: ../baselines/table_kuka_train_ppo2_500000.gif
   :alt: reach-table-kuka