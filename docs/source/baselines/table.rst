.. _table:

Baseline networks
=================
We constantly train networks to provide baselines for the tasks in the toolbox.

If you want to use pretrained baselines models, download them here:

``cd mygym`` ``sh ./utils/download_baselines.sh``

Use ``python test.py`` to see, how the models perform. See :ref:`test_model` for details.

The tables below depict results of baselines models trained using myGym. The 
results are in percentage of successful episodes achieved during evaluation.
The evaluation was done using ``test.py``, performed on 100 episodes.

+--------------+---------------------------------+
|    Reach     |           Ground truth, PPO2    |
+--------------+---------+-----------------------+
|              |  Table  |  Collaborative table  |
+--------------+---------+-----------------------+
| Kuka         |   94%   |          82%          |
+--------------+---------+-----------------------+
| Panda        |   91%   |          73%          |
+--------------+---------+-----------------------+
| Jaco         |   98%   |          86%          |
+--------------+---------+-----------------------+
| UR10         |   91%   |          96%          |
+--------------+---------+-----------------------+

Table:

+---------------------------------------------+-----------------------------------------------+
|.. figure:: table_kuka_train_ppo2_500000.gif | .. figure:: table_panda_train_ppo2_501000.gif |
|   :alt: kuka table                          |    :alt: panda table                          |
+---------------------------------------------+-----------------------------------------------+
|.. figure:: table_jaco_train_ppo2_501000.gif | .. figure:: table_ur10_train_ppo2_500000.gif  |
|   :alt: jaco table                          |    :alt: ur10 table                           |
+---------------------------------------------+-----------------------------------------------+


Collaborative table:

+-----------------------------------------------------+-------------------------------------------------------+
|.. figure:: collabtable_kuka_train_ppo2_500000_2.gif | .. figure:: collabtable_panda_train_ppo2_500000_2.gif |
|   :alt: kuka collabtable                            |    :alt: panda collabtable                            |
+-----------------------------------------------------+-------------------------------------------------------+
|.. figure:: collabtable_jaco_train_ppo2_500000_2.gif | .. figure:: collabtable_ur10_train_ppo2_500000_2.gif  |
|   :alt: jaco collabtable                            |    :alt: ur10 collabtable                             |
+-----------------------------------------------------+-------------------------------------------------------+

+--------------+---------------------------------+
|    Reach     |      Supervised vision, PPO2    |
+--------------+---------+-----------------------+
|              |  Table  |  Collaborative table  |
+--------------+---------+-----------------------+
| Kuka         |         |                       |
+--------------+---------+-----------------------+
| Panda        |         |                       |
+--------------+---------+-----------------------+
| Jaco         |         |                       |
+--------------+---------+-----------------------+ 

