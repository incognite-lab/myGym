.. _mygym_robots:

Robots
======

The toolkit allows you to add various robots to your training environment. Here are presented the implemented robots. 
You can also add custom robot by placing its model and URDF file into envs/robots directory. For details about how 
myGym works with robots in the simulation, see :ref:`robot`.

============ ======== =============== === ==================
Robot        Type     Gripper         DOF Parameter value
============ ======== =============== === ==================
Kuka IIWA    arm      magnetic        7   kuka
Franka-Emica arm      two finger      6   panda
Jaco arm     arm      two finger      6   jaco
UR-3         arm      tactile gripper 6   ur3
UR-5         arm      tactile gripper 6   ur5
UR-10        arm      tactile gripper 6   ur10
Gummiarm     arm      passive palm    6   gummi
Reachy       arm      passive palm    8   reachy
Leachy       arm      passive palm    8   leachy
ReachyLeachy dualarm  passive palms   16  reachy_and_leachy
ABB Yumi     dualarm  two finger      24  yummi
Pepper       humanoid –               –   –
Thiago       humanoid –               –   –
Atlas        humanoid –               –   –
============ ======== =============== === ==================

KUKA LBR iiwa
-------------

.. image:: kuka.png
  :width: 700
  :alt: KUKA LBR iiwa


FRANKA EMICA Panda
------------------

.. image:: panda.png
  :width: 700
  :alt: FRANKA EMICA Panda


KINOVA JACO
-----------

.. image:: jaco.png
  :width: 700
  :alt: KINOVA JACO


GummiFactory GummiArm
---------------------

.. image:: gummi.png
  :width: 700
  :alt: GummiFactory GummiArm


Universal Robots UR3
--------------------

.. image:: ur3.png
  :width: 700
  :alt: Universal Robots UR3


Universal Robots UR5
--------------------

.. image:: ur5.png
  :width: 700
  :alt: Universal Robots UR5


Universal Robots UR10
---------------------

.. image:: ur10.png
  :width: 700
  :alt: Universal Robots UR10


Pollen Robotics Reachy and Leachy
---------------------------------

.. image:: reachy.png
  :width: 700
  :alt: Pollen Robotics Reachy

.. image:: leachy.png
  :width: 700
  :alt: Pollen Robotics Leachy

.. image:: reachy_and_leachy.png
  :width: 700
  :alt: Pollen Robotics Reachy and Leachy


ABB YuMi
--------

.. image:: yumi.png
  :width: 700
  :alt: ABB YuMi
