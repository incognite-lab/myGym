.. _dataset:

Generate dataset
=======================

Mygym provides this useful tool to help you generate dataset for custom training of vision models YOLACT and VAE. 
You can configure the dataset by *json file*, where you first specify the general dataset parameters as type, 
number of images to generate etc. Next, specify the environment, choose cameras to use for rendering images and 
set their resolution. Then choose a robot and objects to appear in the scene. You can further control the objects 
quantity, appearance and location.

A very useful feature that eventualy helps you to achieve better performance with trained vision network is myGym's 
randomizer. Randomizer works as a wrapper to your standart myGym environment that enables more advanced setting of 
the scene. Thanks to randomizer, you can change textures of static and dynamic objects and/or their colors. You can 
change light conditions such as light intensity, direction or color. Camera randomizer slightly changes camera properties. 
Joint randomizer enables robots and dynamic objects to change their configuration.

.. note::
  To be able to use texture randomizer, download the texture dataset first:

  ``cd myGym`` ``sh download_textures.sh``

Dataset json file
-----------------

You can see an example *json file* with all parameters here:

.. literalinclude:: ../../../myGym/configs/dataset_example.json

.. automodule:: myGym.generate_dataset
  :members:
