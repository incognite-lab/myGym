.. _vision:

Vision
======

MyGym enables you to use pre-trained vision models to extend the versatility of your training scenarios. 
The vision models can be used instead of ground truth data from simulator to retrieve information about 
the environment where robot performs its task. Vision models take simulator's camera data (RGB and/or depth image) 
as inputs to inference and return information about observed scene. Thanks to that, your training becomes 
independent on ground truth from simulator and can be therefore easier transfered to real robot tasks.

MyGym integrated two different vision modules - YOLACT and VAE - and you can alternate between ground truth 
and these when specifying the type of source of reward signal in *config file* or as a command line argument: 
**reward_type=** either **gt** (ground truth) or **3dvs** (YOLACT) or **2dvu** (VAE).

YOLACT
------
Mygym implements YOLACT [1]_ for instance segmantation. If **3dvs** is chosen for **reward_type**, the pre-trained YOLACT 
model is used to get observations from the environment. The input into YOLACT inference is RGB image rendered by 
the active camera, the inference results are masks and bounding boxes of detected objects. The vision module further 
calculates the position of centroids of detected objects in pixel space. Lastly, the vision module utilizes the depth 
image from the active camera to project the object's centroid into 3D worl coordinates. This way, the absolute 
position of task objects is obtained only from sensory data without any ground truth inputs.

The current pre-trained model can detect all :ref:`mygym_objects` and three of :ref:`mygym_robots` including 
their grippers (kuka, jaco, panda). .. todo:: how to download weights.

If you would like to train new YOLACT model, you can use prepared dataset generator available in myGym, 
see :ref:`dataset`. For instructions regarding training itself, visit `YOLACT`_ home page.

.. [1] Daniel Bolya, Chong Zhou, Fanyi Xiao, & Yong Jae Lee (2019). YOLACT: Real-time Instance Segmentation. In ICCV.
.. _YOLACT: https://github.com/dbolya/yolact 

VAE
---


The objective of an unsupervised version of the prepared tasks (reach
task, push task, pick and place etc.) is to minimize the difference
between the actual and goal scene images. To measure their difference,
we have implemented a variational autoencoder (VAE) that
compresses each image into an n-dimensional latent vector. Since the VAE
is optimized so that it preserves similarities among images also in the
latent space (scenes with objects close to each other will have their
encoded vectors also closer to each other), it is possible to measure
the euclidean distance between the encoded scenes and use it for reward
calculation - i.e., the smaller the euclidean distance between actual
and goal image, the higher the reward. Pleas note that the limitation of
using VAE is that it works conveniently only with 2D information - i.e.,
it is a very weak source of visal information in 3D tasks such as pick
and place.

We provide a pretrained VAE for some of the task scenarios, but we also
include code for training of your own VAE (including dataset
generation), so that you can create custom experiments.   .. todo:: how to download weights.

.. automodule:: myGym.envs.vision_module
  :members:
